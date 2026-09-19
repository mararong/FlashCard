#!/usr/bin/env python3
"""Build a resumable English vocabulary JSON database."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from ai_providers import AIProviderError, DifficultyEvaluator, OllamaEvaluator

try:
    from nltk.corpus import wordnet as wn
    from wordfreq import top_n_list, zipf_frequency
except ImportError as exc:  # Give a concise setup message instead of a long traceback.
    raise SystemExit(
        f"Missing dependency: {exc.name}. Run: pip install -r requirements.txt"
    ) from exc


# Defaults are intentionally centralized so the full build is easy to tune.
DEFAULT_CANDIDATE_COUNT = 30_000
DEFAULT_LIMIT = 10_000
DEFAULT_BATCH_SIZE = 30
DEFAULT_MAX_SENSES = 8
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_AI_RETRIES = 3
DEFAULT_AI_TIMEOUT_SECONDS = 120.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "english"
VOCABULARY_PATH = DATA_DIR / "english_vocabulary.json"
REJECTED_PATH = DATA_DIR / "rejected_words.json"
ENGLISH_WORD_RE = re.compile(r"^[A-Za-z]+$")

POS_NAMES = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "adjective",
    "r": "adverb",
}


class PipelineError(RuntimeError):
    """A clear, expected pipeline failure."""


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON without risking a half-written checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise PipelineError(f"Expected a JSON array of objects in {path}")
    return value


def frequency_score(zipf: float) -> int:
    if zipf >= 5.0:
        return 0
    if zipf >= 4.0:
        return 1
    if zipf >= 3.0:
        return 2
    return 3


def level_from_score(score: int) -> int:
    if not 0 <= score <= 10:
        raise ValueError(f"Difficulty score out of range: {score}")
    if score <= 3:
        return 1
    if score <= 6:
        return 2
    return 3


def _display_lemma(name: str) -> str:
    return name.replace("_", " ")


def collect_wordnet_data(word: str, max_senses: int) -> dict[str, Any] | None:
    """Collect the leading WordNet senses and normalized dictionary fields."""
    synsets = wn.synsets(word)
    if not synsets:
        return None

    senses: list[dict[str, Any]] = []
    synonyms: list[str] = []
    antonyms: list[str] = []
    seen_synonyms: set[str] = set()
    seen_antonyms: set[str] = set()

    for synset in synsets[:max_senses]:
        sense_synonyms: list[str] = []
        sense_antonyms: list[str] = []
        for lemma in synset.lemmas():
            synonym = _display_lemma(lemma.name())
            if synonym.lower() != word.lower() and synonym.lower() not in seen_synonyms:
                seen_synonyms.add(synonym.lower())
                synonyms.append(synonym)
            if synonym not in sense_synonyms:
                sense_synonyms.append(synonym)
            for antonym_lemma in lemma.antonyms():
                antonym = _display_lemma(antonym_lemma.name())
                if antonym.lower() not in seen_antonyms:
                    seen_antonyms.add(antonym.lower())
                    antonyms.append(antonym)
                if antonym not in sense_antonyms:
                    sense_antonyms.append(antonym)
        senses.append(
            {
                "part_of_speech": POS_NAMES.get(synset.pos(), synset.pos()),
                "definition_en": synset.definition(),
                "examples": list(synset.examples()),
                "synonyms": sense_synonyms,
                "antonyms": sense_antonyms,
            }
        )

    if not senses or not any(sense["definition_en"] for sense in senses):
        return None

    parts_of_speech = list(dict.fromkeys(sense["part_of_speech"] for sense in senses))
    definitions = [sense["definition_en"] for sense in senses]
    examples = list(
        dict.fromkeys(example for sense in senses for example in sense["examples"])
    )
    meanings = [
        {
            "part_of_speech": sense["part_of_speech"],
            "definition_en": sense["definition_en"],
            "examples": sense["examples"],
        }
        for sense in senses
    ]
    return {
        "word": word,
        "parts_of_speech": parts_of_speech,
        "definitions": definitions,
        "senses": senses,
        "meanings": meanings,
        "examples": examples,
        "synonyms": synonyms,
        "antonyms": antonyms,
    }


def upsert_rejection(
    rejected: list[dict[str, Any]],
    word: str,
    stage: str,
    reason: str,
    retryable: bool,
) -> None:
    rejected[:] = [
        item
        for item in rejected
        if not (item.get("word") == word and item.get("stage") == stage)
    ]
    rejected.append(
        {
            "word": word,
            "stage": stage,
            "reason": reason,
            "retryable": retryable,
        }
    )


def remove_ai_rejection(rejected: list[dict[str, Any]], word: str) -> None:
    rejected[:] = [
        item
        for item in rejected
        if not (item.get("word") == word and item.get("stage") == "ai_evaluation")
    ]


def generate_candidates(
    candidate_count: int,
    limit: int,
    max_senses: int,
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate, validate, and enrich up to ``limit`` distinct candidates."""
    if candidate_count < 1 or limit < 1 or max_senses < 1:
        raise PipelineError("candidate-count, limit, and max-senses must be positive")

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_word in top_n_list("en", candidate_count):
        word = raw_word.lower()
        if word in seen:
            continue
        seen.add(word)
        if len(word) < 2 or not ENGLISH_WORD_RE.fullmatch(word):
            continue
        wordnet_data = collect_wordnet_data(word, max_senses)
        if wordnet_data is None:
            upsert_rejection(
                rejected,
                word,
                "wordnet",
                "No usable WordNet definition",
                False,
            )
            continue
        candidates.append(wordnet_data)
        if len(candidates) >= limit:
            break
    return candidates


SCORE_RANGES = {
    "abstractness": {0, 1, 2},
    "semantic_complexity": {0, 1, 2},
    "form_complexity": {0, 1},
    "register": {0, 1, 2},
}


def validate_evaluations(
    batch: list[dict[str, Any]], raw_evaluations: Any
) -> dict[str, dict[str, int]]:
    """Validate exact input coverage while ignoring model-invented words."""
    if not isinstance(raw_evaluations, list):
        raise AIProviderError("AI evaluation must be a JSON array")
    expected_words = {item["word"] for item in batch}
    validated: dict[str, dict[str, int]] = {}
    errors: list[str] = []

    for item in raw_evaluations:
        if not isinstance(item, dict):
            errors.append("array item is not an object")
            continue
        word = item.get("word")
        if word not in expected_words:
            continue  # Explicitly ignore hallucinated additions.
        if word in validated:
            errors.append(f"duplicate word: {word}")
            continue
        scores: dict[str, int] = {}
        for field, allowed in SCORE_RANGES.items():
            value = item.get(field)
            # bool is an int subclass, but is not an acceptable score here.
            if type(value) is not int or value not in allowed:
                errors.append(f"{word}.{field} must be one of {sorted(allowed)}")
                break
            scores[field] = value
        else:
            validated[word] = scores

    missing = expected_words - validated.keys()
    if missing:
        errors.append("missing or invalid words: " + ", ".join(sorted(missing)))
    if errors:
        raise AIProviderError("; ".join(errors))
    return validated


def evaluate_with_retries(
    evaluator: DifficultyEvaluator,
    batch: list[dict[str, Any]],
    retries: int,
) -> dict[str, dict[str, int]]:
    last_error: AIProviderError | None = None
    for attempt in range(1, retries + 1):
        try:
            return validate_evaluations(batch, evaluator.evaluate_words(batch))
        except AIProviderError as exc:
            last_error = exc
            print(
                f"AI batch attempt {attempt}/{retries} failed: {exc}",
                file=sys.stderr,
            )
            if attempt < retries:
                time.sleep(min(attempt, 3))
    assert last_error is not None
    raise last_error


def build_entry(word_data: dict[str, Any], ai_scores: dict[str, int]) -> dict[str, Any]:
    zipf = round(float(zipf_frequency(word_data["word"], "en")), 3)
    components = {"frequency": frequency_score(zipf), **ai_scores}
    total_score = sum(components.values())
    return {
        **word_data,
        "difficulty": {
            "level": level_from_score(total_score),
            "score": total_score,
            "components": components,
            "zipf": zipf,
        },
    }


def chunked(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def run_pipeline(args: argparse.Namespace, evaluator: DifficultyEvaluator) -> int:
    if args.batch_size < 1 or args.ai_retries < 1:
        raise PipelineError("batch-size and ai-retries must be positive")
    if args.reset:
        atomic_write_json(VOCABULARY_PATH, [])
        atomic_write_json(REJECTED_PATH, [])
        print("Reset existing output files.")
    elif not args.resume and (VOCABULARY_PATH.exists() or REJECTED_PATH.exists()):
        raise PipelineError("Output exists. Use --resume (default) or --reset.")

    vocabulary = load_json_list(VOCABULARY_PATH) if args.resume else []
    rejected = load_json_list(REJECTED_PATH) if args.resume else []
    processed_words = {
        item.get("word") for item in vocabulary if isinstance(item.get("word"), str)
    }

    print(
        f"Generating up to {args.limit} valid words from "
        f"{args.candidate_count} wordfreq candidates..."
    )
    candidates = generate_candidates(
        args.candidate_count, args.limit, args.max_senses, rejected
    )
    atomic_write_json(REJECTED_PATH, rejected)
    if not VOCABULARY_PATH.exists():
        atomic_write_json(VOCABULARY_PATH, vocabulary)

    pending = [item for item in candidates if item["word"] not in processed_words]
    print(
        f"Valid candidates: {len(candidates)}; already processed: "
        f"{len(candidates) - len(pending)}; pending: {len(pending)}"
    )
    if not pending:
        print(f"Nothing to do. Vocabulary contains {len(vocabulary)} entries.")
        return 0

    completed_now = 0
    for batch_number, batch in enumerate(chunked(pending, args.batch_size), start=1):
        try:
            evaluations = evaluate_with_retries(evaluator, batch, args.ai_retries)
        except AIProviderError as exc:
            reason = str(exc)
            for item in batch:
                upsert_rejection(
                    rejected, item["word"], "ai_evaluation", reason, True
                )
            atomic_write_json(REJECTED_PATH, rejected)
            atomic_write_json(VOCABULARY_PATH, vocabulary)
            print(
                "Stopped cleanly because the AI batch failed. Existing results were "
                f"preserved. Fix Ollama/model settings and rerun to resume.\nError: {exc}",
                file=sys.stderr,
            )
            return 2

        for item in batch:
            word = item["word"]
            vocabulary.append(build_entry(item, evaluations[word]))
            remove_ai_rejection(rejected, word)
            completed_now += 1
        atomic_write_json(VOCABULARY_PATH, vocabulary)
        atomic_write_json(REJECTED_PATH, rejected)
        print(
            f"Saved batch {batch_number}: {completed_now}/{len(pending)} new words "
            f"({len(vocabulary)} total)."
        )

    print(f"Build complete: {len(vocabulary)} entries in {VOCABULARY_PATH}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATE_COUNT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-senses", type=int, default=DEFAULT_MAX_SENSES)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--ai-retries", type=int, default=DEFAULT_AI_RETRIES)
    parser.add_argument("--ai-timeout", type=float, default=DEFAULT_AI_TIMEOUT_SECONDS)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="replace both output files with empty arrays before building",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    evaluator = OllamaEvaluator(
        base_url=args.ollama_url,
        model=args.model,
        timeout_seconds=args.ai_timeout,
        temperature=0.0,
    )
    try:
        return run_pipeline(args, evaluator)
    except LookupError as exc:
        print(
            "NLTK WordNet data is missing. Run: python -m nltk.downloader wordnet",
            file=sys.stderr,
        )
        print(str(exc), file=sys.stderr)
        return 1
    except (PipelineError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


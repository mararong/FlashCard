from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ai_providers import AIProviderError  # noqa: E402
from build_english_vocabulary import (  # noqa: E402
    build_entry,
    build_pending_entry,
    collect_wordnet_data,
    frequency_score,
    generate_candidates,
    level_from_score,
    normalize_wordnet_lemma,
    validate_evaluations,
)


class ScoringTests(unittest.TestCase):
    def test_frequency_boundaries(self) -> None:
        self.assertEqual(frequency_score(5.0), 0)
        self.assertEqual(frequency_score(4.999), 1)
        self.assertEqual(frequency_score(4.0), 1)
        self.assertEqual(frequency_score(3.999), 2)
        self.assertEqual(frequency_score(3.0), 2)
        self.assertEqual(frequency_score(2.999), 3)

    def test_level_boundaries(self) -> None:
        values = [level_from_score(x) for x in (0, 3, 4, 6, 7, 10)]
        self.assertEqual(values, [1, 1, 2, 2, 3, 3])

    def test_incomplete_entry_has_no_fabricated_ai_scores(self) -> None:
        entry = build_pending_entry(sample_word_data(), model="test-model")
        self.assertEqual(entry["evaluation_status"], "pending_ai")
        self.assertIsNone(entry["difficulty"]["level"])
        self.assertIsNone(entry["difficulty"]["score"])
        self.assertIsNone(entry["difficulty"]["components"]["abstractness"])

    def test_complete_entry_calculates_difficulty(self) -> None:
        entry = build_entry(
            sample_word_data(),
            {
                "abstractness": 1,
                "semantic_complexity": 1,
                "form_complexity": 0,
                "register": 1,
            },
            model="test-model",
        )
        difficulty = entry["difficulty"]
        self.assertEqual(entry["evaluation_status"], "complete")
        self.assertEqual(difficulty["score"], sum(difficulty["components"].values()))
        self.assertIn(difficulty["level"], {1, 2, 3})


class WordNetQualityTests(unittest.TestCase):
    def test_wordnet_morphology_normalizes_inflections(self) -> None:
        expected = {
            "is": "be",
            "was": "be",
            "are": "be",
            "went": "go",
            "running": "run",
            "cars": "car",
            "better": "good",
        }
        self.assertEqual(
            {word: normalize_wordnet_lemma(word) for word in expected}, expected
        )

    def test_case_mismatched_initialism_is_not_a_word_sense(self) -> None:
        self.assertIsNone(collect_wordnet_data("it", 8))

    def test_short_symbol_senses_are_removed(self) -> None:
        data = collect_wordnet_data("in", 8)
        self.assertIsNotNone(data)
        definitions = " ".join(data["definitions"]).lower()
        self.assertNotIn("one twelfth of a foot", definitions)
        self.assertNotIn("metallic element", definitions)
        self.assertNotIn("midwestern united states", definitions)

    def test_normalized_lemma_is_deduplicated(self) -> None:
        rejected: list[dict[str, object]] = []
        candidates = generate_candidates(
            candidate_count=10,
            limit=10,
            max_senses=8,
            rejected=rejected,
            fixed_words=["cars", "car"],
        )
        self.assertEqual([item["word"] for item in candidates], ["car"])
        self.assertTrue(any(item["stage"] == "normalization" for item in rejected))


class ValidationTests(unittest.TestCase):
    def test_validation_ignores_unknown_words(self) -> None:
        batch = [{"word": "cat"}, {"word": "run"}]
        result = validate_evaluations(
            batch,
            [
                score("cat", 0, 0, 0, 0),
                score("invented", 2, 2, 1, 2),
                score("run", 0, 2, 0, 0),
            ],
        )
        self.assertEqual(set(result), {"cat", "run"})

    def test_validation_rejects_missing_or_invalid_values(self) -> None:
        with self.assertRaises(AIProviderError):
            validate_evaluations([{"word": "cat"}], [score("cat", 3, 0, 0, 0)])


def score(
    word: str,
    abstractness: int,
    semantic_complexity: int,
    form_complexity: int,
    register: int,
) -> dict[str, object]:
    return {
        "word": word,
        "abstractness": abstractness,
        "semantic_complexity": semantic_complexity,
        "form_complexity": form_complexity,
        "register": register,
    }


def sample_word_data() -> dict[str, object]:
    return {
        "word": "maintain",
        "parts_of_speech": ["verb"],
        "definitions": ["keep in a condition"],
        "senses": [],
        "meanings": [],
        "examples": [],
        "synonyms": [],
        "antonyms": [],
    }


if __name__ == "__main__":
    unittest.main()

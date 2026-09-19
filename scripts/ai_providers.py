"""AI provider implementations for vocabulary difficulty evaluation."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any


class AIProviderError(RuntimeError):
    """Raised when an AI provider cannot return a usable response."""


class DifficultyEvaluator(ABC):
    """Provider-neutral interface used by the vocabulary pipeline."""

    @abstractmethod
    def evaluate_words(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return one difficulty assessment per input word."""


SYSTEM_PROMPT = """You evaluate English vocabulary difficulty for language learners.
Return JSON only: one JSON array and no Markdown or commentary.
Do not score frequency. Score only these four fields with integers.

Abstractness (0-2):
0 = directly observable concrete object/person/animal/place/action/sensory property.
1 = everyday abstract emotion, mental activity, state, relation, social action, or change.
2 = theoretical, philosophical, ideological, logical, or highly conceptual.
An action is not automatically concrete: verbs about preserving, evaluating, reducing,
managing, or changing a condition (such as maintain or mitigate) normally describe an
everyday abstract relation/state change and should usually receive 1.

Semantic complexity (0-2):
0 = one dominant common meaning, or only closely related meanings.
1 = multiple common but related meanings that context separates easily.
2 = multiple substantially different common meanings, major part-of-speech differences,
    or important figurative/idiomatic uses.
Ignore obsolete or extremely rare senses; do not use the raw WordNet sense count alone.
Common words with substantially different noun/verb meanings (for example, charge or
issue) should not receive 0 merely because each individual sense is easy to understand.

Form complexity (0-1):
0 = spelling/pronunciation and form are reasonably regular.
1 = meaningful learning burden from spelling-pronunciation mismatch, silent letters,
    irregular pronunciation, confusable spelling, difficult phonology, or complex form.
Length alone is not a reason for 1.

Register (0-2):
0 = natural in everyday conversation and daily life.
1 = especially common in school, news, essays, presentations, or general exposition.
2 = mainly academic, scientific, medical, legal, economic, specialist, literary, or formal.
Do not assign 0 merely because a formal word is understandable. Words characteristic of
careful explanatory prose (such as maintain, significant, or mitigate) usually warrant 1;
terms strongly associated with research or formal scholarship (such as empirical) warrant 2.

Calibration: a concrete everyday word such as apple should be near the low end overall;
formal/academic words such as empirical or ubiquitous should score higher on the applicable
components. These are rubric anchors, not permission to ignore the supplied senses.

Every object must have exactly this shape:
{"word":"example","abstractness":0,"semantic_complexity":0,"form_complexity":0,"register":0}
Use each supplied word exactly once and do not add words."""


EVALUATION_FORMAT = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "word": {"type": "string"},
            "abstractness": {"type": "integer", "minimum": 0, "maximum": 2},
            "semantic_complexity": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2,
            },
            "form_complexity": {"type": "integer", "minimum": 0, "maximum": 1},
            "register": {"type": "integer", "minimum": 0, "maximum": 2},
        },
        "required": [
            "word",
            "abstractness",
            "semantic_complexity",
            "form_complexity",
            "register",
        ],
        "additionalProperties": False,
    },
}


class OllamaEvaluator(DifficultyEvaluator):
    """Difficulty evaluator backed by Ollama's local chat API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 300.0,
        temperature: float = 0.0,
        context_window: int = 16_384,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.context_window = context_window

    def evaluate_words(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact_input = [
            {
                "word": item["word"],
                "parts_of_speech": item["parts_of_speech"],
                "definitions": item["definitions"],
                # Definitions and POS are sufficient for scoring. Excluding the
                # examples/synonym lists keeps a 30-word local batch practical.
                "senses": [
                    {
                        "part_of_speech": sense["part_of_speech"],
                        "definition_en": sense["definition_en"],
                    }
                    for sense in item["senses"]
                ],
            }
            for item in batch
        ]
        payload = {
            "model": self.model,
            "stream": False,
            # A schema is more reliable than format="json": some models otherwise
            # wrap the requested array in a top-level object.
            "format": EVALUATION_FORMAT,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.context_window,
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Evaluate this batch:\n"
                    + json.dumps(compact_input, ensure_ascii=False),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AIProviderError(
                f"Ollama HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AIProviderError(
                f"Cannot connect to Ollama at {self.base_url}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise AIProviderError("Ollama returned a non-JSON API response") from exc

        try:
            content = response_data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise AIProviderError("Ollama response has no message.content") from exc

        try:
            evaluations = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            raise AIProviderError("The model response is not valid JSON") from exc
        if not isinstance(evaluations, list):
            raise AIProviderError("The model response must be a JSON array")
        return evaluations

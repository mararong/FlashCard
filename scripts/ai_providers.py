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

Semantic complexity (0-2):
0 = one dominant common meaning, or only closely related meanings.
1 = multiple common but related meanings that context separates easily.
2 = multiple substantially different common meanings, major part-of-speech differences,
    or important figurative/idiomatic uses.
Ignore obsolete or extremely rare senses; do not use the raw WordNet sense count alone.

Form complexity (0-1):
0 = spelling/pronunciation and form are reasonably regular.
1 = meaningful learning burden from spelling-pronunciation mismatch, silent letters,
    irregular pronunciation, confusable spelling, difficult phonology, or complex form.
Length alone is not a reason for 1.

Register (0-2):
0 = natural in everyday conversation and daily life.
1 = especially common in school, news, essays, presentations, or general exposition.
2 = mainly academic, scientific, medical, legal, economic, specialist, literary, or formal.

Every object must have exactly this shape:
{"word":"example","abstractness":0,"semantic_complexity":0,"form_complexity":0,"register":0}
Use each supplied word exactly once and do not add words."""


class OllamaEvaluator(DifficultyEvaluator):
    """Difficulty evaluator backed by Ollama's local chat API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature

    def evaluate_words(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact_input = [
            {
                "word": item["word"],
                "parts_of_speech": item["parts_of_speech"],
                "definitions": item["definitions"],
                "senses": item["senses"],
            }
            for item in batch
        ]
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
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


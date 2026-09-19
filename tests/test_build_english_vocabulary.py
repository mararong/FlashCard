from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ai_providers import AIProviderError  # noqa: E402
from build_english_vocabulary import (  # noqa: E402
    build_entry,
    frequency_score,
    level_from_score,
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
        self.assertEqual([level_from_score(x) for x in (0, 3, 4, 6, 7, 10)], [1, 1, 2, 2, 3, 3])

    def test_validation_ignores_unknown_words(self) -> None:
        batch = [{"word": "cat"}, {"word": "run"}]
        result = validate_evaluations(
            batch,
            [
                {"word": "cat", "abstractness": 0, "semantic_complexity": 0, "form_complexity": 0, "register": 0},
                {"word": "invented", "abstractness": 2, "semantic_complexity": 2, "form_complexity": 1, "register": 2},
                {"word": "run", "abstractness": 0, "semantic_complexity": 2, "form_complexity": 0, "register": 0},
            ],
        )
        self.assertEqual(set(result), {"cat", "run"})

    def test_validation_rejects_missing_or_invalid_values(self) -> None:
        with self.assertRaises(AIProviderError):
            validate_evaluations(
                [{"word": "cat"}],
                [{"word": "cat", "abstractness": 3, "semantic_complexity": 0, "form_complexity": 0, "register": 0}],
            )

    def test_build_entry_calculates_difficulty(self) -> None:
        word_data = {
            "word": "maintain",
            "parts_of_speech": ["verb"],
            "definitions": ["keep in a condition"],
            "senses": [],
            "meanings": [],
            "examples": [],
            "synonyms": [],
            "antonyms": [],
        }
        entry = build_entry(
            word_data,
            {"abstractness": 1, "semantic_complexity": 1, "form_complexity": 0, "register": 1},
        )
        self.assertEqual(entry["difficulty"]["score"], sum(entry["difficulty"]["components"].values()))
        self.assertIn(entry["difficulty"]["level"], {1, 2, 3})


if __name__ == "__main__":
    unittest.main()


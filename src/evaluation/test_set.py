"""
Test set loader for the Clinical Phenotype Extraction and HPO Mapping Pipeline.

Loads the curated evaluation test cases from data/evaluation/test_set.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ExpectedMention:
    """A single expected phenotype mention in a test case."""

    text: str
    hpo_id: str
    negated: bool


@dataclass
class TestCase:
    """A single evaluation test case."""

    note: str
    expected_mentions: list[ExpectedMention] = field(default_factory=list)


def load_test_set(path: str) -> list[TestCase]:
    """Load the curated evaluation test set from a JSON file.

    Args:
        path: Path to the test_set.json file.

    Returns:
        A list of TestCase objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON is malformed or missing required fields.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError(f"Test set file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Test set file is not valid JSON ({path}): {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError(f"Test set must be a JSON array, got {type(raw).__name__}")

    cases: list[TestCase] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Test case at index {i} must be a JSON object")
        if "note" not in entry:
            raise ValueError(f"Test case at index {i} is missing 'note' field")

        expected: list[ExpectedMention] = []
        for j, m in enumerate(entry.get("expected_mentions", [])):
            if not isinstance(m, dict):
                raise ValueError(
                    f"expected_mentions[{j}] in test case {i} must be a JSON object"
                )
            try:
                expected.append(
                    ExpectedMention(
                        text=m["text"],
                        hpo_id=m["hpo_id"],
                        negated=bool(m.get("negated", False)),
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"expected_mentions[{j}] in test case {i} is missing key {exc}"
                ) from exc

        cases.append(TestCase(note=entry["note"], expected_mentions=expected))

    return cases

"""
Rule-based phenotype mention extractor for the Clinical Phenotype Extraction
and HPO Mapping Pipeline.

Uses spaCy's PhraseMatcher to detect HPO term labels and synonyms in clinical
notes, resolving overlapping spans with a longest-wins strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import spacy
import spacy.util
from spacy.matcher import PhraseMatcher

from src.data.build_hpo_db import HPODatabase


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Mention:
    """A phenotype mention detected in a clinical note.

    Attributes:
        text:    The matched text span as it appears in the original note.
        start:   Character offset of the start of the span in the original note.
        end:     Character offset of the end of the span in the original note.
        negated: True if a negation cue was detected for this mention.
    """

    text: str
    start: int   # character offset in original note
    end: int     # character offset in original note
    negated: bool = False


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class Extractor:
    """Rule-based phenotype mention extractor using spaCy PhraseMatcher.

    Builds a case-insensitive PhraseMatcher over all HPO labels and synonyms
    at construction time, then applies it to clinical notes to return Mention
    objects with accurate character offsets.
    """

    def __init__(self, hpo_db: HPODatabase) -> None:
        """Build a case-insensitive PhraseMatcher over all HPO labels and synonyms.

        Uses ``spacy.blank("en")`` and ``PhraseMatcher(nlp.vocab, attr="LOWER")``
        so that matching is case-insensitive regardless of how the term appears
        in the clinical note.

        Args:
            hpo_db: The loaded HPO database providing labels and synonyms.
        """
        self._nlp = spacy.blank("en")
        self._matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")

        # Build patterns from all HPO labels and synonyms.
        # all_labels_and_synonyms() returns [(text, hpo_id), ...]
        patterns: list = []
        for text, _hpo_id in hpo_db.all_labels_and_synonyms():
            if text and text.strip():
                patterns.append(self._nlp.make_doc(text))

        if patterns:
            self._matcher.add("HPO_TERM", patterns)

    def extract(self, clinical_note: str) -> list[Mention]:
        """Run the PhraseMatcher on the clinical note and return Mention objects.

        Overlapping spans are resolved with ``spacy.util.filter_spans`` using a
        longest-wins strategy. Character offsets in the returned Mention objects
        correspond to positions in the original ``clinical_note`` string.

        Args:
            clinical_note: Free-text clinical note to process.

        Returns:
            A list of Mention objects, one per detected HPO term span.
            Returns an empty list (no error) when no mentions are found.
        """
        if not clinical_note:
            return []

        doc = self._nlp(clinical_note)

        # Run the matcher; each match is (match_id, start_token, end_token)
        raw_matches = self._matcher(doc)

        # Convert token-index matches to Span objects
        spans = [doc[start:end] for _match_id, start, end in raw_matches]

        # Resolve overlapping spans: longest span wins (spaCy built-in)
        filtered_spans = spacy.util.filter_spans(spans)

        return [
            Mention(
                text=span.text,
                start=span.start_char,
                end=span.end_char,
            )
            for span in filtered_spans
        ]

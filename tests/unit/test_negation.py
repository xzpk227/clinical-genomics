"""
Unit tests for src/extraction/negation.py — NegationHandler.

Covers:
- Standard negation cues: "no seizures", "without hypotonia", "denies hearing loss"
- Non-negated mention passes through with negated=False
- Negated mention still appears in output list (not dropped)
- Custom cue list is respected
"""

from __future__ import annotations

import spacy

from src.extraction.extractor import Mention
from src.extraction.negation import NegationHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(text: str) -> spacy.tokens.Doc:
    """Create a blank spaCy Doc from raw text (no pipeline components needed)."""
    nlp = spacy.blank("en")
    return nlp(text)


def _make_mention(text: str, note: str) -> Mention:
    """Locate *text* in *note* and return a Mention with correct offsets."""
    start = note.lower().find(text.lower())
    assert start != -1, f"'{text}' not found in note: {note!r}"
    end = start + len(text)
    return Mention(text=note[start:end], start=start, end=end)


# ---------------------------------------------------------------------------
# Fixtures — shared handler instances
# ---------------------------------------------------------------------------


def _default_handler() -> NegationHandler:
    """NegationHandler loaded from the default config/negation_cues.json."""
    return NegationHandler()


# ---------------------------------------------------------------------------
# Tests: standard negation cues
# ---------------------------------------------------------------------------


class TestStandardNegationCues:
    """Verify that common clinical negation cues are detected correctly."""

    def test_no_seizures_is_negated(self) -> None:
        """'no seizures' should produce a mention with negated=True."""
        handler = _default_handler()
        note = "Patient has no seizures."
        doc = _make_doc(note)
        mention = _make_mention("seizures", note)

        result = handler.annotate(doc, [mention])

        assert len(result) == 1
        assert result[0].negated is True

    def test_without_hypotonia_is_negated(self) -> None:
        """'without hypotonia' should produce a mention with negated=True."""
        handler = _default_handler()
        note = "The child presents without hypotonia."
        doc = _make_doc(note)
        mention = _make_mention("hypotonia", note)

        result = handler.annotate(doc, [mention])

        assert len(result) == 1
        assert result[0].negated is True

    def test_denies_hearing_loss_is_negated(self) -> None:
        """'denies hearing loss' should produce a mention with negated=True."""
        handler = _default_handler()
        note = "Patient denies hearing loss."
        doc = _make_doc(note)
        mention = _make_mention("hearing loss", note)

        result = handler.annotate(doc, [mention])

        assert len(result) == 1
        assert result[0].negated is True

    def test_not_present_following_negation(self) -> None:
        """Following negation 'not present' should mark the mention as negated."""
        handler = _default_handler()
        note = "Seizures not present."
        doc = _make_doc(note)
        mention = _make_mention("Seizures", note)

        result = handler.annotate(doc, [mention])

        assert len(result) == 1
        assert result[0].negated is True


# ---------------------------------------------------------------------------
# Tests: non-negated mentions
# ---------------------------------------------------------------------------


class TestNonNegatedMentions:
    """Verify that positive (non-negated) mentions are not incorrectly flagged."""

    def test_positive_mention_has_negated_false(self) -> None:
        """A mention without any negation cue should have negated=False."""
        handler = _default_handler()
        note = "Patient presents with seizures."
        doc = _make_doc(note)
        mention = _make_mention("seizures", note)

        result = handler.annotate(doc, [mention])

        assert len(result) == 1
        assert result[0].negated is False

    def test_positive_mention_text_unchanged(self) -> None:
        """The mention text and offsets must not be modified by annotation."""
        handler = _default_handler()
        note = "Hypotonia was observed."
        doc = _make_doc(note)
        mention = _make_mention("Hypotonia", note)
        original_text = mention.text
        original_start = mention.start
        original_end = mention.end

        result = handler.annotate(doc, [mention])

        assert result[0].text == original_text
        assert result[0].start == original_start
        assert result[0].end == original_end


# ---------------------------------------------------------------------------
# Tests: negated mentions are never dropped
# ---------------------------------------------------------------------------


class TestNegatedMentionsNotDropped:
    """Verify that negated mentions still appear in the output list."""

    def test_negated_mention_still_in_output(self) -> None:
        """A negated mention must appear in the returned list (not dropped)."""
        handler = _default_handler()
        note = "No seizures were observed."
        doc = _make_doc(note)
        mention = _make_mention("seizures", note)

        result = handler.annotate(doc, [mention])

        assert len(result) == 1, "Negated mention must not be dropped from output"
        assert result[0].text.lower() == "seizures"

    def test_mixed_mentions_all_returned(self) -> None:
        """Both negated and non-negated mentions must appear in the output."""
        handler = _default_handler()
        note = "Patient has hypotonia but no seizures."
        doc = _make_doc(note)

        mention_hypotonia = _make_mention("hypotonia", note)
        mention_seizures = _make_mention("seizures", note)

        result = handler.annotate(doc, [mention_hypotonia, mention_seizures])

        assert len(result) == 2, "Both mentions must be returned"

        texts = {m.text.lower() for m in result}
        assert "hypotonia" in texts
        assert "seizures" in texts

        negated_map = {m.text.lower(): m.negated for m in result}
        assert negated_map["hypotonia"] is False
        assert negated_map["seizures"] is True

    def test_empty_mention_list_returns_empty(self) -> None:
        """An empty mention list should return an empty list without error."""
        handler = _default_handler()
        note = "No findings."
        doc = _make_doc(note)

        result = handler.annotate(doc, [])

        assert result == []


# ---------------------------------------------------------------------------
# Tests: custom cue list
# ---------------------------------------------------------------------------


class TestCustomCueList:
    """Verify that a custom cue list is respected by the handler."""

    def test_custom_preceding_cue_triggers_negation(self) -> None:
        """A custom cue supplied at construction time should trigger negation."""
        handler = NegationHandler(cue_list=["never"])
        note = "Patient has never had seizures."
        doc = _make_doc(note)
        mention = _make_mention("seizures", note)

        result = handler.annotate(doc, [mention])

        assert len(result) == 1
        assert result[0].negated is True

    def test_default_cue_not_active_with_custom_list(self) -> None:
        """When a custom cue list is provided, default cues should not apply.

        'no' is a default cue but is NOT in the custom list ['never'], so
        'no seizures' should NOT be negated when using the custom list.
        """
        handler = NegationHandler(cue_list=["never"])
        note = "Patient has no seizures."
        doc = _make_doc(note)
        mention = _make_mention("seizures", note)

        result = handler.annotate(doc, [mention])

        assert len(result) == 1
        # 'no' is not in the custom cue list, so negation should not fire.
        assert result[0].negated is False

    def test_empty_custom_cue_list_no_negation(self) -> None:
        """An empty custom cue list should result in no negation being detected."""
        handler = NegationHandler(cue_list=[])
        note = "No seizures were found."
        doc = _make_doc(note)
        mention = _make_mention("seizures", note)

        result = handler.annotate(doc, [mention])

        assert len(result) == 1
        assert result[0].negated is False

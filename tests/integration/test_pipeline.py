"""
Integration tests for the Pipeline orchestrator.

These tests use a minimal stub pipeline — no real models are loaded.
Dependency injection is achieved by bypassing ``Pipeline.__init__`` via
``object.__new__`` and replacing internal components with lightweight stubs.

Requirements covered: 4.3, 4.5, 5.6, 5.8, 9.2, 9.3
"""

from __future__ import annotations

import pytest
import spacy

from src.extraction.extractor import Mention
from src.mapping.mapper import HPOCandidate
from src.pipeline import (
    ExtractionResult,
    Pipeline,
    PipelineConfig,
    PipelineNotReadyError,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StubExtractor:
    """Returns a fixed list of mentions regardless of the input note."""

    def __init__(self, mentions: list[Mention]) -> None:
        self._mentions = mentions

    def extract(self, clinical_note: str) -> list[Mention]:
        return list(self._mentions)


class StubNegationHandler:
    """Passes mentions through unchanged (no negation detection)."""

    def annotate(self, doc, mentions: list[Mention]) -> list[Mention]:
        return mentions


class NegatingStubHandler:
    """Marks every mention as negated."""

    def annotate(self, doc, mentions: list[Mention]) -> list[Mention]:
        for m in mentions:
            m.negated = True
        return mentions


class StubMapper:
    """Returns a fixed list of HPO candidates for every mention."""

    def __init__(self, candidates: list[HPOCandidate]) -> None:
        self._candidates = candidates
        self._top_k = 3  # mimic the real Mapper attribute

    def map(self, mention: Mention) -> list[HPOCandidate]:
        return list(self._candidates)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def _make_stub_pipeline(
    mentions: list[Mention] | None = None,
    candidates: list[HPOCandidate] | None = None,
    negation_handler=None,
) -> Pipeline:
    """Create a Pipeline with all components stubbed out.

    Bypasses ``Pipeline.__init__`` so no real models are loaded.

    Args:
        mentions:          Mentions the stub extractor will return.
        candidates:        HPO candidates the stub mapper will return.
        negation_handler:  Custom negation handler stub; defaults to
                           ``StubNegationHandler`` (pass-through).

    Returns:
        A fully configured stub Pipeline with ``is_ready=True``.
    """
    if mentions is None:
        mentions = []
    if candidates is None:
        candidates = [
            HPOCandidate(hpo_id="HP:0001250", hpo_label="Seizure", confidence=0.95)
        ]
    if negation_handler is None:
        negation_handler = StubNegationHandler()

    config = PipelineConfig()
    pipeline: Pipeline = object.__new__(Pipeline)
    pipeline.config = config
    pipeline.is_ready = True
    pipeline._nlp = spacy.blank("en")
    pipeline._extractor = StubExtractor(mentions)
    pipeline._negation_handler = negation_handler
    pipeline._mapper = StubMapper(candidates)
    pipeline._llm = None
    return pipeline


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProcessReturnsExtractionResult:
    """Verify that process() returns an ExtractionResult instance."""

    def test_process_returns_extraction_result(self) -> None:
        """process() must return an ExtractionResult for any valid note."""
        pipeline = _make_stub_pipeline(
            mentions=[Mention(text="seizures", start=12, end=20)]
        )
        result = pipeline.process("Patient has seizures.")
        assert isinstance(result, ExtractionResult), (
            f"Expected ExtractionResult, got {type(result)}"
        )

    def test_process_result_has_required_fields(self) -> None:
        """ExtractionResult must expose mentions, mappings, and summary."""
        pipeline = _make_stub_pipeline(
            mentions=[Mention(text="hypotonia", start=0, end=9)]
        )
        result = pipeline.process("hypotonia noted.")
        assert hasattr(result, "mentions")
        assert hasattr(result, "mappings")
        assert hasattr(result, "summary")


class TestProcessRaisesWhenNotReady:
    """Verify that process() raises PipelineNotReadyError when is_ready=False."""

    def test_process_raises_when_not_ready(self) -> None:
        """process() must raise PipelineNotReadyError if is_ready is False."""
        pipeline = _make_stub_pipeline()
        pipeline.is_ready = False

        with pytest.raises(PipelineNotReadyError):
            pipeline.process("Patient has seizures.")

    def test_process_raises_with_descriptive_message(self) -> None:
        """The PipelineNotReadyError message should be non-empty."""
        pipeline = _make_stub_pipeline()
        pipeline.is_ready = False

        with pytest.raises(PipelineNotReadyError, match=r"(?i)not ready|initializ"):
            pipeline.process("Some note.")


class TestProcessEmptyNoteReturnsEmptyMentions:
    """Verify that an empty note (stub extractor returns []) yields empty results."""

    def test_process_empty_note_returns_empty_mentions(self) -> None:
        """When the extractor returns no mentions, mentions list must be empty."""
        pipeline = _make_stub_pipeline(mentions=[])
        result = pipeline.process("")
        assert result.mentions == [], (
            f"Expected empty mentions list, got {result.mentions}"
        )

    def test_process_empty_note_returns_empty_mappings(self) -> None:
        """When the extractor returns no mentions, mappings dict must be empty."""
        pipeline = _make_stub_pipeline(mentions=[])
        result = pipeline.process("")
        assert result.mappings == {}, (
            f"Expected empty mappings dict, got {result.mappings}"
        )

    def test_process_empty_note_summary_is_none(self) -> None:
        """Summary must be None when LLM is disabled (default config)."""
        pipeline = _make_stub_pipeline(mentions=[])
        result = pipeline.process("")
        assert result.summary is None


class TestProcessMapsAllMentions:
    """Verify that every mention returned by the extractor appears in mappings."""

    def test_process_maps_all_mentions(self) -> None:
        """mappings dict must have one key per unique mention text."""
        mentions = [
            Mention(text="seizures", start=12, end=20),
            Mention(text="hypotonia", start=25, end=34),
        ]
        pipeline = _make_stub_pipeline(mentions=mentions)
        result = pipeline.process("Patient has seizures and hypotonia.")

        assert len(result.mappings) == 2, (
            f"Expected 2 keys in mappings, got {len(result.mappings)}: "
            f"{list(result.mappings.keys())}"
        )
        assert "seizures" in result.mappings
        assert "hypotonia" in result.mappings

    def test_process_maps_single_mention(self) -> None:
        """A single mention must produce a mappings dict with exactly one key."""
        mentions = [Mention(text="ataxia", start=0, end=6)]
        pipeline = _make_stub_pipeline(mentions=mentions)
        result = pipeline.process("ataxia observed.")

        assert len(result.mappings) == 1
        assert "ataxia" in result.mappings

    def test_process_mapping_values_are_lists(self) -> None:
        """Each value in mappings must be a list of HPOCandidate objects."""
        mentions = [Mention(text="seizures", start=0, end=8)]
        candidates = [
            HPOCandidate(hpo_id="HP:0001250", hpo_label="Seizure", confidence=0.95)
        ]
        pipeline = _make_stub_pipeline(mentions=mentions, candidates=candidates)
        result = pipeline.process("seizures noted.")

        assert isinstance(result.mappings["seizures"], list)
        assert len(result.mappings["seizures"]) == 1
        assert isinstance(result.mappings["seizures"][0], HPOCandidate)


class TestProcessNegatedMentionPreserved:
    """Verify that negated mentions appear in the output with negated=True."""

    def test_process_negated_mention_preserved(self) -> None:
        """A mention marked negated by the handler must appear in mentions with negated=True."""
        mentions = [Mention(text="hearing loss", start=3, end=15, negated=False)]
        pipeline = _make_stub_pipeline(
            mentions=mentions,
            negation_handler=NegatingStubHandler(),
        )
        result = pipeline.process("No hearing loss noted.")

        assert len(result.mentions) == 1, (
            "Negated mention must not be dropped from the mentions list."
        )
        assert result.mentions[0].negated is True, (
            "Mention should be marked as negated=True."
        )

    def test_process_negated_mention_still_mapped(self) -> None:
        """A negated mention must still appear in the mappings dict (Requirement 3.5)."""
        mentions = [Mention(text="hearing loss", start=3, end=15, negated=False)]
        pipeline = _make_stub_pipeline(
            mentions=mentions,
            negation_handler=NegatingStubHandler(),
        )
        result = pipeline.process("No hearing loss noted.")

        assert "hearing loss" in result.mappings, (
            "Negated mention must still have HPO candidates in mappings."
        )
        assert len(result.mappings["hearing loss"]) > 0

    def test_process_non_negated_mention_has_negated_false(self) -> None:
        """A mention not negated by the handler must have negated=False."""
        mentions = [Mention(text="seizures", start=12, end=20, negated=False)]
        pipeline = _make_stub_pipeline(
            mentions=mentions,
            negation_handler=StubNegationHandler(),
        )
        result = pipeline.process("Patient has seizures.")

        assert result.mentions[0].negated is False

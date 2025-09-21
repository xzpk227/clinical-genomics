"""
Unit tests for src/api/schemas.py Pydantic models.

Covers:
- ExtractionRequest validation (clinical_note length, top_k range)
- ExtractionResponse structure (disclaimer always present, summary default)
- HPOTermResult field presence and types
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    HPOCandidate,
    HPOTermResult,
)

REQUIRED_DISCLAIMER = (
    "This output is for clinical decision support only. "
    "It does not constitute a medical diagnosis. "
    "Always involve a qualified clinician."
)


# ---------------------------------------------------------------------------
# ExtractionRequest — clinical_note validation
# ---------------------------------------------------------------------------


def test_empty_clinical_note_raises_validation_error():
    """Empty clinical_note (min_length=1) must raise ValidationError."""
    with pytest.raises(ValidationError):
        ExtractionRequest(clinical_note="")


def test_clinical_note_too_long_raises_validation_error():
    """clinical_note of 10,001 chars (max_length=10_000) must raise ValidationError."""
    with pytest.raises(ValidationError):
        ExtractionRequest(clinical_note="a" * 10_001)


def test_clinical_note_at_max_length_is_valid():
    """clinical_note of exactly 10,000 chars must pass validation."""
    req = ExtractionRequest(clinical_note="a" * 10_000)
    assert len(req.clinical_note) == 10_000


def test_clinical_note_at_min_length_is_valid():
    """clinical_note of exactly 1 char must pass validation."""
    req = ExtractionRequest(clinical_note="x")
    assert req.clinical_note == "x"


# ---------------------------------------------------------------------------
# ExtractionRequest — top_k validation
# ---------------------------------------------------------------------------


def test_top_k_zero_raises_validation_error():
    """top_k=0 (ge=1) must raise ValidationError."""
    with pytest.raises(ValidationError):
        ExtractionRequest(clinical_note="Patient has seizures.", top_k=0)


def test_top_k_eleven_raises_validation_error():
    """top_k=11 (le=10) must raise ValidationError."""
    with pytest.raises(ValidationError):
        ExtractionRequest(clinical_note="Patient has seizures.", top_k=11)


def test_top_k_default_is_three():
    """top_k defaults to 3 when not provided."""
    req = ExtractionRequest(clinical_note="Patient has seizures.")
    assert req.top_k == 3


def test_top_k_boundary_values_are_valid():
    """top_k=1 and top_k=10 are both valid boundary values."""
    req_min = ExtractionRequest(clinical_note="Patient has seizures.", top_k=1)
    req_max = ExtractionRequest(clinical_note="Patient has seizures.", top_k=10)
    assert req_min.top_k == 1
    assert req_max.top_k == 10


# ---------------------------------------------------------------------------
# ExtractionRequest — valid request
# ---------------------------------------------------------------------------


def test_valid_request_passes_validation():
    """A well-formed request must pass validation with correct field values."""
    req = ExtractionRequest(
        clinical_note="Patient presents with seizures and hypotonia.",
        top_k=5,
    )
    assert req.clinical_note == "Patient presents with seizures and hypotonia."
    assert req.top_k == 5


# ---------------------------------------------------------------------------
# ExtractionResponse — disclaimer and summary
# ---------------------------------------------------------------------------


def test_extraction_response_always_includes_disclaimer():
    """ExtractionResponse must always include the disclaimer field."""
    resp = ExtractionResponse(hpo_terms=[])
    assert hasattr(resp, "disclaimer")
    assert resp.disclaimer == REQUIRED_DISCLAIMER


def test_extraction_response_disclaimer_exact_text():
    """The disclaimer must match the exact required text."""
    resp = ExtractionResponse(hpo_terms=[])
    assert resp.disclaimer == REQUIRED_DISCLAIMER


def test_extraction_response_summary_is_none_by_default():
    """ExtractionResponse.summary must be None when not provided."""
    resp = ExtractionResponse(hpo_terms=[])
    assert resp.summary is None


def test_extraction_response_summary_can_be_set():
    """ExtractionResponse.summary can be set to a string."""
    resp = ExtractionResponse(
        hpo_terms=[],
        summary="Patient has two phenotypes.",
    )
    assert resp.summary == "Patient has two phenotypes."


def test_extraction_response_empty_hpo_terms():
    """ExtractionResponse with an empty hpo_terms list is valid."""
    resp = ExtractionResponse(hpo_terms=[])
    assert resp.hpo_terms == []


# ---------------------------------------------------------------------------
# HPOTermResult — field presence and types
# ---------------------------------------------------------------------------


def _make_candidate(n: int = 1) -> HPOCandidate:
    return HPOCandidate(
        hpo_id=f"HP:{n:07d}",
        hpo_label=f"Label {n}",
        confidence=0.9 / n,
    )


def _make_hpo_term_result() -> HPOTermResult:
    return HPOTermResult(
        text="seizures",
        hpo_id="HP:0001250",
        hpo_label="Seizure",
        confidence=0.95,
        negated=False,
        candidates=[_make_candidate(1), _make_candidate(2)],
    )


def test_hpo_term_result_all_fields_present():
    """HPOTermResult must have all required fields."""
    result = _make_hpo_term_result()
    assert result.text == "seizures"
    assert result.hpo_id == "HP:0001250"
    assert result.hpo_label == "Seizure"
    assert result.confidence == 0.95
    assert result.negated is False
    assert len(result.candidates) == 2


def test_hpo_term_result_field_types():
    """HPOTermResult fields must have the correct types."""
    result = _make_hpo_term_result()
    assert isinstance(result.text, str)
    assert isinstance(result.hpo_id, str)
    assert isinstance(result.hpo_label, str)
    assert isinstance(result.confidence, float)
    assert isinstance(result.negated, bool)
    assert isinstance(result.candidates, list)


def test_hpo_term_result_negated_true():
    """HPOTermResult.negated can be True for negated mentions."""
    result = HPOTermResult(
        text="hearing loss",
        hpo_id="HP:0000365",
        hpo_label="Hearing impairment",
        confidence=0.88,
        negated=True,
        candidates=[],
    )
    assert result.negated is True


def test_hpo_term_result_empty_candidates():
    """HPOTermResult with an empty candidates list is valid."""
    result = HPOTermResult(
        text="seizures",
        hpo_id="HP:0001250",
        hpo_label="Seizure",
        confidence=0.95,
        negated=False,
        candidates=[],
    )
    assert result.candidates == []


def test_extraction_response_with_hpo_terms():
    """ExtractionResponse with populated hpo_terms is valid and disclaimer is present."""
    resp = ExtractionResponse(
        hpo_terms=[_make_hpo_term_result()],
    )
    assert len(resp.hpo_terms) == 1
    assert resp.disclaimer == REQUIRED_DISCLAIMER
    assert resp.summary is None

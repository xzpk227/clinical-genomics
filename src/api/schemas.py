"""
Pydantic v2 request and response schemas for the Clinical Phenotype Extraction
and HPO Mapping Pipeline API.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ExtractionRequest(BaseModel):
    """Request body for POST /extract-phenotypes."""

    clinical_note: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Free-text clinical note (synthetic or de-identified only)",
    )
    top_k: int = Field(default=3, ge=1, le=10)


class HPOCandidate(BaseModel):
    """A single HPO mapping candidate."""

    hpo_id: str
    hpo_label: str
    confidence: float  # [0.0, 1.0]


class HPOTermResult(BaseModel):
    """A single HPO term result for one matched span in the clinical note."""

    text: str                        # matched span from clinical note
    hpo_id: str                      # e.g. "HP:0001250"
    hpo_label: str                   # preferred label e.g. "Seizure"
    confidence: float                # top-1 score, [0.0, 1.0]
    negated: bool                    # True if negation detected
    candidates: list[HPOCandidate]   # top-k results


class ExtractionResponse(BaseModel):
    """Response body for POST /extract-phenotypes."""

    hpo_terms: list[HPOTermResult]
    summary: Optional[str] = None   # only present when LLM_SUMMARY_ENABLED=true
    disclaimer: str = (
        "This output is for clinical decision support only. "
        "It does not constitute a medical diagnosis. "
        "Always involve a qualified clinician."
    )

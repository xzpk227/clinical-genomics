"""
Pydantic v2 request and response schemas for the Clinical Phenotype Extraction
and HPO Mapping Pipeline API, including the oncology extension endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ===========================================================================
# Existing HPO extraction schemas (unchanged)
# ===========================================================================

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


# ===========================================================================
# Oncology extension schemas
# ===========================================================================

# ---------------------------------------------------------------------------
# POST /extract-clinical-concepts
# ---------------------------------------------------------------------------

class ClinicalConceptRequest(BaseModel):
    """Request body for POST /extract-clinical-concepts."""

    clinical_note: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Free-text oncology clinical note (synthetic or de-identified only)",
    )


class OncologyMentionItem(BaseModel):
    """A single oncology symptom or adverse-event mention."""

    text: str
    category: str           # e.g. "fatigue", "neuropathy", "pneumonitis"
    start: int              # character offset (start)
    end: int                # character offset (end)
    negated: bool
    grade: Optional[int] = None    # CTCAE grade 1–4 if detected in context


class ClinicalConceptResponse(BaseModel):
    """Response body for POST /extract-clinical-concepts."""

    mentions: List[OncologyMentionItem]
    categories_present: List[str]   # non-negated categories
    symptom_count: int
    max_grade: Optional[int]
    disclaimer: str = (
        "This output is for clinical decision support only. "
        "It does not constitute a medical diagnosis. "
        "Always involve a qualified clinician."
    )


# ---------------------------------------------------------------------------
# POST /predict-adverse-event-risk
# ---------------------------------------------------------------------------

class PatientFeatures(BaseModel):
    """
    Structured patient features for adverse event risk prediction.

    All laboratory values should be provided in their standard clinical units.
    """

    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=18, le=110)
    sex: str = Field(..., pattern="^[MF]$")
    cancer_type: str = Field(
        ...,
        description="Cancer type identifier, e.g. 'breast', 'lung_nsclc'",
    )
    stage: str = Field(..., pattern="^(I|II|III|IV)$")
    treatment: str = Field(..., description="Treatment regimen identifier")
    cycle_number: int = Field(..., ge=1, le=50)
    ecog_score: int = Field(..., ge=0, le=4)

    # Laboratory values
    wbc:         float = Field(..., ge=0, description="WBC × 10⁹/L")
    anc:         float = Field(..., ge=0, description="ANC × 10⁹/L")
    hemoglobin:  float = Field(..., ge=0, description="Hemoglobin g/dL")
    platelets:   float = Field(..., ge=0, description="Platelets × 10⁹/L")
    creatinine:  float = Field(..., ge=0, description="Creatinine mg/dL")
    alt:         float = Field(..., ge=0, description="ALT U/L")
    bilirubin:   float = Field(..., ge=0, description="Total bilirubin mg/dL")

    # CTCAE-informed symptom scores (0–10)
    fatigue_score:  float = Field(default=0.0, ge=0, le=10)
    nausea_score:   float = Field(default=0.0, ge=0, le=10)
    pain_score:     float = Field(default=0.0, ge=0, le=10)
    dyspnea_score:  float = Field(default=0.0, ge=0, le=10)

    # Optional free-text note (used for NLP feature extraction)
    clinical_note: Optional[str] = Field(
        default=None,
        max_length=10_000,
        description="Optional clinical note for NLP feature extraction",
    )


class RiskPredictionResponse(BaseModel):
    """Response body for POST /predict-adverse-event-risk."""

    patient_id: str
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    risk_tier: str             # "high" | "medium" | "low"
    model_used: str
    disclaimer: str = (
        "Risk scores are for research purposes only and must not be used for "
        "clinical decision-making without validation on real patient data."
    )


# ---------------------------------------------------------------------------
# POST /summarize-risk
# ---------------------------------------------------------------------------

class RiskSummaryRequest(BaseModel):
    """Request body for POST /summarize-risk."""

    patient: PatientFeatures
    clinical_note: Optional[str] = Field(
        default=None,
        max_length=10_000,
        description="Clinical note for symptom extraction (overrides patient.clinical_note)",
    )


class RiskSummaryResponse(BaseModel):
    """Response body for POST /summarize-risk."""

    patient_id: str
    summary: str
    risk_probability: float
    risk_tier: str
    disclaimer: str = (
        "This summary is generated for clinical decision support only. "
        "It does not constitute a diagnosis. Always involve a qualified clinician."
    )

"""
Route handlers for the Clinical Phenotype Extraction and HPO Mapping Pipeline API.

Original endpoints:
  POST /extract-phenotypes      — run the HPO pipeline on a clinical note
  GET  /health                  — pipeline readiness check

Oncology extension endpoints:
  POST /extract-clinical-concepts   — detect symptoms/AEs from oncology notes
  POST /predict-adverse-event-risk  — predict AE risk probability for a patient
  POST /summarize-risk              — generate clinician-facing risk summary
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.schemas import (
    # HPO schemas
    ExtractionRequest,
    ExtractionResponse,
    HPOCandidate,
    HPOTermResult,
    # Oncology schemas
    ClinicalConceptRequest,
    ClinicalConceptResponse,
    OncologyMentionItem,
    PatientFeatures,
    RiskPredictionResponse,
    RiskSummaryRequest,
    RiskSummaryResponse,
)
from src.pipeline import PipelineNotReadyError

logger = logging.getLogger(__name__)

router = APIRouter()

_DISCLAIMER = (
    "This output is for clinical decision support only. "
    "It does not constitute a medical diagnosis. "
    "Always involve a qualified clinician."
)


# ===========================================================================
# Existing HPO endpoint
# ===========================================================================

@router.post("/extract-phenotypes", response_model=ExtractionResponse)
async def extract_phenotypes(
    body: ExtractionRequest,
    request: Request,
) -> ExtractionResponse:
    """Run the HPO phenotype extraction pipeline on a clinical note.

    Returns structured HPO term mappings for all detected phenotype mentions.
    Negated mentions are included with ``negated: true``.

    - **HTTP 200**: Successful extraction.
    - **HTTP 422**: Validation error (empty note, note too long, invalid top_k).
    - **HTTP 503**: Pipeline not yet initialized.
    - **HTTP 500**: Unexpected internal error (non-revealing message).
    """
    if not getattr(request.app.state, "pipeline_ready", False):
        return JSONResponse(
            status_code=503,
            content={"detail": "Service unavailable: pipeline initializing"},
        )

    pipeline = request.app.state.pipeline

    try:
        result = pipeline.process(body.clinical_note, top_k=body.top_k)
    except PipelineNotReadyError:
        return JSONResponse(
            status_code=503,
            content={"detail": "Service unavailable: pipeline initializing"},
        )
    except Exception:
        logger.error("Unexpected error during HPO extraction", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    hpo_terms: list[HPOTermResult] = []
    for mention in result.mentions:
        candidates_raw = result.mappings.get(mention.text, [])
        pydantic_candidates = [
            HPOCandidate(
                hpo_id=c.hpo_id,
                hpo_label=c.hpo_label,
                confidence=c.confidence,
            )
            for c in candidates_raw
        ]
        top = candidates_raw[0] if candidates_raw else None
        hpo_terms.append(
            HPOTermResult(
                text=mention.text,
                hpo_id=top.hpo_id if top else "",
                hpo_label=top.hpo_label if top else "",
                confidence=top.confidence if top else 0.0,
                negated=mention.negated,
                candidates=pydantic_candidates,
            )
        )

    return ExtractionResponse(
        hpo_terms=hpo_terms,
        summary=result.summary,
        disclaimer=_DISCLAIMER,
    )


# ===========================================================================
# Oncology extension endpoints
# ===========================================================================

def _onc_pipeline(request: Request):
    """Return the oncology pipeline from app state, or None."""
    return getattr(request.app.state, "oncology_pipeline", None)


@router.post("/extract-clinical-concepts", response_model=ClinicalConceptResponse)
async def extract_clinical_concepts(
    body: ClinicalConceptRequest,
    request: Request,
) -> ClinicalConceptResponse:
    """
    Extract oncology-specific symptoms and adverse events from a clinical note.

    Detects 18 symptom/AE categories including fatigue, nausea, fever,
    peripheral neuropathy, dyspnea, bleeding, infection, thrombosis,
    pneumonitis, colitis, hepatotoxicity, and more.

    - **HTTP 200**: Successful extraction.
    - **HTTP 503**: Oncology pipeline not yet initialized.
    - **HTTP 500**: Unexpected internal error.
    """
    pipeline = _onc_pipeline(request)
    if pipeline is None or not pipeline.is_ready:
        return JSONResponse(
            status_code=503,
            content={"detail": "Service unavailable: oncology pipeline not ready"},
        )

    try:
        result = pipeline.extract_concepts(body.clinical_note)
    except Exception:
        logger.error("Error in extract_clinical_concepts", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    mentions = [
        OncologyMentionItem(
            text=m.text,
            category=m.category,
            start=m.start,
            end=m.end,
            negated=m.negated,
            grade=m.grade,
        )
        for m in result.mentions
    ]

    return ClinicalConceptResponse(
        mentions=mentions,
        categories_present=sorted(result.categories_present),
        symptom_count=result.symptom_count,
        max_grade=result.max_grade,
    )


@router.post("/predict-adverse-event-risk", response_model=RiskPredictionResponse)
async def predict_adverse_event_risk(
    body: PatientFeatures,
    request: Request,
) -> RiskPredictionResponse:
    """
    Predict the probability of a treatment-related adverse event for a patient.

    Combines structured EHR features (demographics, labs, symptom scores) with
    NLP-derived flags extracted from an optional clinical note.

    Returns a calibrated probability and risk tier (high / medium / low).

    - **HTTP 200**: Successful prediction.
    - **HTTP 503**: Oncology pipeline not ready.
    - **HTTP 500**: Internal server error.
    """
    pipeline = _onc_pipeline(request)
    if pipeline is None or not pipeline.is_ready:
        return JSONResponse(
            status_code=503,
            content={"detail": "Service unavailable: oncology pipeline not ready"},
        )

    try:
        patient_dict = body.model_dump()
        note = patient_dict.pop("clinical_note", None) or ""
        result = pipeline.predict_risk(patient_dict, clinical_note=note)
    except Exception:
        logger.error("Error in predict_adverse_event_risk", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return RiskPredictionResponse(
        patient_id=body.patient_id,
        risk_probability=result.risk_probability,
        risk_tier=result.risk_tier,
        model_used=result.model_used,
    )


@router.post("/summarize-risk", response_model=RiskSummaryResponse)
async def summarize_risk(
    body: RiskSummaryRequest,
    request: Request,
) -> RiskSummaryResponse:
    """
    Generate a clinician-facing risk summary for a patient.

    Combines extracted symptoms, abnormal lab flags, and the predicted risk
    score into a structured plain-language paragraph with a recommended
    next clinical step.

    Optionally augmented by an LLM (when ``LLM_SUMMARY_ENABLED=true``).

    - **HTTP 200**: Summary generated.
    - **HTTP 503**: Oncology pipeline not ready.
    - **HTTP 500**: Internal server error.
    """
    pipeline = _onc_pipeline(request)
    if pipeline is None or not pipeline.is_ready:
        return JSONResponse(
            status_code=503,
            content={"detail": "Service unavailable: oncology pipeline not ready"},
        )

    try:
        patient_dict = body.patient.model_dump()
        note = body.clinical_note or patient_dict.pop("clinical_note", None) or ""

        # Extract concepts
        concepts = pipeline.extract_concepts(note)

        # Predict risk
        risk_result = pipeline.predict_risk(patient_dict, clinical_note=note)

        # Summarize
        summary_result = pipeline.summarize_risk(patient_dict, concepts, risk_result)
    except Exception:
        logger.error("Error in summarize_risk", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return RiskSummaryResponse(
        patient_id=body.patient.patient_id,
        summary=summary_result.summary_text,
        risk_probability=risk_result.risk_probability,
        risk_tier=risk_result.risk_tier,
        disclaimer=summary_result.disclaimer,
    )


# ===========================================================================
# Health check
# ===========================================================================

@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """
    Pipeline readiness check.

    Returns the status of both the HPO pipeline and the oncology pipeline.

    - **HTTP 200**: All initialized pipelines are ready.
    - **HTTP 503**: At least one pipeline is not ready.
    """
    hpo_ready = getattr(request.app.state, "pipeline_ready", False)
    onc_pipeline = _onc_pipeline(request)
    onc_ready = onc_pipeline is not None and onc_pipeline.is_ready

    status = {
        "hpo_pipeline": "ok" if hpo_ready else "unavailable",
        "oncology_pipeline": "ok" if onc_ready else "unavailable",
    }

    if hpo_ready or onc_ready:
        return JSONResponse(status_code=200, content={"status": "ok", **status})
    return JSONResponse(status_code=503, content={"status": "unavailable", **status})

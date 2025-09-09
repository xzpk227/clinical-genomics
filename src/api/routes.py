"""
Route handlers for the Clinical Phenotype Extraction and HPO Mapping Pipeline API.

Endpoints:
  POST /extract-phenotypes  — run the pipeline on a clinical note
  GET  /health              — pipeline readiness check
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from src.api.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    HPOCandidate,
    HPOTermResult,
)
from src.pipeline import PipelineNotReadyError

logger = logging.getLogger(__name__)

router = APIRouter()

_DISCLAIMER = (
    "This output is for clinical decision support only. "
    "It does not constitute a medical diagnosis. "
    "Always involve a qualified clinician."
)


@router.post("/extract-phenotypes", response_model=ExtractionResponse)
async def extract_phenotypes(
    body: ExtractionRequest,
    request: Request,
) -> ExtractionResponse:
    """Run the phenotype extraction pipeline on a clinical note.

    Returns structured HPO term mappings for all detected phenotype mentions.
    Negated mentions are included with ``negated: true``.

    - **HTTP 200**: Successful extraction.
    - **HTTP 422**: Validation error (empty note, note too long, invalid top_k).
    - **HTTP 503**: Pipeline not yet initialized.
    - **HTTP 500**: Unexpected internal error (non-revealing message).
    """
    # Check pipeline readiness
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
        # Log without exposing clinical_note content or internal details
        logger.error("Unexpected error during extraction", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # Build response — map ExtractionResult → ExtractionResponse
    hpo_terms: list[HPOTermResult] = []
    for mention in result.mentions:
        candidates_raw = result.mappings.get(mention.text, [])

        # Build Pydantic HPOCandidate list
        pydantic_candidates = [
            HPOCandidate(
                hpo_id=c.hpo_id,
                hpo_label=c.hpo_label,
                confidence=c.confidence,
            )
            for c in candidates_raw
        ]

        # Top-1 values (fallback to empty strings / 0.0 when no candidates)
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


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Pipeline readiness check.

    - **HTTP 200**: Pipeline is initialized and ready.
    - **HTTP 503**: Pipeline is not yet ready.
    """
    if getattr(request.app.state, "pipeline_ready", False):
        return JSONResponse(status_code=200, content={"status": "ok"})
    return JSONResponse(status_code=503, content={"status": "unavailable"})

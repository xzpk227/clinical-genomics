"""
FastAPI application entry point for the Clinical Genomics + Oncology AI Pipeline.

Exposes:
  POST /extract-phenotypes          — HPO term extraction from clinical notes
  POST /extract-clinical-concepts   — Oncology symptom/AE extraction
  POST /predict-adverse-event-risk  — Adverse event risk prediction
  POST /summarize-risk              — Clinician-facing risk summary
  GET  /health                      — Readiness check for all pipelines
  GET  /docs                        — Auto-generated OpenAPI UI (FastAPI default)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.pipeline import Pipeline, PipelineConfig
from src.oncology_pipeline import OncologyConfig, OncologyPipeline

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Initializes both the HPO pipeline and the oncology pipeline on startup.
    Failures in either pipeline are logged but do not block startup — the
    /health endpoint reports which pipelines are available.
    """
    # --- HPO pipeline ---
    hpo_config = PipelineConfig()
    try:
        logger.info("Initializing HPO pipeline …")
        hpo_pipeline = Pipeline(hpo_config)
        app.state.pipeline = hpo_pipeline
        app.state.pipeline_ready = True
        logger.info("HPO pipeline ready.")
    except Exception as exc:
        logger.error("HPO pipeline initialization failed: %s", exc)
        app.state.pipeline = None
        app.state.pipeline_ready = False

    # --- Oncology pipeline ---
    onc_config = OncologyConfig()
    try:
        logger.info("Initializing Oncology pipeline …")
        onc_pipeline = OncologyPipeline(onc_config)
        app.state.oncology_pipeline = onc_pipeline
        logger.info("Oncology pipeline ready (is_ready=%s).", onc_pipeline.is_ready)
    except Exception as exc:
        logger.error("Oncology pipeline initialization failed: %s", exc)
        app.state.oncology_pipeline = None

    yield

    # --- shutdown ---
    app.state.pipeline_ready = False
    logger.info("Pipelines shut down.")


app = FastAPI(
    title="Clinical Genomics + Oncology AI Pipeline",
    description=(
        "Accepts free-text clinical notes and structured patient data. "
        "Returns HPO term mappings, oncology symptom extraction, "
        "adverse event risk predictions, and clinician-facing summaries. "
        "For research and decision support only — not a diagnostic tool."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

from src.api.routes import router  # noqa: E402 — must come after app is defined

app.include_router(router)

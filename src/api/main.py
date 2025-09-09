"""
FastAPI application entry point for the Clinical Phenotype Extraction
and HPO Mapping Pipeline.

Exposes:
  POST /extract-phenotypes  — main extraction endpoint
  GET  /health              — readiness check
  GET  /docs                — auto-generated OpenAPI UI (FastAPI default)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.pipeline import Pipeline, PipelineConfig

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager.

    Initializes the Pipeline on startup and marks it ready.
    On shutdown, marks the pipeline as not ready.
    If initialization fails, the app still starts but /health returns 503.
    """
    # --- startup ---
    config = PipelineConfig()
    try:
        logger.info("Initializing pipeline …")
        pipeline = Pipeline(config)
        app.state.pipeline = pipeline
        app.state.pipeline_ready = True
        logger.info("Pipeline ready.")
    except Exception as exc:
        logger.error("Pipeline initialization failed: %s", exc)
        app.state.pipeline = None
        app.state.pipeline_ready = False

    yield

    # --- shutdown ---
    app.state.pipeline_ready = False
    logger.info("Pipeline shut down.")


app = FastAPI(
    title="Clinical Phenotype Extraction and HPO Mapping Pipeline",
    description=(
        "Accepts free-text clinical notes and returns structured HPO term mappings. "
        "For decision support only — not a diagnostic tool."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

from src.api.routes import router  # noqa: E402 — must come after app is defined

app.include_router(router)

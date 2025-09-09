"""
Pipeline orchestrator for the Clinical Phenotype Extraction and HPO Mapping Pipeline.

This module defines PipelineConfig and the Pipeline class that coordinates all
pipeline components: Extractor, NegationHandler, Mapper, and optional LLM summary.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import spacy.tokens
    from sentence_transformers import SentenceTransformer

    from src.data.build_hpo_db import HPODatabase
    from src.extraction.extractor import Extractor, Mention
    from src.extraction.negation import NegationHandler
    from src.mapping.mapper import HPOCandidate, Mapper

logger = logging.getLogger(__name__)

_LLM_DISCLAIMER = (
    "This output is for decision support only and should not be used to diagnose patients."
)

_LLM_PROMPT_TEMPLATE = (
    "You are a clinical genomics assistant. Given the following list of HPO terms "
    "extracted from a clinical note, provide a brief plain-language summary for a "
    "clinician. Include only the confirmed (non-negated) findings. "
    "Always end your response with: '{disclaimer}'\n\n"
    "HPO terms:\n{hpo_terms}\n\nSummary:"
)


# ---------------------------------------------------------------------------
# LLM Summarizer
# ---------------------------------------------------------------------------


class LLMSummarizer:
    """Optional post-processing step that generates a plain-language summary.

    Loads a local open medical language model (e.g., MedGemma) and generates
    a clinician-friendly summary from the structured HPO term list.

    The summary always includes the required disclaimer string.
    On failure, logs the error and returns None — never exposes model internals.
    """

    def __init__(self, model_name: str) -> None:
        """Load the LLM model.

        Args:
            model_name: HuggingFace model identifier (e.g., "google/medgemma-4b-it").

        Raises:
            LLMLoadError: If the model fails to load.
        """
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import]

            self._pipe = hf_pipeline(
                "text-generation",
                model=model_name,
                max_new_tokens=256,
                do_sample=False,
            )
            self._model_name = model_name
            logger.info("LLM summarizer loaded: %s", model_name)
        except Exception as exc:
            raise LLMLoadError(
                f"Failed to load LLM model '{model_name}': {exc}"
            ) from exc

    def summarize(self, hpo_terms: list[dict]) -> str | None:
        """Generate a plain-language summary from a list of HPO term dicts.

        Each dict should have keys: ``text``, ``hpo_id``, ``hpo_label``, ``negated``.

        On failure, logs the error and returns None without raising.

        Args:
            hpo_terms: List of HPO term result dicts.

        Returns:
            A summary string including the disclaimer, or None on failure.
        """
        try:
            # Format the HPO terms for the prompt
            confirmed = [t for t in hpo_terms if not t.get("negated", False)]
            if not confirmed:
                return (
                    f"No confirmed phenotypes were identified. {_LLM_DISCLAIMER}"
                )

            terms_text = "\n".join(
                f"- {t['hpo_label']} ({t['hpo_id']})" for t in confirmed
            )
            prompt = _LLM_PROMPT_TEMPLATE.format(
                disclaimer=_LLM_DISCLAIMER,
                hpo_terms=terms_text,
            )

            output = self._pipe(prompt)
            generated = output[0]["generated_text"]

            # Extract only the generated portion after the prompt
            if prompt in generated:
                summary = generated[len(prompt):].strip()
            else:
                summary = generated.strip()

            # Ensure disclaimer is present
            if _LLM_DISCLAIMER not in summary:
                summary = f"{summary}\n\n{_LLM_DISCLAIMER}"

            return summary

        except Exception as exc:
            # Log without exposing model internals or note content
            logger.error(
                "LLM summary generation failed (model=%s): %s",
                self._model_name,
                type(exc).__name__,
            )
            return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the clinical phenotype extraction pipeline.

    All fields can be overridden via environment variables. The env var name
    is the uppercase version of the field name (e.g., ``LLM_SUMMARY_ENABLED``
    maps to ``llm_summary_enabled``).
    """

    hpo_database_path: str = "data/hpo_database.json"
    faiss_index_path: str = "data/hpo_index.faiss"
    id_map_path: str = "data/hpo_id_map.json"
    embedding_model_name: str = "FremyCompany/BioLORD-2023"
    negation_cues_path: str = "config/negation_cues.json"
    top_k_default: int = 3
    llm_summary_enabled: bool = False
    llm_model_name: str = "google/medgemma-4b-it"
    regression_accuracy_threshold: float = 0.70

    def __post_init__(self) -> None:
        """Apply environment variable overrides after dataclass initialization."""
        # String fields
        for str_field in (
            "hpo_database_path",
            "faiss_index_path",
            "id_map_path",
            "embedding_model_name",
            "negation_cues_path",
            "llm_model_name",
        ):
            env_val = os.environ.get(str_field.upper())
            if env_val is not None:
                object.__setattr__(self, str_field, env_val)

        # Integer fields
        for int_field in ("top_k_default",):
            env_val = os.environ.get(int_field.upper())
            if env_val is not None:
                object.__setattr__(self, int_field, int(env_val))

        # Boolean fields
        for bool_field in ("llm_summary_enabled",):
            env_val = os.environ.get(bool_field.upper())
            if env_val is not None:
                object.__setattr__(self, bool_field, env_val.lower() in ("1", "true", "yes"))

        # Float fields
        for float_field in ("regression_accuracy_threshold",):
            env_val = os.environ.get(float_field.upper())
            if env_val is not None:
                object.__setattr__(self, float_field, float(env_val))


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class PipelineNotReadyError(RuntimeError):
    """Raised when Pipeline.process() is called before initialization completes."""


class HPODatabaseError(RuntimeError):
    """Raised when the HPO database file is missing or malformed."""


class ModelLoadError(RuntimeError):
    """Raised when the embedding model fails to load."""


class IndexNotFoundError(RuntimeError):
    """Raised when the FAISS index file is missing."""


class LLMLoadError(RuntimeError):
    """Raised when the LLM model fails to load (only when LLM summary is enabled)."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """Result returned by Pipeline.process().

    Attributes:
        mentions:  All detected phenotype mentions (negated and non-negated).
        mappings:  Mapping from mention text to a ranked list of HPO candidates.
        summary:   Optional plain-language summary generated by the LLM layer.
                   ``None`` when LLM summary is disabled or unavailable.
    """

    mentions: list  # list[Mention]
    mappings: dict  # dict[str, list[HPOCandidate]]  — mention.text → candidates
    summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Coordinates all pipeline components.

    Initialization loads every required component in order.  If any step
    fails, the appropriate custom exception is raised and ``is_ready`` remains
    ``False``.  Once all components are loaded successfully, ``is_ready`` is
    set to ``True``.

    Usage::

        config = PipelineConfig()
        pipeline = Pipeline(config)
        result = pipeline.process("Patient has seizures and hypotonia.")
    """

    is_ready: bool

    def __init__(self, config: PipelineConfig) -> None:
        """Load and initialize all pipeline components.

        Steps (in order):
        1. Load HPO database.
        2. Initialize spaCy blank model.
        3. Initialize Extractor.
        4. Load BioLORD-2023 SentenceTransformer.
        5. Load FAISS index and ID map.
        6. Initialize Mapper.
        7. Initialize NegationHandler.
        8. Optionally load LLM (feature-flagged).
        9. Set ``self.is_ready = True``.

        Args:
            config: Pipeline configuration.

        Raises:
            HPODatabaseError: If the HPO database file is missing or malformed.
            ModelLoadError: If the embedding model fails to load.
            IndexNotFoundError: If the FAISS index or ID map file is missing.
            LLMLoadError: If the LLM model is explicitly configured and fails to load.
        """
        self.config = config
        self.is_ready = False

        # ------------------------------------------------------------------
        # Step 1: Load HPO database
        # ------------------------------------------------------------------
        logger.info("Loading HPO database from %s …", config.hpo_database_path)
        try:
            from src.data.build_hpo_db import load_hpo_database
            hpo_db = load_hpo_database(config.hpo_database_path)
            logger.info(
                "HPO database loaded: version=%s, term_count=%d",
                hpo_db.version,
                len(hpo_db.terms),
            )
        except HPODatabaseError:
            raise
        except Exception as exc:
            raise HPODatabaseError(
                f"Failed to load HPO database from {config.hpo_database_path}: {exc}"
            ) from exc

        self._hpo_db = hpo_db

        # ------------------------------------------------------------------
        # Step 2: Initialize spaCy blank model
        # ------------------------------------------------------------------
        logger.info("Initializing spaCy blank English model …")
        try:
            import spacy
            self._nlp = spacy.blank("en")
            logger.info("spaCy blank model initialized.")
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to initialize spaCy blank model: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Step 3: Initialize Extractor
        # ------------------------------------------------------------------
        logger.info("Initializing Extractor …")
        try:
            from src.extraction.extractor import Extractor
            self._extractor = Extractor(hpo_db)
            logger.info("Extractor initialized.")
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to initialize Extractor: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Step 4: Load BioLORD-2023 embedding model
        # ------------------------------------------------------------------
        logger.info(
            "Loading embedding model %s …", config.embedding_model_name
        )
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.embedding_model_name)
            logger.info(
                "Embedding model loaded: %s", config.embedding_model_name
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load embedding model '{config.embedding_model_name}': {exc}"
            ) from exc

        self._model = model

        # ------------------------------------------------------------------
        # Step 5: Load FAISS index and ID map
        # ------------------------------------------------------------------
        logger.info(
            "Loading FAISS index from %s and ID map from %s …",
            config.faiss_index_path,
            config.id_map_path,
        )
        try:
            from src.mapping.build_index import load_faiss_index
            index, id_map = load_faiss_index(
                config.faiss_index_path, config.id_map_path
            )
            logger.info(
                "FAISS index loaded: ntotal=%d, id_map_entries=%d",
                index.ntotal,
                len(id_map),
            )
        except IndexNotFoundError:
            raise
        except Exception as exc:
            raise IndexNotFoundError(
                f"Failed to load FAISS index: {exc}"
            ) from exc

        self._index = index
        self._id_map = id_map

        # ------------------------------------------------------------------
        # Step 6: Initialize Mapper
        # ------------------------------------------------------------------
        logger.info("Initializing Mapper (top_k=%d) …", config.top_k_default)
        try:
            from src.mapping.mapper import Mapper
            self._mapper = Mapper(
                model=model,
                index=index,
                id_map=id_map,
                top_k=config.top_k_default,
            )
            logger.info("Mapper initialized.")
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to initialize Mapper: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Step 7: Initialize NegationHandler
        # ------------------------------------------------------------------
        logger.info("Initializing NegationHandler …")
        try:
            from src.extraction.negation import NegationHandler
            self._negation_handler = NegationHandler()
            logger.info("NegationHandler initialized.")
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to initialize NegationHandler: {exc}"
            ) from exc

        # ------------------------------------------------------------------
        # Step 8: Optionally load LLM
        # ------------------------------------------------------------------
        self._llm: LLMSummarizer | None = None
        if config.llm_summary_enabled:
            if config.llm_model_name:
                logger.info(
                    "Loading LLM summarizer (model=%s) …", config.llm_model_name
                )
                try:
                    self._llm = LLMSummarizer(config.llm_model_name)
                    logger.info("LLM summarizer loaded.")
                except LLMLoadError:
                    raise
                except Exception as exc:
                    raise LLMLoadError(
                        f"Failed to load LLM model '{config.llm_model_name}': {exc}"
                    ) from exc
            else:
                logger.warning(
                    "LLM summary is enabled but no model name is configured. "
                    "The summary field will be None for all responses."
                )

        # ------------------------------------------------------------------
        # Step 9: Mark pipeline as ready
        # ------------------------------------------------------------------
        self.is_ready = True
        logger.info("Pipeline initialization complete. is_ready=True")

    def process(
        self,
        clinical_note: str,
        top_k: int = 3,
    ) -> ExtractionResult:
        """Run the full extraction pipeline on a clinical note.

        Steps:
        1. Raise ``PipelineNotReadyError`` if not ready.
        2. Extract phenotype mentions with the Extractor.
        3. Process the note through spaCy to produce a Doc.
        4. Annotate mentions with negation via the NegationHandler.
        5. Map each mention to HPO candidates via the Mapper.
        6. Build the mappings dict (mention.text → candidates).
        7. Optionally generate an LLM summary (stub: returns None).
        8. Return an ExtractionResult.

        Args:
            clinical_note: Free-text clinical note to process.
            top_k: Number of HPO candidates to return per mention.
                   Overrides the mapper's default for this call.

        Returns:
            An ExtractionResult with mentions, mappings, and optional summary.

        Raises:
            PipelineNotReadyError: If the pipeline has not been initialized.
        """
        # Step 1: Guard — pipeline must be ready
        if not self.is_ready:
            raise PipelineNotReadyError(
                "Pipeline is not ready. Ensure initialization completed successfully."
            )

        # Step 2: Extract phenotype mentions
        logger.debug("Extracting mentions from clinical note (length=%d) …", len(clinical_note))
        mentions = self._extractor.extract(clinical_note)
        logger.debug("Extracted %d mention(s).", len(mentions))

        # Step 3: Process the note through spaCy to produce a Doc
        doc = self._nlp(clinical_note)

        # Step 4: Annotate mentions with negation
        if mentions:
            mentions = self._negation_handler.annotate(doc, mentions)
            logger.debug(
                "Negation annotation complete. Negated: %d/%d",
                sum(1 for m in mentions if m.negated),
                len(mentions),
            )

        # Step 5 & 6: Map each mention to HPO candidates and build mappings dict
        # If top_k differs from the mapper's default, temporarily override it.
        original_top_k = self._mapper._top_k
        if top_k != original_top_k:
            self._mapper._top_k = top_k

        mappings: dict = {}
        try:
            for mention in mentions:
                candidates = self._mapper.map(mention)
                mappings[mention.text] = candidates
                logger.debug(
                    "Mapped mention %r → %d candidate(s).", mention.text, len(candidates)
                )
        finally:
            # Restore original top_k
            if top_k != original_top_k:
                self._mapper._top_k = original_top_k

        # Step 7: Optionally generate LLM summary
        summary: Optional[str] = None
        if self.config.llm_summary_enabled and self._llm is not None:
            hpo_terms_for_llm = [
                {
                    "text": mention.text,
                    "hpo_id": result_mappings[0].hpo_id if result_mappings else "",
                    "hpo_label": result_mappings[0].hpo_label if result_mappings else "",
                    "negated": mention.negated,
                }
                for mention in mentions
                for result_mappings in [mappings.get(mention.text, [])]
            ]
            summary = self._llm.summarize(hpo_terms_for_llm)

        # Step 8: Return ExtractionResult
        return ExtractionResult(
            mentions=mentions,
            mappings=mappings,
            summary=summary,
        )

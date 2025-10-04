"""
Oncology clinical AI pipeline orchestrator.

Coordinates three processing stages:
  1. Clinical concept extraction  — OncologyExtractor detects symptoms and AEs
     from free-text clinical notes; NegationHandler annotates negations.
  2. Adverse event risk prediction — AdverseEventModel returns calibrated
     probability estimates for each patient.
  3. Clinician-facing risk summary — structured template (+ optional LLM) that
     combines extracted symptoms, abnormal lab flags, and the risk score into
     a plain-language paragraph with a recommended next step.

Design goals
------------
  - Additive / backward-compatible: existing HPO pipeline is unchanged.
  - Fail-safe: each stage degrades gracefully; errors are logged, not raised.
  - Stateless inference: the pipeline loads models once at startup and is
    called concurrently by FastAPI request handlers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Recommended-next-step lookup by top risk tier
_NEXT_STEP_BY_RISK = {
    "high":   "Urgent clinical review recommended. Consider same-day assessment, "
              "CBC with differential, and dose modification per institutional protocol.",
    "medium": "Close monitoring indicated. Schedule follow-up within 48–72 hours "
              "and repeat labs at next visit.",
    "low":    "Continue current treatment. Routine monitoring at scheduled visit.",
}

# Lab abnormality thresholds (simplified CTCAE-aligned)
_LAB_THRESHOLDS: Dict[str, tuple] = {
    "anc":        (None, 1.0),     # ANC < 1.0 = neutropenia
    "hemoglobin": (None, 10.0),    # Hgb < 10.0 = anemia
    "platelets":  (None, 100.0),   # Plt < 100 = thrombocytopenia
    "alt":        (80.0, None),    # ALT > 80 = hepatic elevation
    "creatinine": (1.3, None),     # Creat > 1.3 = renal impairment
    "bilirubin":  (1.5, None),     # Bili > 1.5 = elevated
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OncologyConfig:
    """
    Configuration for the oncology AI pipeline.

    All string/bool fields can be overridden via environment variables
    (uppercase field name, e.g. AE_MODEL_DIR → ae_model_dir).
    """
    ae_model_dir: str = "data/model_outputs"
    risk_threshold_high:   float = 0.60
    risk_threshold_medium: float = 0.35
    llm_summary_enabled: bool = False
    llm_model_name: str = "google/medgemma-4b-it"

    def __post_init__(self) -> None:
        for str_field in ("ae_model_dir", "llm_model_name"):
            val = os.environ.get(str_field.upper())
            if val:
                object.__setattr__(self, str_field, val)
        for float_field in ("risk_threshold_high", "risk_threshold_medium"):
            val = os.environ.get(float_field.upper())
            if val:
                object.__setattr__(self, float_field, float(val))
        for bool_field in ("llm_summary_enabled",):
            val = os.environ.get(bool_field.upper())
            if val:
                object.__setattr__(self, bool_field, val.lower() in ("1", "true", "yes"))


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClinicalConceptResult:
    """Output of the concept extraction stage."""
    mentions: list         # list[OncologyMention]
    categories_present: set
    symptom_count: int
    max_grade: Optional[int]
    has_flags: Dict[str, bool] = field(default_factory=dict)


@dataclass
class RiskPredictionResult:
    """Output of the risk prediction stage."""
    risk_probability: float
    risk_tier: str          # "high" | "medium" | "low"
    model_used: str


@dataclass
class RiskSummaryResult:
    """Output of the risk summary stage."""
    summary_text: str
    disclaimer: str = (
        "This summary is generated for clinical decision support only. "
        "It does not constitute a diagnosis. Always involve a qualified clinician."
    )


# ---------------------------------------------------------------------------
# Oncology Pipeline
# ---------------------------------------------------------------------------

class OncologyPipeline:
    """
    End-to-end oncology clinical AI pipeline.

    Usage::

        config = OncologyConfig()
        pipeline = OncologyPipeline(config)

        # Concept extraction
        concepts = pipeline.extract_concepts(clinical_note)

        # Risk prediction (requires structured patient data dict)
        risk = pipeline.predict_risk(patient_dict, clinical_note)

        # Clinician summary
        summary = pipeline.summarize_risk(patient_dict, concepts, risk)
    """

    is_ready: bool

    def __init__(self, config: OncologyConfig) -> None:
        self.config = config
        self.is_ready = False
        self._extractor = None
        self._negation_handler = None
        self._ae_model = None
        self._nlp = None

        # Step 1: spaCy
        try:
            import spacy
            self._nlp = spacy.blank("en")
        except Exception as exc:
            logger.error("spaCy init failed: %s", exc)

        # Step 2: NegationHandler
        try:
            from src.extraction.negation import NegationHandler
            self._negation_handler = NegationHandler()
        except Exception as exc:
            logger.warning("NegationHandler unavailable: %s", exc)

        # Step 3: OncologyExtractor
        try:
            from src.extraction.oncology_extractor import OncologyExtractor
            self._extractor = OncologyExtractor()
            logger.info("OncologyExtractor initialized.")
        except Exception as exc:
            logger.error("OncologyExtractor init failed: %s", exc)

        # Step 4: AdverseEventModel (optional — only if saved model exists)
        model_dir = Path(config.ae_model_dir)
        if (model_dir / "xgb_pipeline.joblib").exists() or (model_dir / "lr_pipeline.joblib").exists():
            try:
                from src.models.adverse_event_model import AdverseEventModel
                self._ae_model = AdverseEventModel.load(str(model_dir))
                logger.info("AdverseEventModel loaded from %s.", config.ae_model_dir)
            except Exception as exc:
                logger.warning("AdverseEventModel load failed: %s", exc)
        else:
            logger.info("No trained AE model found at %s; risk scoring will use heuristic.", config.ae_model_dir)

        self.is_ready = self._extractor is not None

    # ------------------------------------------------------------------
    # Stage 1: Clinical concept extraction
    # ------------------------------------------------------------------

    def extract_concepts(self, clinical_note: str) -> ClinicalConceptResult:
        """
        Extract oncology symptoms and adverse events from a clinical note.

        Args:
            clinical_note: Free-text clinical note.

        Returns:
            ClinicalConceptResult with detected mentions and category flags.
        """
        if not self._extractor:
            logger.warning("OncologyExtractor not initialized; returning empty result.")
            return ClinicalConceptResult(
                mentions=[], categories_present=set(), symptom_count=0, max_grade=None
            )

        structured = self._extractor.extract_structured(
            clinical_note,
            negation_handler=self._negation_handler,
        )

        has_flags = {k: v for k, v in structured.items() if k.startswith("has_")}

        return ClinicalConceptResult(
            mentions=structured["mentions"],
            categories_present=structured["categories_present"],
            symptom_count=structured["symptom_count"],
            max_grade=structured.get("max_grade"),
            has_flags=has_flags,
        )

    # ------------------------------------------------------------------
    # Stage 2: Risk prediction
    # ------------------------------------------------------------------

    def predict_risk(
        self,
        patient: Dict[str, Any],
        clinical_note: str = "",
    ) -> RiskPredictionResult:
        """
        Predict adverse event risk probability for a patient.

        If a trained AdverseEventModel is available, it is used.
        Otherwise a heuristic scoring function is applied.

        Args:
            patient:       Dict with patient fields (age, sex, cancer_type, etc.).
            clinical_note: Clinical note text (used to extract NLP flags).

        Returns:
            RiskPredictionResult with probability and risk tier.
        """
        nlp_flags = None
        if clinical_note:
            concepts = self.extract_concepts(clinical_note)
            nlp_flags_dict = {k: v for k, v in concepts.has_flags.items()}
            nlp_flags_dict["symptom_count"] = concepts.symptom_count
            nlp_flags_dict["patient_id"] = patient.get("patient_id", "UNKNOWN")

        try:
            import pandas as pd

            if self._ae_model is not None:
                df = pd.DataFrame([patient])
                nlp_df = pd.DataFrame([nlp_flags_dict]) if nlp_flags_dict else None
                prob = float(self._ae_model.predict_proba(df, nlp_df)[0])
                model_used = "xgboost"
            else:
                prob = self._heuristic_risk(patient, nlp_flags_dict or {})
                model_used = "heuristic"
        except Exception as exc:
            logger.error("Risk prediction failed: %s", exc)
            prob = self._heuristic_risk(patient, nlp_flags_dict or {})
            model_used = "heuristic_fallback"

        tier = self._risk_tier(prob)
        return RiskPredictionResult(
            risk_probability=round(prob, 4),
            risk_tier=tier,
            model_used=model_used,
        )

    def _heuristic_risk(self, patient: Dict[str, Any], nlp_flags: Dict[str, Any]) -> float:
        """Simple heuristic risk score when no trained model is available."""
        score = 0.30   # base rate
        stage = patient.get("stage", "II")
        if stage == "IV":
            score += 0.12
        elif stage == "III":
            score += 0.06
        ecog = int(patient.get("ecog_score", 1))
        score += ecog * 0.05
        age = int(patient.get("age", 60))
        if age > 70:
            score += 0.05
        # Lab signals
        if float(patient.get("anc", 2.0)) < 1.0:
            score += 0.10
        if float(patient.get("hemoglobin", 12.0)) < 10.0:
            score += 0.07
        # NLP flags
        score += nlp_flags.get("symptom_count", 0) * 0.02
        if nlp_flags.get("has_fever"):
            score += 0.08
        if nlp_flags.get("has_infection"):
            score += 0.10
        return float(np.clip(score, 0.0, 0.99))

    def _risk_tier(self, prob: float) -> str:
        if prob >= self.config.risk_threshold_high:
            return "high"
        if prob >= self.config.risk_threshold_medium:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Stage 3: Clinician summary
    # ------------------------------------------------------------------

    def summarize_risk(
        self,
        patient: Dict[str, Any],
        concepts: ClinicalConceptResult,
        risk: RiskPredictionResult,
    ) -> RiskSummaryResult:
        """
        Generate a structured, clinician-facing risk summary.

        Combines extracted symptoms, abnormal lab flags, and risk score into
        a plain-language paragraph with a recommended next step.

        Args:
            patient:  Patient data dict.
            concepts: Output of extract_concepts().
            risk:     Output of predict_risk().

        Returns:
            RiskSummaryResult with summary_text and disclaimer.
        """
        summary = self._build_summary_text(patient, concepts, risk)

        # Optionally enhance with LLM
        if self.config.llm_summary_enabled:
            llm_text = self._llm_enhance(summary, patient, concepts, risk)
            if llm_text:
                summary = llm_text

        return RiskSummaryResult(summary_text=summary)

    def _build_summary_text(
        self,
        patient: Dict[str, Any],
        concepts: ClinicalConceptResult,
        risk: RiskPredictionResult,
    ) -> str:
        """Build a structured plain-language summary from structured inputs."""
        age  = patient.get("age", "unknown")
        sex  = "male" if patient.get("sex", "M") == "M" else "female"
        cancer = str(patient.get("cancer_type", "cancer")).replace("_", " ").upper()
        treatment = str(patient.get("treatment", "treatment")).replace("_", "/")
        pid  = patient.get("patient_id", "")

        # Symptom list (non-negated)
        confirmed = sorted({m.category for m in concepts.mentions if not m.negated})
        symptom_str = (
            ", ".join(c.replace("_", " ") for c in confirmed)
            if confirmed else "none documented"
        )

        # Abnormal labs
        abnormal_labs = []
        for lab, (upper, lower) in _LAB_THRESHOLDS.items():
            val = patient.get(lab)
            if val is None:
                continue
            val = float(val)
            if lower is not None and val < lower:
                abnormal_labs.append(f"{lab} {val} (below threshold {lower})")
            elif upper is not None and val > upper:
                abnormal_labs.append(f"{lab} {val} (above threshold {upper})")
        labs_str = (", ".join(abnormal_labs) if abnormal_labs else "within normal limits")

        # Risk text
        prob_pct = f"{risk.risk_probability * 100:.0f}%"
        next_step = _NEXT_STEP_BY_RISK.get(risk.risk_tier, _NEXT_STEP_BY_RISK["medium"])

        lines = [
            f"Patient {pid}: {age}-year-old {sex} with {cancer}, currently on {treatment}.",
            f"Extracted symptoms: {symptom_str}.",
            f"Laboratory findings: {labs_str}.",
            f"Estimated adverse event risk: {prob_pct} ({risk.risk_tier.upper()} tier, "
            f"model: {risk.model_used}).",
            f"Recommended next step: {next_step}",
        ]
        return "\n".join(lines)

    def _llm_enhance(
        self,
        base_summary: str,
        patient: Dict[str, Any],
        concepts: ClinicalConceptResult,
        risk: RiskPredictionResult,
    ) -> Optional[str]:
        """
        Optionally enhance the base summary with an LLM.
        Returns None if LLM unavailable or fails.
        """
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import]
            prompt = (
                "You are an oncology clinical AI assistant. Given the following "
                "structured patient summary, write a concise, clear clinician-facing "
                "paragraph. Do not add clinical diagnoses or treatment recommendations "
                "beyond what is provided. Always end with: "
                "'For decision support only — verify with attending clinician.'\n\n"
                f"Summary:\n{base_summary}\n\nClinician paragraph:"
            )
            pipe = hf_pipeline(
                "text-generation",
                model=self.config.llm_model_name,
                max_new_tokens=200,
                do_sample=False,
            )
            output = pipe(prompt)[0]["generated_text"]
            if prompt in output:
                return output[len(prompt):].strip()
            return output.strip()
        except Exception as exc:
            logger.warning("LLM enhancement failed: %s", type(exc).__name__)
            return None

"""
Adverse event risk prediction model.

Predicts the probability of a treatment-related adverse event (any grade) for
oncology patients, combining structured EHR features with NLP-derived symptom
flags extracted from free-text clinical notes.

Model architecture
------------------
  Baseline   : Logistic Regression with L2 regularization
  Primary    : XGBoost gradient boosted trees
  Ensemble   : Soft-vote average of LR + XGBoost probabilities

Feature groups
--------------
  Demographic   : age, sex (binary)
  Clinical      : ECOG score, stage (ordinal 1–4), cycle number
  Oncology      : cancer_type (one-hot), treatment (one-hot)
  Laboratory    : WBC, ANC, hemoglobin, platelets, creatinine, ALT, bilirubin
  Symptom scores: fatigue_score, nausea_score, pain_score, dyspnea_score
  NLP flags     : has_fatigue, has_nausea, has_fever, has_neuropathy,
                  has_dyspnea, has_pain, has_bleeding, has_infection,
                  has_thrombosis, has_pneumonitis, has_colitis,
                  has_hepatotoxicity, has_mucositis, has_neutropenia,
                  has_anemia, has_thrombocytopenia, symptom_count

Evaluation metrics
------------------
  AUROC, AUPRC (primary for imbalanced setting), Brier score,
  calibration plot (Platt scaling via CalibratedClassifierCV),
  subgroup AUROC by age_group / sex / cancer_type.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

try:
    import xgboost as xgb  # type: ignore[import]
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "age", "cycle_number", "ecog_score", "stage_ordinal",
    "wbc", "anc", "hemoglobin", "platelets", "creatinine", "alt", "bilirubin",
    "fatigue_score", "nausea_score", "pain_score", "dyspnea_score",
    "symptom_count",
]

BINARY_NLP_FEATURES = [
    "has_fatigue", "has_nausea", "has_fever", "has_neuropathy",
    "has_dyspnea", "has_pain", "has_bleeding", "has_infection",
    "has_thrombosis", "has_pneumonitis", "has_colitis",
    "has_hepatotoxicity", "has_mucositis", "has_neutropenia",
    "has_anemia", "has_thrombocytopenia",
]

BINARY_DEMO_FEATURES = ["sex_binary"]

CATEGORICAL_FEATURES = ["cancer_type", "treatment"]

ALL_FEATURE_COLS = (
    NUMERIC_FEATURES
    + BINARY_NLP_FEATURES
    + BINARY_DEMO_FEATURES
    + CATEGORICAL_FEATURES
)

# Stage ordinal mapping
_STAGE_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4}


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    nlp_flags: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Prepare the feature matrix from a patient DataFrame.

    Args:
        df:        Raw patient DataFrame (from generate_ehr).
        nlp_flags: Optional DataFrame with per-patient NLP-derived binary flags
                   (columns: patient_id, has_fatigue, …, symptom_count).
                   If None, all NLP features default to 0.

    Returns:
        Feature DataFrame aligned to ALL_FEATURE_COLS.
    """
    feats = df.copy()

    # Ordinal stage
    feats["stage_ordinal"] = feats["stage"].map(_STAGE_MAP).fillna(2).astype(int)

    # Binary sex
    feats["sex_binary"] = (feats["sex"] == "M").astype(int)

    # Merge NLP flags
    if nlp_flags is not None:
        nlp_cols = ["patient_id"] + [c for c in nlp_flags.columns if c != "patient_id"]
        feats = feats.merge(nlp_flags[nlp_cols], on="patient_id", how="left")

    # Fill missing NLP flags with 0
    for col in BINARY_NLP_FEATURES + ["symptom_count"]:
        if col not in feats.columns:
            feats[col] = 0
        feats[col] = feats[col].fillna(0)

    feats = feats.fillna(0)
    return feats


def build_preprocessor() -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer:
      - StandardScaler for numeric features
      - OneHotEncoder (drop='first') for categorical features
      - Passthrough for binary features
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
             CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_NLP_FEATURES + BINARY_DEMO_FEATURES),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    auroc: float
    auprc: float
    brier: float
    calibration_fractions: List[float] = field(default_factory=list)
    calibration_mean_predicted: List[float] = field(default_factory=list)
    subgroup_auroc: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cv_auroc_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auroc":   round(self.auroc, 4),
            "auprc":   round(self.auprc, 4),
            "brier":   round(self.brier, 4),
            "cv_auroc_mean": round(float(np.mean(self.cv_auroc_scores)), 4) if self.cv_auroc_scores else None,
            "cv_auroc_std":  round(float(np.std(self.cv_auroc_scores)), 4) if self.cv_auroc_scores else None,
            "subgroup_auroc": self.subgroup_auroc,
        }


def _subgroup_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
) -> Dict[str, float]:
    """Compute AUROC for each unique group value."""
    results = {}
    for group_val in groups.unique():
        mask = groups == group_val
        if mask.sum() < 10 or y_true[mask].sum() < 2:
            continue
        try:
            results[str(group_val)] = round(roc_auc_score(y_true[mask], y_prob[mask]), 4)
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class AdverseEventModel:
    """
    Oncology adverse event risk predictor.

    Trains a Logistic Regression baseline and an XGBoost primary model,
    evaluates with stratified CV, and exposes calibrated probability estimates.

    Usage::

        model = AdverseEventModel()
        model.fit(df_train, df_test, nlp_flags_train, nlp_flags_test)
        probs = model.predict_proba(df_new)
        print(model.metrics["xgboost"].to_dict())
    """

    def __init__(self, n_cv_folds: int = 5, random_state: int = 42) -> None:
        self.n_cv_folds = n_cv_folds
        self.random_state = random_state
        self.lr_pipeline: Optional[Pipeline] = None
        self.xgb_pipeline: Optional[Pipeline] = None
        self.metrics: Dict[str, ModelMetrics] = {}
        self._feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        nlp_flags_train: Optional[pd.DataFrame] = None,
        nlp_flags_test: Optional[pd.DataFrame] = None,
    ) -> "AdverseEventModel":
        """
        Train LR and XGBoost models and compute evaluation metrics.

        Args:
            df_train / df_test: Patient DataFrames (from generate_ehr).
            nlp_flags_train / nlp_flags_test: Optional NLP-derived binary flags.

        Returns:
            self (for method chaining).
        """
        X_train = prepare_features(df_train, nlp_flags_train)
        X_test  = prepare_features(df_test,  nlp_flags_test)

        y_train = df_train["adverse_event"].astype(int).values
        y_test  = df_test["adverse_event"].astype(int).values

        logger.info(
            "Training AE model: %d train / %d test; AE rate train=%.1f%% test=%.1f%%",
            len(y_train), len(y_test),
            100 * y_train.mean(), 100 * y_test.mean(),
        )

        preprocessor = build_preprocessor()

        # --- Logistic Regression baseline ---
        lr_clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.lr_pipeline = Pipeline([
            ("pre", preprocessor),
            ("clf", CalibratedClassifierCV(lr_clf, cv=3, method="isotonic")),
        ])
        self.lr_pipeline.fit(X_train[ALL_FEATURE_COLS], y_train)

        lr_prob_test = self.lr_pipeline.predict_proba(X_test[ALL_FEATURE_COLS])[:, 1]
        self.metrics["logistic_regression"] = self._evaluate(
            y_test, lr_prob_test, X_test,
            X_train[ALL_FEATURE_COLS], y_train,
            model=self.lr_pipeline,
        )
        logger.info(
            "LR AUROC=%.4f  AUPRC=%.4f",
            self.metrics["logistic_regression"].auroc,
            self.metrics["logistic_regression"].auprc,
        )

        # --- XGBoost primary ---
        if _XGB_AVAILABLE:
            pos_weight = max(1.0, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
            xgb_clf = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=self.random_state,
                verbosity=0,
            )
            # Build new preprocessor for XGBoost pipeline
            xgb_pre = build_preprocessor()
            self.xgb_pipeline = Pipeline([
                ("pre", xgb_pre),
                ("clf", CalibratedClassifierCV(xgb_clf, cv=3, method="isotonic")),
            ])
            self.xgb_pipeline.fit(X_train[ALL_FEATURE_COLS], y_train)

            xgb_prob_test = self.xgb_pipeline.predict_proba(X_test[ALL_FEATURE_COLS])[:, 1]
            self.metrics["xgboost"] = self._evaluate(
                y_test, xgb_prob_test, X_test,
                X_train[ALL_FEATURE_COLS], y_train,
                model=self.xgb_pipeline,
            )
            logger.info(
                "XGBoost AUROC=%.4f  AUPRC=%.4f",
                self.metrics["xgboost"].auroc,
                self.metrics["xgboost"].auprc,
            )
        else:
            logger.warning("XGBoost not installed; skipping XGBoost model.")

        return self

    def _evaluate(
        self,
        y_test: np.ndarray,
        y_prob: np.ndarray,
        X_test_df: pd.DataFrame,
        X_train_transformed: pd.DataFrame,
        y_train: np.ndarray,
        model: Pipeline,
    ) -> ModelMetrics:
        """Compute AUROC, AUPRC, Brier, calibration, subgroup AUROC, and CV."""
        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)

        # Calibration curve
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)

        # Subgroup AUROC
        subgroup_metrics: Dict[str, Dict[str, float]] = {}
        for groupby_col in ("cancer_type", "sex"):
            if groupby_col in X_test_df.columns:
                subgroup_metrics[groupby_col] = _subgroup_auroc(
                    y_test, y_prob, X_test_df[groupby_col]
                )
        # Age group
        if "age" in X_test_df.columns:
            age_groups = pd.cut(
                X_test_df["age"],
                bins=[0, 50, 65, 75, 200],
                labels=["<50", "50-65", "65-75", ">75"],
            ).astype(str)
            subgroup_metrics["age_group"] = _subgroup_auroc(y_test, y_prob, age_groups)

        # 5-fold CV AUROC on training data
        cv = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        for train_idx, val_idx in cv.split(X_train_transformed, y_train):
            x_tr, x_val = X_train_transformed.iloc[train_idx], X_train_transformed.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            model.fit(x_tr, y_tr)
            p_val = model.predict_proba(x_val)[:, 1]
            cv_scores.append(roc_auc_score(y_val, p_val))

        return ModelMetrics(
            auroc=auroc,
            auprc=auprc,
            brier=brier,
            calibration_fractions=frac_pos.tolist(),
            calibration_mean_predicted=mean_pred.tolist(),
            subgroup_auroc=subgroup_metrics,
            cv_auroc_scores=cv_scores,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        df: pd.DataFrame,
        nlp_flags: Optional[pd.DataFrame] = None,
        model_name: str = "xgboost",
    ) -> np.ndarray:
        """
        Return predicted AE probability for each patient row.

        Args:
            df:         Patient DataFrame.
            nlp_flags:  Optional NLP flags DataFrame.
            model_name: "xgboost", "logistic_regression", or "ensemble".

        Returns:
            1-D numpy array of probabilities in [0, 1].
        """
        X = prepare_features(df, nlp_flags)

        if model_name == "ensemble":
            probs = []
            if self.lr_pipeline:
                probs.append(self.lr_pipeline.predict_proba(X[ALL_FEATURE_COLS])[:, 1])
            if self.xgb_pipeline:
                probs.append(self.xgb_pipeline.predict_proba(X[ALL_FEATURE_COLS])[:, 1])
            if not probs:
                raise RuntimeError("No trained models available.")
            return np.mean(probs, axis=0)

        pipeline = {"xgboost": self.xgb_pipeline, "logistic_regression": self.lr_pipeline}.get(model_name)
        if pipeline is None:
            raise ValueError(f"Model '{model_name}' not trained or unavailable.")
        return pipeline.predict_proba(X[ALL_FEATURE_COLS])[:, 1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Serialize trained pipelines and metrics to directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        if self.lr_pipeline:
            joblib.dump(self.lr_pipeline, path / "lr_pipeline.joblib")
        if self.xgb_pipeline:
            joblib.dump(self.xgb_pipeline, path / "xgb_pipeline.joblib")

        metrics_dict = {k: v.to_dict() for k, v in self.metrics.items()}
        (path / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))
        logger.info("Model saved to %s", directory)

    @classmethod
    def load(cls, directory: str) -> "AdverseEventModel":
        """Load a previously saved AdverseEventModel."""
        path = Path(directory)
        model = cls()

        lr_path = path / "lr_pipeline.joblib"
        if lr_path.exists():
            model.lr_pipeline = joblib.load(lr_path)

        xgb_path = path / "xgb_pipeline.joblib"
        if xgb_path.exists():
            model.xgb_pipeline = joblib.load(xgb_path)

        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            raw = json.loads(metrics_path.read_text())
            for k, v in raw.items():
                model.metrics[k] = ModelMetrics(
                    auroc=v.get("auroc", 0.0),
                    auprc=v.get("auprc", 0.0),
                    brier=v.get("brier", 0.0),
                    subgroup_auroc=v.get("subgroup_auroc", {}),
                    cv_auroc_scores=[],
                )
        return model

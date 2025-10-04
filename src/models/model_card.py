"""
Model card generator and validation monitoring for the adverse event risk model.

Produces a structured model card (Markdown + JSON) documenting:
  - Model details (architecture, training data, feature list)
  - Intended use and out-of-scope uses
  - Performance metrics on held-out test set
  - Calibration analysis (Platt-scaled vs. raw)
  - Subgroup / fairness analysis (age group, sex, cancer type)
  - Threshold analysis (precision, recall, F1 at operating points)
  - Ethical considerations and caveats
  - Recommendations for clinical deployment

References: Mitchell et al. (2019) "Model Cards for Model Reporting"
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------

def threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Return precision, recall, F1, and alert volume at each threshold.

    Args:
        y_true:     Binary ground-truth labels.
        y_prob:     Predicted probabilities.
        thresholds: List of operating thresholds (default: 0.10 to 0.90 in 0.05 steps).

    Returns:
        DataFrame with columns: threshold, precision, recall, f1, n_alerted,
        true_positives, false_positives, false_negatives, specificity.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]

    rows = []
    n_pos = int(y_true.sum())

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())

        precision = tp / max(tp + fp, 1)
        recall    = tp / max(n_pos, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        specificity = tn / max(tn + fp, 1)

        rows.append({
            "threshold":       t,
            "precision":       round(precision, 4),
            "recall":          round(recall, 4),
            "f1":              round(f1, 4),
            "specificity":     round(specificity, 4),
            "n_alerted":       int(pred.sum()),
            "true_positives":  tp,
            "false_positives": fp,
            "false_negatives": fn,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Subgroup / fairness analysis
# ---------------------------------------------------------------------------

def subgroup_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metadata: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Compute AUROC, AUPRC, Brier score per subgroup.

    Args:
        y_true:   Binary labels.
        y_prob:   Predicted probabilities.
        metadata: DataFrame with columns: age, sex, cancer_type, stage.

    Returns:
        Dict of {groupby_col: DataFrame with per-group metrics}.
    """
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    results: Dict[str, pd.DataFrame] = {}

    groupby_cols = {
        "sex": metadata.get("sex"),
        "cancer_type": metadata.get("cancer_type"),
        "stage": metadata.get("stage"),
        "age_group": pd.cut(
            metadata["age"],
            bins=[0, 50, 65, 75, 200],
            labels=["<50", "50–65", "65–75", ">75"],
        ).astype(str) if "age" in metadata.columns else None,
    }

    for group_col, group_series in groupby_cols.items():
        if group_series is None:
            continue
        rows = []
        for val in group_series.unique():
            mask = group_series == val
            n = int(mask.sum())
            n_pos = int(y_true[mask].sum())
            if n < 10 or n_pos < 2 or n_pos == n:
                continue
            try:
                rows.append({
                    "group":  val,
                    "n":      n,
                    "n_pos":  n_pos,
                    "ae_rate": round(n_pos / n, 4),
                    "auroc":  round(roc_auc_score(y_true[mask], y_prob[mask]), 4),
                    "auprc":  round(average_precision_score(y_true[mask], y_prob[mask]), 4),
                    "brier":  round(brier_score_loss(y_true[mask], y_prob[mask]), 4),
                })
            except Exception as exc:
                logger.warning("Subgroup %s=%s failed: %s", group_col, val, exc)
        if rows:
            results[group_col] = pd.DataFrame(rows).sort_values("auroc", ascending=False)

    return results


# ---------------------------------------------------------------------------
# Model card writer
# ---------------------------------------------------------------------------

def generate_model_card(
    metrics: Dict[str, Any],
    threshold_df: pd.DataFrame,
    subgroup_results: Dict[str, pd.DataFrame],
    output_dir: str = "data/model_outputs",
    model_name: str = "XGBoost",
) -> str:
    """
    Write a Markdown model card to output_dir/model_card.md and
    a machine-readable JSON to output_dir/model_card.json.

    Args:
        metrics:           dict from ModelMetrics.to_dict().
        threshold_df:      Output of threshold_analysis().
        subgroup_results:  Output of subgroup_analysis().
        output_dir:        Directory to write outputs.
        model_name:        Display name of the primary model.

    Returns:
        Path to the generated model_card.md file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Best threshold (max F1) ---
    best_row = threshold_df.loc[threshold_df["f1"].idxmax()]

    # --- Markdown ---
    lines = [
        "# Model Card: Oncology Adverse Event Risk Predictor",
        "",
        "> **Version:** 1.0.0  |  **Date:** 2025-09  |  **Framework:** scikit-learn + XGBoost",
        "> **Disclaimer:** For research purposes only. Not validated for clinical use.",
        "",
        "---",
        "",
        "## 1. Model Details",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Architecture | {model_name} + Logistic Regression baseline (Platt-calibrated) |",
        "| Task | Binary classification — predicts treatment-related AE (any grade) |",
        "| Training data | Synthetic oncology EHR (2,000 patients, 80/20 train/test split) |",
        "| Feature count | ~70 (after one-hot encoding of cancer type and treatment) |",
        "| Label prevalence | ~38% adverse events |",
        "| Calibration | Isotonic regression via CalibratedClassifierCV |",
        "",
        "---",
        "",
        "## 2. Intended Use",
        "",
        "**Primary use:** Research demonstration of oncology ML pipelines.  "
        "Identifies patients at elevated risk of treatment-related adverse events "
        "to support prioritisation of clinical monitoring.",
        "",
        "**Out-of-scope:** Clinical diagnosis, treatment selection, regulatory decision-making.  "
        "This model was trained on *synthetic* data and has not been validated on real patients.",
        "",
        "---",
        "",
        "## 3. Performance on Held-Out Test Set",
        "",
        f"| Metric | {model_name} |",
        "|--------|--------|",
        f"| AUROC  | **{metrics.get('auroc', 0):.4f}** |",
        f"| AUPRC  | **{metrics.get('auprc', 0):.4f}** |",
        f"| Brier score | {metrics.get('brier', 0):.4f} |",
        f"| CV AUROC (5-fold) | {metrics.get('cv_auroc_mean', 'N/A')} ± {metrics.get('cv_auroc_std', 'N/A')} |",
        "",
        f"**Recommended operating threshold:** {best_row['threshold']:.2f}  "
        f"(Precision={best_row['precision']:.3f}, Recall={best_row['recall']:.3f}, "
        f"F1={best_row['f1']:.3f})",
        "",
        "---",
        "",
        "## 4. Threshold Analysis",
        "",
        "| Threshold | Precision | Recall | F1 | Alerts | TP | FP |",
        "|-----------|-----------|--------|-----|--------|-----|-----|",
    ]

    for _, row in threshold_df[threshold_df["threshold"].isin(
        [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
    )].iterrows():
        lines.append(
            f"| {row['threshold']:.2f} | {row['precision']:.3f} | {row['recall']:.3f} "
            f"| {row['f1']:.3f} | {int(row['n_alerted'])} "
            f"| {int(row['true_positives'])} | {int(row['false_positives'])} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Subgroup / Fairness Analysis",
        "",
        "> AUROC parity assessed across sex, cancer type, age group, and disease stage.",
        "",
    ]

    for group_col, df_sg in subgroup_results.items():
        lines.append(f"### {group_col.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Group | N | AE Rate | AUROC | AUPRC |")
        lines.append("|-------|---|---------|-------|-------|")
        for _, row in df_sg.iterrows():
            lines.append(
                f"| {row['group']} | {row['n']} | {row['ae_rate']:.1%} "
                f"| {row['auroc']:.4f} | {row['auprc']:.4f} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 6. Ethical Considerations",
        "",
        "- **Synthetic data:** All patients are algorithmically generated; no real patient data used.",
        "- **Class imbalance:** Addressed via `class_weight='balanced'` (LR) and `scale_pos_weight` (XGBoost).",
        "- **Subgroup parity:** AUROC was checked across sex, age group, cancer type, and stage "
        "  to surface differential performance.",
        "- **Clinical deployment:** Any clinical use would require prospective validation, "
        "  regulatory approval, and robust governance.",
        "",
        "---",
        "",
        "## 7. Training Data",
        "",
        "| Field | Value |",
        "|-------|-------|",
        "| Source | Synthetic (generated by `src/data/generate_ehr.py`) |",
        "| Size | 2,000 patients |",
        "| Split | 80% train / 20% test (stratified) |",
        "| Features | Demographics, labs, symptom scores, NLP flags from oncology notes |",
        "| Label | Binary adverse_event (any CTCAE grade ≥ 1) |",
        "",
        "---",
        "",
        "*Generated by `src/models/model_card.py`.*",
    ]

    card_md = "\n".join(lines)
    md_path = output_path / "model_card.md"
    md_path.write_text(card_md)

    # --- JSON ---
    json_data = {
        "metrics": metrics,
        "best_threshold": {
            "threshold": float(best_row["threshold"]),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1": float(best_row["f1"]),
        },
        "threshold_analysis": threshold_df.to_dict(orient="records"),
        "subgroup_analysis": {k: v.to_dict(orient="records") for k, v in subgroup_results.items()},
    }
    json_path = output_path / "model_card.json"
    json_path.write_text(json.dumps(json_data, indent=2))

    logger.info("Model card written to %s", md_path)
    return str(md_path)

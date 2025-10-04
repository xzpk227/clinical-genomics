#!/usr/bin/env python3
"""
Train the adverse event risk prediction model.

Steps:
  1. Load (or generate) the oncology EHR dataset.
  2. Extract NLP feature flags from clinical notes using OncologyExtractor.
  3. Split into 80/20 train/test (stratified).
  4. Train LR baseline and XGBoost primary model.
  5. Evaluate on test set: AUROC, AUPRC, Brier, calibration, subgroup AUROC.
  6. Generate threshold analysis and model card.
  7. Save models and artifacts to data/model_outputs/.

Usage:
  python scripts/train_adverse_event_model.py [--data-path PATH] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_nlp_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run OncologyExtractor over each row's clinical_note and
    return a DataFrame of binary NLP flags.
    """
    from src.extraction.oncology_extractor import OncologyExtractor

    extractor = OncologyExtractor()
    rows = []
    logger.info("Extracting NLP features from %d notes …", len(df))

    for i, row in df.iterrows():
        note = row.get("clinical_note", "") or ""
        structured = extractor.extract_structured(note)
        flag_row = {"patient_id": row["patient_id"], "symptom_count": structured["symptom_count"]}
        flag_row.update({k: int(v) for k, v in structured.items() if k.startswith("has_")})
        rows.append(flag_row)

        if (i + 1) % 500 == 0:
            logger.info("  … processed %d / %d patients", i + 1, len(df))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train adverse event risk model")
    parser.add_argument(
        "--data-path", type=str, default="data/oncology_ehr.parquet",
        help="Path to the oncology EHR dataset (default: data/oncology_ehr.parquet)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/model_outputs",
        help="Output directory for saved models and artifacts (default: data/model_outputs/)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.info("Dataset not found at %s; generating …", data_path)
        from src.data.generate_ehr import generate_oncology_ehr
        df = generate_oncology_ehr(n_patients=2000, seed=args.seed)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path, index=False)
        df.to_csv(data_path.with_suffix(".csv"), index=False)
    else:
        logger.info("Loading dataset from %s …", data_path)
        df = pd.read_parquet(data_path)

    logger.info(
        "Loaded %d patients; AE prevalence = %.1f%%",
        len(df), 100 * df["adverse_event"].mean()
    )

    # ------------------------------------------------------------------
    # 2. Extract NLP flags
    # ------------------------------------------------------------------
    nlp_flags = extract_nlp_flags(df)

    # ------------------------------------------------------------------
    # 3. Train/test split (stratified)
    # ------------------------------------------------------------------
    df_train, df_test = train_test_split(
        df,
        test_size=args.test_size,
        stratify=df["adverse_event"],
        random_state=args.seed,
    )
    nlp_train = nlp_flags[nlp_flags["patient_id"].isin(df_train["patient_id"])]
    nlp_test  = nlp_flags[nlp_flags["patient_id"].isin(df_test["patient_id"])]

    logger.info(
        "Split: %d train / %d test | AE rate: train=%.1f%%, test=%.1f%%",
        len(df_train), len(df_test),
        100 * df_train["adverse_event"].mean(),
        100 * df_test["adverse_event"].mean(),
    )

    # ------------------------------------------------------------------
    # 4. Train models
    # ------------------------------------------------------------------
    from src.models.adverse_event_model import AdverseEventModel

    model = AdverseEventModel(n_cv_folds=5, random_state=args.seed)
    model.fit(df_train, df_test, nlp_train, nlp_test)

    # ------------------------------------------------------------------
    # 5. Save models
    # ------------------------------------------------------------------
    model.save(args.output_dir)
    logger.info("Models saved to %s", args.output_dir)

    # ------------------------------------------------------------------
    # 6. Print summary
    # ------------------------------------------------------------------
    for model_name, metrics in model.metrics.items():
        d = metrics.to_dict()
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"  AUROC:          {d['auroc']:.4f}")
        print(f"  AUPRC:          {d['auprc']:.4f}")
        print(f"  Brier score:    {d['brier']:.4f}")
        if d.get("cv_auroc_mean"):
            print(f"  CV AUROC:       {d['cv_auroc_mean']:.4f} ± {d['cv_auroc_std']:.4f}")
        if d.get("subgroup_auroc"):
            print("  Subgroup AUROC:")
            for grp, sub_dict in d["subgroup_auroc"].items():
                vals = ", ".join(f"{k}={v:.3f}" for k, v in sub_dict.items())
                print(f"    {grp}: {vals}")

    # ------------------------------------------------------------------
    # 7. Threshold analysis + model card
    # ------------------------------------------------------------------
    from src.models.adverse_event_model import prepare_features
    from src.models.model_card import generate_model_card, threshold_analysis, subgroup_analysis

    # Use best model (XGBoost if available, else LR) for threshold analysis
    model_name = "xgboost" if model.xgb_pipeline is not None else "logistic_regression"
    X_test_feats = prepare_features(df_test, nlp_test)

    from src.models.adverse_event_model import ALL_FEATURE_COLS
    pipeline = model.xgb_pipeline if model_name == "xgboost" else model.lr_pipeline
    y_test  = df_test["adverse_event"].astype(int).values
    y_prob  = pipeline.predict_proba(X_test_feats[ALL_FEATURE_COLS])[:, 1]

    thresh_df = threshold_analysis(y_test, y_prob)
    subgroup_results = subgroup_analysis(y_test, y_prob, df_test.reset_index(drop=True))

    card_path = generate_model_card(
        metrics=model.metrics[model_name].to_dict(),
        threshold_df=thresh_df,
        subgroup_results=subgroup_results,
        output_dir=args.output_dir,
        model_name=model_name.replace("_", " ").title(),
    )
    logger.info("Model card written to %s", card_path)

    # Save threshold analysis CSV
    thresh_csv = Path(args.output_dir) / "threshold_analysis.csv"
    thresh_df.to_csv(thresh_csv, index=False)
    logger.info("Threshold analysis saved to %s", thresh_csv)

    print(f"\nDone. All outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()

"""
Regression test for the Clinical Phenotype Extraction and HPO Mapping Pipeline.

Runs the full Evaluation Suite against the curated test set and asserts that
top-1 HPO mapping accuracy meets the configured threshold.

The threshold defaults to 0.70 and can be overridden via the
REGRESSION_ACCURACY_THRESHOLD environment variable.

Run with:
    pytest tests/regression/test_regression.py -v

Requirements: 7.6, 8.1, 8.2
"""

from __future__ import annotations

import os

import pytest

from src.evaluation.evaluator import Evaluator
from src.pipeline import Pipeline, PipelineConfig


def test_top1_accuracy_above_threshold() -> None:
    """Regression test: top-1 HPO mapping accuracy must meet the configured threshold.

    Initializes the full pipeline (requires data/hpo_database.json,
    data/hpo_index.faiss, and data/hpo_id_map.json to be pre-built), runs the
    Evaluator against the curated test set, and asserts that top-1 accuracy
    is at or above the threshold.

    Fails with a descriptive message showing actual vs. threshold accuracy.
    """
    config = PipelineConfig()

    # Allow threshold override via environment variable
    threshold_env = os.environ.get("REGRESSION_ACCURACY_THRESHOLD")
    if threshold_env is not None:
        try:
            threshold = float(threshold_env)
        except ValueError:
            pytest.fail(
                f"REGRESSION_ACCURACY_THRESHOLD env var is not a valid float: "
                f"{threshold_env!r}"
            )
    else:
        threshold = config.regression_accuracy_threshold

    pipeline = Pipeline(config)
    evaluator = Evaluator(test_set_path="data/evaluation/test_set.json")
    result = evaluator.run(pipeline)

    assert result.top1_accuracy >= threshold, (
        f"Regression failure: top-1 accuracy {result.top1_accuracy:.3f} is below "
        f"the required threshold of {threshold:.3f}.\n"
        f"Full metrics:\n"
        f"  extraction_precision = {result.extraction_precision:.3f}\n"
        f"  extraction_recall    = {result.extraction_recall:.3f}\n"
        f"  extraction_f1        = {result.extraction_f1:.3f}\n"
        f"  top1_accuracy        = {result.top1_accuracy:.3f}\n"
        f"  top3_accuracy        = {result.top3_accuracy:.3f}\n"
        f"  negation_fp_rate     = {result.negation_fp_rate:.3f}\n"
        f"  total_cases          = {result.total_cases}"
    )

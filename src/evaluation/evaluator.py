"""
Evaluation suite for the Clinical Phenotype Extraction and HPO Mapping Pipeline.

Computes six metrics against the curated test set:
  - extraction_precision
  - extraction_recall
  - extraction_f1
  - top1_accuracy
  - top3_accuracy
  - negation_fp_rate
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

from src.evaluation.test_set import load_test_set, TestCase, ExpectedMention

if TYPE_CHECKING:
    from src.pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Metrics produced by the Evaluator.

    All float metrics are in [0.0, 1.0].
    """

    extraction_precision: float
    extraction_recall: float
    extraction_f1: float
    top1_accuracy: float
    top3_accuracy: float
    negation_fp_rate: float
    total_cases: int


class Evaluator:
    """Runs the pipeline against the curated test set and computes metrics."""

    def __init__(self, test_set_path: str = "data/evaluation/test_set.json") -> None:
        self._test_set_path = test_set_path

    def run(self, pipeline: "Pipeline") -> EvaluationResult:
        """Run the pipeline against the curated test set and compute all metrics.

        Args:
            pipeline: An initialized Pipeline instance.

        Returns:
            An EvaluationResult with all six metrics.
        """
        cases: list[TestCase] = load_test_set(self._test_set_path)
        total = len(cases)
        logger.info("Running evaluation on %d test cases …", total)

        # Counters for extraction precision/recall
        total_predicted = 0
        total_expected = 0
        total_true_positive = 0

        # Counters for mapping accuracy
        top1_correct = 0
        top3_correct = 0
        mapping_total = 0  # expected mentions that have an hpo_id to check

        # Counters for negation false-positive rate
        negation_fp = 0       # negated mentions returned as non-negated
        negation_total = 0    # total expected negated mentions

        for case in cases:
            try:
                result = pipeline.process(case.note, top_k=3)
            except Exception as exc:
                logger.warning("Pipeline failed on test case %r: %s", case.note[:50], exc)
                # Count all expected mentions as missed
                total_expected += len(case.expected_mentions)
                continue

            predicted_texts = {m.text.lower() for m in result.mentions}
            expected_texts = {e.text.lower() for e in case.expected_mentions}

            # Extraction TP: predicted mention text matches an expected mention text
            tp = len(predicted_texts & expected_texts)
            total_true_positive += tp
            total_predicted += len(predicted_texts)
            total_expected += len(expected_texts)

            # Build a lookup: mention_text → candidates list
            candidates_by_text = {
                m.text.lower(): result.mappings.get(m.text, [])
                for m in result.mentions
            }

            # Mapping accuracy and negation FP rate
            for expected in case.expected_mentions:
                exp_text = expected.text.lower()
                exp_hpo = expected.hpo_id

                # Only evaluate mapping when the mention was actually predicted
                if exp_text in predicted_texts:
                    candidates = candidates_by_text.get(exp_text, [])
                    mapping_total += 1

                    # Top-1 accuracy
                    if candidates and candidates[0].hpo_id == exp_hpo:
                        top1_correct += 1

                    # Top-3 accuracy
                    top3_ids = {c.hpo_id for c in candidates[:3]}
                    if exp_hpo in top3_ids:
                        top3_correct += 1

                # Negation FP rate: expected negated but returned as non-negated
                if expected.negated:
                    negation_total += 1
                    # Find the predicted mention for this text
                    predicted_mention = next(
                        (m for m in result.mentions if m.text.lower() == exp_text),
                        None,
                    )
                    if predicted_mention is not None and not predicted_mention.negated:
                        negation_fp += 1

        # Compute metrics
        precision = total_true_positive / total_predicted if total_predicted > 0 else 0.0
        recall = total_true_positive / total_expected if total_expected > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        top1_acc = top1_correct / mapping_total if mapping_total > 0 else 0.0
        top3_acc = top3_correct / mapping_total if mapping_total > 0 else 0.0
        neg_fp_rate = negation_fp / negation_total if negation_total > 0 else 0.0

        result_obj = EvaluationResult(
            extraction_precision=round(precision, 4),
            extraction_recall=round(recall, 4),
            extraction_f1=round(f1, 4),
            top1_accuracy=round(top1_acc, 4),
            top3_accuracy=round(top3_acc, 4),
            negation_fp_rate=round(neg_fp_rate, 4),
            total_cases=total,
        )

        logger.info(
            "Evaluation complete: precision=%.3f recall=%.3f f1=%.3f "
            "top1=%.3f top3=%.3f neg_fp=%.3f",
            result_obj.extraction_precision,
            result_obj.extraction_recall,
            result_obj.extraction_f1,
            result_obj.top1_accuracy,
            result_obj.top3_accuracy,
            result_obj.negation_fp_rate,
        )

        return result_obj

    def save_report(self, result: EvaluationResult, path: str) -> None:
        """Write the evaluation result to a JSON file.

        Args:
            result: The EvaluationResult to serialize.
            path:   Output file path.
        """
        report = asdict(result)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Evaluation report written to %s", path)

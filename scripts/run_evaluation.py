"""Run the HPO extraction pipeline evaluation and print metrics."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import Pipeline, PipelineConfig
from src.evaluation.evaluator import Evaluator


def main() -> None:
    print("Loading pipeline...")
    config = PipelineConfig()
    pipeline = Pipeline(config)

    print("Running evaluation...")
    evaluator = Evaluator()
    result = evaluator.run(pipeline)

    print("\n--- Evaluation Results ---")
    print(f"Total test cases   : {result.total_cases}")
    print(f"Extraction precision: {result.extraction_precision:.4f}")
    print(f"Extraction recall   : {result.extraction_recall:.4f}")
    print(f"Extraction F1       : {result.extraction_f1:.4f}")
    print(f"Top-1 HPO accuracy  : {result.top1_accuracy:.4f}")
    print(f"Top-3 HPO accuracy  : {result.top3_accuracy:.4f}")
    print(f"Negation FP rate    : {result.negation_fp_rate:.4f}")

    output_path = Path("data/model_outputs/evaluation_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.save_report(result, str(output_path))
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()

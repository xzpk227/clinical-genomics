#!/usr/bin/env python3
"""
Generate the synthetic oncology EHR dataset.

Saves:
  data/oncology_ehr.csv      — CSV for human inspection
  data/oncology_ehr.parquet  — Parquet for model training

Usage:
  python scripts/generate_oncology_ehr.py [--n-patients N] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generate_ehr import generate_oncology_ehr


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic oncology EHR dataset")
    parser.add_argument("--n-patients", type=int, default=2000, help="Number of patients (default 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory (default: data/)")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic oncology EHR: n={args.n_patients}, seed={args.seed} …")
    df = generate_oncology_ehr(n_patients=args.n_patients, seed=args.seed)

    csv_path = out / "oncology_ehr.csv"
    parquet_path = out / "oncology_ehr.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    n_ae = int(df["adverse_event"].sum())
    print(f"\n{'='*50}")
    print(f"Patients:          {len(df):,}")
    print(f"AE prevalence:     {n_ae} / {len(df)} ({n_ae/len(df):.1%})")
    print(f"\nAE type breakdown:")
    print(df["ae_type"].value_counts().to_string())
    print(f"\nCancer type breakdown:")
    print(df["cancer_type"].value_counts().to_string())
    print(f"\nSaved to: {csv_path}  and  {parquet_path}")


if __name__ == "__main__":
    main()

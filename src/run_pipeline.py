#!/usr/bin/env python
"""
run_pipeline.py
---------------
End-to-end reproducibility script: preprocessing → training → evaluation.

Run from the project root:
    python run_pipeline.py

Or via the Makefile:
    make run
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import pandas as pd

from src.preprocess import run_preprocessing_pipeline
from src.pipeline import run_pipeline_comparison


def main():
    start = time.time()

    print("=" * 60)
    print("TWITTER SENTIMENT ANALYSIS — END-TO-END PIPELINE")
    print("=" * 60)

    # ── Step 1: Preprocess ────────────────────────────────────
    print("\n[1/2] Running preprocessing pipeline...")
    df_binary, df_multiclass = run_preprocessing_pipeline(save_output=False)
    print(f"      Binary dataset   : {len(df_binary):,} rows")
    print(f"      Multiclass dataset: {len(df_multiclass):,} rows")

    # ── Step 2: Train & evaluate all pipelines ────────────────
    print("\n[2/2] Running model pipeline comparison...")
    results = run_pipeline_comparison(df_binary, df_multiclass)

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - start
    best = max(results, key=lambda r: r["f1"])

    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"Total time  : {elapsed:.1f}s")
    print(f"Best model  : {best['name']}")
    print(f"Best F1     : {best['f1']:.4f}")
    print(f"Best Accuracy: {best['accuracy']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

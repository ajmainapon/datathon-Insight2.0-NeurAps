"""Validate a submission file against the competition's sample submission.

Usage: python src/check_submission.py submissions/sub_001_baseline.csv data/raw/sample_submission.csv
"""
import sys

import numpy as np
import pandas as pd


def check(sub_path: str, sample_path: str) -> None:
    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(sample_path)
    errors = []

    if list(sub.columns) != list(sample.columns):
        errors.append(f"Columns differ: {list(sub.columns)} vs {list(sample.columns)}")
    if len(sub) != len(sample):
        errors.append(f"Row count differs: {len(sub)} vs {len(sample)}")

    id_col = sample.columns[0]
    if id_col in sub.columns and len(sub) == len(sample):
        if not sub[id_col].equals(sample[id_col]):
            if set(sub[id_col]) == set(sample[id_col]):
                errors.append(f"IDs match as a set but ORDER differs on '{id_col}'")
            else:
                errors.append(f"ID sets differ on '{id_col}'")

    pred_cols = [c for c in sample.columns if c != id_col and c in sub.columns]
    for c in pred_cols:
        if sub[c].isna().any():
            errors.append(f"NaNs in '{c}': {sub[c].isna().sum()} rows")
        if pd.api.types.is_numeric_dtype(sub[c]):
            vals = sub[c]
            if not np.isfinite(vals.dropna()).all():
                errors.append(f"Non-finite values in '{c}'")
            print(f"  {c}: min={vals.min():.6g} max={vals.max():.6g} mean={vals.mean():.6g}")
        else:
            allowed = set(sample[c].dropna().unique())
            bad = set(sub[c].dropna().unique()) - allowed
            if bad:
                errors.append(f"Labels in '{c}' not present in sample submission: {bad}")
            print(f"  {c}: value counts {sub[c].value_counts().to_dict()}")

    if errors:
        print(f"\nFAILED {sub_path}:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    print(f"\nOK: {sub_path} passes all hard checks against {sample_path}")


if __name__ == "__main__":
    check(sys.argv[1], sys.argv[2])

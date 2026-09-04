"""Day 4: exact-duplicate leak fix + fine-grained decision-threshold scan.

No model is retrained here. Both actions operate on probabilities already saved by
day1_05_pseudo_label_blend.py (data/processed/test_v4_blend.npy and oof_v4_blend.npy).

1. `apply_duplicate_fix` — 283 test patients are exact copies of a training patient
   on every one of the 35 shared feature columns (see experiments/log.md row 16).
   Their prediction is overridden with the known training label. This is the fix
   that turned sub_005 -> sub_015 and sub_014 -> sub_016 (the file we ultimately
   selected as our honest-CV final).

2. `scan_thresholds` — re-thresholds the finished v4 blend probabilities at a fine
   grid of cuts to reproduce the sub_017 / sub_023-026 / sub_018 probes (row 26).

Usage:
    .venv/bin/python src/day4_2026-08-31_leakfix_and_final_selection/day4_01_leakfix_and_threshold_scan.py
"""
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))


def find_duplicate_test_rows(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Return a Series indexed like `test`, holding the train label ('Dead'/'Alive')
    for every test row that exactly matches a train row on all shared feature columns,
    and NaN for every row with no match."""
    feature_cols = [c for c in test.columns if c != "patient_id"]
    dedup_train = train[feature_cols + ["vital_status"]].drop_duplicates(feature_cols)
    merged = test.merge(dedup_train, on=feature_cols, how="left")
    return merged["vital_status"]


def apply_duplicate_fix(pred_df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """pred_df: a submission-shaped frame with columns [patient_id, vital_status].
    Overrides predictions for exact-duplicate test rows with the true train label."""
    assert (pred_df["patient_id"].values == test["patient_id"].values).all(), \
        "pred_df must be in the same row order as `test`"
    true_label = find_duplicate_test_rows(train, test)
    fixed = pred_df.copy()
    mask = true_label.notna()
    fixed.loc[mask.values, "vital_status"] = true_label[mask].values
    n_changed = (fixed["vital_status"] != pred_df["vital_status"]).sum()
    print(f"duplicate fix: {mask.sum()} exact-duplicate test rows found "
          f"({mask.sum() / len(test):.4%}); {n_changed} prediction(s) actually changed")
    return fixed


def scan_thresholds(p_test: np.ndarray, patient_ids, thresholds, out_dir: str, tag: str):
    """Write one submission file per threshold in `thresholds`, all from the same
    finished probability array (no retraining)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for t in thresholds:
        labels = np.where(p_test > t, "Dead", "Alive")
        path = os.path.join(out_dir, f"sub_{tag}_thr{str(t).replace('.', '')}.csv")
        pd.DataFrame({"patient_id": patient_ids, "vital_status": labels}).to_csv(path, index=False)
        print(f"  t={t:.4f}  Dead rate={np.mean(labels == 'Dead'):.4f}  -> {path}")
        paths.append(path)
    return paths


def main():
    train = pd.read_csv(os.path.join(REPO_ROOT, "data/raw/train.csv"))
    test = pd.read_csv(os.path.join(REPO_ROOT, "data/raw/test.csv"))

    # --- 1. duplicate fix, applied to an existing submission-shaped file ---
    v4_blend_path = os.path.join(REPO_ROOT, "data/processed/test_v4_blend.npy")
    if os.path.exists(v4_blend_path):
        p_test = np.load(v4_blend_path)
        base_labels = np.where(p_test > 0.575, "Dead", "Alive")  # e.g. sub_014-style
        base_pred = pd.DataFrame({"patient_id": test["patient_id"], "vital_status": base_labels})
        apply_duplicate_fix(base_pred, train, test)
    else:
        print(f"skip duplicate-fix demo: {v4_blend_path} not found "
              f"(run day1_05_pseudo_label_blend.py first to produce it)")

    # --- 2. fine threshold scan on the same finished probabilities ---
    if os.path.exists(v4_blend_path):
        p_test = np.load(v4_blend_path)
        scan_thresholds(
            p_test, test["patient_id"],
            thresholds=[0.560, 0.565, 0.5675, 0.570, 0.5725, 0.582],
            out_dir=os.path.join(REPO_ROOT, "submissions"),
            tag="day4scan",
        )


if __name__ == "__main__":
    main()

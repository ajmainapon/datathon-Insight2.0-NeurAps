"""Sweep class weights (scale_pos_weight) and year-based sample weights on LGBM, 10-fold.

Positive class = Dead (majority). spw < 1 shifts capacity toward the Alive minority.
Year weight: w = 1 + alpha * (year - 2013) / 10 upweights recent (hard) years.
"""
import sys
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)                              # sibling files in this day's folder
sys.path.insert(0, os.path.dirname(_THIS_DIR))              # src/ -> for common.features
from common.features import build_features  # noqa: E402

SEED = 42
N_FOLDS = 10
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1)


def best_wf1(y, p):
    ts = np.arange(0.20, 0.85, 0.005)
    scores = [f1_score(y, (p > t).astype(int), average="weighted") for t in ts]
    i = int(np.argmax(scores))
    return ts[i], scores[i]


def run(X, y, folds, sample_weight=None, spw=None, tag=""):
    oof = np.zeros(len(y))
    params = dict(PARAMS)
    if spw is not None:
        params["scale_pos_weight"] = spw
    for i_tr, i_va in folds:
        w = sample_weight[i_tr] if sample_weight is not None else None
        m = lgb.train(params, lgb.Dataset(X.iloc[i_tr], y[i_tr], weight=w),
                      num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X.iloc[i_va], y[i_va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[i_va] = m.predict(X.iloc[i_va], num_iteration=m.best_iteration)
    t, s = best_wf1(y, oof)
    print(f"{tag}: wF1={s:.5f} @ t={t:.3f}", flush=True)
    np.save(f"data/processed/oof_sweep_{tag}.npy", oof)
    return s


def main():
    tr = pd.read_csv("data/raw/train.csv")
    y = (tr["vital_status"] == "Dead").astype(int).values
    X = build_features(tr)
    years = tr["year_of_diagnosis"].values
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))

    print("=== class weight sweep (scale_pos_weight, pos=Dead) ===", flush=True)
    for spw in [0.4, 0.6, 0.8, 1.0, 1.25]:
        run(X, y, folds, spw=spw, tag=f"spw{str(spw).replace('.','p')}")

    print("=== year-weight sweep (w = 1 + a*(year-2013)/10) ===", flush=True)
    for a in [0.5, 1.0, 2.0]:
        sw = 1.0 + a * (years - 2013) / 10.0
        run(X, y, folds, sample_weight=sw, tag=f"yw{str(a).replace('.','p')}")


if __name__ == "__main__":
    main()

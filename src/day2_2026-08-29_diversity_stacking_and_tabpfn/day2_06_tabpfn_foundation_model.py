"""TabPFN OOF + test predictions on MPS.

OOF: 10-fold, one 8k-subsample fit per fold, predict validation only.
Test: 4 fits on distinct 10k subsamples of full train, average predictions.
"""
import sys
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)                              # sibling files in this day's folder
sys.path.insert(0, os.path.dirname(_THIS_DIR))              # src/ -> for common.features
from common.features import build_features  # noqa: E402

SEED = 42
N_FOLDS = 10


def prep():
    tr = pd.read_csv("data/raw/train.csv")
    te = pd.read_csv("data/raw/test.csv")
    y = (tr["vital_status"] == "Dead").astype(int).values
    X = build_features(tr)
    X_te = build_features(te)[X.columns]
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    for c in cat_cols:
        cats = pd.api.types.union_categoricals([X[c], X_te[c]]).categories
        X[c] = X[c].cat.set_categories(cats).cat.codes
        X_te[c] = X_te[c].cat.set_categories(cats).cat.codes
    return X.astype(np.float32).values, X_te.astype(np.float32).values, y, len(te)


def main():
    from tabpfn import TabPFNClassifier
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    X, X_te, y, n_te = prep()
    rng = np.random.RandomState(SEED)

    oof = np.zeros(len(y))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for f, (i_tr, i_va) in enumerate(skf.split(X, y)):
        t0 = time.time()
        sub = rng.choice(i_tr, 8000, replace=False)
        clf = TabPFNClassifier(device=dev, ignore_pretraining_limits=True)
        clf.fit(X[sub], y[sub])
        oof[i_va] = clf.predict_proba(X[i_va])[:, 1]
        print(f"fold {f} done in {time.time()-t0:.0f}s", flush=True)
    ts = np.arange(0.35, 0.75, 0.005)
    scores = [f1_score(y, (oof > t).astype(int), average="weighted") for t in ts]
    i = int(np.argmax(scores))
    print(f"TabPFN OOF wF1={scores[i]:.5f} @ t={ts[i]:.3f}", flush=True)
    np.save("data/processed/oof_tabpfn.npy", oof)

    tep = np.zeros(n_te)
    N_TEST_FITS = 4
    for k in range(N_TEST_FITS):
        t0 = time.time()
        sub = rng.choice(len(y), 10000, replace=False)
        clf = TabPFNClassifier(device=dev, ignore_pretraining_limits=True)
        clf.fit(X[sub], y[sub])
        tep += clf.predict_proba(X_te)[:, 1] / N_TEST_FITS
        print(f"test fit {k} done in {time.time()-t0:.0f}s", flush=True)
    np.save("data/processed/test_tabpfn.npy", tep)
    print("saved oof_tabpfn.npy / test_tabpfn.npy", flush=True)


if __name__ == "__main__":
    main()

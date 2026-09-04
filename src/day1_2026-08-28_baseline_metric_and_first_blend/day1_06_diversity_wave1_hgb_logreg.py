"""Diversity models: HistGradientBoosting + one-hot LogisticRegression, 10-fold OOF.

Saves OOF/test predictions for blending with the GBM family.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)                              # sibling files in this day's folder
sys.path.insert(0, os.path.dirname(_THIS_DIR))              # src/ -> for common.features
from common.features import build_features  # noqa: E402

SEED = 42
N_FOLDS = 10


def best_wf1(y, p):
    ts = np.arange(0.35, 0.75, 0.005)
    scores = [f1_score(y, (p > t).astype(int), average="weighted") for t in ts]
    i = int(np.argmax(scores))
    return ts[i], scores[i]


def main():
    tr = pd.read_csv("data/raw/train.csv")
    te = pd.read_csv("data/raw/test.csv")
    y = (tr["vital_status"] == "Dead").astype(int).values

    X = build_features(tr)
    X_te = build_features(te)[X.columns]
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in cat_cols:
        cats = pd.api.types.union_categoricals([X[c], X_te[c]]).categories
        X[c] = X[c].cat.set_categories(cats)
        X_te[c] = X_te[c].cat.set_categories(cats)

    # HistGB: native categorical on integer codes
    Xh = X.copy(); Xh_te = X_te.copy()
    for c in cat_cols:
        Xh[c] = Xh[c].cat.codes
        Xh_te[c] = Xh_te[c].cat.codes
    cat_idx = [Xh.columns.get_loc(c) for c in cat_cols]

    # LogReg: one-hot cats + scaled numerics
    Xs = X.copy(); Xs_te = X_te.copy()
    for c in cat_cols:
        Xs[c] = Xs[c].astype(str).where(Xs[c].notna(), "NA")
        Xs_te[c] = Xs_te[c].astype(str).where(Xs_te[c].notna(), "NA")

    lr_pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
    ])

    oof = {"hgb": np.zeros(len(tr)), "lr": np.zeros(len(tr))}
    tep = {"hgb": np.zeros(len(te)), "lr": np.zeros(len(te))}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for f, (i_tr, i_va) in enumerate(skf.split(X, y)):
        hgb = HistGradientBoostingClassifier(
            max_iter=1000, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40,
            l2_regularization=1.0, categorical_features=cat_idx,
            early_stopping=True, n_iter_no_change=50, validation_fraction=0.15,
            random_state=SEED)
        hgb.fit(Xh.iloc[i_tr], y[i_tr])
        oof["hgb"][i_va] = hgb.predict_proba(Xh.iloc[i_va])[:, 1]
        tep["hgb"] += hgb.predict_proba(Xh_te)[:, 1] / N_FOLDS

        lr = Pipeline([("pre", lr_pre),
                       ("clf", LogisticRegression(C=0.1, max_iter=2000, random_state=SEED))])
        lr.fit(Xs.iloc[i_tr], y[i_tr])
        oof["lr"][i_va] = lr.predict_proba(Xs.iloc[i_va])[:, 1]
        tep["lr"] += lr.predict_proba(Xs_te)[:, 1] / N_FOLDS
        print(f"fold {f} done", flush=True)

    for m in oof:
        t, s = best_wf1(y, oof[m])
        print(f"{m}: OOF wF1={s:.5f} @ t={t:.3f}")
        np.save(f"data/processed/oof_div_{m}.npy", oof[m])
        np.save(f"data/processed/test_div_{m}.npy", tep[m])


if __name__ == "__main__":
    main()

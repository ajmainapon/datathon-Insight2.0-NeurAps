"""Last untried ideas, same-config 10-fold LGBM comparisons:
1. Focal loss (gamma 1, 2) via custom objective.
2. OOF target encoding for histologic_type_icdo3 + site_stage added to features.
Baseline: plain binary, same folds/params.
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
PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1)


def best_wf1(y, p):
    ts = np.arange(0.20, 0.85, 0.005)
    scores = [f1_score(y, (p > t).astype(int), average="weighted") for t in ts]
    i = int(np.argmax(scores))
    return ts[i], scores[i]


def focal_obj(gamma):
    def obj(preds, data):
        y = data.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        pt = np.where(y == 1, p, 1 - p)
        # d/dz of focal loss FL = -(1-pt)^g log(pt); numeric-stable approximation
        g = (y - p) * ((1 - pt) ** gamma - gamma * pt * (1 - pt) ** (gamma - 1) * np.log(np.clip(pt, 1e-9, 1)))
        grad = -g
        hess = p * (1 - p) * ((1 - pt) ** gamma + 1e-3) + 1e-6
        return grad, hess
    return obj


def run_lgb(X, y, folds, params=None, fobj=None, tag=""):
    oof = np.zeros(len(y))
    pr = dict(PARAMS)
    pr.update(params or {})
    for i_tr, i_va in folds:
        dtr = lgb.Dataset(X.iloc[i_tr], y[i_tr])
        dva = lgb.Dataset(X.iloc[i_va], y[i_va])
        if fobj:
            pr2 = dict(pr); pr2["objective"] = fobj
            m = lgb.train(pr2, dtr, num_boost_round=1500, valid_sets=[dva],
                          feval=lambda pred, d: ("wf1", best_wf1(d.get_label(), 1/(1+np.exp(-pred)))[1], True),
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[i_va] = 1 / (1 + np.exp(-m.predict(X.iloc[i_va], num_iteration=m.best_iteration)))
        else:
            pr2 = dict(pr); pr2["objective"] = "binary"
            m = lgb.train(pr2, dtr, num_boost_round=3000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[i_va] = m.predict(X.iloc[i_va], num_iteration=m.best_iteration)
    t, s = best_wf1(y, oof)
    print(f"{tag}: wF1={s:.5f} @ t={t:.3f}", flush=True)
    return oof


def add_te(X, tr, folds, y):
    """OOF target encoding for high-card cats; returns augmented copy."""
    X = X.copy()
    hist = tr["histologic_type_icdo3"].astype(str)
    ss = tr["primary_site"].astype(str) + "|" + tr["summary_stage"].astype(str)
    prior = y.mean()
    for name, col in [("te_hist", hist), ("te_sitestage", ss)]:
        te_vals = np.full(len(tr), np.nan)
        for i_tr, i_va in folds:
            stats = pd.DataFrame({"c": col.iloc[i_tr], "y": y[i_tr]}).groupby("c")["y"].agg(["mean", "count"])
            smooth = (stats["mean"] * stats["count"] + prior * 20) / (stats["count"] + 20)
            te_vals[i_va] = col.iloc[i_va].map(smooth).fillna(prior).values
        X[name] = te_vals
    return X


def main():
    tr = pd.read_csv("data/raw/train.csv")
    y = (tr["vital_status"] == "Dead").astype(int).values
    X = build_features(tr)
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))

    run_lgb(X, y, folds, tag="baseline-binary")
    for g in [1.0, 2.0]:
        run_lgb(X, y, folds, fobj=focal_obj(g), tag=f"focal-g{g}")
    Xte_feat = add_te(X, tr, folds, y)
    run_lgb(Xte_feat, y, folds, tag="with-target-encoding")


if __name__ == "__main__":
    main()

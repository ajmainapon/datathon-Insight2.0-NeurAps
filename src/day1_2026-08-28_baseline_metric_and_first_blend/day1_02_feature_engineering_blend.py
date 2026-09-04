"""Insight 2.0 v2: engineered features + LGBM/XGB/CatBoost blend, weighted-F1 threshold tuning.

Usage: .venv/bin/python src/train_v2.py <sub_number> <description>
"""
import sys
import os

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))              # src/ -> for common.features
from common.features import build_features  # noqa: E402
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5

def wf1(y, p, t):
    return f1_score(y, (p > t).astype(int), average="weighted")


def best_threshold(y, p):
    ts = np.arange(0.30, 0.75, 0.005)
    scores = [wf1(y, p, t) for t in ts]
    i = int(np.argmax(scores))
    return ts[i], scores[i]


def main():
    tr = pd.read_csv("data/raw/train.csv")
    te = pd.read_csv("data/raw/test.csv")
    y = (tr["vital_status"] == "Dead").astype(int).values

    X = build_features(tr)
    X_te = build_features(te)[X.columns]
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    for c in cat_cols:
        cats = pd.api.types.union_categoricals([X[c], X_te[c]]).categories
        X[c] = X[c].cat.set_categories(cats)
        X_te[c] = X_te[c].cat.set_categories(cats)

    # integer-coded copies for xgb/catboost
    Xi = X.copy()
    Xi_te = X_te.copy()
    for c in cat_cols:
        Xi[c] = Xi[c].cat.codes
        Xi_te[c] = Xi_te[c].cat.codes

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))
    models = ["lgb", "xgb", "cat"]
    oof = {m: np.zeros(len(tr)) for m in models}
    tep = {m: np.zeros(len(te)) for m in models}

    for f, (i_tr, i_va) in enumerate(folds):
        m = lgb.train(
            dict(objective="binary", learning_rate=0.05, num_leaves=63, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=40, seed=SEED, verbose=-1),
            lgb.Dataset(X.iloc[i_tr], y[i_tr]), num_boost_round=3000,
            valid_sets=[lgb.Dataset(X.iloc[i_va], y[i_va])],
            callbacks=[lgb.early_stopping(100, verbose=False)])
        oof["lgb"][i_va] = m.predict(X.iloc[i_va], num_iteration=m.best_iteration)
        tep["lgb"] += m.predict(X_te, num_iteration=m.best_iteration) / N_FOLDS

        xm = xgb.XGBClassifier(
            n_estimators=3000, learning_rate=0.05, max_depth=7, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=10, eval_metric="logloss",
            early_stopping_rounds=100, random_state=SEED, verbosity=0, n_jobs=-1)
        xm.fit(Xi.iloc[i_tr], y[i_tr], eval_set=[(Xi.iloc[i_va], y[i_va])], verbose=False)
        oof["xgb"][i_va] = xm.predict_proba(Xi.iloc[i_va])[:, 1]
        tep["xgb"] += xm.predict_proba(Xi_te)[:, 1] / N_FOLDS

        cm = cb.CatBoostClassifier(
            iterations=3000, learning_rate=0.05, depth=7, l2_leaf_reg=5,
            random_seed=SEED, eval_metric="Logloss", early_stopping_rounds=100, verbose=0)
        Xc = X.copy()
        Xc_te = X_te.copy()
        for c in cat_cols:
            Xc[c] = Xc[c].astype(str).where(Xc[c].notna(), "NA")
            Xc_te[c] = Xc_te[c].astype(str).where(Xc_te[c].notna(), "NA")
        cm.fit(Xc.iloc[i_tr], y[i_tr], eval_set=(Xc.iloc[i_va], y[i_va]),
               cat_features=cat_cols)
        oof["cat"][i_va] = cm.predict_proba(Xc.iloc[i_va])[:, 1]
        tep["cat"] += cm.predict_proba(Xc_te)[:, 1] / N_FOLDS
        print(f"fold {f} done")

    print("\n=== per-model OOF weighted F1 (tuned threshold) ===")
    for m in models:
        t, s = best_threshold(y, oof[m])
        print(f"{m}: wF1={s:.5f} @ t={t:.3f}")
        np.save(f"data/processed/oof_v2_{m}.npy", oof[m])
        np.save(f"data/processed/test_v2_{m}.npy", tep[m])

    def neg_blend(w):
        w = np.abs(w) / np.abs(w).sum()
        p = sum(wi * oof[m] for wi, m in zip(w, models))
        return -best_threshold(y, p)[1]

    res = minimize(neg_blend, x0=np.ones(len(models)) / len(models), method="Nelder-Mead")
    w = np.abs(res.x) / np.abs(res.x).sum()
    p_blend = sum(wi * oof[m] for wi, m in zip(w, models))
    t_blend, s_blend = best_threshold(y, p_blend)
    print(f"\nblend weights {dict(zip(models, w.round(3)))}: wF1={s_blend:.5f} @ t={t_blend:.3f}")

    p_test = sum(wi * tep[m] for wi, m in zip(w, models))
    labels = np.where(p_test > t_blend, "Dead", "Alive")
    sub_num, desc = sys.argv[1], sys.argv[2]
    path = f"submissions/sub_{sub_num}_{desc}_cv{s_blend:.5f}.csv"
    pd.DataFrame({"patient_id": te["patient_id"], "vital_status": labels}).to_csv(path, index=False)
    print(f"pred Dead rate={np.mean(labels == 'Dead'):.4f} (train {y.mean():.4f})")
    print(f"saved {path}")


if __name__ == "__main__":
    main()

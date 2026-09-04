"""Insight 2.0 v3: 10-fold x 3-seed LGBM/XGB/CatBoost, tuned params, robust threshold.

Usage: .venv/bin/python src/train_v3.py <sub_number> <description>
"""
import json
import os
import sys

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)                              # sibling files in this day's folder
sys.path.insert(0, os.path.dirname(_THIS_DIR))              # src/ -> for common.features
from common.features import build_features  # noqa: E402

N_FOLDS = 10
SEEDS = [42, 2026, 7]

LGB_TUNED = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
                 feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
if os.path.exists("data/processed/lgb_tuned.json"):
    LGB_TUNED.update(json.load(open("data/processed/lgb_tuned.json")))
    print("loaded tuned LGBM params:", LGB_TUNED)


def wf1(y, p, t):
    return f1_score(y, (p > t).astype(int), average="weighted")


def best_threshold(y, p, lo=0.35, hi=0.75, step=0.005):
    ts = np.arange(lo, hi, step)
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

    Xi = X.copy(); Xi_te = X_te.copy()
    for c in cat_cols:
        Xi[c] = Xi[c].cat.codes
        Xi_te[c] = Xi_te[c].cat.codes
    Xc = X.copy(); Xc_te = X_te.copy()
    for c in cat_cols:
        Xc[c] = Xc[c].astype(str).where(Xc[c].notna(), "NA")
        Xc_te[c] = Xc_te[c].astype(str).where(Xc_te[c].notna(), "NA")

    models = ["lgb", "xgb", "cat"]
    oof = {m: np.zeros(len(tr)) for m in models}
    tep = {m: np.zeros(len(te)) for m in models}
    fold_of_row = np.zeros(len(tr), dtype=int)

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for f, (i_tr, i_va) in enumerate(skf.split(X, y)):
            if seed == SEEDS[0]:
                fold_of_row[i_va] = f
            p = dict(LGB_TUNED); p["seed"] = seed
            m = lgb.train(p, lgb.Dataset(X.iloc[i_tr], y[i_tr]), num_boost_round=3000,
                          valid_sets=[lgb.Dataset(X.iloc[i_va], y[i_va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof["lgb"][i_va] += m.predict(X.iloc[i_va], num_iteration=m.best_iteration) / len(SEEDS)
            tep["lgb"] += m.predict(X_te, num_iteration=m.best_iteration) / (N_FOLDS * len(SEEDS))

            xm = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=7,
                                   subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                                   eval_metric="logloss", early_stopping_rounds=100,
                                   random_state=seed, verbosity=0, n_jobs=-1)
            xm.fit(Xi.iloc[i_tr], y[i_tr], eval_set=[(Xi.iloc[i_va], y[i_va])], verbose=False)
            oof["xgb"][i_va] += xm.predict_proba(Xi.iloc[i_va])[:, 1] / len(SEEDS)
            tep["xgb"] += xm.predict_proba(Xi_te)[:, 1] / (N_FOLDS * len(SEEDS))

            cm = cb.CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=7,
                                       l2_leaf_reg=5, random_seed=seed, eval_metric="Logloss",
                                       early_stopping_rounds=100, verbose=0)
            cm.fit(Xc.iloc[i_tr], y[i_tr], eval_set=(Xc.iloc[i_va], y[i_va]), cat_features=cat_cols)
            oof["cat"][i_va] += cm.predict_proba(Xc.iloc[i_va])[:, 1] / len(SEEDS)
            tep["cat"] += cm.predict_proba(Xc_te)[:, 1] / (N_FOLDS * len(SEEDS))
        print(f"seed {seed} done", flush=True)

    print("\n=== per-model OOF wF1 (seed-averaged, tuned threshold) ===")
    for m in models:
        t, s = best_threshold(y, oof[m])
        print(f"{m}: wF1={s:.5f} @ t={t:.3f}")
        np.save(f"data/processed/oof_v3_{m}.npy", oof[m])
        np.save(f"data/processed/test_v3_{m}.npy", tep[m])

    def neg_blend(w):
        w = np.abs(w) / np.abs(w).sum()
        p = sum(wi * oof[m] for wi, m in zip(w, models))
        return -best_threshold(y, p)[1]

    res = minimize(neg_blend, x0=np.ones(len(models)) / len(models), method="Nelder-Mead")
    w = np.abs(res.x) / np.abs(res.x).sum()
    p_blend = sum(wi * oof[m] for wi, m in zip(w, models))
    p_test = sum(wi * tep[m] for wi, m in zip(w, models))
    np.save("data/processed/oof_v3_blend.npy", p_blend)
    np.save("data/processed/test_v3_blend.npy", p_test)

    t_global, s_global = best_threshold(y, p_blend)
    fold_ts = []
    for f in range(N_FOLDS):
        mask = fold_of_row == f
        tf, _ = best_threshold(y[mask], p_blend[mask])
        fold_ts.append(tf)
    t_robust = float(np.median(fold_ts))
    s_robust = wf1(y, p_blend, t_robust)
    print(f"\nblend weights {dict(zip(models, w.round(3)))}")
    print(f"global threshold: wF1={s_global:.5f} @ t={t_global:.3f}")
    print(f"per-fold thresholds {np.round(fold_ts, 3)} -> median t={t_robust:.3f}, wF1={s_robust:.5f}")

    t_use = t_robust
    labels = np.where(p_test > t_use, "Dead", "Alive")
    sub_num, desc = sys.argv[1], sys.argv[2]
    path = f"submissions/sub_{sub_num}_{desc}_cv{s_robust:.5f}.csv"
    pd.DataFrame({"patient_id": te["patient_id"], "vital_status": labels}).to_csv(path, index=False)
    print(f"pred Dead rate={np.mean(labels == 'Dead'):.4f} (train {y.mean():.4f})")
    print(f"saved {path}")


if __name__ == "__main__":
    main()

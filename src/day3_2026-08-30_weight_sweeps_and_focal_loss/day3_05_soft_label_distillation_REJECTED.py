"""Full-test soft-label distillation: train on train (hard labels) + ALL test rows
(soft labels = v4 blend probabilities, sample weight 0.4), 10-fold x 2 seeds.

Uses cross-entropy objectives that accept probabilistic targets:
LGBM 'cross_entropy', XGB 'binary:logistic', CatBoost 'CrossEntropy'.

Usage: .venv/bin/python src/train_distill.py <sub_number> <description>
"""
import sys
import os

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
SEEDS = [42, 2026]
TEST_W = 0.4


def wf1(y, p, t):
    return f1_score(y, (p > t).astype(int), average="weighted")


def best_threshold(y, p):
    ts = np.arange(0.30, 0.80, 0.005)
    scores = [wf1(y, p, t) for t in ts]
    i = int(np.argmax(scores))
    return ts[i], scores[i]


def main():
    sub_num, desc = sys.argv[1], sys.argv[2]
    tr = pd.read_csv("data/raw/train.csv")
    te = pd.read_csv("data/raw/test.csv")
    y = (tr["vital_status"] == "Dead").astype(int).values
    y_soft_te = np.load("data/processed/test_v4_blend.npy")

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

    models = ["lgb", "xgb"]
    oof = {m: np.zeros(len(tr)) for m in models}
    tep = {m: np.zeros(len(te)) for m in models}
    fold_of_row = np.zeros(len(tr), dtype=int)
    w_te = np.full(len(te), TEST_W)

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for f, (i_tr, i_va) in enumerate(skf.split(X, y)):
            if seed == SEEDS[0]:
                fold_of_row[i_va] = f
            X_aug = pd.concat([X.iloc[i_tr], X_te], ignore_index=True)
            y_aug = np.concatenate([y[i_tr].astype(float), y_soft_te])
            w_aug = np.concatenate([np.ones(len(i_tr)), w_te])

            p = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=63,
                     min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
                     bagging_freq=1, seed=seed, verbose=-1)
            m = lgb.train(p, lgb.Dataset(X_aug, y_aug, weight=w_aug), num_boost_round=3000,
                          valid_sets=[lgb.Dataset(X.iloc[i_va], y[i_va].astype(float))],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof["lgb"][i_va] += m.predict(X.iloc[i_va], num_iteration=m.best_iteration) / len(SEEDS)
            tep["lgb"] += m.predict(X_te, num_iteration=m.best_iteration) / (N_FOLDS * len(SEEDS))

            Xi_aug = pd.concat([Xi.iloc[i_tr], Xi_te], ignore_index=True)
            xm = xgb.XGBRegressor(objective="binary:logistic", n_estimators=3000,
                                  learning_rate=0.05, max_depth=7, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=10,
                                  eval_metric="logloss", early_stopping_rounds=100,
                                  random_state=seed, verbosity=0, n_jobs=-1)
            xm.fit(Xi_aug, y_aug, sample_weight=w_aug,
                   eval_set=[(Xi.iloc[i_va], y[i_va].astype(float))], verbose=False)
            oof["xgb"][i_va] += xm.predict(Xi.iloc[i_va]) / len(SEEDS)
            tep["xgb"] += xm.predict(Xi_te) / (N_FOLDS * len(SEEDS))
        print(f"seed {seed} done", flush=True)

    print("\n=== per-model OOF wF1 (distill) ===", flush=True)
    for m in models:
        t, s = best_threshold(y, oof[m])
        print(f"{m}: wF1={s:.5f} @ t={t:.3f}", flush=True)
        np.save(f"data/processed/oof_distill_{m}.npy", oof[m])
        np.save(f"data/processed/test_distill_{m}.npy", tep[m])

    def neg(w):
        w = np.abs(w) / np.abs(w).sum()
        return -best_threshold(y, sum(wi * oof[m] for wi, m in zip(w, models)))[1]

    res = minimize(neg, np.ones(len(models)) / len(models), method="Nelder-Mead",
                   options={"maxiter": 40})
    w = np.abs(res.x) / np.abs(res.x).sum()
    p_blend = sum(wi * oof[m] for wi, m in zip(w, models))
    p_test = sum(wi * tep[m] for wi, m in zip(w, models))
    np.save("data/processed/oof_distill_blend.npy", p_blend)
    np.save("data/processed/test_distill_blend.npy", p_test)
    t_g, s_g = best_threshold(y, p_blend)
    fold_ts = [best_threshold(y[fold_of_row == f], p_blend[fold_of_row == f])[0]
               for f in range(N_FOLDS)]
    t_r = float(np.median(fold_ts)); s_r = wf1(y, p_blend, t_r)
    print(f"blend w={dict(zip(models, w.round(3)))} global {s_g:.5f}@{t_g:.3f} robust {s_r:.5f}@{t_r:.3f}", flush=True)

    labels = np.where(p_test > t_r, "Dead", "Alive")
    path = f"submissions/sub_{sub_num}_{desc}_cv{s_r:.5f}.csv"
    pd.DataFrame({"patient_id": te["patient_id"], "vital_status": labels}).to_csv(path, index=False)
    print(f"saved {path} Dead={np.mean(labels == 'Dead'):.4f}", flush=True)


if __name__ == "__main__":
    main()

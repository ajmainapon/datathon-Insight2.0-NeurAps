"""Insight 2.0 v11: focal gamma=1 + narrow pseudo-labels (v4 preds), 10-fold x 3 seeds.

Also evaluates blending with saved v4 components and picks the best OOF combo.

Usage: .venv/bin/python src/train_v10.py <sub_number> <description>
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
SEEDS = [42, 2026, 7]
GAMMA = 1.0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def focal_grad_hess(y, p, gamma):
    pt = np.where(y == 1, p, 1 - p)
    g = (y - p) * ((1 - pt) ** gamma - gamma * pt * (1 - pt) ** (gamma - 1) * np.log(np.clip(pt, 1e-9, 1)))
    grad = -g
    hess = p * (1 - p) * ((1 - pt) ** gamma + 1e-3) + 1e-6
    return grad, hess


def lgb_focal(preds, data):
    return focal_grad_hess(data.get_label(), sigmoid(preds), GAMMA)


def xgb_focal(preds, dtrain):
    return focal_grad_hess(dtrain.get_label(), sigmoid(preds), GAMMA)


def wf1(y, p, t):
    return f1_score(y, (p > t).astype(int), average="weighted")


def best_threshold(y, p):
    ts = np.arange(0.20, 0.85, 0.005)
    scores = [wf1(y, p, t) for t in ts]
    i = int(np.argmax(scores))
    return ts[i], scores[i]


def main():
    sub_num, desc = sys.argv[1], sys.argv[2]
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
    p_src = np.load("data/processed/test_v4_blend.npy")
    conf_mask = (p_src > 0.99) | (p_src < 0.02)
    y_ps = (p_src[conf_mask] > 0.5).astype(int)
    X_ps = X_te[conf_mask].reset_index(drop=True)
    print(f"pseudo rows: {conf_mask.sum()}", flush=True)

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

            p = dict(objective=lgb_focal, learning_rate=0.05, num_leaves=63,
                     min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
                     bagging_freq=1, seed=seed, verbose=-1)
            X_aug = pd.concat([X.iloc[i_tr], X_ps], ignore_index=True)
            y_aug = np.concatenate([y[i_tr], y_ps]).astype(float)
            m = lgb.train(p, lgb.Dataset(X_aug, y_aug),
                          num_boost_round=1500,
                          valid_sets=[lgb.Dataset(X.iloc[i_va], y[i_va].astype(float))],
                          feval=lambda pr, d: ("wf1", best_threshold(d.get_label(), sigmoid(pr))[1], True),
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof["lgb"][i_va] += sigmoid(m.predict(X.iloc[i_va], num_iteration=m.best_iteration)) / len(SEEDS)
            tep["lgb"] += sigmoid(m.predict(X_te, num_iteration=m.best_iteration)) / (N_FOLDS * len(SEEDS))

            Xi_aug = pd.concat([Xi.iloc[i_tr], Xi_te[conf_mask.nonzero()[0]].reset_index(drop=True) if False else Xi_te[conf_mask].reset_index(drop=True)], ignore_index=True)
            dtr = xgb.DMatrix(Xi_aug, label=np.concatenate([y[i_tr], y_ps])); dva = xgb.DMatrix(Xi.iloc[i_va], label=y[i_va])
            dte = xgb.DMatrix(Xi_te)
            xparams = dict(max_depth=7, eta=0.05, subsample=0.8, colsample_bytree=0.8,
                           min_child_weight=10, seed=seed, nthread=-1)
            bst = xgb.train(xparams, dtr, num_boost_round=1500, obj=xgb_focal,
                            evals=[(dva, "va")],
                            custom_metric=lambda pr, d: ("wf1", -best_threshold(d.get_label(), sigmoid(pr))[1]),
                            early_stopping_rounds=100, verbose_eval=False)
            oof["xgb"][i_va] += sigmoid(bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))) / len(SEEDS)
            tep["xgb"] += sigmoid(bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))) / (N_FOLDS * len(SEEDS))

            cm = cb.CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=7,
                                       l2_leaf_reg=5, random_seed=seed,
                                       loss_function="Focal:focal_alpha=0.5;focal_gamma=1.0",
                                       eval_metric="Logloss",
                                       early_stopping_rounds=100, verbose=0)
            Xc_aug = pd.concat([Xc.iloc[i_tr], Xc_te[conf_mask].reset_index(drop=True)], ignore_index=True)
            cm.fit(Xc_aug, np.concatenate([y[i_tr], y_ps]), eval_set=(Xc.iloc[i_va], y[i_va]), cat_features=cat_cols)
            oof["cat"][i_va] += cm.predict_proba(Xc.iloc[i_va])[:, 1] / len(SEEDS)
            tep["cat"] += cm.predict_proba(Xc_te)[:, 1] / (N_FOLDS * len(SEEDS))
        print(f"seed {seed} done", flush=True)

    print("\n=== per-model OOF wF1 (v10 focal) ===", flush=True)
    for m in models:
        t, s = best_threshold(y, oof[m])
        print(f"{m}: wF1={s:.5f} @ t={t:.3f}", flush=True)
        np.save(f"data/processed/oof_v11_{m}.npy", oof[m])
        np.save(f"data/processed/test_v11_{m}.npy", tep[m])

    # blend focal trio, then try adding v4 blend as a component
    comp = {m: oof[m] for m in models}
    comp_te = {m: tep[m] for m in models}
    comp["v4"] = np.load("data/processed/oof_v4_blend.npy")
    comp_te["v4"] = np.load("data/processed/test_v4_blend.npy")

    def opt(keys):
        def neg(w):
            w = np.abs(w) / np.abs(w).sum()
            return -best_threshold(y, sum(wi * comp[k] for wi, k in zip(w, keys)))[1]
        r = minimize(neg, np.ones(len(keys)) / len(keys), method="Nelder-Mead",
                     options={"maxiter": 80})
        w = np.abs(r.x) / np.abs(r.x).sum()
        p = sum(wi * comp[k] for wi, k in zip(w, keys))
        t, s = best_threshold(y, p)
        print(f"blend {keys}: {s:.5f} @ t={t:.3f} w={w.round(3)}", flush=True)
        return s, w, keys, t

    results = [opt(["lgb", "xgb", "cat"]), opt(["lgb", "xgb", "cat", "v4"])]
    s, w, keys, t_g = max(results, key=lambda r: r[0])
    p_blend = sum(wi * comp[k] for wi, k in zip(w, keys))
    p_test = sum(wi * comp_te[k] for wi, k in zip(w, keys))
    np.save("data/processed/oof_v11_blend.npy", p_blend)
    np.save("data/processed/test_v11_blend.npy", p_test)

    fold_ts = [best_threshold(y[fold_of_row == f], p_blend[fold_of_row == f])[0]
               for f in range(N_FOLDS)]
    t_r = float(np.median(fold_ts)); s_r = wf1(y, p_blend, t_r)
    print(f"best combo {keys}: global {s:.5f}@{t_g:.3f} robust {s_r:.5f}@{t_r:.3f}", flush=True)

    labels = np.where(p_test > t_r, "Dead", "Alive")
    path = f"submissions/sub_{sub_num}_{desc}_cv{s_r:.5f}.csv"
    pd.DataFrame({"patient_id": te["patient_id"], "vital_status": labels}).to_csv(path, index=False)
    print(f"saved {path} Dead={np.mean(labels == 'Dead'):.4f}", flush=True)


if __name__ == "__main__":
    main()

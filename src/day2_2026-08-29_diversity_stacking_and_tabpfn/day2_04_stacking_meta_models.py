"""True stacking: meta-LGBM on component OOF probabilities + context features.

Meta-features: OOF probs of saved components + year, stage ordinals, age.
Nested 10-fold CV for honest meta evaluation (meta trained OOF over the same folds).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)                              # sibling files in this day's folder
sys.path.insert(0, os.path.dirname(_THIS_DIR))              # src/ -> for common.features
from common.features import build_features  # noqa: E402

SEED = 42
N_FOLDS = 10

COMPONENTS = {  # oof path fragment -> test path fragment
    "v4_lgb": "v4_lgb", "v4_xgb": "v4_xgb", "v4_cat": "v4_cat",
    "v8_012_lgb": "v8_012_lgb", "v8_012_xgb": "v8_012_xgb", "v8_012_cat": "v8_012_cat",
    "v7_lgb": "v7_lgb", "v7_xgb": "v7_xgb", "v7_cat": "v7_cat",
    "div_hgb": "div_hgb", "div_lr": "div_lr",
}


def best_wf1(y, p):
    ts = np.arange(0.35, 0.75, 0.005)
    scores = [f1_score(y, (p > t).astype(int), average="weighted") for t in ts]
    i = int(np.argmax(scores))
    return ts[i], scores[i]


def main():
    tr = pd.read_csv("data/raw/train.csv")
    te = pd.read_csv("data/raw/test.csv")
    y = (tr["vital_status"] == "Dead").astype(int).values

    Xf = build_features(tr)
    Xf_te = build_features(te)[Xf.columns]
    ctx_cols = ["years_since_dx", "age_num", "tnm_sum", "stage_ord", "mets_count",
                "surgery_done", "radiation_given", "tumor_size_best"]
    ctx_cols = [c for c in ctx_cols if c in Xf.columns]

    M = pd.DataFrame({k: np.load(f"data/processed/oof_{v}.npy") for k, v in COMPONENTS.items()})
    M_te = pd.DataFrame({k: np.load(f"data/processed/test_{v}.npy") for k, v in COMPONENTS.items()})
    for c in ctx_cols:
        M[c] = Xf[c].values
        M_te[c] = Xf_te[c].values
    print(f"meta matrix: {M.shape}, components: {list(COMPONENTS)}", flush=True)

    for name, make in {
        "meta_lgb": lambda: None,
        "meta_lr": lambda: LogisticRegression(C=1.0, max_iter=2000),
    }.items():
        oof_meta = np.zeros(len(tr))
        te_meta = np.zeros(len(te))
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
        for i_tr, i_va in skf.split(M, y):
            if name == "meta_lgb":
                m = lgb.train(
                    dict(objective="binary", learning_rate=0.03, num_leaves=15,
                         min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.7,
                         bagging_freq=1, lambda_l2=5.0, seed=SEED, verbose=-1),
                    lgb.Dataset(M.iloc[i_tr], y[i_tr]), num_boost_round=2000,
                    valid_sets=[lgb.Dataset(M.iloc[i_va], y[i_va])],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
                oof_meta[i_va] = m.predict(M.iloc[i_va], num_iteration=m.best_iteration)
                te_meta += m.predict(M_te, num_iteration=m.best_iteration) / N_FOLDS
            else:
                probs_only = list(COMPONENTS)
                mm = make()
                mm.fit(M.iloc[i_tr][probs_only], y[i_tr])
                oof_meta[i_va] = mm.predict_proba(M.iloc[i_va][probs_only])[:, 1]
                te_meta += mm.predict_proba(M_te[probs_only])[:, 1] / N_FOLDS
        t, s = best_wf1(y, oof_meta)
        print(f"{name}: OOF wF1={s:.5f} @ t={t:.3f}", flush=True)
        np.save(f"data/processed/oof_stack_{name}.npy", oof_meta)
        np.save(f"data/processed/test_stack_{name}.npy", te_meta)
        labels = np.where(te_meta > t, "Dead", "Alive")
        path = f"submissions/sub_stack_{name}_cv{s:.5f}.csv"
        pd.DataFrame({"patient_id": te["patient_id"], "vital_status": labels}).to_csv(path, index=False)
        print(f"saved {path} Dead={np.mean(labels=='Dead'):.4f}", flush=True)


if __name__ == "__main__":
    main()

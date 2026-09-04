"""Insight 2.0 baseline: LightGBM, StratifiedKFold(5), OOF metric analysis + threshold tuning.

Usage: .venv/bin/python src/train.py <sub_number> <description>
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
rng = np.random.RandomState(SEED)

NUMERIC_CODED = ["tumor_size_overtime", "tumor_size_summary", "cs_tumor_size20042015"]


def parse_coded_numeric(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x[x >= 990] = np.nan  # 999/998 etc = unknown codes
    return x


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=["patient_id", "vital_status"], errors="ignore").copy()
    for c in NUMERIC_CODED:
        out[c + "_num"] = parse_coded_numeric(out[c])
        out[c + "_missing"] = out[c + "_num"].isna().astype(int)
        out = out.drop(columns=[c])
    # best available tumor size across the three era-specific fields
    out["tumor_size_best"] = (
        out["tumor_size_summary_num"]
        .fillna(out["tumor_size_overtime_num"])
        .fillna(out["cs_tumor_size20042015_num"])
    )
    out["nodes_ratio"] = np.where(
        df["regional_nodes_examined"] > 0,
        df["regional_nodes_positive"] / df["regional_nodes_examined"].clip(lower=1),
        np.nan,
    )
    out["years_since_dx"] = 2024 - df["year_of_diagnosis"]
    for c in out.columns:
        if pd.api.types.is_string_dtype(out[c]) or out[c].dtype == object:
            out[c] = out[c].astype("category")
    return out


def main():
    tr = pd.read_csv("data/raw/train.csv")
    te = pd.read_csv("data/raw/test.csv")
    y = (tr["vital_status"] == "Dead").astype(int).values

    X = build_features(tr)
    X_te = build_features(te)
    X_te = X_te[X.columns]
    for c in X.columns:
        if str(X[c].dtype) == "category":
            cats = pd.api.types.union_categoricals([X[c], X_te[c]]).categories
            X[c] = X[c].cat.set_categories(cats)
            X_te[c] = X_te[c].cat.set_categories(cats)

    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=40,
        seed=SEED,
        verbose=-1,
    )

    oof = np.zeros(len(tr))
    test_pred = np.zeros(len(te))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (i_tr, i_va) in enumerate(skf.split(X, y)):
        dtr = lgb.Dataset(X.iloc[i_tr], y[i_tr])
        dva = lgb.Dataset(X.iloc[i_va], y[i_va])
        model = lgb.train(
            params, dtr, num_boost_round=3000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[i_va] = model.predict(X.iloc[i_va], num_iteration=model.best_iteration)
        test_pred += model.predict(X_te, num_iteration=model.best_iteration) / N_FOLDS
        print(f"fold {fold}: best_iter={model.best_iteration} "
              f"macroF1@0.5={f1_score(y[i_va], (oof[i_va] > 0.5).astype(int), average='macro'):.5f}")

    np.save("data/processed/oof_baseline.npy", oof)
    np.save("data/processed/test_pred_baseline.npy", test_pred)

    print("\n=== OOF metric analysis (probability threshold sweep) ===")
    best = {}
    for name, fn in {
        "macro_f1": lambda yt, yp: f1_score(yt, yp, average="macro"),
        "f1_dead": lambda yt, yp: f1_score(yt, yp, pos_label=1),
        "f1_alive": lambda yt, yp: f1_score(yt, yp, pos_label=0),
        "weighted_f1": lambda yt, yp: f1_score(yt, yp, average="weighted"),
    }.items():
        scores = [(t, fn(y, (oof > t).astype(int))) for t in np.arange(0.05, 0.95, 0.01)]
        t_best, s_best = max(scores, key=lambda x: x[1])
        s_05 = fn(y, (oof > 0.5).astype(int))
        best[name] = (t_best, s_best)
        print(f"{name:12s}: @0.5={s_05:.5f}  best={s_best:.5f} @ threshold={t_best:.2f}")

    # submission using macro-F1-optimal threshold (leading hypothesis for the metric)
    t_macro = best["macro_f1"][0]
    sub_num, desc = sys.argv[1], sys.argv[2]
    labels = np.where(test_pred > t_macro, "Dead", "Alive")
    sub = pd.DataFrame({"patient_id": te["patient_id"], "vital_status": labels})
    path = f"submissions/sub_{sub_num}_{desc}_cv{best['macro_f1'][1]:.5f}.csv"
    sub.to_csv(path, index=False)
    print(f"\npred rate Dead={np.mean(labels == 'Dead'):.4f} (train rate {y.mean():.4f})")
    print(f"saved {path}")


if __name__ == "__main__":
    main()

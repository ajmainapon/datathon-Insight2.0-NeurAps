"""Second diversity wave: RandomForest, ExtraTrees, MLP — 10-fold OOF for blend testing."""
import sys
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    # trees: integer codes + median-imputed numerics
    Xt = X.copy(); Xt_te = X_te.copy()
    for c in cat_cols:
        Xt[c] = Xt[c].cat.codes
        Xt_te[c] = Xt_te[c].cat.codes
    med = Xt[num_cols].median()
    Xt[num_cols] = Xt[num_cols].fillna(med)
    Xt_te[num_cols] = Xt_te[num_cols].fillna(med)

    # MLP: one-hot + scaled
    Xs = X.copy(); Xs_te = X_te.copy()
    for c in cat_cols:
        Xs[c] = Xs[c].astype(str).where(Xs[c].notna(), "NA")
        Xs_te[c] = Xs_te[c].astype(str).where(Xs_te[c].notna(), "NA")
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
    ])

    defs = {
        "rf": (Xt, Xt_te, lambda: RandomForestClassifier(
            n_estimators=800, min_samples_leaf=5, max_features="sqrt",
            n_jobs=-1, random_state=SEED)),
        "et": (Xt, Xt_te, lambda: ExtraTreesClassifier(
            n_estimators=800, min_samples_leaf=5, max_features="sqrt",
            n_jobs=-1, random_state=SEED)),
        "mlp": (Xs, Xs_te, lambda: Pipeline([("pre", pre), ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3,
            batch_size=512, max_iter=60, early_stopping=True, n_iter_no_change=8,
            random_state=SEED))])),
    }

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for name, (Xa, Xa_te, make) in defs.items():
        oof = np.zeros(len(tr)); tep = np.zeros(len(te))
        for f, (i_tr, i_va) in enumerate(skf.split(Xa, y)):
            m = make()
            m.fit(Xa.iloc[i_tr], y[i_tr])
            oof[i_va] = m.predict_proba(Xa.iloc[i_va])[:, 1]
            tep += m.predict_proba(Xa_te)[:, 1] / N_FOLDS
            print(f"{name} fold {f} done", flush=True)
        t, s = best_wf1(y, oof)
        print(f"{name}: OOF wF1={s:.5f} @ t={t:.3f}", flush=True)
        np.save(f"data/processed/oof_div2_{name}.npy", oof)
        np.save(f"data/processed/test_div2_{name}.npy", tep)


if __name__ == "__main__":
    main()

"""Optuna tuning for LGBM (and CatBoost) on weighted F1 with per-fold threshold tuning."""
import sys
import os

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)                              # sibling files in this day's folder
sys.path.insert(0, os.path.dirname(_THIS_DIR))              # src/ -> for common.features
from common.features import build_features  # noqa: E402

SEED = 42

tr = pd.read_csv("data/raw/train.csv")
y = (tr["vital_status"] == "Dead").astype(int).values
X = build_features(tr)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))


def fold_wf1(y_va, p):
    ts = np.arange(0.35, 0.75, 0.01)
    return max(f1_score(y_va, (p > t).astype(int), average="weighted") for t in ts)


def objective(trial):
    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 200, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=1,
        lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
        lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
        min_gain_to_split=trial.suggest_float("min_gain_to_split", 0, 0.5),
        seed=SEED,
        verbose=-1,
    )
    scores = []
    for i_tr, i_va in folds:
        m = lgb.train(params, lgb.Dataset(X.iloc[i_tr], y[i_tr]), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(X.iloc[i_va], y[i_va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        scores.append(fold_wf1(y[i_va], m.predict(X.iloc[i_va], num_iteration=m.best_iteration)))
    return float(np.mean(scores))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=int(sys.argv[1]) if len(sys.argv) > 1 else 40)
    print("best value:", round(study.best_value, 5))
    print("best params:", study.best_params)

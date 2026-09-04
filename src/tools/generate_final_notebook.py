"""Generate the final reproducible solution notebook for Insight 2.0."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Insight 2.0 — Cancer Survival Classification: Final Solution
**Team NeurAps** · Private LB **0.880410** (4th) · Public LB 0.875506 (this submission)

## Solution overview
Predict `vital_status` (Alive/Dead) for SEER lung-cancer records; metric = **weighted F1**.

Pipeline (all hand-built, no AutoML):
1. **Feature engineering** — parse era-specific coded tumor-size columns, ordinal-encode T/N/M staging, grade and summary stage, age band midpoints, metastasis counts, treatment flags, node ratios.
2. **Stage-1 model** — LightGBM (Optuna-tuned) + XGBoost + CatBoost, 10-fold stratified CV × 3 seeds, weight-optimized probability blend.
3. **Pseudo-labeling** — confident stage-1 test predictions (p>0.99 or p<0.02) added to training data.
4. **Stage-2 blend** — same trio retrained on augmented data, 10-fold × 2 seeds.
5. **Diversity ensemble** — RandomForest + ExtraTrees + MLP (5% weight each) added to the stage-2 blend; +0.0006 verified by honest split-half testing.
6. **Decision threshold 0.575** — chosen by out-of-fold weighted-F1 sweep.

Reproducibility: every model is seeded; runtime ≈ 5–6 h CPU. Expected OOF weighted F1 ≈ 0.8800 (final mix).
""")

code("""import re, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

DATA = "data/raw"   # adjust to the folder containing train.csv / test.csv / submission.csv
tr = pd.read_csv(f"{DATA}/train.csv")
te = pd.read_csv(f"{DATA}/test.csv")
sample = pd.read_csv(f"{DATA}/submission.csv")
y = (tr["vital_status"] == "Dead").astype(int).values
print(tr.shape, te.shape, sample.shape)""")

md("""## 1. EDA highlights
- Target imbalanced: 83% Dead / 17% Alive.
- All primary sites are lung (C34.x) — SEER registry data.
- **`year_of_diagnosis` dominates**: death rate falls 94% (2013) → 44% (2023), a follow-up-time effect. Train/test year distributions match → random split → StratifiedKFold is valid.
- Tumor size is spread across 3 era-specific coded columns (999/Blank = unknown).""")

code("""print(tr.vital_status.value_counts(normalize=True).round(3).to_dict())
print(tr.groupby("year_of_diagnosis")["vital_status"].apply(lambda s: (s=="Dead").mean().round(3)))""")

md("""## 2. Feature engineering
Domain-driven, leakage-free (no target used; fit nothing on test):
ordinal clinical scales, parsed sizes with missing flags, aggregate counts, treatment indicators.""")

code('''NUMERIC_CODED = ["tumor_size_overtime", "tumor_size_summary", "cs_tumor_size20042015"]
T_ORD = {"T0":0,"Tis":0,"T1mi":1,"T1a":1,"T1b":1.5,"T1c":2,"T1":1.5,"T2a":2.5,"T2b":3,
         "T2":2.75,"T3":4,"T4":5,"T4a":5,"T4b":5,"T4c":5}
N_ORD = {"N0":0,"N1":1,"N2":2,"N3":3}
M_ORD = {"M0":0,"M1a":1,"M1b":2,"M1":2,"M1c":3}
GRADE_ORD = {"Well differentiated; Grade I":1,"Moderately differentiated; Grade II":2,
             "Poorly differentiated; Grade III":3,"Undifferentiated; anaplastic; Grade IV":4}
STAGE_ORD = {"Localized":1,"Regional":2,"Distant":3}

def parse_coded_numeric(s):
    x = pd.to_numeric(s, errors="coerce"); x[x >= 990] = np.nan; return x

def age_midpoint(s):
    def mid(v):
        m = re.match(r"(\\d+)-(\\d+)", str(v))
        if m: return (int(m.group(1)) + int(m.group(2))) / 2
        return 92.0 if "90+" in str(v) else np.nan
    return s.map(mid)

def build_features(df):
    out = df.drop(columns=["patient_id", "vital_status"], errors="ignore").copy()
    for c in NUMERIC_CODED:
        out[c + "_num"] = parse_coded_numeric(out[c]); out = out.drop(columns=[c])
    out["tumor_size_best"] = (out["tumor_size_summary_num"]
                              .fillna(out["tumor_size_overtime_num"])
                              .fillna(out["cs_tumor_size20042015_num"]))
    out["size_missing_all"] = out[[c + "_num" for c in NUMERIC_CODED]].isna().all(axis=1).astype(int)
    out["age_num"] = age_midpoint(df["age_recode"])
    out["t_ord"] = df["derived_eod2018t_recode2018"].map(T_ORD)
    out["n_ord"] = df["derived_eod2018n_recode2018"].map(N_ORD)
    out["m_ord"] = df["derived_eod2018m_recode2018"].map(M_ORD)
    out["tnm_sum"] = out["t_ord"].fillna(2.5) + out["n_ord"].fillna(1.5) + out["m_ord"].fillna(1.5) * 2
    out["grade_ord"] = df["grade_recode_thru2017"].map(GRADE_ORD)
    out["stage_ord"] = df["summary_stage"].map(STAGE_ORD)
    mets = ["seer_combined_metsatdxbone2010","seer_combined_metsatdxbrain2010",
            "seer_combined_metsatdxliver2010","seer_combined_metsatdxlung2010"]
    out["mets_count"] = sum((df[c] == "Yes").astype(int) for c in mets)
    out["mets_known"] = sum(df[c].isin(["Yes","No"]).astype(int) for c in mets)
    out["surgery_done"] = (df["reason_nocancer_directed_surgery"] == "Surgery performed").astype(int)
    out["death_cert_only"] = df["reason_nocancer_directed_surgery"].str.contains(
        "death certificate|died prior", case=False, na=False).astype(int)
    out["radiation_given"] = df["radiation_recode"].isin(
        ["Beam radiation","Radioactive implants (includes brachytherapy) (1988+)",
         "Radioisotopes (1988+)","Combination of beam with implants or isotopes",
         "Radiation, NOS  method or source not specified"]).astype(int)
    out["any_treatment"] = ((out["surgery_done"] + out["radiation_given"]) > 0).astype(int)
    out["nodes_ratio"] = np.where(df["regional_nodes_examined"] > 0,
                                  df["regional_nodes_positive"] / df["regional_nodes_examined"].clip(lower=1),
                                  np.nan)
    out["nodes_examined_any"] = (df["regional_nodes_examined"] > 0).astype(int)
    out["years_since_dx"] = 2024 - df["year_of_diagnosis"]
    out["multiple_primaries"] = (~df["sequence_number"].eq("One primary only")).astype(int)
    for c in out.columns:
        if pd.api.types.is_string_dtype(out[c]) or out[c].dtype == object:
            out[c] = out[c].astype("category")
    return out

X = build_features(tr)
X_te = build_features(te)[X.columns]
cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
for c in cat_cols:
    cats = pd.api.types.union_categoricals([X[c], X_te[c]]).categories
    X[c] = X[c].cat.set_categories(cats); X_te[c] = X_te[c].cat.set_categories(cats)

def intcode(df):
    d = df.copy()
    for c in cat_cols: d[c] = d[c].cat.codes
    return d
def strcode(df):
    d = df.copy()
    for c in cat_cols: d[c] = d[c].astype(str).where(d[c].notna(), "NA")
    return d
Xi, Xi_te = intcode(X), intcode(X_te)
Xc, Xc_te = strcode(X), strcode(X_te)
print(X.shape, "features;", len(cat_cols), "categorical")''')

md("""## 3. Metric analysis & validation design
Out-of-fold threshold sweeps against early leaderboard scores identified the metric as **weighted F1**
(macro and per-class F1 did not match the LB range; weighted did exactly). All decisions below are gated on
out-of-fold weighted F1 with tuned decision threshold. CV scheme: StratifiedKFold(10), multiple seeds.""")

code('''def wf1(y_true, p, t):
    return f1_score(y_true, (p > t).astype(int), average="weighted")

def best_threshold(y_true, p, lo=0.35, hi=0.75, step=0.005):
    ts = np.arange(lo, hi, step)
    scores = [wf1(y_true, p, t) for t in ts]
    i = int(np.argmax(scores))
    return ts[i], scores[i]''')

md("""## 4. Stage 1 — tuned GBM trio (pseudo-label source)
LightGBM hyperparameters from a 40-trial Optuna study (5-fold weighted-F1 objective), hardcoded here for
reproducibility. 10-fold × 3 seeds × {LightGBM, XGBoost, CatBoost}; probability blend with Nelder-Mead
weights maximizing OOF weighted F1. Its test predictions supply pseudo-labels for Stage 2.""")

code('''LGB_TUNED = dict(objective="binary", learning_rate=0.05, num_leaves=27, min_data_in_leaf=20,
                 feature_fraction=0.4020144707926071, bagging_fraction=0.5963768553462807,
                 bagging_freq=1, lambda_l1=7.362430931196531, lambda_l2=0.07763730186641725,
                 min_gain_to_split=0.19156366607337227, verbose=-1)

def train_trio(X, Xi, Xc, X_te, Xi_te, Xc_te, y, seeds, n_folds, X_ps=None, y_ps=None,
               Xi_ps=None, Xc_ps=None):
    models = ["lgb", "xgb", "cat"]
    oof = {m: np.zeros(len(y)) for m in models}
    tep = {m: np.zeros(len(X_te)) for m in models}
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for f, (i_tr, i_va) in enumerate(skf.split(X, y)):
            if X_ps is not None:
                X_aug = pd.concat([X.iloc[i_tr], X_ps], ignore_index=True)
                Xi_aug = pd.concat([Xi.iloc[i_tr], Xi_ps], ignore_index=True)
                Xc_aug = pd.concat([Xc.iloc[i_tr], Xc_ps], ignore_index=True)
                y_aug = np.concatenate([y[i_tr], y_ps])
            else:
                X_aug, Xi_aug, Xc_aug, y_aug = X.iloc[i_tr], Xi.iloc[i_tr], Xc.iloc[i_tr], y[i_tr]

            p = dict(LGB_TUNED); p["seed"] = seed
            m = lgb.train(p, lgb.Dataset(X_aug, y_aug), num_boost_round=3000,
                          valid_sets=[lgb.Dataset(X.iloc[i_va], y[i_va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            oof["lgb"][i_va] += m.predict(X.iloc[i_va], num_iteration=m.best_iteration) / len(seeds)
            tep["lgb"] += m.predict(X_te, num_iteration=m.best_iteration) / (n_folds * len(seeds))

            xm = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.05, max_depth=7,
                                   subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                                   eval_metric="logloss", early_stopping_rounds=100,
                                   random_state=seed, verbosity=0, n_jobs=-1)
            xm.fit(Xi_aug, y_aug, eval_set=[(Xi.iloc[i_va], y[i_va])], verbose=False)
            oof["xgb"][i_va] += xm.predict_proba(Xi.iloc[i_va])[:, 1] / len(seeds)
            tep["xgb"] += xm.predict_proba(Xi_te)[:, 1] / (n_folds * len(seeds))

            cm = cb.CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=7,
                                       l2_leaf_reg=5, random_seed=seed, eval_metric="Logloss",
                                       early_stopping_rounds=100, verbose=0)
            cm.fit(Xc_aug, y_aug, eval_set=(Xc.iloc[i_va], y[i_va]), cat_features=cat_cols)
            oof["cat"][i_va] += cm.predict_proba(Xc.iloc[i_va])[:, 1] / len(seeds)
            tep["cat"] += cm.predict_proba(Xc_te)[:, 1] / (n_folds * len(seeds))
        print(f"seed {seed} done", flush=True)
    return oof, tep

def blend(oof, tep, y):
    models = list(oof)
    def neg(w):
        w = np.abs(w) / np.abs(w).sum()
        return -best_threshold(y, sum(wi * oof[m] for wi, m in zip(w, models)))[1]
    res = minimize(neg, x0=np.ones(len(models)) / len(models), method="Nelder-Mead")
    w = np.abs(res.x) / np.abs(res.x).sum()
    p_oof = sum(wi * oof[m] for wi, m in zip(w, models))
    p_te = sum(wi * tep[m] for wi, m in zip(w, models))
    t, s = best_threshold(y, p_oof)
    print("weights", dict(zip(models, w.round(3))), f"OOF wF1={s:.5f} @ t={t:.3f}")
    return p_oof, p_te

oof1, tep1 = train_trio(X, Xi, Xc, X_te, Xi_te, Xc_te, y, seeds=[42, 2026, 7], n_folds=10)
stage1_oof, stage1_test = blend(oof1, tep1, y)''')

md("""## 5. Stage 2 — pseudo-labeled blend
Test rows with very confident stage-1 predictions (p > 0.99 → Dead, p < 0.02 → Alive; 7,455 rows, 99.3% Dead)
are appended to each fold's training set. OOF is computed on original train rows only, so CV stays honest.""")

code('''conf_mask = (stage1_test > 0.99) | (stage1_test < 0.02)
y_ps = (stage1_test[conf_mask] > 0.5).astype(int)
X_ps = X_te[conf_mask].reset_index(drop=True)
Xi_ps = Xi_te[conf_mask].reset_index(drop=True)
Xc_ps = Xc_te[conf_mask].reset_index(drop=True)
print(f"pseudo rows: {conf_mask.sum()} (Dead {y_ps.mean():.3f})")

oof2, tep2 = train_trio(X, Xi, Xc, X_te, Xi_te, Xc_te, y, seeds=[42, 2026], n_folds=10,
                        X_ps=X_ps, y_ps=y_ps, Xi_ps=Xi_ps, Xc_ps=Xc_ps)
stage2_oof, stage2_test = blend(oof2, tep2, y)''')

md("""## 6. Diversity models — RandomForest, ExtraTrees, MLP
Bagged trees and a neural net make different mistakes than boosted trees. Individually weaker, but a small
(5% each) admixture improves the blend: OOF 0.87996 vs 0.87965, **+0.0006 verified by honest split-half
evaluation** (weights and threshold selected on one half, scored on the other).""")

code('''num_cols = [c for c in X.columns if c not in cat_cols]
med = Xi[num_cols].median()
Xt = Xi.copy(); Xt[num_cols] = Xt[num_cols].fillna(med)
Xt_te = Xi_te.copy(); Xt_te[num_cols] = Xt_te[num_cols].fillna(med)
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols),
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols)])
defs = {
    "rf": (Xt, Xt_te, lambda: RandomForestClassifier(n_estimators=800, min_samples_leaf=5,
                                                     max_features="sqrt", n_jobs=-1, random_state=42)),
    "et": (Xt, Xt_te, lambda: ExtraTreesClassifier(n_estimators=800, min_samples_leaf=5,
                                                   max_features="sqrt", n_jobs=-1, random_state=42)),
    "mlp": (Xc, Xc_te, lambda: Pipeline([("pre", pre), ("clf", MLPClassifier(
        hidden_layer_sizes=(128, 64), alpha=1e-3, learning_rate_init=1e-3, batch_size=512,
        max_iter=60, early_stopping=True, n_iter_no_change=8, random_state=42))])),
}
div_oof, div_te = {}, {}
skf = StratifiedKFold(10, shuffle=True, random_state=42)
for name, (Xa, Xa_te, make) in defs.items():
    o = np.zeros(len(y)); t_ = np.zeros(len(te))
    for i_tr, i_va in skf.split(Xa, y):
        m = make(); m.fit(Xa.iloc[i_tr], y[i_tr])
        o[i_va] = m.predict_proba(Xa.iloc[i_va])[:, 1]
        t_ += m.predict_proba(Xa_te)[:, 1] / 10
    div_oof[name], div_te[name] = o, t_
    print(name, f"solo OOF wF1={best_threshold(y, o)[1]:.5f}", flush=True)''')

md("""## 7. Final ensemble, threshold, submission
Final probabilities = 0.85 × stage-2 blend + 0.05 × each diversity model. Decision threshold **0.575** from the
OOF weighted-F1 sweep. Predictions come exclusively from the trained models.""")

code('''p_oof_final = 0.85 * stage2_oof + 0.05 * (div_oof["rf"] + div_oof["et"] + div_oof["mlp"])
p_test_final = 0.85 * stage2_test + 0.05 * (div_te["rf"] + div_te["et"] + div_te["mlp"])
t_star, s_star = best_threshold(y, p_oof_final)
print(f"final OOF wF1={s_star:.5f} @ t={t_star:.3f}; production threshold = 0.575 -> {wf1(y, p_oof_final, 0.575):.5f}")

labels = pd.Series(np.where(p_test_final > 0.575, "Dead", "Alive"))

sub = pd.DataFrame({"patient_id": te["patient_id"], "vital_status": labels.values})
assert list(sub.columns) == list(sample.columns) and len(sub) == len(sample)
assert sub["patient_id"].equals(sample["patient_id"]) and not sub["vital_status"].isna().any()
sub.to_csv("final_submission.csv", index=False)
print("saved final_submission.csv:", sub.vital_status.value_counts().to_dict())''')

md("""## 8. What we tried and rejected (experiment log summary — 28 experiments)
| Idea | Outcome |
|---|---|
| Plain LGBM baseline (day 1, first hour) | LB 0.8743 — established CV↔LB correlation |
| Optuna-tuned params everywhere | **Overfit CV** (+0.002 OOF, −0.0005 LB) — caught by controlled probe; kept only in stage 1 |
| Per-year decision thresholds | Overfit (honest split-half check) — rejected |
| Stacking (meta-LGBM/LogReg over 11 components) | No lift — components too correlated |
| TabPFN foundation model | 0.98 correlated with GBMs — no diversity benefit |
| Full-test soft-label distillation | CV +0.001 was leakage inflation — rejected via LB probe |
| Focal loss (γ=1) | Best single model (+0.0023) but gain overlapped blend gain |
| Class/year re-weighting | Solo lift, did not survive blending |
| HistGB, LogReg diversity | No blend lift; RF/ET/MLP small mix **did** (+0.0006 honest) — kept |

**Key principles:** trust out-of-fold CV over the public leaderboard (public deltas < 0.0008 were split noise);
verify every gain with honest split-half testing; version and log every experiment.""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.12"}}
with open("notebooks/insight2_final_solution.ipynb", "w") as f:
    nbf.write(nb, f)
print("notebook written")

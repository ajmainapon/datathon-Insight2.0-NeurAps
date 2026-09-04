# Day 1 — 28 Aug 2026: baseline, metric identification, first blend

Scripts run in this order:

1. `day1_01_baseline_lgbm.py` — single LightGBM, 5-fold, minimal features.
   Produced sub_001 and the first out-of-fold threshold sweep, which is what
   told us the metric was **weighted F1** (not macro, not Dead-class F1) —
   see `experiments/log.md` row 1.
2. `day1_02_feature_engineering_blend.py` — the 18 engineered features
   (`common/features.py` holds the shared function) + LightGBM/XGBoost/CatBoost
   blend. Produced sub_002, our first top-3 finish (row 2).
3. `day1_03_optuna_tune_lgbm.py` — 40-trial Optuna search for LightGBM.
4. `day1_04_tuned_trio_blend.py` — the tuned trio, 10-fold x 3 seeds (sub_003/004).
   CV and the public leaderboard diverged here (row 3) — the first sign that
   tuning could overfit cross-validation.
5. `day1_05_pseudo_label_blend.py` — first pseudo-labelling pass, using
   `day1_04`'s test predictions as the confidence source (sub_005, row 5).
   **This is also the script used again on Day 2/3** with different confidence
   bands and seed counts — it is the core pipeline for the rest of the project.
6. `day1_06_diversity_wave1_hgb_logreg.py` — HistGradientBoosting + logistic
   regression, evaluated for blending (their OOF run is reported on Day 2,
   `experiments/log.md` row 10 — the code was written the evening before).
7. `day1_07_round2_pseudo_default_params.py` — a second pseudo-labelling script,
   default (untuned) parameters, confidence bands as CLI arguments. Written this
   evening but **run on Day 2** with narrow (0.02/0.99) and wide (0.05/0.97)
   bands to produce sub_007 and sub_008 (rows 8-9).

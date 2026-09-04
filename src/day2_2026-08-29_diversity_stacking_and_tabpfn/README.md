# Day 2 — 29 Aug 2026: round-2 pseudo-labels, new features, diversity, stacking, TabPFN

1. `day2_01_features_v7.py` — an extended feature set (histology as categorical,
   interaction columns, frequency encodings) on top of `common/features.py`.
2. `day2_02_v7_features_pseudo_blend.py` — the v7 feature set on the day1_05
   pseudo-label recipe (sub_009, row 11). Scored flat — the first sign the
   feature signal was saturating.
3. `day2_03_v8_five_seed_consolidation.py` — 5 seeds, lower learning rate,
   more rounds: a pure variance-reduction test (sub_012, row 13).
4. `day2_04_stacking_meta_models.py` — meta-LightGBM / meta-logistic regression
   over 11 component OOF predictions (row 15). No lift — components too correlated.
5. `day2_05_diversity_wave2_rf_et_mlp.py` — RandomForest, ExtraTrees, MLP.
   This produced the diversity mixture (+0.0006, split-half verified) that
   became part of the **final submitted pipeline** (row 17, sub_014/016).
6. `day2_06_tabpfn_foundation_model.py` — a tabular foundation model, run on
   the laptop GPU (row 18). Correlated 0.98 with the GBM blend — evidence the
   dataset's signal was saturating around OOF 0.880.

Also produced this day: round-2 pseudo-labelling narrow/wide bands, using
`day1_07_round2_pseudo_default_params.py` (written the evening before, run
this day — see that folder's README) to produce sub_007 and sub_008
(rows 8-9), plus the exact-duplicate leakage check (row 16, a one-off pandas
merge, not saved as a script — later formalized in `day4_01_leakfix_and_threshold_scan.py`).

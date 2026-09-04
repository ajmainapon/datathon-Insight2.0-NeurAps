# Submission manifest

Every file scored on the Kaggle leaderboard, in submission order, with the script that
produced it. The actual `.csv` files are not tracked in this repository (see
`.gitignore`) — each row's OOF/public/private score is reproducible by running the
named script and applying the given threshold. `sub_016` and `sub_024` were the two
files selected as our scored finals; `sub_016` is marked **selected (honest-CV pick)**.

| Sub | Script | OOF wF1 | Public | Private | Notes |
|---|---|---|---|---|---|
| 001 | `day1_01_baseline_lgbm.py`, t=0.54 | 0.87696 | 0.874348 | 0.878678 | baseline; confirmed the metric is weighted F1 |
| 002 | `day1_02_feature_engineering_blend.py`, t=0.59 | 0.87768 | 0.876023 | 0.879663 | first top-3 finish |
| 003 | `day1_04_tuned_trio_blend.py`, t=0.56 | 0.87964 | 0.875246 | 0.880395 | Optuna-tuned trio; CV/LB diverged (see day1 README) |
| 004 | same probs as 003, t=0.59 (probe) | 0.87862 | 0.875493 | 0.879243 | isolates threshold from model effect |
| 005 | `day1_05_pseudo_label_blend.py`, t=0.573 | 0.87942 | 0.876190 | 0.879515 | first pseudo-labelling pass |
| 006 | `day1_04`-style, default params, t=0.593 | 0.87818 | 0.875322 | 0.879248 | default vs tuned params control |
| 007 | `day1_07_round2_pseudo_default_params.py`, narrow bands | 0.87847 | 0.876123 | 0.879408 | round-2 pseudo, run on day 2 |
| 008 | `day1_07_round2_pseudo_default_params.py`, wide bands | 0.87833 | not submitted (file only) | — | wider bands, no CV gain |
| 009 | `day2_02_v7_features_pseudo_blend.py` | 0.87882 | not submitted (file only) | — | extended features, flat |
| 010 | v4 blend, t=0.555 | 0.87832 | 0.876015 | 0.880051 | threshold probe |
| 011 | v4 blend, t=0.592 | 0.87913 | 0.876162 | 0.879303 | threshold probe |
| 012 | `day2_03_v8_five_seed_consolidation.py` | 0.87824 | not submitted (file only) | — | variance-reduction only |
| 013 | v4+v8 combined, t=0.570 | 0.87949 | 0.875770 | 0.879710 | plateau confirmed |
| 014 | `day2_05_diversity_wave2_rf_et_mlp.py` + v4, t=0.575 | 0.87996 | 0.875506 | 0.880370 | best honest CV before the leak fix |
| 015 | leak fix of 005 (`day4_01_leakfix_and_threshold_scan.py`) | — | 0.876190 | 0.879555 | |
| 016 | leak fix of 014 | 0.87996 | 0.875506 | **0.880410** | **selected (honest-CV pick) — 4th of 79** |
| 017 | v4 blend, t=0.565, leak-fixed | — | 0.876227 | 0.880138 | |
| 018 | v4 blend, t=0.582, leak-fixed | — | 0.875889 | 0.879299 | |
| 019 | `day3_05_soft_label_distillation_REJECTED.py` | 0.88075* | 0.875330 | 0.878816 | *contaminated: soft labels leaked validation info |
| 020 | `day3_03_v9_weighted_pseudo_blend.py` | 0.87871 | not submitted (file only) | — | class/year weighting, didn't survive blending |
| 021 | `day3_04_v10_focal_loss_blend.py` | 0.87973 | 0.875972 | 0.879168 | focal loss across the trio |
| 022 | `day3_06_v11_focal_plus_pseudo.py` | 0.87929 | 0.875646 | 0.879305 | highest global OOF (0.88005), unstable per-fold threshold |
| 023 | v4 blend, t=0.560, leak-fixed | — | 0.875861 | 0.880251 | |
| 024 | v4 blend, t=0.570, leak-fixed | — | **0.876311** | 0.879732 | **selected (public-best pick)** |
| 025 | v4 blend, t=0.5675, leak-fixed | — | 0.876227 | 0.879912 | |
| 026 | v4 blend, t=0.5725, leak-fixed | — | 0.876190 | 0.879555 | |
| 027 | diversity mix rate-matched to t=0.570 | — | 0.875747 | 0.880394 | |
| stack_meta_lgb | `day2_04_stacking_meta_models.py` | 0.87841 | not submitted | — | |
| stack_meta_lr | `day2_04_stacking_meta_models.py` | 0.87931 | not submitted | — | |

**Result:** sub_016 scored **0.880410** privately — 4th of 79 teams, 0.000075 behind
3rd. sub_024, the public leaderboard's own favorite, would have scored 0.879732 and
finished around 9th. Full narrative in `experiments/log.md` and `memory/insights.md`.

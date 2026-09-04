# Experiment Log

Every experiment gets a row. Never delete rows — mark them dropped.

| # | Date | Change | CV (mean ± std) | LB | Sub file | Verdict |
|---|------|--------|-----------------|-----|----------|---------|
| 0 | — | (example) LightGBM baseline, raw features | — | — | — | — |
| 1 | 2026-08-28 | LGBM baseline, 5-fold SKF, native cats, coded-numeric parsing, threshold 0.54 | wF1 0.87696 (folds macro@0.5: .769/.774/.765/.759/.752) | **0.874348** | sub_001 | keep — baseline established; metric = weighted F1 confirmed |
| 2 | 2026-08-28 | +FE (ordinal T/N/M/grade/stage, age midpoint, mets count, treatment flags, nodes ratio) + LGBM/XGB/CatBoost equal-ish blend, thr 0.59 | wF1 0.87768 (lgb .87704 / xgb .87672 / cat .87721) | **0.876023 → 3rd place** | sub_002 | keep — CV delta +0.0007 gave LB +0.0017 |
| 3 | 2026-08-28 | Optuna-tuned LGBM + XGB + Cat, 10-fold × 3-seed, thr 0.56 | wF1 0.87964 (lgb .87773 / xgb .87885 / cat .87871) | 0.875246 | sub_003 | **CV↔LB diverged** (−0.0044 gap vs −0.0017 for sub_002). Suspect thr 0.56 too aggressive or Optuna CV-overfit. Probe threshold next. |
| 4 | 2026-08-28 | Probe: v3 blend probs, thr 0.59 (isolate threshold vs model effect) | wF1 0.87862 | 0.875493 | sub_004 | thr 0.59 > 0.56 on LB (+0.0002), but v3 probs @0.59 still < v2 @0.59 (−0.0005) → **Optuna params overfit CV**. Prefer default params. |
| 5 | 2026-08-28 | Pseudo-labeling (v3 blend conf>0.99/<0.02) + tuned blend, 10f×2seed, thr 0.573 | wF1 0.87942 (global 0.87965) | **0.876190 — NEW BEST, 3rd** | sub_005 | keep — pseudo-labeling works on LB even though CV said flat |
| 6 | 2026-08-28 | Default-params LGB+XGB+Cat, 10f×3seed, thr 0.593 | wF1 0.87818 (global 0.87913) | 0.875322 | sub_006 | public LB gaps 0.8752–0.8756 across variants ≈ split noise; don't over-read |
| 7 | 2026-08-28 | Mega-blends across generations (OOF eval only) | best v5+v4 0.87935 < v4 alone 0.87965 | — | — | components too correlated; simple averaging across gens doesn't help |
| 8 | 2026-08-29 | Round-2 pseudo narrow (sub_007) | 0.87907 global / 0.87847 robust | 0.876123 | sub_007 | ≈ sub_005; not additive |
| 9 | 2026-08-29 | Round-2 pseudo wide 0.05/0.97 (sub_008 file) | 0.87913 global / 0.87833 robust | not submitted | sub_008 (file only) | below v4 CV; skip LB slot |
| 10 | 2026-08-29 | Diversity models HistGB/LogReg in blend (OOF eval) | all ≤ 0.87956 (v4 alone) | — | — | dropped — no blend lift |
| 11 | 2026-08-29 | v7 features (hist_type cat, interactions, freq enc) + pseudo recipe | 0.87899 global / 0.87882 robust | not submitted | sub_009 (file only) | below v4; features saturated |
| 12 | 2026-08-29 | Threshold probes on v4 blend: t=0.555 / t=0.592 | 0.87832 / 0.87913 | 0.876015 / 0.876162 | sub_010, sub_011 | public LB threshold curve peaks at sub_005's t=0.573 → threshold already optimal |
| 13 | 2026-08-29 | v8: v4 recipe, 5 seeds, lr 0.03 | 0.87956 global / 0.87824 robust | not submitted alone | sub_012 (file only) | CV ≡ v4; variance reduction only |
| 14 | 2026-08-29 | v4+v8 combined (7-seed equiv), t=0.570 | 0.87949 | 0.875770 | sub_013 | public LB still prefers sub_005; plateau confirmed |
| 15 | 2026-08-29 | Stacking meta-LGB / meta-LR over 11 components | 0.87841 / 0.87931 | — | files only | flat; components too correlated |
| 16 | 2026-08-29 | Train↔test exact-dup leak check | 283 matches; sub_005 already right on 282 | — | — | negligible |
| 17 | 2026-08-29 | RF/ET/MLP diversity wave: v4 + 5% each, t=0.575 | 0.87996 (honest 2-fold: .87934 vs v4 .87876 → real +0.0006) | 0.875506 | sub_014 | **best honest CV** → aggressive final pick candidate |
| 18 | 2026-08-29 | TabPFN (MPS, 8k ctx OOF, 4×10k test fits) | solo 0.87644; corr 0.98 w/ v4; blend lift ZERO (honest identical) | — | — | signal fully saturated — even transformer ICL sees the same |
| 19 | 2026-08-30 | Leak-fixed finals + micro threshold probes (all leakfixed) | — | 015: 0.876190, 016: 0.875506, 017 (t=0.565): **0.876227 NEW BEST**, 018 (t=0.582): 0.875889 | sub_015–018 | public peak ≈ t 0.565; curve noise ±0.0003 |
| 20 | 2026-08-30 | Class-weight sweep (spw 0.4–1.25) + year-weight sweep, LGBM 10f | spw0.6: 0.87732 (+0.0009 vs same-config 0.87646); yw1.0: +0.0007 | — | — | single-model lift real |
| 21 | 2026-08-30 | v9 = pseudo + spw0.6 + yw1.0 full blend | 0.87917 global / 0.87871 robust | not submitted | sub_020 (file) | gains didn't survive blending; drop |
| 22 | 2026-08-30 | Full-test soft distillation LGB+XGB | 0.88075 (CONTAMINATED — soft labels leak validation info) | 0.875330 | sub_019 | LB flat → CV gain was distillation leakage; drop |
| 23 | 2026-08-30 | Focal loss γ=1 single LGBM (clean same-config) | 0.87875 vs 0.87646 (+0.0023!) | — | — | biggest clean single-model lift; γ=2 flat; TE +0.0018 but fold-leaky eval |
| 24 | 2026-08-30 | v10: focal γ=1 LGB+XGB+Cat blend, 10f×2s | lgb solo 0.87982 (best single ever); blend 0.87973 | 0.875972 | sub_021 | focal gain overlaps blend gain; public still prefers sub_017 |
| 25 | 2026-08-30 | v11: focal γ=1 + pseudo narrow, 3 seeds; +v4 blend | **0.88005 global** (first >0.880) / 0.87929 robust | queued for 00:00 UTC Aug 31 | sub_022 | best global OOF ever; robust threshold less stable |
| 26 | 2026-08-31 | Fine threshold scan on v4 blend (leakfix): .560/.570/.5675/.5725 | — | .87586/.87631/.87623/.87619 | sub_023–026 | **peak t=0.570 → 0.876311 NEW BEST** |
| 27 | 2026-08-31 | Diversity mix rate-matched to 0.570 peak | — | 0.875747 | sub_027 | public prefers pure v4 ranking; sub_022 focal+pseudo public 0.875646 |
| 28 | 2026-09-01 | **FINAL RESULT: 4th/50 private, 0.880410 (sub_016 honest-CV hedge scored; +0.0007 over public-best pick, ≈ measured honest lift). CV discipline decided the placement.** | | | | |

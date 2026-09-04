# Insights

High-value discoveries about THIS competition. Update immediately when found — insights outlive code.

## Data insights
- **Metric = WEIGHTED F1** (confirmed sub_001: OOF wF1 0.87696 → LB 0.874348; macro/dead-F1 don't match LB range). Optimal threshold ≈ 0.54 on OOF.
- Lung cancer SEER data (all primary_site C34.x). Target: 83% Dead / 17% Alive.
- **year_of_diagnosis is the dominant covariate**: death rate 94% (2013) → 44% (2023) — follow-up-time effect, not treatment. Train/test year dists match → random split, StratifiedKFold(5) valid.
- Tumor size lives in 3 era-specific coded columns (999/Blank = unknown); parsed numeric + missing flags + coalesced `tumor_size_best`.
- rx_summ_scope_reglnsur2003 is 66.5% null; mets columns ~0.3% null; test has similar null pattern.
- No duplicate rows/IDs.

## CV ↔ LB
- sub_001: OOF 0.87696 → LB 0.874348 (gap −0.0026, direction consistent). Trust OOF deltas.
- LB is packed: #1 0.8768, #17 0.8744 → gains of +0.001 move several ranks.
- **Public LB deltas < ~0.0008 are mostly split noise** (subs 002–006 span 0.8752–0.8762 with 99% identical labels while OOF ordering disagrees). Use public LB only for big signals; select finals by CV + one LB-lucky pick.
- Optuna-tuned LGBM params overfit CV (OOF +0.002, LB −0.0005 vs defaults). Pseudo-labeling: CV flat but LB best (0.876190, sub_005).
- Current best: sub_005 pseudo-label blend (LB 0.876190, 3rd). Gap to #1 0.00064, to #2 0.00034.

- **Per-year thresholds OVERFIT** (tested on v2 OOF): in-sample 0.8805 but honest 2-fold eval 0.87548 < global-threshold 0.87722. Use ONE global tuned threshold.
- Errors concentrate in recent diagnosis years (2023: 23% err, 2013: 6%) — hardest rows are recent patients where survival is genuinely uncertain.
- Optuna LGBM (40 trials): strong regularization wins — num_leaves 27, min_data 20, feature_fraction 0.40, bagging 0.60, λ1 7.36, λ2 0.078, min_gain 0.19 → fold-mean wF1 0.87911 (params in data/processed/lgb_tuned.json).

## Ideas backlog
- [ ] Ordinal-encode staged codes: T/N/M recodes, grade, summary_stage, age midpoint from age_recode bands
- [ ] Count of metastasis sites (bone+brain+liver+lung), any-surgery flag, any-radiation flag
- [ ] CatBoost + XGBoost for diversity → rank-average / weighted blend on OOF
- [ ] Seed averaging (3 seeds) + 10-fold CV for final
- [ ] Per-year threshold or year-interaction features (death rate varies hugely by year)
- [ ] Optuna tune LGBM (num_leaves, min_data_in_leaf, feature/bagging fraction, lambdas)
- [ ] Pseudo-labeling: test is 1.5× train, confident test preds could augment training
- [ ] Threshold robustness: pick threshold maximizing wF1 averaged across folds, not global OOF
- Diversity models (HistGB 0.87502, LogReg 0.87326) do NOT improve the v4 GBM blend (all combos ≤ 0.87956 vs v4 alone). GBM trio already spans the signal. Dropped.
- Round-2 pseudo-labeling (narrow) ≈ equal to round-1 on LB (0.876123 vs 0.876190); not additive.
- histologic_type_icdo3 was fed as NUMERIC but is a categorical ICD-O-3 code — fix in v7.
- Public LB threshold curve on v4 blend: t 0.555→0.876015, 0.573→0.876190 (peak), 0.592→0.876162. CV-chosen threshold is LB-optimal; no headroom there.
- v7 features (hist_type categorical, interactions, freq enc) flat; v4↔v7 OOF corr 0.9965 → GBM signal saturated on this data.
- ENDGAME: sub_005 = safe+best pick. v8 (5-seed, lr 0.03 consolidation) is the last-variance-reduction candidate.
- FINAL STATE (Aug 29): sub_005 best public 0.876190 (3rd). All top candidates CV-equal (~0.8795±0.0002); public deltas noise. RECOMMENDED FINAL PICKS: sub_005 (safe, public-best) + sub_013 (most-averaged, stability hedge). Final selection must be done in Kaggle UI before deadline Sep 1 05:59 UTC.
- RF+ET+MLP at 5% each on v4 = first real CV lift (OOF 0.87996, honest +0.0006). Public LB didn't reward it (0.875506) but public deltas are noise. UPDATED FINAL PICKS: sub_005 (public best) + sub_014 (honest-CV best).
- Leaders (0.8772+) edge unexplained: stacking/leak/threshold/diversity all ruled out locally. External SEER lookup remains prime suspect.
- TabPFN: solo OOF 0.87644, corr 0.98 with v4 blend, zero blend lift. CONCLUSION: ~0.880 OOF is this dataset's ceiling for legitimate modeling. Leaders above 0.8772 public are either using external SEER lookup or overfitting the public split via probing — if the latter, they collapse on private and our CV-first finals win.

## FINAL (2026-08-31)
- Finals SELECTED by user: sub_024 (v4 blend t=0.570 leakfix, public 0.876311) + sub_016 (diversity blend leakfix, honest CV 0.87996).
- Ended ~20th/50 public with 0.876311; leaders 0.8774. Private reveal pending after Sep 1 05:59 UTC.
- 27 experiments logged. Key lessons: metric detective work via OOF threshold sweeps; Optuna/distillation CV inflation traps caught by LB probes; focal loss best single-model trick (+0.0023); public deltas <0.0008 were noise; pseudo-labeling & diversity blending real but small.

## RESULT (2026-09-01): 4th PLACE 🏆
- Private: 0.880410 (sub_016, the honest-CV diversity hedge) vs 0.879732 (sub_024, public-best). The hedge scored — manual final selection was worth ~5 places. 0.000075 (~3 rows) from 3rd.
- OOF estimate 0.87996 vs private 0.880410 — CV accurate to 0.0005. Public climbers (Sh54 0.8801 public) collapsed on private. Every strategic call validated.
- 2026-09-01: Reproducibility notebook submitted on time (verified 100% reproduction of sub_014 pipeline; sub_016 confirmed as the selected+scoring final, 0.880410). Next: finalists (top 5) announced Sep 3; Grand Finale presentations Sep 5 (40% of total score). Deck: presentations/NeurAps_Insight2_Finale_v2.pptx (28 slides). Study material: NeurAps Playbook artifact.
- CORRECTION from full leaderboard CSV: 79 teams total (not ~50). Final ranks: public 30th → private 4th. Team = mdajmainistiakapon + jahanesrat. Deck updated.
- 2026-09-04 (finale prep): deck is FINAL — 24-slide beamer (`presentations/neuraps_finale.tex` → `.pdf`), and `NeurAps_Insight2_Finale_FINAL.pptx` is the same 24 pages as images. Verified rendered pages: no overflow, tables legible. Added timed speaker notes (10:13 total, cumulative clock per slide, 7-min cut list, two-speaker hand-off at slide 12) in `NeurAps_Insight2_Finale_FINAL_notes.pptx` + printable `speaker_script.md`. Playbook artifact corrected to "4th of 79 teams (30th public)". Finale is Sep 5; Q&A prep = Playbook ch. 13.
- 2026-09-04: Built "NeurAps Dossier" study artifact (https://claude.ai/code/artifact/e204a11f-7e10-4428-8cb3-b78ff03788d4): day-by-day story, exact pipeline params, all 23 submissions with public+private scores (from Kaggle API), intuition entries, 31-question self-test drill. Private-score post-mortem: sub_003 (Optuna trio) private 0.880395 ≈ winner → "Optuna hurt" verdict was public noise; private threshold optimum for v4 blend was 0.560 (0.880251) not 0.570; distillation (0.878816) and focal (0.8792) confirmed weak; diversity-mix files were our 3 best private. Public↔private Spearman across our subs = 0.17. Final OOF mix @0.575: acc 0.884, F1 dead 0.931, F1 alive 0.630; @0.5 same accuracy, wF1 0.87518.
- 2026-09-04 (finale format known): 30 min = 10 presentation + 10 Q&A + 10 technical procedure. Built Part-3 deck `presentations/neuraps_technical.tex` (14 slides: workflow loop, ledger, feature code, validation protocol, exact model configs, blend/threshold, pseudo-label code, reproducibility checklist, traceability) → `NeurAps_Insight2_Technical_FINAL(.pptx / _notes.pptx)`, 9:39 script. Main deck retimed to 9:58. `speaker_script.md` now has run-of-show + Part 1 + Q&A plan + Part 3. Rebuild everything with scratchpad build_notes.py (needs both PDFs). CORRECTION propagated everywhere: final pipeline uses the Optuna-tuned LightGBM in BOTH stages (train_v4.py loads lgb_tuned.json; notebook hard-codes it) — deck slide 18, Playbook 6.2/11.1 and Dossier fixed; never say "defaults kept for stage 2".

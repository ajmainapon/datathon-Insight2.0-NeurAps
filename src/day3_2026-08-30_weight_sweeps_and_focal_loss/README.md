# Day 3 — 30 Aug 2026: weight sweeps, the two rejected traps, focal loss

1. `day3_01_class_and_year_weight_sweep.py` — scale_pos_weight and
   diagnosis-year sample-weight sweeps on a single LightGBM (row 20).
   Real single-model lift; did not survive blending (see day3_03).
2. `day3_02_focal_loss_and_target_encoding_probe.py` — focal loss (gamma 1, 2)
   and out-of-fold target encoding, same-config comparison (row 23).
   Focal loss gamma=1 was the best clean single-model gain of the project
   (+0.0023).
3. `day3_03_v9_weighted_pseudo_blend.py` — combines the day3_01 weighting
   with the pseudo-label recipe and blends all three models (row 21).
   The single-model gains did not survive blending — dropped.
4. `day3_04_v10_focal_loss_blend.py` — focal loss across the full trio
   (row 24, sub_021).
5. `day3_05_soft_label_distillation_REJECTED.py` — full-test soft-label
   distillation (row 22, sub_019). **Named REJECTED deliberately**: it raised
   out-of-fold score to 0.88075, our highest number of the week, but the
   teacher model had seen every fold's validation rows through the soft
   labels, so the gain was leakage, not signal. The public score was flat,
   confirming the diagnosis. Kept in the repo as a worked example of a false
   positive our verification process caught.
6. `day3_06_v11_focal_plus_pseudo.py` — focal loss + pseudo-labels + the day1_05
   blend combined (row 25, sub_022). Highest global OOF ever recorded
   (0.88005) but an unstable per-fold threshold — a warning sign, not adopted
   as the final pipeline.

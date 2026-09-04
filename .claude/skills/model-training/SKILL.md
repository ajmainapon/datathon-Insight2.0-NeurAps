---
name: model-training
description: Model selection, cross-validation design, hyperparameter tuning, and ensembling strategy. Use when training models or improving CV scores.
---

# Model Training & Validation

## Validation design (do this FIRST, before any model)

The CV scheme must mirror the train→test relationship:

| Situation | Scheme |
|---|---|
| i.i.d. rows, classification | StratifiedKFold (5 folds, fixed seed) |
| i.i.d. rows, regression | KFold, or stratify on binned target |
| Entities repeat across rows | GroupKFold on entity ID (if test has unseen entities) |
| Temporal data, test is future | TimeSeriesSplit or single time-based holdout matching the test horizon |
| Small data | Repeated k-fold, multiple seeds, report mean ± std |

Always report per-fold scores. High fold variance = unstable CV; fix that before trusting any experiment. Track CV↔LB correlation from your first submissions; if they disagree, the CV scheme is wrong — stop and redesign it.

## Model ladder (tabular)

1. Dummy baseline (mean/mode) — sanity-check metric implementation.
2. Ridge/Logistic on basic features — fast signal check.
3. **LightGBM with near-defaults** — the workhorse. `learning_rate=0.05`, early stopping on the fold's validation set, `num_leaves` 31–127.
4. XGBoost + CatBoost with the same features — diversity for later ensembling (CatBoost especially with many categoricals).
5. Neural net (MLP with embeddings / TabM) only if data is large or GBMs plateau.

For NLP: TF-IDF+linear baseline → fine-tuned transformer (deberta-v3 family). For CV: pretrained backbone (timm) + light augmentation, progressive resizing.

## Tuning

- Tune only after features stabilize. Use Optuna, 30–100 trials, on ONE fold or a subsample first, confirm the best config on full CV.
- LightGBM priority order: `num_leaves`, `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`, `lambda_l1/l2`. Lower `learning_rate` (0.01) + more rounds only for the final fit.
- Diminishing returns past ~1–2 hours of tuning; go back to features.

## Ensembling (endgame)

- Save **out-of-fold (OOF) predictions** for every model — they are the currency of ensembling.
- Start with weighted averaging of diverse models (optimize weights on OOF via scipy). Rank-average for AUC-like metrics.
- Stacking: level-2 Ridge/Logistic on OOF predictions; keep level 2 simple.
- Diversity beats strength: 3 different architectures > 5 seeds of one model. Different feature sets, different model families.
- Seed-average your best single model (3–5 seeds) — cheap, reliable gain.

## Final submission selection

Pick two: (1) best stable CV ensemble, (2) an aggressive/diverse variant. Never pick by public LB alone — the private split punishes it.

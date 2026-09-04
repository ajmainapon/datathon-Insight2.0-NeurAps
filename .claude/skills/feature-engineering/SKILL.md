---
name: feature-engineering
description: Feature engineering playbook for tabular, time-series, text, and categorical data. Use when improving model performance after a baseline exists.
---

# Feature Engineering

Every feature idea gets tested through the standard CV loop — add features in small batches, measure ΔCV, log in `experiments/log.md`, keep only what helps. Cache computed feature sets to `data/processed/*.parquet` keyed by a version name.

## Playbook by data type

**Numeric**
- Ratios, differences, and products of domain-related pairs (price/area, income/dependents).
- Binning + count encoding for non-monotonic relationships; log1p for skewed positives.
- Row-level aggregates: sum/mean/std/nan-count across related column groups.

**Categorical**
- Frequency encoding (cheap, safe), one-hot only for low cardinality.
- Target encoding with **out-of-fold computation + smoothing** — never fit on the full train (leakage). CatBoost handles this natively.
- Interactions: concatenate 2–3 categoricals into a combined key, then frequency/target encode.

**Group aggregates (usually the biggest win)**
- For each entity (user, store, product): mean/std/min/max/count of numerics per group; deviation of the row from its group mean (`x - group_mean_x`).

**Time series / temporal**
- Lags, rolling mean/std/min/max (windows at natural periods: 7, 28, 365), expanding means.
- Calendar: day-of-week, month, holidays, time-since-event, time-to-event.
- STRICT rule: all rolling/lag features must use only past data relative to the prediction time.

**Text**
- Start with TF-IDF (word + char n-grams) → linear/SVM; length, punctuation, casing stats as side features.
- Sentence-transformer embeddings + GBM/linear head when semantics matter.

**Dates/geo**
- Haversine distances to key points, lat/lon rounding + count encoding, clustering (KMeans) region IDs.

## Selection & hygiene

- Prune with permutation importance or null importances, not raw gain (gain over-ranks high-cardinality features).
- Drop features that top adversarial validation (train/test shift drivers) if CV/LB gap is large.
- Watch feature count vs. rows: aggressive feature explosion + small data = overfit CV. Prefer fewer, well-motivated features.

## Anti-patterns

- Fitting any encoder/imputer/scaler on train+test or full train before CV split.
- Keeping a feature because CV improved 0.0001 on one seed — require improvement consistent across folds/seeds.
- Engineering features the metric can't reward (e.g., calibrating probabilities when metric is rank-based AUC).

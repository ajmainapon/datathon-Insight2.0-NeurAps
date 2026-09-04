---
name: eda
description: Fast, systematic exploratory data analysis for a new competition dataset. Use when data is first loaded or when a new data file appears.
---

# Exploratory Data Analysis

Run EDA as a script (`src/eda.py` or a notebook), saving findings to `memory/eda.md`. Time-box: 30–45 minutes for the first pass.

## Checklist (in order)

1. **Basics:** shape, dtypes, memory usage, head/tail of every file. Column meanings vs. the data dictionary.
2. **Target:** distribution (histogram / value counts), skew, outliers, class imbalance. For regression, check if log-transform normalizes it.
3. **Missingness:** per-column null %, missingness patterns (MCAR vs. informative — does a null correlate with the target?). A "missing" indicator is often a feature.
4. **Duplicates:** exact duplicate rows, duplicate IDs, near-duplicates between train and test (leak opportunity or contamination).
5. **Train/test shift:** compare feature distributions train vs. test (KS test or simple overlaid histograms). Build a quick adversarial validation model (classify train-vs-test with LightGBM); AUC ≫ 0.5 means shift → CV design must account for it, and top adversarial features may need dropping.
6. **ID structure:** are IDs sequential/temporal? Does row order carry signal? Is test interleaved or after train in time?
7. **Cardinality:** unique counts for categoricals; flag high-cardinality columns for target/frequency encoding.
8. **Correlations:** feature–target correlations, feature–feature redundancy. For a single feature suspiciously predictive, investigate leakage.
9. **Groups/hierarchy:** entities appearing in multiple rows (users, stores, patients)? If groups span train and test decisions about GroupKFold follow.

## Output

Write `memory/eda.md` with: key facts, anomalies, leakage suspicions, CV scheme recommendation, and a ranked list of feature ideas. Plots go in `notebooks/` — but conclusions in the markdown, since plots are lost context next session.

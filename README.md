# Insight 2.0 — Cancer Survival Classification

**Team NeurAps** · private weighted F1 **0.880410** · **4th of 79** teams
(30th on the public board — see [Endgame](#endgame))

Predicting `vital_status` (Alive / Dead) for 36,000 lung-cancer patients from the
SEER cancer registry, trained on 24,000 labelled records with 36 clinical features.
Metric: **weighted F1** on a hidden test set split 40% public / 60% private.

> If you remember one thing about this repository: we won our rank with validation
> discipline, not an exotic model. Every component here is standard; what mattered
> was that every decision was gated by honest out-of-fold cross-validation, and the
> public leaderboard was used only as a noisy instrument, never as a target.

## Result in one sentence

Engineered clinical features → a blend of three gradient-boosted tree models
(LightGBM, XGBoost, CatBoost), trained twice — once plainly, once with
pseudo-labelled test rows — mixed with 5% each of RandomForest, ExtraTrees and a
small neural network, thresholded at 0.575. Our out-of-fold forecast was 0.87996,
made before the reveal; the private result was 0.880410 — accurate to 0.0004.

## Repository layout

```
src/
  common/features.py          canonical feature-engineering function, used by every script
  day1_.../                   28 Aug — baseline, metric identification, first blend
  day2_.../                   29 Aug — diversity models, stacking, TabPFN
  day3_.../                   30 Aug — weight sweeps, focal loss, two rejected traps
  day4_.../                   31 Aug — leak fix, threshold scan, final selection
  tools/                      submission checker, notebook/deck generators
notebooks/                    the reproducible end-to-end solution notebook
experiments/log.md            every experiment, kept or dropped, one row each
memory/                       competition brief + running insights log
submissions/MANIFEST.md       every submission's OOF/public/private score
presentations/                Grand Finale decks (main + technical) and speaker script
data/raw/, data/processed/    not tracked in git — see Setup below
```

Each `dayN_.../README.md` explains that day's scripts in the order they were run,
with a pointer to the exact `experiments/log.md` row and submission each one produced.

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Place the competition's `train.csv`, `test.csv` and `submission.csv` in `data/raw/`
(not included — see [Data](#data)). Kaggle credentials, if you want to submit, go in
a local `.env` (never committed):

```
KAGGLE_USERNAME=...
KAGGLE_API_TOKEN=...
```

Every script assumes it is run from the repository root, e.g.:

```bash
python src/day1_2026-08-28_baseline_metric_and_first_blend/day1_01_baseline_lgbm.py 001 baseline
```

Reproduce the full final pipeline (~1.5 h on an 8-core laptop, CPU only) with
`notebooks/neuraps_notebook.ipynb`. It re-derives the metric, builds features from
`common/features.py`, trains both GBM stages, adds the diversity mixture, and writes
a submission that reproduces `sub_016` (see `submissions/MANIFEST.md`).

## Data

SEER (Surveillance, Epidemiology, and End Results) is the US National Cancer
Institute's population cancer registry. Each row is one lung-cancer patient
(primary site C34.x), coded at diagnosis, with vital status at last follow-up as the
target. The raw files are competition-provided and are **not redistributed here**;
`data/raw/` and `data/processed/` are gitignored.

Key finding: death rate falls from 93.7% (2013 diagnoses) to 44.4% (2023) — not
medical progress at that scale, but right-censoring: a 2023 patient has been
followed for months, a 2013 patient for a decade. This is why `years_since_dx` is a
feature, and why a random stratified split is the honest cross-validation design
(train and test share the same year distribution).

## Method summary

- **Metric identification:** the brief only said "F1 Score." An all-Dead baseline
  would top the leaderboard if the metric were Dead-class F1; our macro F1 sat far
  below the observed leaderboard range. Weighted F1, swept over thresholds, matched
  it exactly.
- **Validation:** stratified 10-fold, seeds 42/2026/7, out-of-fold scoring, and
  **split-half verification** for every selected quantity (choose on half the OOF
  rows, score the frozen choice on the other half) — this is what caught two false
  positives (see `day3_.../README.md`) and certified one real gain.
- **Features:** 35 raw columns → 53, via ordinal staging scales, a coalesced
  tumour-size field across three coding eras, metastasis counts, treatment flags,
  node ratios and a censoring-proxy time feature. `src/common/features.py`.
- **Models:** LightGBM + XGBoost + CatBoost (Optuna-tuned LightGBM, used in both
  training stages), blended by Nelder–Mead on out-of-fold weighted F1; +5% each of
  RandomForest, ExtraTrees, and an MLP for decorrelated errors.
- **Pseudo-labelling:** 7,455 test rows the stage-1 blend was >99%/<2% confident
  about, added to the training side of every fold only — evaluation never touches
  them, so the out-of-fold estimate stays honest.
- **Threshold:** swept on out-of-fold predictions; 0.575, worth about +0.005
  weighted F1 over the default 0.5 (accuracy is identical at both cuts — only the
  metric moves).

## Endgame

Two submissions are scored at the end: one chosen by public leaderboard score
(sub_024), one by honest cross-validation (sub_016). They differed on 102 of 36,000
patients. The public board preferred sub_024; our own out-of-fold estimate preferred
sub_016 by 0.0006. The private reveal agreed with the out-of-fold estimate, not the
public board: sub_016 scored 0.880410 (4th); sub_024 scored 0.879732 (would have
finished ≈9th). Full account in `memory/insights.md`.

## What failed, and why

More than half of the 28 logged experiments were rejections, each with a named
cause: aggressive Optuna tuning (CV overfitting via multiple comparisons on the same
folds), full-test soft-label distillation (the teacher had seen every fold's
validation rows through its soft labels — a leak, not a gain), and per-year decision
thresholds (eleven small optimisations, eleven chances to fit noise). Each is a
script in `day3_.../`, deliberately kept and named so the failure is visible, not
just the successes.

# NeurAps Grand Finale · speaker script

Three 10-minute segments. Decks: `NeurAps_Insight2_Finale_FINAL_notes.pptx` (Part 1) and `NeurAps_Insight2_Technical_FINAL_notes.pptx` (Part 3); each carries these notes in Presenter View. `clock` = cumulative time at the END of that slide. Pace ≈ 2.5 words/sec.

## Run of show (30 min)

| Segment | Time | Material | Speaker suggestion |
|---|---|---|---|
| Part 1 · Presentation | 10:00 | 24-slide main deck | A: slides 1–11 · B: slides 12–24 |
| Part 2 · Q&A | 10:00 | Dossier drill + Playbook ch. 13 | whoever owns the topic answers; the other adds the number |
| Part 3 · Technical procedure | 10:00 | 14-slide technical deck | B: workflow + methodology (1–9) · A: reproducibility (10–14) |

## Part 1 · Presentation (target 10:00)

### Slide 1  ·  0:15  ·  clock 0:15

Good morning. We are Team NeurAps. Our entry finished 4th of 79 teams on the private leaderboard with a weighted F1 of 0.880410. The story we want to tell in ten minutes is how a cross-validation-first discipline got us there, and why it moved us from 30th on the public board to 4th on the private one.

### Slide 2  ·  0:15  ·  clock 0:30

We'll walk the eight required components in order: summary, EDA, feature selection, feature engineering, model selection, evaluation, comparison, and findings. One thread runs through all eight: every decision was gated by honest out-of-fold evidence.

### Slide 3  ·  0:03  ·  clock 0:33

Section one: summary.

### Slide 4  ·  0:45  ·  clock 1:18

The task: SEER registry records of lung-cancer patients; predict vital status, Alive or Dead. 24,000 training rows, 36,000 test rows, 36 clinical features, six submissions a day. Metric: weighted F1 on a hidden 40/60 public-private split. Our approach in one sentence: engineered clinical features, a blend of three gradient-boosted tree models trained twice, plain and then with pseudo-labelled test rows, mixed with five percent each of RandomForest, ExtraTrees and an MLP, thresholded at 0.575. Nothing exotic. What was unusual was the verification behind each of those choices.

### Slide 5  ·  0:03  ·  clock 1:21

Section two: exploratory data analysis.

### Slide 6  ·  0:30  ·  clock 1:51

Eighty-three percent of patients are Dead. That matters because weighted F1 weights each class by its frequency, but the Alive F1 is far lower, around 0.63 versus 0.93 for Dead, so the minority class is where the score moves. Two consequences: the decision threshold becomes a model parameter we tune, and stratified folds are mandatory.

### Slide 7  ·  0:45  ·  clock 2:36

The single most important EDA finding. Death rate by diagnosis year falls from 94 percent in 2013 to 44 percent in 2023. That is not medical progress at that scale; it is right-censoring. A 2013 patient has had a decade to die; a 2023 patient, months. Three things followed. Train and test share this year distribution, so the split is random and stratified K-fold is the honest CV scheme. Errors concentrate in recent years, 23 percent in 2023 versus 6 percent in 2013; those are the genuinely uncertain patients. And years-since-diagnosis becomes a first-class feature, the follow-up-window proxy.

### Slide 8  ·  0:30  ·  clock 3:06

SEER doesn't use NaN; it uses sentinel codes: 999, Blank, Unknown. Tumor size lives in three era-specific columns. We parsed the numerics, flagged missingness explicitly, and coalesced the eras instead of discarding either. We audited every high-power feature for leakage. Categories like 'death certificate only' are directly informative, but they exist in both splits as provided data, so they are legitimate signal, not leakage.

### Slide 9  ·  0:03  ·  clock 3:09

Section three: features. [Suggested hand-off point if two speakers: speaker A has covered problem and data; speaker B takes features, models and results. Alternatively hand off at slide 12.]

### Slide 10  ·  0:30  ·  clock 3:39

We kept all 36 provided features plus about 15 engineered ones, and that retention was a tested decision. Gradient-boosted trees select implicitly: uninformative features never get chosen for splits. Every time we tried dropping low-importance features, CV got worse. The importance ranking is clinically coherent: surgery-decision codes, stage, year, age, tumor size, metastasis. Train and test distributions align, so nothing was sacrificed for stability.

### Slide 11  ·  0:35  ·  clock 4:14

Each engineered feature has a one-sentence clinical story. T, N and M staging mapped to ordered scales, because staging is ordinal. A composite TNM score with metastasis double-weighted. The coalesced tumor size. A count of metastatic sites, a graded severity signal. The positive-to-examined node ratio, which normalizes out surgical thoroughness. Treatment flags, which encode physician prognosis. And years since diagnosis. What we deliberately avoided: target encoding outside folds, anything fitted on test, and AutoML.

### Slide 12  ·  0:03  ·  clock 4:17

Section four: model selection and evaluation.

### Slide 13  ·  0:45  ·  clock 5:02

Why three gradient-boosted models? LightGBM grows leaf-wise with native categoricals. XGBoost grows depth-wise with second-order gradients. CatBoost uses ordered target statistics, built for our 114-level histology codes. They correlate 0.97, but the three percent where they disagree sits exactly on the uncertain rows where F1 is decided, and the blend beats every member. Then a diversity mixture: RandomForest, ExtraTrees and an MLP at five percent each, weaker alone but with decorrelated errors, verified at plus 0.0006 by split-half testing. Why not deep learning? We measured rather than assumed: the MLP scored 0.874 solo and TabPFN correlated 0.98 with our blend. At 24,000 rows, trees win.

### Slide 14  ·  0:40  ·  clock 5:42

The full pipeline. Engineered features feed a stage-one trio, ten folds by three seeds. Its most confident test predictions, above 0.99 or below 0.02, become 7,455 pseudo-labelled rows. A stage-two trio retrains on train plus pseudo-rows. We add the diversity models at five percent each and threshold at 0.575, chosen on out-of-fold predictions. The evaluation protocol was identical for every candidate: stratified ten-fold OOF weighted F1 with a tuned threshold, multiple seeds, split-half verification for every selected quantity, and the public leaderboard used only for controlled single-variable probes.

### Slide 15  ·  0:35  ·  clock 6:17

Pseudo-labelling was our biggest single win, and the detail that makes it honest is the CV-integrity rule: pseudo-rows join only the training side of each fold; evaluation stays on original labelled rows, so the estimate stays unbiased. The test set is 1.5 times our training data, so 36,000 in-distribution rows were too valuable to ignore. Only extreme confidences cross over, with expected label noise of one to two percent. We tested wider bands and a second round; neither helped, and both rejections are logged.

### Slide 16  ·  0:03  ·  clock 6:20

Section five: model comparison and experiments.

### Slide 17  ·  0:30  ·  clock 6:50

Every model we tried, head to head, on the same protocol. The ladder runs from ExtraTrees at 0.8727, through logistic regression, the MLP, RandomForest, HistGradientBoosting and TabPFN, a default LightGBM at 0.8765, up to the seed-averaged CatBoost and XGBoost near 0.8788. Focal-loss LightGBM was our best single model at 0.8798, but its gain overlapped what blending already recovered. The GBM trio with pseudo-labels reached 0.87965, and the final mixture 0.87996, which scored 0.880410 on private.

### Slide 18  ·  0:45  ·  clock 7:35

More than half of our logged experiments were rejections, and each has a named failure mode. Aggressive Optuna tuning: plus 0.002 OOF, minus 0.0005 on the leaderboard. Forty trials on the same folds is CV overfitting by multiple comparisons, proven with a controlled probe, so we stopped tuning there. Soft-label distillation: a plus 0.001 CV gain that was the teacher leaking validation information through soft labels; leaderboard flat. Per-year thresholds: better in-sample, worse under split-half verification, eleven chances to fit noise. And more model families: everything correlated 0.97 or above with the blend. The dataset's signal saturates near 0.880, a measured ceiling.

### Slide 19  ·  0:45  ·  clock 8:20

This is the decision that made 4th place. Comparing our own submissions, public deltas under 0.0008 were split noise: label agreement above 99 percent, with an ordering that contradicted OOF. So we selected one final by public score and one by honest CV. Grey bars are the public split, navy bars the private split. The CV pick was punished by the public board at 0.8755, and scored 0.880410 on private. Auto-selection would have finished around ninth. One informed click, worth five places.

### Slide 20  ·  0:03  ·  clock 8:23

Section six: best findings.

### Slide 21  ·  0:30  ·  clock 8:53

Our OOF forecast before the reveal was 0.87996. The private result was 0.880410; forecast error 0.0004. We went from 30th on the public board to 4th on private, 0.000075 behind third, about three patients out of 36,000. Several teams above us on the public board fell out of the private top ten, the signature of public-split overfitting.

### Slide 22  ·  0:35  ·  clock 9:28

Four principles decided this result. Identify the exact metric before modelling: one OOF sweep revealed weighted F1 and made the threshold a parameter. Trust out-of-fold CV and treat the leaderboard as an instrument; never chase it. Verify every gain adversarially: split-half testing killed three seductive improvements. And log everything: a complete experiment ledger made the endgame a calculation, not a gamble.

### Slide 23  ·  0:20  ·  clock 9:48

Limitations, honestly. Censoring is implicit, not modelled; a survival-analysis formulation is the statistically superior next step. Per-era calibration could sharpen the global threshold. SHAP interpretation of the ensemble remains future work. Everything is seeded; the submitted notebook re-runs end to end in about ninety minutes and regenerates our predictions exactly. No external data.

### Slide 24  ·  0:10  ·  clock 9:58

Thank you. Every claim is reproducible from our notebook and experiment ledger. We welcome questions.

## Part 2 · Q&A (10 min) · how to run it
- Number first, reason second. Point at the ledger row or submission that decided the thing.
- Keep the Dossier drill open on a phone: https://claude.ai/code/artifact/e204a11f-7e10-4428-8cb3-b78ff03788d4 (Section 8) and the Playbook question bank (chapter 13).
- If you do not know: "We did not test that within the four days. My expectation is X because Y, and it is the first thing I would run." Never invent a number.
- Likely openers: why weighted F1 and why 0.575; why 30th public and 4th private; is pseudo-labelling circular; what leaked in distillation; what would you do with more time; would you deploy this.
- If a judge asks about tuning: the Optuna-tuned LightGBM parameters stayed in the final pipeline (both stages); our probe showed no leaderboard gain from tuning, so we stopped searching. Do not say "we reverted to defaults".

## Part 3 · Technical procedure (target 10:00)

### Slide 1  ·  0:15  ·  clock 0:15

Part three: the technical procedure. Everything on these slides is a file in our repository or a cell in the submitted notebook, and we're happy to open any of them.

### Slide 2  ·  0:03  ·  clock 0:18

Section one: workflow.

### Slide 3  ·  1:00  ·  clock 1:18

This is the loop we ran for every idea, for four days. An idea becomes a script that follows one contract: it reads the raw CSVs, which we never modified; it builds features through one shared function; it runs ten-fold stratified cross-validation and saves the out-of-fold probabilities to disk, so every model we ever trained is still available for blending. It prints per-model OOF, blend weights and the tuned threshold, and that result goes into the ledger before the next idea starts. Only if the honest CV improved does a versioned submission file get written, named with its description and CV score, and only after the format checker passes does it go to Kaggle, at most six a day, and only as single-variable probes.

### Slide 4  ·  1:00  ·  clock 2:18

The ledger is one row per experiment, kept or dropped. Here are three real rows. Row two: engineered features plus the three-model blend, CV up seven ten-thousandths, leaderboard up seventeen. Row three: the Optuna-tuned trio, where CV and leaderboard diverged, and the verdict says what we did next: probe the threshold. Row seventeen: the diversity mixture, with the split-half numbers that made it the final pick. The technical point is that the endgame decision was made by sorting this table on the honest-CV column, and the public score sits in the same row so every CV-to-leaderboard gap is visible per experiment.

### Slide 5  ·  0:03  ·  clock 2:21

Section two: methodology.

### Slide 6  ·  1:00  ·  clock 3:21

Feature construction, as code. Sentinel decoding: coerce to numeric, and treat anything at or above 990 as missing, because 999 and 998 are registry codes for unknown. Tumor size is coalesced across the three era columns in a fixed order, with a flag when all three are missing. The ordinal maps are explicit dictionaries; you can see the T values, with T1b at one and a half and T2 at two and three quarters where the code is ambiguous. The composite TNM score doubles metastasis. Counts and flags come from string matches on the treatment columns. Categoricals are pandas categories with levels unioned over train and test, so every library sees identical codes, and we keep three encodings because LightGBM, XGBoost and CatBoost each want a different one. Thirty-five raw predictors become fifty-three columns, and nothing is fitted on the target outside a fold or on test.

### Slide 7  ·  1:10  ·  clock 4:31

The validation protocol as implemented. Stratified ten-fold with shuffling, seeds 42, 2026 and 7. In each fold we train on nine parts, early-stop on the tenth, and predict the tenth, so every training patient gets exactly one out-of-fold probability per seed; we average across seeds. Test probabilities are the average over all fold models and seeds; we never refit on the full set, because the fold models are early-stopped and averaging them is a free ensemble. The threshold is swept from 0.35 to 0.75 in steps of 0.005 on pooled OOF, and we report both the global argmax and the median of per-fold optima; when those disagree, the score is fragile. Split-half verification is four lines: random halves A and B, choose on A, score the frozen choice on B, compare with the incumbent on B. It certified the diversity mixture and rejected per-year thresholds and stacking.

### Slide 8  ·  1:05  ·  clock 5:36

The model configurations, exactly. LightGBM uses the Optuna parameters in both stages: 27 leaves, 20 rows per leaf minimum, feature fraction 0.40, bagging 0.60, L1 7.36, up to three thousand rounds with early stopping at a hundred. XGBoost: depth 7, subsample and column sample 0.8, min child weight 10, on integer codes. CatBoost: depth 7, L2 leaf regularisation 5, on string categories with an explicit NA level. RandomForest and ExtraTrees: 800 trees, at least five rows per leaf, square-root features. The MLP: 128 then 64 ReLU units on one-hot categoricals and standardised numerics, early stopping. The Optuna study was 40 trials of TPE on five folds, objective the mean per-fold best-threshold weighted F1. XGBoost and CatBoost were never tuned; the controlled probe showed no leaderboard gain from tuning, so no more budget went there.

### Slide 9  ·  0:50  ·  clock 6:26

Blending and the decision. The blend weights maximise best-threshold weighted F1 of the weighted average of OOF probabilities; because thresholded F1 is a step function, we use Nelder-Mead, a derivative-free simplex search, on absolute normalised weights. They came out near a third each. The final probability is 0.85 times the GBM blend plus 0.05 times each of RandomForest, ExtraTrees and the MLP. The threshold sweep gave 0.575, with per-fold optima between 0.570 and 0.575. Label Dead when p exceeds 0.575; predicted Dead rate 85.8 percent. One detail judges like: accuracy is 88.4 percent at both 0.5 and 0.575; only the weighted F1 moves, from 0.875 to 0.880. And the duplicate correction: 283 test rows are exact copies of train rows and take the train label, which changed one prediction.

### Slide 10  ·  0:55  ·  clock 7:21

Pseudo-labelling inside the fold, as code. The mask selects test rows with stage-one probability above 0.99 or below 0.02; 7,455 rows, 99.3 percent Dead. Inside each fold, the augmented training set is the fold's training rows plus those pseudo-rows, and the model early-stops on the fold's real validation rows. The validation fold and the OOF array never contain a pseudo-row, so the estimate stays unbiased. Stage two is the same trio, ten folds by two seeds, retrained on the augmented folds. We tested wider bands, 0.05 and 0.97, which doubled the pool to 14,566 rows, and a second round from stage-two predictions. Neither improved honest CV, and both are in the ledger.

### Slide 11  ·  0:03  ·  clock 7:24

Section three: reproducibility.

### Slide 12  ·  1:10  ·  clock 8:34

The reproducibility checklist. Determinism: fold seeds 42, 2026 and 7; every model seeded with its fold seed; the tuned parameters are hard-coded so no live Optuna run is needed; the same feature function is used in every script and in the notebook. Environment: Python 3.12, LightGBM 4.7, XGBoost 3.4, CatBoost 1.2, scikit-learn 1.9, Optuna 4.9, pandas 3.0, NumPy 2.5, on one Apple M2 laptop with eight cores and eight gigabytes, CPU only. Verification: the seventeen-cell notebook was re-run in a clean kernel before submission and reproduced sub_014 on all 36,000 rows; the selected sub_016 is that file plus the one documented duplicate correction. Runtime about ninety minutes end to end. The submission checker asserts identical columns, row count and id order, no missing values, and only labels present in the sample. No external data, no manual labels, no AutoML.

### Slide 13  ·  0:50  ·  clock 9:24

Traceability. For each headline number in part one, this table says where a reviewer can check it: the OOF score is a printed line of the final ensemble cell and ledger row seventeen, with the probability arrays on disk; the private and public scores are Kaggle entries for a named file; the year effect is a one-line groupby in the EDA cell; the pseudo-label count is printed by the stage-two cell; the split-half result is ledger row seventeen; the threshold is the final cell; the tuned parameters are a JSON file produced by the tuning script; the duplicate count is a merge of test against train. In one sentence: every number on a slide is either a printed line of a seeded script or a leaderboard entry, and the file that produced it is named.

### Slide 14  ·  0:15  ·  clock 9:39

Thank you. The repository, the notebook and the ledger are open; pick any file and we'll walk through it.

## Seven-minute cut for Part 1 (only if the limit turns out to be shorter than 10 min)
- Slide 2 (outline): one sentence. Slide 10 (feature selection): skip; open slide 11 with "We kept all 36 features; dropping any made CV worse."
- Slide 4: the four stats plus the one-sentence approach. Slide 8: one sentence on sentinel codes, one on the leakage audit.
- Slide 15: keep only the CV-integrity rule and the confidence bands. Slide 17: point at the bold final row and the focal-loss row.
- Slide 22: read the four bold headings. Slide 23: one sentence on survival analysis, one on reproducibility.
- Never cut slides 13 and 18; they carry the model-selection and model-comparison criteria.

---
name: submission-check
description: Pre-submission validation checklist. ALWAYS run before producing or delivering any submission file.
---

# Submission Check

Run these checks programmatically against the competition's sample submission file before every submission. A failed submission wastes a daily attempt.

## Hard checks (script them once in `src/check_submission.py`)

1. **Columns:** exact names, exact order as sample submission.
2. **Row count:** matches sample submission exactly.
3. **IDs:** same set AND same order as sample submission (some scorers are order-sensitive).
4. **No NaN/inf** in prediction columns.
5. **Dtype/range sanity:** probabilities in [0,1]; classification labels from the allowed label set; regression predictions within a plausible range (compare to train target min/max — flag anything wildly outside).
6. **File format:** correct delimiter, header present, no index column accidentally written (`index=False`), correct file extension/compression if specified.

## Soft checks

- Prediction distribution vs. train target distribution — a huge mismatch (e.g., mean prediction 0.02 when train positive rate is 0.3) usually means a pipeline bug.
- Spot-check 5 rows manually against raw features: do predictions make directional sense?
- Diff against the previous best submission: correlation and mean absolute difference. ~1.0 correlation means the change did nothing; very low correlation on a small change means a bug.

## Bookkeeping

- Save as `submissions/sub_XXX_<short-desc>_cv<score>.csv` (XXX = incrementing number).
- Log in `experiments/log.md`: submission number, generating script/commit, CV score, then the LB score once known.
- Update the CV↔LB correlation note in `memory/insights.md`.

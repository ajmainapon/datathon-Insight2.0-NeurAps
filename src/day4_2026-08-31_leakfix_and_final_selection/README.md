# Day 4 — 31 Aug 2026: leak fix, threshold scan, final selection

No new model was trained this day. The day-1 pseudo-label pipeline
(`day1_05_pseudo_label_blend.py`) had already produced the components we
needed; this day's activity was correcting a small labelling bug and doing
a fine-grained decision-threshold search on the finished probabilities.

`day4_01_leakfix_and_threshold_scan.py` reconstructs that day's two actions,
reading only the already-saved blend probabilities (no retraining):

1. **Duplicate-row fix.** 283 test patients are exact copies of a training
   patient on every one of the 35 shared feature columns. Their prediction
   is overridden with the known training label. This is the fix applied to
   produce sub_015 (from sub_005) and **sub_016** (from sub_014) —
   row 16 and rows 19+ in `experiments/log.md`.
2. **Fine threshold scan.** The finished v4 blend probabilities are
   re-thresholded at 0.560 / 0.565 / 0.5675 / 0.570 / 0.5725 / 0.582 to
   produce sub_017, sub_023–026, sub_018 (row 26).

**The final step of the day was not code**: two submissions were selected
in the Kaggle UI as the competition's two scored finals —
`sub_016` (our honest out-of-fold pick) and `sub_024` (the public
leaderboard's preferred pick). See `submissions/MANIFEST.md` for every
file's public and private score, and `memory/insights.md` for why the
honest-CV pick was chosen for one of the two slots.

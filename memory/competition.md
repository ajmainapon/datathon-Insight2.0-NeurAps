# Competition Brief — Insight 2.0

- **Name / host:** Insight 2.0 (private Kaggle Community competition, id 162355) — https://www.kaggle.com/competitions/insight-2-0
- **Task type:** Binary classification — "Leveraging Predictive Models for Cancer Survival Classification"
- **Metric (exact):** F1 Score (sklearn f1_score — per competition tag; assume binary/positive-class F1 unless data says otherwise. THRESHOLD TUNING MATTERS.)
- **Data files:** train.csv (11.3 MB), test.csv (16.8 MB — larger than train!), submission.csv (450 KB sample)
- **Target column:** TBD from data (survival-related)
- **Test set structure:** TBD — test bigger than train; check adversarial validation
- **Submission format:** per submission.csv sample
- **Submission limit per day:** 6
- **Deadline:** 2026-09-01 05:59:59 UTC (= Sep 1, ~11:59 AM Dhaka time). Team merger deadline same.
- **Team size:** max 3
- **Leaderboard (2026-08-28):** 42 teams; #1 = 0.876829 (Attension_Seeker); top 15 packed within 0.8768–0.8746 — tiny margins, threshold optimization and ensembling will decide ranks.
- **Chosen CV scheme + why:** TBD — default StratifiedKFold(5) unless EDA shows groups/time.

## Notes
- Competition page is private (404 unauthenticated); all access must go through authenticated Kaggle API (`.env` token).
- Enabled 2026-08-26; user has entered; current user rank: not on visible top of leaderboard yet.
- F1 metric: predicted labels (not probabilities) likely required — optimize decision threshold on OOF predictions.

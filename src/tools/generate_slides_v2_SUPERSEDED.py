"""NeurAps Grand Finale deck v2 — academic style, ~26 slides, 16:9."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BURG = RGBColor(0x8C, 0x2F, 0x2F)
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x55, 0x5F, 0x66)
RULE = RGBColor(0xD6, 0xDB, 0xE0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)

HEAD = "Georgia"
BODY = "Cambria"
MONO = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
PAGE = [0]


def base(section=""):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    PAGE[0] += 1
    # footer rule + text
    ln = s.shapes.add_shape(1, Inches(0.9), Inches(7.02), Inches(11.53), Pt(1.2))
    ln.fill.solid(); ln.fill.fore_color.rgb = RULE; ln.line.fill.background()
    tf = s.shapes.add_textbox(Inches(0.9), Inches(7.06), Inches(9), Inches(0.35)).text_frame
    r = tf.paragraphs[0].add_run()
    r.text = "Insight 2.0 — Datathon 2026   ·   Team NeurAps" + ("   ·   " + section if section else "")
    f = r.font; f.name = BODY; f.size = Pt(10.5); f.color.rgb = MUTED
    tf2 = s.shapes.add_textbox(Inches(11.9), Inches(7.06), Inches(0.9), Inches(0.35)).text_frame
    p = tf2.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    r = p.add_run(); r.text = str(PAGE[0])
    f = r.font; f.name = MONO; f.size = Pt(10.5); f.color.rgb = MUTED
    return s


def tb(s, l, t, w, h):
    tf = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    return tf


def title_bar(s, text, size=32):
    tf = tb(s, 0.9, 0.5, 11.6, 1.0)
    r = tf.paragraphs[0].add_run(); r.text = text
    f = r.font; f.name = HEAD; f.size = Pt(size); f.bold = True; f.color.rgb = NAVY
    ln = s.shapes.add_shape(1, Inches(0.9), Inches(1.32), Inches(2.2), Pt(2.5))
    ln.fill.solid(); ln.fill.fore_color.rgb = BURG; ln.line.fill.background()


def para(tf, text, size=15, color=None, bold=False, name=BODY, before=6, bullet=False):
    p = tf.paragraphs[0] if (len(tf.paragraphs) == 1 and not tf.paragraphs[0].runs) else tf.add_paragraph()
    p.space_before = Pt(before)
    r = p.add_run(); r.text = ("•  " + text) if bullet else text
    f = r.font; f.name = name; f.size = Pt(size); f.bold = bold
    f.color.rgb = color if color else INK
    return p


def divider(num, name, sub):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = NAVY
    PAGE[0] += 1
    tf = tb(s, 1.1, 2.5, 11, 2.6)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = f"Section {num}"
    f = r.font; f.name = MONO; f.size = Pt(16); f.color.rgb = RGBColor(0xC8, 0xD4, 0xE0)
    p2 = tf.add_paragraph(); p2.space_before = Pt(10)
    r = p2.add_run(); r.text = name
    f = r.font; f.name = HEAD; f.size = Pt(44); f.bold = True; f.color.rgb = WHITE
    p3 = tf.add_paragraph(); p3.space_before = Pt(12)
    r = p3.add_run(); r.text = sub
    f = r.font; f.name = BODY; f.size = Pt(17); f.color.rgb = RGBColor(0xC8, 0xD4, 0xE0)
    tf2 = tb(s, 11.9, 7.0, 0.9, 0.4)
    p = tf2.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    r = p.add_run(); r.text = str(PAGE[0])
    f = r.font; f.name = MONO; f.size = Pt(10.5); f.color.rgb = RGBColor(0xC8, 0xD4, 0xE0)


def stat(s, l, t, num, label, color=NAVY, size=32, w=3.4):
    tf = tb(s, l, t, w, 1.3)
    r = tf.paragraphs[0].add_run(); r.text = num
    f = r.font; f.name = MONO; f.size = Pt(size); f.bold = True; f.color.rgb = color
    p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = label
    f2 = r2.font; f2.name = BODY; f2.size = Pt(12.5); f2.color.rgb = MUTED


def box(s, l, t, w, h, head, text, tag=None, accent=NAVY):
    shp = s.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = LIGHT
    shp.line.color.rgb = RULE; shp.line.width = Pt(1)
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.16); tf.margin_right = Inches(0.16); tf.margin_top = Inches(0.1)
    p = tf.paragraphs[0]
    if tag:
        rt = p.add_run(); rt.text = tag + "\n"
        ft = rt.font; ft.name = MONO; ft.size = Pt(10); ft.bold = True; ft.color.rgb = BURG
    r = p.add_run(); r.text = head
    f = r.font; f.name = BODY; f.size = Pt(14.5); f.bold = True; f.color.rgb = accent
    p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = text
    f2 = r2.font; f2.name = BODY; f2.size = Pt(12); f2.color.rgb = INK


def table(s, rows, col_w, l=0.9, t=2.0, row_h=0.42, fs=13, hl_rows=()):
    tbl = s.shapes.add_table(len(rows), len(rows[0]), Inches(l), Inches(t),
                             Inches(sum(col_w)), Inches(row_h * len(rows))).table
    for j, w in enumerate(col_w):
        tbl.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i, j); c.text = str(val)
            pr = c.text_frame.paragraphs[0].runs[0].font
            pr.name = BODY if j == 0 else MONO
            pr.size = Pt(fs); pr.bold = (i == 0)
            pr.color.rgb = WHITE if i == 0 else (NAVY if i in hl_rows else INK)
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(0xE9, 0xEE, 0xF4) if i in hl_rows else WHITE)
    return tbl


# ============ 1 TITLE ============
s = prs.slides.add_slide(BLANK)
s.background.fill.solid(); s.background.fill.fore_color.rgb = WHITE
PAGE[0] += 1
ln = s.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(0.18))
ln.fill.solid(); ln.fill.fore_color.rgb = NAVY; ln.line.fill.background()
tf = tb(s, 1.0, 1.15, 11.3, 0.5)
r = tf.paragraphs[0].add_run()
r.text = "INSIGHT 2.0 — DATATHON 2026  ·  GRAND FINALE  ·  IASDS STUDENTS' CLUB, UNIVERSITY OF DHAKA"
f = r.font; f.name = MONO; f.size = Pt(13); f.color.rgb = BURG; f.bold = True
tf = tb(s, 1.0, 1.9, 11.3, 2.2)
r = tf.paragraphs[0].add_run(); r.text = "Survival Classification of Lung-Cancer Patients:"
f = r.font; f.name = HEAD; f.size = Pt(40); f.bold = True; f.color.rgb = NAVY
p2 = tf.add_paragraph()
r = p2.add_run(); r.text = "A Cross-Validation-First Approach on SEER Registry Data"
f = r.font; f.name = HEAD; f.size = Pt(40); f.bold = True; f.color.rgb = NAVY
tf = tb(s, 1.0, 4.15, 11, 0.6)
r = tf.paragraphs[0].add_run(); r.text = "Team NeurAps"
f = r.font; f.name = BODY; f.size = Pt(20); f.bold = True; f.color.rgb = INK
stat(s, 1.0, 5.1, "0.880410", "private weighted F1", color=BURG)
stat(s, 4.6, 5.1, "4th / 50", "final private rank")
stat(s, 8.2, 5.1, "28", "logged experiments")

# ============ 2 OUTLINE ============
s = base()
title_bar(s, "Outline — covering all eight required components")
tf = tb(s, 1.1, 1.75, 11.4, 5.0)
for i, (sec, det) in enumerate([
    ("Summary of Our Work", "§I — overall approach and headline result"),
    ("Exploratory Data Analysis", "§II — imbalance, the diagnosis-year effect, data quality"),
    ("Feature Selection", "§III — what we kept, what we audited, and why"),
    ("Feature Engineering", "§III — features created and transformed, with rationale"),
    ("Model Selection", "§III — models considered and the reasoning behind choices"),
    ("Model Evaluation", "§III–IV — validation design and metric-faithful evaluation"),
    ("Model Comparison", "§IV — head-to-head performance of every model tried"),
    ("Best Findings", "§IV–V — key insights and final conclusions"),
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.space_before = Pt(8)
    r = p.add_run(); r.text = f"{i+1}.   "
    f = r.font; f.name = HEAD; f.size = Pt(17); f.bold = True; f.color.rgb = BURG
    r = p.add_run(); r.text = sec
    f = r.font; f.name = BODY; f.size = Pt(17); f.bold = True; f.color.rgb = INK
    r = p.add_run(); r.text = "   —   " + det
    f = r.font; f.name = BODY; f.size = Pt(13.5); f.color.rgb = MUTED

# ============ SECTION I ============
divider("I", "Problem & Data", "Task definition, evaluation metric, and dataset structure")

# 4 problem
s = base("Problem & Data")
title_bar(s, "Problem statement")
tf = tb(s, 0.9, 1.7, 11.5, 1.6)
para(tf, "Given de-identified SEER cancer-registry records of lung-cancer patients "
         "(all primary sites C34.x), predict each patient's vital status:  Alive  or  Dead.", 17)
para(tf, "Evaluation: weighted F1 score on a hidden test set (40% public / 60% private split).", 17)
stat(s, 0.9, 3.6, "24,000", "training records")
stat(s, 4.2, 3.6, "36,000", "test records (1.5× train)")
stat(s, 7.5, 3.6, "36", "clinical features")
stat(s, 10.2, 3.6, "6 / day", "submission budget", w=2.4)
tf = tb(s, 0.9, 5.2, 11.5, 1.4)
para(tf, "Constraints: 5-day online round; no AutoML; reproducible notebook required; "
         "final score = 60% private leaderboard + 40% jury presentation.", 14, MUTED)

# 5 data overview
s = base("Problem & Data")
title_bar(s, "The feature space")
table(s, [
    ("Feature group", "Examples", "Count"),
    ("Demographics", "age band, sex, race, origin, marital status", "6"),
    ("Tumor descriptors", "primary site, histologic type (ICD-O-3), laterality, grade", "7"),
    ("Staging", "derived T/N/M recodes, summary stage, EOD extension", "7"),
    ("Tumor size", "three era-specific coded columns (1988–2015, 2016+)", "3"),
    ("Nodes & metastasis", "nodes examined/positive, mets at dx: bone/brain/liver/lung", "6"),
    ("Treatment", "surgery codes, radiation, surgery-refusal reasons, sequence", "7"),
], [3.2, 6.6, 1.4], t=1.9, row_h=0.55)
tf = tb(s, 0.9, 5.9, 11.5, 0.9)
para(tf, "Nearly all features are categorical registry codes; three size columns hide numeric values "
         "behind sentinel codes (999 / Blank). Correct decoding is itself feature engineering.", 14, MUTED)

# ============ SECTION II ============
divider("II", "Exploratory Analysis", "What the data told us before any model was fit")

# 7 target
s = base("Exploratory Analysis")
title_bar(s, "The target is heavily imbalanced")
stat(s, 0.9, 2.1, "83.0%", "of training patients are Dead", color=NAVY, size=40)
stat(s, 5.4, 2.1, "17.0%", "are Alive — the minority class", color=BURG, size=40)
tf = tb(s, 0.9, 3.9, 11.5, 2.6)
para(tf, "Why this matters for weighted F1:", 16, INK, bold=True)
para(tf, "A trivial all-Dead classifier reaches 0.907 on Dead-class F1 but only 0.75 weighted F1 — "
         "the Alive class, though small, carries decisive weight.", 15, bullet=True)
para(tf, "The decision threshold therefore becomes a first-class model parameter (§ Methodology).", 15, bullet=True)
para(tf, "Class imbalance also dictates stratified resampling in cross-validation.", 15, bullet=True)

# 8 year effect
s = base("Exploratory Analysis")
title_bar(s, "The dominant covariate: year of diagnosis")
cd = CategoryChartData()
cd.categories = [str(yr) for yr in range(2013, 2024)]
cd.add_series("Death rate (%)", (93.7, 92.6, 92.2, 90.2, 88.5, 87.0, 84.3, 78.7, 76.2, 67.4, 44.4))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.9), Inches(1.8), Inches(7.6), Inches(4.7), cd)
ch = gf.chart; ch.has_legend = False
ser = ch.plots[0].series[0]
ser.format.line.color.rgb = NAVY; ser.format.line.width = Pt(2.5)
tf = tb(s, 8.9, 2.0, 3.6, 4.5)
para(tf, "93.7% → 44.4%", 22, BURG, bold=True, name=MONO)
para(tf, "Death rate by diagnosis year. Recent patients \"survive\" more only because they have "
         "not been followed as long — a censoring artifact, not a treatment effect.", 13)
para(tf, "Train and test share the same year distribution → the split is random → "
         "StratifiedKFold is the honest CV scheme.", 13)
para(tf, "Model errors concentrate in recent years (23% in 2023 vs 6% in 2013).", 13)

# 9 data quality
s = base("Exploratory Analysis")
title_bar(s, "Data quality and coded missingness")
box(s, 0.9, 1.9, 5.7, 1.55, "Sentinel codes, not NaN",
    "Tumor size uses 999 / \"Blank(s)\" / \"Unknown or size unreasonable\"; 66.5% of one surgery-scope "
    "column is null by era definition. Decoded into numeric values + explicit missingness flags.")
box(s, 6.8, 1.9, 5.7, 1.55, "Era-dependent columns",
    "Size and staging columns switch coding systems across diagnosis eras; we coalesce them into "
    "single best-available features rather than dropping either era.")
box(s, 0.9, 3.65, 5.7, 1.55, "No ID or ordering leakage",
    "Patient IDs are sequential and carry no signal; no duplicated IDs; row order uninformative.")
box(s, 6.8, 3.65, 5.7, 1.55, "Informative categories",
    "\"Death certificate only\" and \"died prior to recommended surgery\" categories directly "
    "signal the outcome and appear in both train and test — legitimate, powerful features.")

# ============ SECTION III ============
divider("III", "Methodology", "Metric identification · validation design · features · architecture")

# 11 metric identification
s = base("Methodology")
title_bar(s, "Step 0 — identify the exact metric")
tf = tb(s, 0.9, 1.7, 11.5, 0.9)
para(tf, "\"F1 score\" is ambiguous under imbalance. We swept our out-of-fold predictions over every F1 "
         "variant and compared against the observable leaderboard range:", 15)
table(s, [
    ("Candidate metric", "Our OOF (tuned thr.)", "LB range", "Verdict"),
    ("Macro F1", "0.776", "0.874 – 0.877", "too low — rejected"),
    ("F1, Dead class", "0.931", "0.874 – 0.877", "too high — rejected"),
    ("Weighted F1", "0.874 – 0.877", "0.874 – 0.877", "exact match — adopted"),
], [3.4, 3.4, 2.6, 2.8], t=2.75, row_h=0.5, hl_rows=(3,))
tf = tb(s, 0.9, 5.4, 11.5, 1.2)
para(tf, "Consequence: we optimize the decision threshold on out-of-fold predictions for every model "
         "(final threshold: 0.575, not the default 0.5) — worth ≈ +0.003 weighted F1 by itself.", 15, NAVY, bold=True)

# 12 validation design
s = base("Methodology")
title_bar(s, "Validation design")
box(s, 0.9, 1.8, 5.7, 1.6, "Stratified 10-fold CV × 2–3 seeds",
    "Out-of-fold predictions pooled across folds; per-fold scores monitored for stability. "
    "Stratification preserves the 83/17 class ratio in every fold.")
box(s, 6.8, 1.8, 5.7, 1.6, "Threshold tuned on OOF only",
    "The decision cut is swept on out-of-fold probabilities; per-fold threshold medians "
    "checked against the global optimum for robustness.")
box(s, 0.9, 3.6, 5.7, 1.6, "Honest split-half verification",
    "Any selected quantity (blend weights, thresholds, mixtures) is re-chosen on half the OOF data "
    "and scored on the other half — a guard against selection bias.")
box(s, 6.8, 3.6, 5.7, 1.6, "Leaderboard used as an instrument",
    "Public submissions answer one isolated question each (metric identity, tuning generalization). "
    "Never used for model selection.")
tf = tb(s, 0.9, 5.5, 11.5, 0.8)
para(tf, "Result: our CV↔LB correlation was established on day 1 and monitored throughout — "
         "the foundation for every later decision.", 14, MUTED)

# 12b feature selection
s = base("Methodology")
title_bar(s, "Feature selection — what we kept, and why")
tf = tb(s, 0.9, 1.7, 11.5, 1.0)
para(tf, "Starting point: all 36 provided clinical features. Final model: all 36, plus ~15 engineered "
         "features. Retention was a deliberate, tested decision — not a default.", 15)
box(s, 0.9, 2.75, 5.7, 1.75, "Why not hard filtering?",
    "Gradient-boosted trees perform implicit selection: uninformative features are simply never chosen "
    "for splits, and regularization (feature subsampling, min-leaf constraints) suppresses noise. "
    "We tested dropping low-importance features — CV worsened or was flat every time.")
box(s, 6.8, 2.75, 5.7, 1.75, "Importance-ranked signal",
    "Permutation-checked importance concentrated in: surgery-decision codes, summary stage / TNM, "
    "year of diagnosis, age, tumor size, and metastasis flags — all clinically coherent, "
    "which itself is a sanity check on the pipeline.")
box(s, 0.9, 4.7, 5.7, 1.6, "Leakage audit as selection",
    "Every high-power feature was individually audited (single-feature discriminative power, "
    "presence and distribution in test). All passed — the strong features are legitimately strong.")
box(s, 6.8, 4.7, 5.7, 1.6, "No shift-based drops needed",
    "Train and test distributions align (matching year profiles; no adversarial separability), "
    "so no feature had to be sacrificed for train/test stability.")

# 13 features 1
s = base("Methodology")
title_bar(s, "Feature engineering (i) — clinical ordinality")
box(s, 0.9, 1.85, 5.7, 1.7, "Staging → ordered scales",
    "T-stage: T0/Tis < T1mi < T1a…< T4 mapped to a numeric scale; N0–N3; M0 < M1a < M1b < M1c; "
    "grade I–IV; summary stage Localized < Regional < Distant.")
box(s, 6.8, 1.85, 5.7, 1.7, "Composite burden",
    "TNM sum with imputed midpoints for unknowns; metastasis count over bone/brain/liver/lung; "
    "positive-node ratio with examined-nodes guard.")
box(s, 0.9, 3.75, 5.7, 1.7, "Tumor size reconstruction",
    "Three era-coded columns parsed (999/Blank → missing), coalesced into tumor_size_best, "
    "plus per-column and all-missing indicator flags.")
box(s, 6.8, 3.75, 5.7, 1.7, "Temporal context",
    "Years since diagnosis (follow-up window proxy), age-band midpoints, "
    "multiple-primaries indicator from sequence number.")

# 14 features 2
s = base("Methodology")
title_bar(s, "Feature engineering (ii) — treatment & leakage discipline")
box(s, 0.9, 1.85, 5.7, 1.7, "Treatment indicators",
    "Surgery-performed, radiation-given, any-treatment; refusal and not-recommended categories "
    "retained as levels — treatment decisions encode physician prognosis.")
box(s, 6.8, 1.85, 5.7, 1.7, "Directly informative categories",
    "\"Death certificate / autopsy only\" and \"died prior to recommended surgery\" flags — "
    "present in test, honestly usable, individually validated.")
box(s, 0.9, 3.75, 5.7, 1.7, "Categorical handling",
    "Native categorical splits in LightGBM/CatBoost; integer codes for XGBoost; one-hot with "
    "min-frequency pruning for linear/NN models. Consistent category alignment across train/test.")
box(s, 6.8, 3.75, 5.7, 1.7, "Anti-leakage rules",
    "No target statistics fitted outside CV folds; nothing fitted on test; any feature with "
    "suspicious single-feature power audited before use.")

# 15 architecture
s = base("Methodology")
title_bar(s, "Model architecture — two-stage ensemble")
steps = [("1 · Stage-1 trio", "Tuned LightGBM + XGBoost + CatBoost\n10 folds × 3 seeds; weight-optimized blend"),
         ("2 · Pseudo-labelling", "7,455 high-confidence test rows\n(p > 0.99 or p < 0.02) join training"),
         ("3 · Stage-2 trio", "Same trio retrained on augmented data\n10 folds × 2 seeds; re-blended"),
         ("4 · Diversity mixture", "85% GBM blend + 5% each of\nRandomForest · ExtraTrees · MLP"),
         ("5 · Decision rule", "Threshold 0.575 from OOF\nweighted-F1 sweep"),
         ("6 · Output", "36,000 Alive/Dead predictions\nOOF weighted F1 = 0.87996")]
for k, (h_, x_) in enumerate(steps):
    l = 0.9 + (k % 3) * 4.0; t = 1.95 + (k // 3) * 2.15
    shp = s.shapes.add_shape(1, Inches(l), Inches(t), Inches(3.7), Inches(1.85))
    shp.fill.solid(); shp.fill.fore_color.rgb = LIGHT
    shp.line.color.rgb = NAVY if k in (0, 2, 4) else RULE
    shp.line.width = Pt(1.75 if k in (0, 2, 4) else 1)
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.15); tf.margin_top = Inches(0.1)
    r = tf.paragraphs[0].add_run(); r.text = h_
    f = r.font; f.name = BODY; f.size = Pt(15); f.bold = True; f.color.rgb = NAVY
    p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = x_
    f2 = r2.font; f2.name = BODY; f2.size = Pt(12); f2.color.rgb = INK
tf = tb(s, 0.9, 6.25, 11.6, 0.6)
para(tf, "Hyperparameters: 40-trial Optuna study for LightGBM (stage 1 only — see § Experiments for why); "
         "library defaults elsewhere.", 13, MUTED)

# 16 pseudo-labeling detail
s = base("Methodology")
title_bar(s, "Pseudo-labelling — the honest way")
tf = tb(s, 0.9, 1.8, 11.5, 4.6)
para(tf, "Motivation: the test set is 1.5× the training set — a large pool of unlabelled, in-distribution data.", 15, bullet=True)
para(tf, "Only extremely confident predictions cross the boundary (p > 0.99 → Dead, p < 0.02 → Alive): "
         "7,455 of 36,000 rows, 99.3% of them Dead-confident.", 15, bullet=True)
para(tf, "Augmented rows join only the training side of each fold; out-of-fold evaluation remains on "
         "original training rows — CV stays unbiased.", 15, bullet=True)
para(tf, "Verified empirically: wider confidence bands (0.05/0.97) and a second pseudo-labelling round "
         "were both tested and rejected — no additional gain.", 15, bullet=True)
para(tf, "Effect: the single largest leaderboard improvement of our campaign "
         "(public 0.87532 → 0.87619 at the time of adoption).", 15, NAVY, bold=True, bullet=True)

# 17 diversity
s = base("Methodology")
title_bar(s, "Diversity mixture — small dose, verified honestly")
tf = tb(s, 0.9, 1.8, 11.5, 2.6)
para(tf, "Boosted trees dominate tabular data, but three GBMs make correlated mistakes (pairwise OOF "
         "correlation ≥ 0.97). Bagged trees and a neural network disagree precisely on uncertain rows.", 15, bullet=True)
para(tf, "RandomForest (0.8743), ExtraTrees (0.8727), MLP (0.8739) are individually weaker — "
         "but a 5%-each admixture over the GBM blend improves OOF weighted F1: 0.87965 → 0.87996.", 15, bullet=True)
para(tf, "Split-half verification: choosing the mixture on half the OOF data and scoring the other half "
         "confirms +0.0006 — a real effect, not selection noise.", 15, bullet=True)
stat(s, 0.9, 4.9, "+0.0006", "honest (split-half) gain from diversity", color=BURG, size=30, w=5)
stat(s, 6.4, 4.9, "0.87996", "full-pipeline OOF weighted F1", size=30, w=5)

# ============ SECTION IV ============
divider("IV", "Experiments & Results", "28 logged experiments · negative results · the final-selection decision")

# 19 experiment overview
s = base("Experiments & Results")
title_bar(s, "The experiment ledger")
table(s, [
    ("Day", "Experiments", "Key outcome"),
    ("1", "baseline, metric ID, features, first blend", "public 0.8743 → 0.8760; metric identified; 3rd place"),
    ("2", "tuning, threshold probes, 10-fold, pseudo-labels", "pseudo-labelling adopted — public 0.87619"),
    ("3", "stacking, TabPFN, distillation, diversity waves", "diversity mixture adopted; three traps rejected"),
    ("4", "focal loss, re-weighting, fine probes, finals", "public peak found; finals selected by CV + LB"),
], [1.2, 5.3, 5.8], t=1.9, row_h=0.62)
tf = tb(s, 0.9, 5.4, 11.5, 1.2)
para(tf, "Every experiment logged with CV score, leaderboard score (when spent), and a keep/drop verdict. "
         "The ledger is what made the endgame decisions evidence-based rather than intuitive.", 14, MUTED)

# 20 negative results 1
s = base("Experiments & Results")
title_bar(s, "Negative results (i) — two validation traps")
box(s, 0.9, 1.9, 5.7, 2.3, "Aggressive hyperparameter tuning",
    "A 40-trial Optuna study improved OOF by +0.002 — and worsened the leaderboard by 0.0005. "
    "A controlled probe (same predictions, one variable isolated) confirmed the tuned parameters "
    "were overfitting the CV folds. We kept library defaults for stage 2.", tag="REJECTED — CV OVERFITTING")
box(s, 6.8, 1.9, 5.7, 2.3, "Soft-label distillation",
    "Training on all 36,000 test rows with soft targets from our best blend inflated CV by +0.001. "
    "The gain was validation information leaking through the soft labels — leaderboard flat. "
    "The CV number was provably a lie, and we discarded it.", tag="REJECTED — LEAKAGE INFLATION")
tf = tb(s, 0.9, 4.5, 11.6, 1.6)
para(tf, "Both traps produce beautiful CV numbers. Both were caught by the same discipline: "
         "one isolated leaderboard probe per hypothesis, and honest split-half re-verification.", 15, NAVY, bold=True)

# 21 negative results 2
s = base("Experiments & Results")
title_bar(s, "Negative results (ii) — the saturation evidence")
box(s, 0.9, 1.9, 5.7, 1.8, "Per-year decision thresholds",
    "In-sample +0.001; honest split-half evaluation shows it worse than a single global threshold. "
    "Eleven per-year cuts = eleven chances to overfit.", tag="REJECTED — SELECTION BIAS")
box(s, 6.8, 1.9, 5.7, 1.8, "More model families",
    "TabPFN (tabular foundation model), meta-model stacking, HistGB, logistic regression, focal loss: "
    "every candidate correlated ≥ 0.97 with the GBM blend. No usable diversity remained.", tag="REJECTED — NO DIVERSITY")
tf = tb(s, 0.9, 4.0, 11.6, 2.2)
para(tf, "Conclusion we could defend: the dataset's extractable signal saturates near OOF ≈ 0.880.", 16, INK, bold=True)
para(tf, "This is a measured ceiling — five model families, two pseudo-labelling rounds, three re-weighting "
         "schemes and a foundation model all converge on it. Knowing the ceiling exists is what justified "
         "our conservative endgame instead of chasing the public leaderboard.", 14, MUTED)

# 21b model comparison
s = base("Experiments & Results")
title_bar(s, "Model comparison — every model, head to head")
table(s, [
    ("Model / configuration", "OOF weighted F1", "Role in final solution"),
    ("Logistic regression (one-hot)", "0.87326", "diversity candidate — rejected"),
    ("ExtraTrees (800 trees)", "0.87266", "5% of final mixture"),
    ("MLP 128×64 (one-hot + scaled)", "0.87393", "5% of final mixture"),
    ("RandomForest (800 trees)", "0.87432", "5% of final mixture"),
    ("HistGradientBoosting", "0.87502", "diversity candidate — rejected"),
    ("TabPFN (foundation model)", "0.87644", "rejected — 0.98 corr. with blend"),
    ("LightGBM (single, default)", "0.87646", "baseline reference"),
    ("CatBoost (10f × 3 seeds)", "0.87871", "blend member"),
    ("XGBoost (10f × 3 seeds)", "0.87885", "blend member"),
    ("LightGBM, focal loss γ=1", "0.87982", "best single model — gain overlapped blend"),
    ("GBM trio blend + pseudo-labels", "0.87965", "stage-2 core"),
    ("Final: + RF/ET/MLP mixture", "0.87996", "submitted — private 0.880410"),
], [5.0, 2.6, 4.0], t=1.75, row_h=0.36, fs=11.5, hl_rows=(13,))
tf = tb(s, 0.9, 6.55, 11.6, 0.5)
para(tf, "Identical validation protocol for every row: stratified 10-fold OOF, threshold tuned per model. "
         "The ordering is what justified every architectural choice.", 12, MUTED)

# 22 public vs private
s = base("Experiments & Results")
title_bar(s, "The decisive call — final submission selection")
cd = CategoryChartData()
cd.categories = ["CV pick (selected & scored)", "Public-best pick (selected)"]
cd.add_series("public split", (0.875506, 0.876311))
cd.add_series("private split", (0.880410, 0.879732))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.9), Inches(1.85), Inches(7.4), Inches(4.3), cd)
ch = gf.chart; ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
ch.plots[0].series[0].format.fill.solid()
ch.plots[0].series[0].format.fill.fore_color.rgb = RGBColor(0x9A, 0xA7, 0xB3)
ch.plots[0].series[1].format.fill.solid()
ch.plots[0].series[1].format.fill.fore_color.rgb = NAVY
ch.value_axis.minimum_scale = 0.874; ch.value_axis.maximum_scale = 0.881
tf = tb(s, 8.7, 2.0, 3.8, 4.4)
para(tf, "Measured: public deltas < 0.0008 between our submissions were split noise.", 13, bullet=True)
para(tf, "Strategy: select one final by public score, one by honest CV.", 13, bullet=True)
para(tf, "The CV pick — punished by the public board — scored 0.880410 and delivered 4th.", 13, NAVY, bold=True, bullet=True)
para(tf, "Auto-selection (two best public) would have finished ≈ 9th.", 13, bullet=True)

# 23 results
s = base("Experiments & Results")
title_bar(s, "Final results")
stat(s, 0.9, 2.0, "0.87996", "out-of-fold forecast (pre-reveal)", size=36, w=4)
stat(s, 5.1, 2.0, "0.880410", "private leaderboard score", color=BURG, size=36, w=4)
stat(s, 9.3, 2.0, "0.0004", "forecast error", size=36, w=3.2)
stat(s, 0.9, 4.1, "21st → 4th", "public rank → private rank", size=36, w=4)
stat(s, 5.1, 4.1, "0.000075", "gap to 3rd place (≈ 3 patients of 36,000)", size=36, w=4.6)
tf = tb(s, 0.9, 5.9, 11.5, 0.9)
para(tf, "The private reshuffle rewarded validation-first teams: several teams above us on the public "
         "board fell out of the top ten entirely.", 14, MUTED)

# ============ SECTION V ============
divider("V", "Conclusions", "Principles, limitations, and where this work goes next")

# 25 principles
s = base("Conclusions")
title_bar(s, "Four principles that decided this result")
prin = [("Identify the exact metric before modelling.",
         "One OOF sweep on day 1 revealed weighted F1 and promoted the decision threshold to a model parameter."),
        ("Trust out-of-fold CV; treat the public leaderboard as an instrument.",
         "Read it through controlled single-variable probes; never chase it."),
        ("Verify every gain adversarially.",
         "Split-half verification killed three seductive improvements that were bias or leakage."),
        ("Log everything — negative results compound.",
         "28 recorded experiments made the endgame a calculation instead of a gamble.")]
for k, (h_, x_) in enumerate(prin):
    l = 0.9 + (k % 2) * 6.0; t = 1.95 + (k // 2) * 2.25
    tf = tb(s, l, t, 5.6, 2.0)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = f"{k+1}.  "
    f = r.font; f.name = HEAD; f.size = Pt(22); f.bold = True; f.color.rgb = BURG
    r = p.add_run(); r.text = h_
    f = r.font; f.name = BODY; f.size = Pt(16); f.bold = True; f.color.rgb = INK
    p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = x_
    f2 = r2.font; f2.name = BODY; f2.size = Pt(13); f2.color.rgb = MUTED

# 26 limitations
s = base("Conclusions")
title_bar(s, "Limitations & future work")
box(s, 0.9, 1.9, 5.7, 2.1, "Censoring is implicit, not modelled",
    "Vital status conflates death with insufficient follow-up. A survival-analysis formulation "
    "(Cox PH, gradient-boosted survival, discrete-time hazard) would model time-to-event directly.")
box(s, 6.8, 1.9, 5.7, 2.1, "Calibration under distribution drift",
    "Death rates drift by diagnosis year; probability calibration per era could sharpen the "
    "single global threshold we settled on.")
box(s, 0.9, 4.2, 5.7, 2.1, "Interpretability",
    "Judged useful in clinical contexts: SHAP-based attribution of the final ensemble and "
    "per-feature partial dependence remain future work.")
box(s, 6.8, 4.2, 5.7, 2.1, "Compute-bounded search",
    "Four days on a single laptop bounded the ensemble size; more seeds and deeper "
    "stacking under a proper time budget may add a small margin.")

# 27 tools
s = base("Conclusions")
title_bar(s, "Tools & reproducibility")
tf = tb(s, 0.9, 1.8, 11.5, 3.8)
para(tf, "Python 3.12 · pandas · NumPy · scikit-learn (CV, metrics, RF/ET/MLP)", 15, bullet=True)
para(tf, "LightGBM 4.x · XGBoost 3.x · CatBoost 1.x — gradient-boosted trees", 15, bullet=True)
para(tf, "Optuna (stage-1 LightGBM hyperparameter study, 40 trials)", 15, bullet=True)
para(tf, "All seeds fixed (42 / 2026 / 7); submitted notebook re-runs end-to-end and regenerates the "
         "selected predictions (≈ 5–6 h CPU).", 15, bullet=True)
para(tf, "Data: competition-provided SEER lung-cancer extract. No external data used.", 15, bullet=True)
para(tf, "Experiment ledger and submission history available on request.", 15, bullet=True)

# 28 thanks
s = prs.slides.add_slide(BLANK)
s.background.fill.solid(); s.background.fill.fore_color.rgb = NAVY
PAGE[0] += 1
tf = tb(s, 1.1, 2.6, 11, 2.4)
r = tf.paragraphs[0].add_run(); r.text = "Thank you."
f = r.font; f.name = HEAD; f.size = Pt(52); f.bold = True; f.color.rgb = WHITE
p2 = tf.add_paragraph(); p2.space_before = Pt(14)
r = p2.add_run(); r.text = "Questions welcome — every claim is reproducible from our notebook and experiment ledger."
f = r.font; f.name = BODY; f.size = Pt(18); f.color.rgb = RGBColor(0xC8, 0xD4, 0xE0)
p3 = tf.add_paragraph(); p3.space_before = Pt(26)
r = p3.add_run(); r.text = "Team NeurAps  ·  Insight 2.0 — Datathon 2026  ·  private weighted F1 0.880410 (4th of 50)"
f = r.font; f.name = MONO; f.size = Pt(14); f.color.rgb = RGBColor(0xC8, 0xD4, 0xE0)

prs.save("presentations/NeurAps_Insight2_Finale_v3.pptx")
print(f"saved presentations/NeurAps_Insight2_Finale_v3.pptx — {PAGE[0]} slides")

"""Generate the NeurAps Grand Finale PowerPoint deck (16:9)."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

INK = RGBColor(0x16, 0x26, 0x2E)
MUTED = RGBColor(0x5B, 0x6E, 0x77)
TEAL = RGBColor(0x0B, 0x75, 0x68)
TEAL_BRIGHT = RGBColor(0x00, 0xA0, 0x8F)
ORANGE = RGBColor(0xC2, 0x57, 0x1A)
GROUND = RGBColor(0xF7, 0xF9, 0xFA)
RULE = RGBColor(0xD8, 0xE1, 0xE5)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def bg(slide):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = GROUND


def tb(slide, l, t, w, h):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    box.text_frame.word_wrap = True
    return box.text_frame


def eyebrow(slide, text):
    tf = tb(slide, 0.9, 0.55, 11.5, 0.4)
    p = tf.paragraphs[0]; r = p.add_run(); r.text = text.upper()
    f = r.font; f.name = "Consolas"; f.size = Pt(13); f.color.rgb = TEAL; f.bold = True


def headline(slide, text, size=40, top=1.05, color=INK, width=11.5):
    tf = tb(slide, 0.9, top, width, 1.4)
    p = tf.paragraphs[0]; r = p.add_run(); r.text = text
    f = r.font; f.name = "Georgia"; f.size = Pt(size); f.bold = True; f.color.rgb = color


def body(slide, text, top, size=17, width=11.0, color=MUTED, left=0.9):
    tf = tb(slide, left, top, width, 1.4)
    p = tf.paragraphs[0]; r = p.add_run(); r.text = text
    f = r.font; f.name = "Calibri"; f.size = Pt(size); f.color.rgb = color
    return tf


def stat(slide, left, top, num, label, accent=False, num_size=34):
    tf = tb(slide, left, top, 3.4, 1.3)
    p = tf.paragraphs[0]; r = p.add_run(); r.text = num
    f = r.font; f.name = "Consolas"; f.size = Pt(num_size); f.bold = True
    f.color.rgb = TEAL if accent else INK
    p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = label
    f2 = r2.font; f2.name = "Calibri"; f2.size = Pt(13); f2.color.rgb = MUTED


def card(slide, l, t, w, h, title, text, tag=None):
    shp = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    shp.line.color.rgb = RULE; shp.line.width = Pt(1)
    tf = shp.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.18); tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.12); tf.margin_bottom = Inches(0.1)
    p = tf.paragraphs[0]
    if tag:
        rt = p.add_run(); rt.text = tag + "\n"
        ft = rt.font; ft.name = "Consolas"; ft.size = Pt(10.5); ft.color.rgb = ORANGE; ft.bold = True
    r = p.add_run(); r.text = title
    f = r.font; f.name = "Calibri"; f.size = Pt(16); f.bold = True; f.color.rgb = INK
    p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = text
    f2 = r2.font; f2.name = "Calibri"; f2.size = Pt(12.5); f2.color.rgb = MUTED


def new(eyebrow_text, headline_text, hsize=38):
    s = prs.slides.add_slide(BLANK); bg(s)
    eyebrow(s, eyebrow_text); headline(s, headline_text, size=hsize)
    return s


# ---- 1 TITLE ----
s = prs.slides.add_slide(BLANK); bg(s)
eyebrow(s, "Insight 2.0 · Grand Finale · IASDS Students' Club, University of Dhaka")
headline(s, "Trusting the Cross-Validation", size=54, top=1.5)
body(s, "Lung-cancer survival classification on SEER registry data — how a disciplined validation "
        "process took Team NeurAps from 21st on the public leaderboard to 4th on the private one.",
     top=3.0, size=18, width=10.5)
stat(s, 0.9, 4.6, "0.880410", "private weighted F1", accent=True)
stat(s, 4.6, 4.6, "4th / 50", "final private rank")
stat(s, 8.3, 4.6, "28", "logged experiments in 4 days")

# ---- 2 PROBLEM ----
s = new("The problem", "Predict who survives lung cancer")
body(s, "SEER registry records of lung-cancer patients (all primary sites C34.x): 36 clinical features — "
        "staging codes, tumor size, treatment, demographics. Predict vital_status ∈ {Alive, Dead}. "
        "Metric: weighted F1.", top=2.3, size=17)
stat(s, 0.9, 3.9, "24,000", "training patients")
stat(s, 4.0, 3.9, "36,000", "test patients — 1.5× train")
stat(s, 7.1, 3.9, "83 / 17", "Dead / Alive — heavy imbalance")
stat(s, 10.2, 3.9, "6 /day", "submission budget")

# ---- 3 METRIC ----
s = new("Day one · metric identification", "Which F1? The leaderboard told us.")
body(s, "\"F1 Score\" is ambiguous under imbalance. A trivial all-Dead baseline scores 0.907 on Dead-class F1 — "
        "above the leaderboard top, so that isn't the metric. We swept out-of-fold predictions across every "
        "F1 variant and matched ranges:", top=2.15, size=16)
rows = [("candidate metric", "our OOF (tuned threshold)", "leaderboard range", "verdict"),
        ("Macro F1", "0.776", "—", "too low"),
        ("F1 (Dead)", "0.931", "—", "too high"),
        ("Weighted F1", "0.874 – 0.877", "0.874 – 0.877", "exact match")]
table = s.shapes.add_table(4, 4, Inches(0.9), Inches(3.4), Inches(11.5), Inches(2.2)).table
for j, wd in enumerate([3.2, 3.4, 3.0, 1.9]):
    table.columns[j].width = Inches(wd)
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        cell = table.cell(i, j); cell.text = val
        pr = cell.text_frame.paragraphs[0].runs[0].font
        pr.name = "Calibri"; pr.size = Pt(14)
        pr.bold = (i == 0 or i == 3); pr.color.rgb = INK if i in (0, 3) else MUTED
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0xE2, 0xF2, 0xEF) if i == 3 else RGBColor(0xFF, 0xFF, 0xFF)
body(s, "Consequence: the decision threshold is part of the model. Every submission used a threshold tuned "
        "on out-of-fold predictions (final: 0.575), never the default 0.5.", top=6.15, size=14)

# ---- 4 YEAR EFFECT (native line chart) ----
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
s = new("EDA · the dominant covariate", "One variable explains most of the difficulty")
cd = CategoryChartData()
cd.categories = [str(yr) for yr in range(2013, 2024)]
cd.add_series("Death rate", (93.7, 92.6, 92.2, 90.2, 88.5, 87.0, 84.3, 78.7, 76.2, 67.4, 44.4))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.9), Inches(2.2),
                        Inches(8.0), Inches(4.6), cd)
chart = gf.chart; chart.has_legend = False
ser = chart.plots[0].series[0]
ser.format.line.color.rgb = TEAL_BRIGHT; ser.format.line.width = Pt(2.5)
body(s, "Death rate by year of diagnosis: 93.7% (2013) → 44.4% (2023). Recent patients survive more only "
        "because they haven't been followed as long.\n\nTrain/test year distributions match → random split → "
        "StratifiedKFold(10) is the honest validation scheme.\n\nErrors concentrate in recent years "
        "(23% in 2023 vs 6% in 2013) — the genuinely uncertain patients.", top=2.4, size=14, width=3.4, left=9.2)

# ---- 5 FEATURES ----
s = new("Feature engineering · hand-built, no AutoML", "Clinical structure, encoded deliberately")
feats = [("Staging → ordinal scales", "T/N/M codes mapped to clinically ordered scales; grade I–IV; summary stage; combined TNM burden score."),
         ("Tumor size across eras", "Size lives in 3 era-coded columns (999/Blank = unknown). Parsed, missing-flagged, coalesced into tumor_size_best."),
         ("Disease burden", "Metastasis count across bone/brain/liver/lung; positive-node ratio; nodes-examined indicator."),
         ("Treatment signals", "Surgery, radiation, any-treatment; the \"death certificate / died prior to surgery\" category."),
         ("Time", "Years since diagnosis (follow-up window), age-band midpoints, multiple-primaries indicator."),
         ("Leakage discipline", "No target in any encoder; nothing fitted on test; sanity checks on anything \"too good\".")]
for k, (t_, x_) in enumerate(feats):
    card(s, 0.9 + (k % 3) * 4.0, 2.3 + (k // 3) * 2.3, 3.7, 2.05, t_, x_)

# ---- 6 DISCIPLINE ----
s = new("Method · validation discipline", "Every change gated by out-of-fold CV")
cards6 = [("10-fold × multi-seed OOF", "Stratified 10-fold over 2–3 seeds. Every change evaluated on pooled OOF weighted F1 with a tuned threshold — never on the public leaderboard."),
          ("Honest split-half verification", "Blend weights, thresholds and mixtures re-verified by choosing them on one half of OOF and scoring the other. Killed several \"improvements\" that were selection bias."),
          ("Controlled leaderboard probes", "Public submissions answered single questions (which metric? does tuning generalize?) — one variable at a time, never score-chasing."),
          ("28 experiments, all logged", "Every experiment recorded with CV, LB and a keep/drop verdict. The log is why the final-selection call was made with confidence.")]
for k, (t_, x_) in enumerate(cards6):
    card(s, 0.9 + (k % 2) * 6.0, 2.3 + (k // 2) * 2.35, 5.7, 2.1, t_, x_)

# ---- 7 ARCHITECTURE ----
s = new("The final model", "Two-stage ensemble with pseudo-labeling")
steps = [("Engineered features", "ordinals · sizes · burden · treatment"),
         ("Stage 1 trio", "tuned LGBM + XGB + CatBoost\n10 folds × 3 seeds → blend"),
         ("Pseudo-labels", "7,455 test rows with\np>0.99 or p<0.02 join training"),
         ("Stage 2 trio", "retrained on augmented data\n10 folds × 2 seeds → blend"),
         ("Diversity mix", "85% blend + 5% each\nRF · ExtraTrees · MLP"),
         ("Decision", "threshold 0.575 from\nOOF weighted-F1 sweep")]
for k, (t_, x_) in enumerate(steps):
    l = 0.55 + (k % 3) * 4.25; t = 2.4 + (k // 3) * 2.1
    shp = s.shapes.add_shape(1, Inches(l), Inches(t), Inches(3.9), Inches(1.75))
    shp.fill.solid(); shp.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    accent = k in (1, 3, 5)
    shp.line.color.rgb = TEAL_BRIGHT if accent else RULE
    shp.line.width = Pt(2 if accent else 1)
    tf = shp.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = f"{k+1}.  {t_}"
    f = r.font; f.name = "Calibri"; f.size = Pt(15); f.bold = True; f.color.rgb = INK
    p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = x_
    f2 = r2.font; f2.name = "Calibri"; f2.size = Pt(12); f2.color.rgb = MUTED
body(s, "The 5%-each diversity admixture is small by design: bagged trees and a neural net disagree with boosted "
        "trees on exactly the uncertain rows (+0.0006, verified by split-half). Full-pipeline OOF: 0.87996.",
     top=6.6, size=13.5)

# ---- 8 NEGATIVE RESULTS ----
s = new("Rigor · what we proved doesn't work", "Rejected with evidence, not intuition")
neg = [("REJECTED — CV OVERFITTING", "Aggressive Optuna tuning", "+0.002 OOF, −0.0005 on the leaderboard. Caught with a controlled probe. Defaults generalized better."),
       ("REJECTED — LEAKAGE INFLATION", "Full-test soft-label distillation", "+0.001 CV was validation information leaking through soft labels; leaderboard flat. The CV number lied — and we proved it."),
       ("REJECTED — SELECTION BIAS", "Per-year decision thresholds", "Looked great in-sample (+0.001); honest split-half evaluation showed it worse than one global threshold."),
       ("REJECTED — NO DIVERSITY", "TabPFN, stacking, HistGB, focal-only", "Every extra family correlated ≥0.97 with the GBM blend. The dataset's signal saturates near 0.880 OOF — measured, not assumed.")]
for k, (tag, t_, x_) in enumerate(neg):
    card(s, 0.9 + (k % 2) * 6.0, 2.3 + (k // 2) * 2.35, 5.7, 2.1, t_, x_, tag=tag)

# ---- 9 PUBLIC VS PRIVATE (native bar chart) ----
s = new("The decisive call · final selection", "Public leaderboard was noise. CV wasn't.")
cd = CategoryChartData()
cd.categories = ["CV pick (scored)", "public pick"]
cd.add_series("public split", (0.875506, 0.876311))
cd.add_series("private split", (0.880410, 0.879732))
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.9), Inches(2.2),
                        Inches(7.6), Inches(4.4), cd)
chart = gf.chart
chart.has_legend = True; chart.legend.position = XL_LEGEND_POSITION.BOTTOM
chart.legend.include_in_layout = False
chart.plots[0].series[0].format.fill.solid()
chart.plots[0].series[0].format.fill.fore_color.rgb = RGBColor(0x8F, 0xA0, 0xA8)
chart.plots[0].series[1].format.fill.solid()
chart.plots[0].series[1].format.fill.fore_color.rgb = TEAL_BRIGHT
chart.value_axis.minimum_scale = 0.874; chart.value_axis.maximum_scale = 0.881
body(s, "Public deltas <0.0008 were measured to be split noise, so we selected one final by public score and "
        "one by honest CV.\n\nThe CV pick — punished by the public board — scored 0.880410 and delivered 4th "
        "place. CV-vs-private gap: 0.0004.\n\nTeams that chased the public split fell; validation-first teams "
        "rose.", top=2.5, size=14, width=3.8, left=8.8)

# ---- 10 RESULTS ----
s = new("Results", "The CV estimate was the truth")
stat(s, 0.9, 2.5, "0.87996", "out-of-fold forecast, made before the reveal", num_size=40)
stat(s, 5.2, 2.5, "0.880410", "private leaderboard result", accent=True, num_size=40)
stat(s, 9.6, 2.5, "0.0004", "forecast error", num_size=40)
stat(s, 0.9, 4.6, "21st → 4th", "public rank → private rank", num_size=40)
stat(s, 5.2, 4.6, "0.000075", "gap to 3rd place — about 3 patients of 36,000", num_size=40)

# ---- 11 PRINCIPLES ----
s = new("What we'd tell any team", "Four principles that decided this result")
prin = [("Identify the exact metric before modeling.", "One OOF threshold sweep on day one revealed weighted F1 and made the decision threshold a first-class model parameter."),
        ("Trust OOF CV; treat the public LB as a noisy instrument.", "Read it only through controlled, single-variable probes — never chase it."),
        ("Verify every gain adversarially.", "Honest split-half testing killed three seductive \"improvements\". A gain you can't reproduce on held-out selection isn't a gain."),
        ("Log everything — negative results compound.", "28 recorded experiments meant the final call was made on evidence. Rejected ideas were as valuable as accepted ones.")]
for k, (t_, x_) in enumerate(prin):
    l = 0.9 + (k % 2) * 6.1; t = 2.4 + (k // 2) * 2.2
    tf = tb(s, l, t, 5.6, 2.0)
    p = tf.paragraphs[0]; r = p.add_run(); r.text = f"{k+1}.  "
    f = r.font; f.name = "Georgia"; f.size = Pt(24); f.bold = True; f.color.rgb = TEAL_BRIGHT
    r = p.add_run(); r.text = t_
    f = r.font; f.name = "Calibri"; f.size = Pt(17); f.bold = True; f.color.rgb = INK
    p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = x_
    f2 = r2.font; f2.name = "Calibri"; f2.size = Pt(13); f2.color.rgb = MUTED

# ---- 12 THANKS ----
s = prs.slides.add_slide(BLANK); bg(s)
eyebrow(s, "Insight 2.0 · Chasing the Significance")
headline(s, "Thank you. Questions welcome.", size=48, top=1.6)
body(s, "Team NeurAps — every claim in this presentation is reproducible from our submitted notebook "
        "and experiment log.", top=3.0, size=18, width=10)
stat(s, 0.9, 4.5, "4", "days")
stat(s, 3.4, 4.5, "28", "experiments")
stat(s, 5.9, 4.5, "27", "submissions")
stat(s, 8.4, 4.5, "0.880410", "private weighted F1 · 4th place", accent=True)

prs.save("presentations/NeurAps_Insight2_Finale.pptx")
print("saved presentations/NeurAps_Insight2_Finale.pptx")

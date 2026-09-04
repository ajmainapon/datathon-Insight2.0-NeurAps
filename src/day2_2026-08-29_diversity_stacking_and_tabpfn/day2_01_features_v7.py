"""v7 feature set: v2 features + histologic type as categorical + interactions + count encodings."""
import sys
import os

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))              # src/ -> for common.features
from common.features import build_features as build_v2  # noqa: E402


def build_features_v7(df: pd.DataFrame, freq_maps: dict | None = None):
    """freq_maps: precomputed on train+test concat (unsupervised, no target -> no leakage)."""
    out = build_v2(df)

    hist = df["histologic_type_icdo3"].astype(str)
    out["hist_type_cat"] = hist.astype("category")
    out["site_stage"] = (df["primary_site"].astype(str) + "|"
                         + df["summary_stage"].astype(str)).astype("category")
    out["stage_treat"] = (df["summary_stage"].astype(str) + "|"
                          + df["reason_nocancer_directed_surgery"].astype(str)).astype("category")
    out["grade_stage"] = (df["grade_recode_thru2017"].astype(str) + "|"
                          + df["summary_stage"].astype(str)).astype("category")

    if freq_maps is not None:
        for col, mp in freq_maps.items():
            src = (df[col].astype(str) if col != "site_stage"
                   else df["primary_site"].astype(str) + "|" + df["summary_stage"].astype(str))
            out[f"{col}_freq"] = src.map(mp).astype(float)
    return out


def make_freq_maps(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    maps = {}
    for col in ["histologic_type_icdo3", "rx_summ_surgprim_site19982022"]:
        s = pd.concat([tr[col], te[col]]).astype(str)
        maps[col] = s.value_counts(normalize=True).to_dict()
    s = pd.concat([tr["primary_site"].astype(str) + "|" + tr["summary_stage"].astype(str),
                   te["primary_site"].astype(str) + "|" + te["summary_stage"].astype(str)])
    maps["site_stage"] = s.value_counts(normalize=True).to_dict()
    return maps

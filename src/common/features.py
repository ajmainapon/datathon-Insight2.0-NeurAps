"""Canonical feature-engineering function, shared by every training script.

Extracted verbatim from the script that first defined it (day1_02, formerly
train_v2.py) so every later experiment (day1-day3) builds features identically —
one definition, no drift between versions.
"""
import re

import numpy as np
import pandas as pd


NUMERIC_CODED = ["tumor_size_overtime", "tumor_size_summary", "cs_tumor_size20042015"]

T_ORD = {"T0": 0, "Tis": 0, "T1mi": 1, "T1a": 1, "T1b": 1.5, "T1c": 2, "T1": 1.5,
         "T2a": 2.5, "T2b": 3, "T2": 2.75, "T3": 4, "T4": 5, "T4a": 5, "T4b": 5, "T4c": 5}
N_ORD = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}
M_ORD = {"M0": 0, "M1a": 1, "M1b": 2, "M1": 2, "M1c": 3}
GRADE_ORD = {"Well differentiated; Grade I": 1, "Moderately differentiated; Grade II": 2,
             "Poorly differentiated; Grade III": 3, "Undifferentiated; anaplastic; Grade IV": 4}
STAGE_ORD = {"Localized": 1, "Regional": 2, "Distant": 3}


def parse_coded_numeric(s):
    x = pd.to_numeric(s, errors="coerce")
    x[x >= 990] = np.nan
    return x


def age_midpoint(s):
    def mid(v):
        m = re.match(r"(\d+)-(\d+)", str(v))
        if m:
            return (int(m.group(1)) + int(m.group(2))) / 2
        return 92.0 if "90+" in str(v) else np.nan
    return s.map(mid)


def build_features(df):
    out = df.drop(columns=["patient_id", "vital_status"], errors="ignore").copy()
    for c in NUMERIC_CODED:
        out[c + "_num"] = parse_coded_numeric(out[c])
        out = out.drop(columns=[c])
    out["tumor_size_best"] = (out["tumor_size_summary_num"]
                              .fillna(out["tumor_size_overtime_num"])
                              .fillna(out["cs_tumor_size20042015_num"]))
    out["size_missing_all"] = out[[c + "_num" for c in NUMERIC_CODED]].isna().all(axis=1).astype(int)

    out["age_num"] = age_midpoint(df["age_recode"])
    out["t_ord"] = df["derived_eod2018t_recode2018"].map(T_ORD)
    out["n_ord"] = df["derived_eod2018n_recode2018"].map(N_ORD)
    out["m_ord"] = df["derived_eod2018m_recode2018"].map(M_ORD)
    out["tnm_sum"] = out["t_ord"].fillna(2.5) + out["n_ord"].fillna(1.5) + out["m_ord"].fillna(1.5) * 2
    out["grade_ord"] = df["grade_recode_thru2017"].map(GRADE_ORD)
    out["stage_ord"] = df["summary_stage"].map(STAGE_ORD)

    mets = ["seer_combined_metsatdxbone2010", "seer_combined_metsatdxbrain2010",
            "seer_combined_metsatdxliver2010", "seer_combined_metsatdxlung2010"]
    out["mets_count"] = sum((df[c] == "Yes").astype(int) for c in mets)
    out["mets_known"] = sum(df[c].isin(["Yes", "No"]).astype(int) for c in mets)

    out["surgery_done"] = (df["reason_nocancer_directed_surgery"] == "Surgery performed").astype(int)
    out["death_cert_only"] = df["reason_nocancer_directed_surgery"].str.contains(
        "death certificate|died prior", case=False, na=False).astype(int)
    out["radiation_given"] = df["radiation_recode"].isin(
        ["Beam radiation", "Radioactive implants (includes brachytherapy) (1988+)",
         "Radioisotopes (1988+)", "Combination of beam with implants or isotopes",
         "Radiation, NOS  method or source not specified"]).astype(int)
    out["any_treatment"] = ((out["surgery_done"] + out["radiation_given"]) > 0).astype(int)

    out["nodes_ratio"] = np.where(df["regional_nodes_examined"] > 0,
                                  df["regional_nodes_positive"] / df["regional_nodes_examined"].clip(lower=1),
                                  np.nan)
    out["nodes_examined_any"] = (df["regional_nodes_examined"] > 0).astype(int)
    out["years_since_dx"] = 2024 - df["year_of_diagnosis"]
    out["multiple_primaries"] = (~df["sequence_number"].eq("One primary only")).astype(int)

    for c in out.columns:
        if pd.api.types.is_string_dtype(out[c]) or out[c].dtype == object:
            out[c] = out[c].astype("category")
    return out


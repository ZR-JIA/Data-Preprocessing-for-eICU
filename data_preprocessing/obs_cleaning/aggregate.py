
import pandas as pd
import numpy as np

# Columns that appear in eICU vitals tables but carry no physiological meaning.
# The config whitelist should already keep them out; this is a second guard so a
# future change to the loader cannot silently put a database row identifier into
# the feature matrix. vitalperiodicid is an auto-increment primary key, so its
# mean/std/slope encode record ordering and monitoring density, not patient
# state (AUDIT.md EICU-01).
NON_PHYSIOLOGICAL = frozenset({
    "vitalperiodicid",
    "vitalaperiodicid",
    "labid",
    "observationoffset",
    "labresultoffset",
})

# Feature aggregation strategy: keep temporal dynamics, stay interpretable.
VITAL_STATS = ["mean", "std"]  # paired with slope to express the trend
LAB_STATS = ["mean", "std"]    # level and variability, robust to single extremes

def _trend_slope(group: pd.DataFrame, value_col: str, time_col: str) -> float:
    g = group[[time_col, value_col]].dropna().sort_values(time_col)
    if len(g) < 2:
        return np.nan
    dt = g[time_col].iloc[-1] - g[time_col].iloc[0]
    if dt == 0:
        return np.nan
    dv = g[value_col].iloc[-1] - g[value_col].iloc[0]
    return float(dv / dt)

def aggregate_vitals(vital: pd.DataFrame, patient_key: str = "patientunitstayid") -> pd.DataFrame:
    if vital.empty:
        return pd.DataFrame()
    excluded = NON_PHYSIOLOGICAL | {patient_key}
    value_cols = [c for c in vital.columns if c not in excluded]
    value_cols = [c for c in value_cols if pd.api.types.is_numeric_dtype(vital[c])]
    if not value_cols:
        return pd.DataFrame()

    # Vitals: mean + std + slope, so instability over the window is visible.
    agg_funcs = {c: VITAL_STATS for c in value_cols}
    base = vital.groupby(patient_key).agg(agg_funcs)
    base.columns = [f"{v}_{s}" for v, s in base.columns]

    # Slope captures direction of change, which carries most of the mortality signal.
    slopes = {}
    for c in value_cols:
        slopes[f"{c}_slope"] = vital.groupby(patient_key).apply(
            lambda g: _trend_slope(g, value_col=c, time_col="observationoffset")
        )
    slopes = pd.DataFrame(slopes)

    return base.join(slopes)

def aggregate_labs(lab: pd.DataFrame, patient_key: str = "patientunitstayid") -> pd.DataFrame:
    if lab.empty:
        return pd.DataFrame()
    lab_w = lab.pivot_table(index=[patient_key, "labresultoffset"], columns="labname", values="labresult", aggfunc="mean")
    lab_w = lab_w.reset_index()
    excluded = NON_PHYSIOLOGICAL | {patient_key}
    value_cols = [c for c in lab_w.columns if c not in excluded]
    if not value_cols:
        return pd.DataFrame()

    # Labs: mean + std only; single outlying draws should not dominate.
    agg_funcs = {c: LAB_STATS for c in value_cols}
    base = lab_w.groupby(patient_key).agg(agg_funcs)
    base.columns = [f"lab_{v}_{s}" for v, s in base.columns]

    return base

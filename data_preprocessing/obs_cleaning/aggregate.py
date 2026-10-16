import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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

MIN_POINTS_FOR_SLOPE = 3


def _trend_slope(group: pd.DataFrame, value_col: str, time_col: str) -> float:
    """
    Least-squares slope of value against time.

    This was a two-point secant, (last - first) / dt, which ignored every
    sample in between and was maximally sensitive to two noisy endpoints -- on
    a 72h window of 5-minute vitals, roughly 860 readings informed nothing
    (AUDIT.md R-3). A regression over all points is both steadier and easier to
    defend as a "trend".
    """
    g = group[[time_col, value_col]].dropna()
    if len(g) < MIN_POINTS_FOR_SLOPE:
        return np.nan
    t = g[time_col].to_numpy(dtype=float)
    y = g[value_col].to_numpy(dtype=float)
    t_centred = t - t.mean()
    denom = float((t_centred ** 2).sum())
    if denom == 0.0:          # every reading at the same instant
        return np.nan
    return float((t_centred * (y - y.mean())).sum() / denom)


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


def harmonise_lab_units(lab: pd.DataFrame, unit_col: str = "labmeasurenamesystem"):
    """
    Keep one unit per analyte.

    eICU documents labname as hospital-specific and records the unit
    separately. Pivoting on labname alone averages, say, mg/dL and mmol/L
    readings of the same analyte into one number (AUDIT.md EICU-08). Rather
    than invent conversion factors, keep the unit each analyte is most often
    reported in and drop the rest, reporting what was dropped.
    """
    report = {"unit_column_present": unit_col in lab.columns}
    if not report["unit_column_present"]:
        logger.warning(
            "%s is absent, so lab units cannot be checked. Values recorded in "
            "different units will be aggregated together.", unit_col
        )
        return lab, report

    counts = lab.groupby(["labname", unit_col]).size()
    dominant = counts.groupby(level=0).idxmax().map(lambda ix: ix[1])
    mixed = {
        name: list(grp.index.get_level_values(1))
        for name, grp in counts.groupby(level=0) if len(grp) > 1
    }
    keep = lab[unit_col] == lab["labname"].map(dominant)

    report.update({
        "analytes": int(lab["labname"].nunique()),
        "analytes_with_mixed_units": mixed,
        "rows_before": len(lab),
        "rows_dropped": int((~keep).sum()),
    })
    if mixed:
        logger.warning("%d analyte(s) reported in more than one unit: %s",
                       len(mixed), mixed)
    return lab[keep].copy(), report


def aggregate_labs(lab: pd.DataFrame, patient_key: str = "patientunitstayid") -> pd.DataFrame:
    if lab.empty:
        return pd.DataFrame()
    lab_w = lab.pivot_table(index=[patient_key, "labresultoffset"], columns="labname",
                            values="labresult", aggfunc="mean")
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

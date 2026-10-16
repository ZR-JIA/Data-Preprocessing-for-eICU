
import pandas as pd
from typing import Dict, Tuple

def apply_outlier_nan(df: pd.DataFrame, limits: Dict[str, Tuple[float, float]]):
    """Replace values outside [lo, hi] with NaN for listed columns. Returns df and counts."""
    df = df.copy()
    counts = {}
    for col, lim in limits.items():
        if col not in df.columns:
            continue
        if not isinstance(lim, (list, tuple)) or len(lim) != 2:
            continue
        lo, hi = lim
        mask = (df[col] < lo) | (df[col] > hi)
        counts[col] = int(mask.sum())
        df.loc[mask, col] = pd.NA
    return df, counts


def apply_lab_outlier_nan(lab: pd.DataFrame, limits: Dict[str, Tuple[float, float]]):
    """
    Blank implausible lab results.

    The lab table is long -- one analyte per row, named in `labname` -- so the
    wide-format helper above does not reach it, and labs previously received no
    range checking at all (AUDIT.md EICU-08).
    """
    if lab.empty or "labname" not in lab.columns:
        return lab, {}
    lab = lab.copy()
    name = lab["labname"].astype("string").str.strip().str.lower()
    value = pd.to_numeric(lab["labresult"], errors="coerce")
    counts = {}
    for analyte, lim in limits.items():
        if not isinstance(lim, (list, tuple)) or len(lim) != 2:
            continue
        lo, hi = lim
        mask = (name == analyte.lower()) & ((value < lo) | (value > hi))
        n = int(mask.sum())
        if n:
            counts[analyte] = n
        lab.loc[mask, "labresult"] = pd.NA
    return lab, counts

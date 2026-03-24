
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


import pandas as pd

MIN_PER_HOUR = 60

def align_time_window(vital: pd.DataFrame, lab: pd.DataFrame, window_hours: int):
    """Keep records within [0, window_hours] minutes from ICU admit (offsets in minutes)."""
    upper = window_hours * MIN_PER_HOUR
    v = vital[(vital["observationoffset"] >= 0) & (vital["observationoffset"] <= upper)].copy() if not vital.empty else vital
    l = lab[(lab["labresultoffset"] >= 0) & (lab["labresultoffset"] <= upper)].copy() if not lab.empty else lab
    return v, l

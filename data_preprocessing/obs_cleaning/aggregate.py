
import pandas as pd
import numpy as np

# Feature aggregation strategy: 保留时间动态信息，提升可解释性
VITAL_STATS = ["mean", "std"]  # vital保留mean和std，配合slope反映变化趋势
LAB_STATS = ["mean", "std"]    # lab保留均值和变异，避免极值敏感

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
    value_cols = [c for c in vital.columns if c not in (patient_key, "observationoffset")]
    value_cols = [c for c in value_cols if pd.api.types.is_numeric_dtype(vital[c])]
    if not value_cols:
        return pd.DataFrame()

    # Vital特征: mean + std + slope (保留时间动态，反映不稳定病情)
    agg_funcs = {c: VITAL_STATS for c in value_cols}
    base = vital.groupby(patient_key).agg(agg_funcs)
    base.columns = [f"{v}_{s}" for v, s in base.columns]

    # 计算slope反映趋势变化（对预测死亡特别关键）
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
    value_cols = [c for c in lab_w.columns if c not in (patient_key, "labresultoffset")]
    if not value_cols:
        return pd.DataFrame()

    # Lab特征: mean + std (保留主要变异，避免极值干扰)
    agg_funcs = {c: LAB_STATS for c in value_cols}
    base = lab_w.groupby(patient_key).agg(agg_funcs)
    base.columns = [f"lab_{v}_{s}" for v, s in base.columns]

    return base

"""
Regression tests for AUDIT.md EICU-08 and R-3.

`_trend_slope` keeps its name and signature, so the slope tests run against
either version and show the difference directly.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data_preprocessing"))

from obs_cleaning.aggregate import _trend_slope


# --- R-3: slope over all samples, not just the endpoints ------------------

def test_slope_uses_every_sample_not_only_the_endpoints():
    """
    A series that rises steadily but happens to start and end at the same
    value. The two-point secant reports no trend at all; a regression over the
    whole window sees the rise.
    """
    g = pd.DataFrame({"t": [0, 1, 2, 3, 4, 5, 6],
                      "v": [5, 1, 2, 3, 4, 5, 5]})
    slope = _trend_slope(g, value_col="v", time_col="t")
    assert slope == pytest.approx(10 / 28, abs=1e-9), (
        f"expected the least-squares slope 0.357, got {slope} "
        "(0.0 means the endpoint-only secant is still in use)"
    )


def test_slope_of_a_flat_series_is_zero():
    g = pd.DataFrame({"t": [0, 1, 2, 3], "v": [7.0, 7.0, 7.0, 7.0]})
    assert _trend_slope(g, value_col="v", time_col="t") == pytest.approx(0.0)


def test_slope_of_a_clean_ramp_is_its_gradient():
    g = pd.DataFrame({"t": [0, 10, 20, 30], "v": [0.0, 5.0, 10.0, 15.0]})
    assert _trend_slope(g, value_col="v", time_col="t") == pytest.approx(0.5)


def test_slope_needs_enough_points():
    g = pd.DataFrame({"t": [0, 5], "v": [1.0, 9.0]})
    assert pd.isna(_trend_slope(g, value_col="v", time_col="t"))


def test_slope_is_undefined_when_all_readings_share_one_timestamp():
    g = pd.DataFrame({"t": [4, 4, 4], "v": [1.0, 2.0, 3.0]})
    assert pd.isna(_trend_slope(g, value_col="v", time_col="t"))


# --- EICU-08: labs get range checks and one unit per analyte ---------------

def _labs():
    return pd.DataFrame({
        "patientunitstayid": [1, 1, 1, 2, 2],
        "labname":   ["glucose", "glucose", "sodium", "glucose", "sodium"],
        "labresult": [120.0, 99999.0, 140.0, 95.0, 141.0],
        "labresultoffset": [10, 20, 30, 10, 20],
        "labmeasurenamesystem": ["mg/dL", "mg/dL", "mmol/L", "mg/dL", "mmol/L"],
    })


def test_implausible_lab_values_are_blanked():
    from obs_cleaning.outliers import apply_lab_outlier_nan
    out, counts = apply_lab_outlier_nan(_labs(), {"glucose": (10, 2000)})
    assert counts == {"glucose": 1}
    assert pd.isna(out.loc[1, "labresult"])
    assert out.loc[0, "labresult"] == 120.0


def test_analytes_not_listed_are_left_alone():
    from obs_cleaning.outliers import apply_lab_outlier_nan
    out, counts = apply_lab_outlier_nan(_labs(), {"potassium": (1, 10)})
    assert counts == {}
    assert out["labresult"].notna().all()


def test_mixed_units_are_reduced_to_the_dominant_one():
    from obs_cleaning.aggregate import harmonise_lab_units
    lab = _labs()
    lab.loc[len(lab)] = [3, "glucose", 7.0, 15, "mmol/L"]   # minority unit
    out, report = harmonise_lab_units(lab)
    assert report["analytes_with_mixed_units"] == {"glucose": ["mg/dL", "mmol/L"]}
    assert report["rows_dropped"] == 1
    assert set(out.loc[out["labname"] == "glucose", "labmeasurenamesystem"]) == {"mg/dL"}


def test_missing_unit_column_is_reported_not_fatal():
    from obs_cleaning.aggregate import harmonise_lab_units
    out, report = harmonise_lab_units(_labs().drop(columns=["labmeasurenamesystem"]))
    assert report["unit_column_present"] is False
    assert len(out) == 5

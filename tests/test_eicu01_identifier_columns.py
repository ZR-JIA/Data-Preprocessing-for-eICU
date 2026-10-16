"""
Regression tests for AUDIT.md EICU-01 / EICU-05 / EICU-06.

Before the fix, load_data() read every column of every raw CSV and
aggregate_vitals() aggregated every numeric column it found. The vitalPeriodic
primary key `vitalperiodicid` therefore became three model features
(mean/std/slope) describing database row ordering rather than patient state.

These tests pin the two independent guards that now prevent that:
  1. the config whitelist is enforced at load time, and
  2. aggregate_vitals refuses known non-physiological columns even if they
     somehow reach it.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data_preprocessing"))

from obs_cleaning.aggregate import aggregate_vitals
from obs_cleaning.load_data import load_data  # noqa: E402


def _write_vitals(tmp_path: Path) -> Path:
    """Minimal raw-data directory shaped like the real eICU extract."""
    pd.DataFrame({
        "vitalperiodicid": [9001, 9002, 9003, 9004],
        "patientunitstayid": [1, 1, 2, 2],
        "observationoffset": [10, 70, 10, 70],
        "heartrate": [80, 92, 101, 97],
        "sao2": [98, 96, 95, 94],
    }).to_csv(tmp_path / "vitalPeriodic.csv", index=False)
    for name, frame in {
        "patient": pd.DataFrame({"patientunitstayid": [1, 2], "age": ["64", "71"]}),
        "diagnosis": pd.DataFrame({"patientunitstayid": [1], "icd9code": ["434.91"]}),
        "lab": pd.DataFrame({"patientunitstayid": [1], "labname": ["glucose"],
                             "labresult": [5.1], "labresultoffset": [30]}),
    }.items():
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    return tmp_path


def test_aggregate_vitals_excludes_the_row_identifier():
    vital = pd.DataFrame({
        "vitalperiodicid": [9001, 9002, 9003, 9004],
        "patientunitstayid": [1, 1, 2, 2],
        "observationoffset": [10, 70, 10, 70],
        "heartrate": [80, 92, 101, 97],
    })
    out = aggregate_vitals(vital)

    leaked = [c for c in out.columns if c.startswith("vitalperiodicid")]
    assert leaked == [], f"database identifier reached the feature matrix: {leaked}"
    assert {"heartrate_mean", "heartrate_std", "heartrate_slope"} <= set(out.columns)


def test_offsets_are_not_aggregated_as_measurements():
    """observationoffset is time, not a vital sign; it must not become a feature."""
    vital = pd.DataFrame({
        "patientunitstayid": [1, 1],
        "observationoffset": [10, 70],
        "heartrate": [80, 92],
    })
    out = aggregate_vitals(vital)
    assert not [c for c in out.columns if c.startswith("observationoffset")]


def test_whitelist_keeps_the_identifier_out_of_memory(tmp_path):
    data_dir = _write_vitals(tmp_path)
    tables = {
        "patient": ["patientunitstayid", "age"],
        "diagnosis": ["patientunitstayid", "icd9code"],
        "vitalPeriodic": ["patientunitstayid", "observationoffset", "heartrate", "sao2"],
        "lab": ["patientunitstayid", "labname", "labresult", "labresultoffset"],
    }
    _, _, vital, _ = load_data(data_dir, tables=tables)
    assert "vitalperiodicid" not in vital.columns


def test_unknown_column_fails_loudly_instead_of_being_skipped(tmp_path):
    """
    'map' and 'spo2' do not exist in eICU vitalPeriodic (they are 'systemicmean'
    and 'sao2'). The old pipeline skipped such names in silence, so their outlier
    rules never ran. A wrong name must now raise. (AUDIT.md EICU-06)
    """
    data_dir = _write_vitals(tmp_path)
    tables = {"vitalPeriodic": ["patientunitstayid", "map", "spo2"]}
    with pytest.raises(KeyError, match="map"):
        load_data(data_dir, tables=tables)

"""
Regression tests for AUDIT.md EICU-03, EICU-04 and EICU-07.

EICU-07's two predicates (`_icd9_hit`, `_kw_hit`) keep the same names and
signatures as before the fix, so those tests run against either version and
show the behavioural difference directly.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data_preprocessing"))

from obs_cleaning.cohort import _icd9_hit, _kw_hit


# --- EICU-07: cohort definition -------------------------------------------

@pytest.mark.parametrize("code", ["433.01", "43401", "434.11", "430.0", "431", "436"])
def test_acute_cerebrovascular_codes_are_selected(code):
    assert _icd9_hit(code), f"{code} should qualify"


@pytest.mark.parametrize("code", ["433.00", "43400", "434.10"])
def test_occlusion_without_infarction_is_rejected(code):
    """x.x0 is chronic stenosis, not a stroke."""
    assert not _icd9_hit(code), f"{code} has no infarction and must not qualify"


@pytest.mark.parametrize("code", ["437.0", "437.3", "4377"])
def test_ill_defined_cerebrovascular_disease_is_rejected(code):
    """437 covers atherosclerosis, unruptured aneurysm, transient amnesia."""
    assert not _icd9_hit(code), f"{code} is not an acute stroke"


@pytest.mark.parametrize("text", [
    "heat stroke", "heatstroke", "Heat Stroke with rhabdomyolysis", "sun stroke",
])
def test_heat_stroke_is_not_a_cerebrovascular_event(text):
    assert not _kw_hit(text), f"{text!r} is not a stroke"


@pytest.mark.parametrize("text", [
    "stroke", "acute CVA", "ICH with midline shift", "SAH, aneurysmal",
])
def test_real_stroke_text_still_matches(text):
    assert _kw_hit(text), f"{text!r} should qualify"


# --- EICU-03: label construction ------------------------------------------

def _patient(statuses):
    return pd.DataFrame({
        "patientunitstayid": range(len(statuses)),
        "hospitaldischargestatus": statuses,
    })


def test_missing_discharge_status_is_dropped_not_called_survival():
    from obs_cleaning.cohort import extract_mortality_label
    df, report = extract_mortality_label(
        _patient(["Alive", "Expired", None, "", "Unknown"])
    )
    assert report["dropped_unusable_status"] == 3
    assert len(df) == 2
    assert df["mortality"].tolist() == [0, 1]


def test_label_report_carries_prevalence():
    from obs_cleaning.cohort import extract_mortality_label
    _, report = extract_mortality_label(_patient(["Alive"] * 9 + ["Expired"]))
    assert report["deaths"] == 1
    assert report["prevalence"] == 0.1


# --- EICU-04: observation window ------------------------------------------

def test_stays_shorter_than_the_window_are_excluded():
    from obs_cleaning.cohort import apply_observation_window
    patient = pd.DataFrame({
        "patientunitstayid": [1, 2, 3, 4],
        "unitdischargeoffset": [360, 4320, 10000, 4319],   # 6h, 72h, ~7d, 71.98h
    })
    kept, report = apply_observation_window(patient, window_minutes=72 * 60)
    assert kept["patientunitstayid"].tolist() == [2, 3]
    assert report["stays_shorter_than_window"] == 2


def test_window_filter_requires_the_length_column():
    from obs_cleaning.cohort import apply_observation_window
    with pytest.raises(KeyError, match="unitdischargeoffset"):
        apply_observation_window(pd.DataFrame({"patientunitstayid": [1]}), 4320)


def test_late_diagnoses_do_not_define_the_cohort():
    from obs_cleaning.cohort import select_cohort
    patient = pd.DataFrame({"patientunitstayid": [1, 2]})
    diagnosis = pd.DataFrame({
        "patientunitstayid": [1, 2],
        "icd9code": ["434.11", "434.11"],
        "diagnosisstring": ["stroke", "stroke"],
        "diagnosisoffset": [120, 13000],       # 2h vs ~9 days
    })
    cohort, _ = select_cohort(patient, diagnosis, window_minutes=72 * 60)
    assert cohort["patientunitstayid"].tolist() == [1]

"""
Regression tests for AUDIT.md EICU-02.

The pipeline advertised a "patient-level stratified split", but keyed it on
patientunitstayid -- one ICU *stay*. eICU assigns one uniquepid several stays
when a patient is admitted more than once, so the same person could appear in
train and test at once. The grouping key also included site_id, which split a
patient admitted to two hospitals into two groups and defeated the grouping.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data_preprocessing"))

from obs_trainReady.data_splitting import PatientLevelSplitter


def _cohort(n_patients: int = 120) -> pd.DataFrame:
    """Every patient has two ICU stays, each at a different hospital."""
    rows = []
    for i in range(n_patients):
        pid = f"P{i:04d}"
        label = i % 7 == 0           # ~14% mortality, as in the paper
        for stay, site in enumerate(("H1", "H2")):
            rows.append({
                "patient_id": pid,
                "patientunitstayid": f"{i:04d}{stay}",
                "site_id": site,
                "mortality": int(label),
                "heartrate_mean": 80.0 + i % 23,
            })
    return pd.DataFrame(rows)


def _split_membership(splits):
    """patient_id -> set of split names it appears in."""
    seen = {}
    for name, frame in splits.items():
        for pid in frame["patient_id"].unique():
            seen.setdefault(pid, set()).add(name)
    return seen


def test_no_patient_appears_in_two_splits():
    splits = PatientLevelSplitter(random_state=42).split(_cohort())
    straddling = {p: s for p, s in _split_membership(splits).items() if len(s) > 1}
    assert not straddling, (
        f"{len(straddling)} patients appear in more than one split, "
        f"e.g. {dict(list(straddling.items())[:3])}"
    )


def test_every_stay_is_kept():
    """Grouping must not silently drop rows."""
    df = _cohort()
    splits = PatientLevelSplitter(random_state=42).split(df)
    assert sum(len(f) for f in splits.values()) == len(df)


def test_splitter_does_not_mutate_the_caller_frame():
    df = _cohort()
    before = list(df.columns)
    PatientLevelSplitter(random_state=42).split(df)
    assert list(df.columns) == before, "_group_key was left on the caller's frame"


def test_default_grouping_is_the_person_not_the_site():
    assert PatientLevelSplitter().group_cols == ["patient_id"]

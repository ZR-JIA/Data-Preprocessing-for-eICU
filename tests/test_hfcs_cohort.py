"""Tests for the HF-CS cohort definition, treatment flags and APACHE baseline."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_preprocessing"))

from obs_cleaning.cohort_hfcs import (  # noqa: E402
    apache_iv_prediction, filter_adults, keep_first_stay, parse_age, select_hfcs_cohort)
from obs_cleaning.treatments import treatment_flags  # noqa: E402

CS = ("cardiovascular|shock / hypotension|shock|cardiogenic", "785.51, R57.0")
HF = ("cardiovascular|ventricular disorders|congestive heart failure", "428.0, I50.9")
AMI = ("cardiovascular|chest pain / ASHD|acute coronary syndrome|acute myocardial "
       "infarction (no ST elevation)", "410.71, I21.4")


def _patient(ids):
    return pd.DataFrame({"patientunitstayid": ids, "uniquepid": [f"p{i}" for i in ids],
                         "age": ["60"] * len(ids)})


def _dx(rows):
    return pd.DataFrame(rows, columns=["patientunitstayid", "diagnosisstring",
                                       "icd9code", "diagnosisoffset"])


def test_parse_age_maps_over_89_and_rejects_blank():
    out = parse_age(pd.Series(["> 89", "45", "", None]))
    assert out.iloc[0] == 90 and out.iloc[1] == 45
    assert out.iloc[2:].isna().all()


def test_filter_adults_drops_minors_and_unknown_age():
    p = pd.DataFrame({"patientunitstayid": [1, 2, 3], "age": ["17", "18", ""]})
    kept, report = filter_adults(p)
    assert kept["patientunitstayid"].tolist() == [2]
    assert report["under_min_age"] == 1 and report["age_missing"] == 1


def test_hfcs_requires_both_shock_and_heart_failure():
    dx = _dx([(1, *CS, 10), (1, *HF, 10), (2, *CS, 10), (3, *HF, 10)])
    kept, _ = select_hfcs_cohort(_patient([1, 2, 3]), dx, window_minutes=1440)
    assert kept["patientunitstayid"].tolist() == [1]


def test_icd_codes_alone_qualify():
    dx = _dx([(1, "unrelated text", "785.51", 10), (1, "unrelated text", "I50.9", 10)])
    kept, _ = select_hfcs_cohort(_patient([1]), dx, window_minutes=1440)
    assert kept["patientunitstayid"].tolist() == [1]


def test_shock_recorded_after_window_does_not_qualify():
    dx = _dx([(1, *CS, 3000), (1, *HF, 10)])
    kept, _ = select_hfcs_cohort(_patient([1]), dx, window_minutes=1440)
    assert kept.empty


def test_ami_anywhere_in_stay_excludes():
    dx = _dx([(1, *CS, 10), (1, *HF, 10), (1, *AMI, 5000)])
    kept, report = select_hfcs_cohort(_patient([1]), dx, window_minutes=1440)
    assert kept.empty and report["excluded_ami"] == 1


def test_operative_admission_excluded_and_admissiondx_adds_shock():
    dx = _dx([(1, *HF, 10), (2, *HF, 10)])
    adx = pd.DataFrame({"patientunitstayid": [1, 2], "admitdxpath": [
        "admission diagnosis|All Diagnosis|Non-operative|Diagnosis|Cardiovascular|Shock, cardiogenic",
        "admission diagnosis|All Diagnosis|Operative|Diagnosis|Cardiovascular|Shock, cardiogenic"]})
    kept, report = select_hfcs_cohort(_patient([1, 2]), dx, 1440, admissiondx=adx)
    assert kept["patientunitstayid"].tolist() == [1]
    assert report["excluded_operative_admission"] == 1


def test_report_counts_only_stays_in_play():
    dx = _dx([(1, *CS, 10), (1, *HF, 10), (9, *CS, 10), (9, *HF, 10)])
    kept, report = select_hfcs_cohort(_patient([1]), dx, window_minutes=1440)
    assert report["stays_selected"] == len(kept) == 1


def test_keep_first_stay_orders_by_year_then_stay():
    p = pd.DataFrame({"patientunitstayid": [11, 10, 20], "uniquepid": ["a", "a", "b"],
                      "hospitaldischargeyear": [2015, 2014, 2014],
                      "patienthealthsystemstayid": [2, 1, 3], "unitvisitnumber": [1, 1, 1]})
    kept, report = keep_first_stay(p)
    assert sorted(kept["patientunitstayid"]) == [10, 20]
    assert report["repeat_stays_dropped"] == 1


def test_apache_prefers_iva_and_falls_back_to_iv():
    a = pd.DataFrame({"patientunitstayid": [1, 1, 2, 2, 3],
                      "apacheversion": ["IV", "IVa", "IV", "IVa", "IV"],
                      "predictedhospitalmortality": [0.3, -1, 0.2, 0.4, -1]})
    s = apache_iv_prediction(a)
    assert s[1] == 0.3 and s[2] == 0.4 and 3 not in s.index


def test_treatment_flags_window_and_agents():
    inf = pd.DataFrame({"patientunitstayid": [1, 1, 2], "infusionoffset": [100, 200, 3000],
                        "drugname": ["Norepinephrine (mcg/min)", "Dobutamine (mcg/kg/min)",
                                     "Norepinephrine (mcg/min)"]})
    trt = pd.DataFrame({"patientunitstayid": [1, 3], "treatmentoffset": [-30, 50],
                        "treatmentstring": [
                            "pulmonary|ventilation and oxygenation|mechanical ventilation",
                            "cardiovascular|non-operative procedures|intraaortic balloon pump"]})
    f = treatment_flags([1, 2, 3], 1440, infusion=inf, treatment=trt)
    assert f.loc[1].tolist() == [1, 1, 2, 1, 0]
    assert f.loc[2].sum() == 0          # infusion after the window
    assert f.loc[3, "mcs_24h"] == 1


def test_epinephrine_pattern_does_not_match_norepinephrine():
    inf = pd.DataFrame({"patientunitstayid": [1], "infusionoffset": [0],
                        "drugname": ["Norepinephrine (mcg/min)"]})
    f = treatment_flags([1], 1440, infusion=inf)
    assert f.loc[1, "n_vasoactive_24h"] == 1


def test_chunked_loader_matches_full_filter(tmp_path):
    from obs_cleaning.load_data import load_table_for_stays
    df = pd.DataFrame({"patientunitstayid": np.repeat([1, 2, 3], 5),
                       "observationoffset": np.tile([-10, 0, 700, 1440, 2000], 3),
                       "heartrate": np.arange(15, dtype=float)})
    df.to_csv(tmp_path / "vitalPeriodic.csv.gz", index=False)
    got = load_table_for_stays(tmp_path, "vitalPeriodic", list(df.columns), {1, 3},
                               "observationoffset", 1440, chunksize=4)
    want = df[df["patientunitstayid"].isin({1, 3})
              & df["observationoffset"].between(0, 1440)].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want)

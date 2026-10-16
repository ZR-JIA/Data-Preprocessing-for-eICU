"""
Tests for AUDIT.md EICU-09, the missing-data treatment on this branch.

These cover behaviour that does not exist on `fix`: indicators are an addition,
not a correction, and they change the feature count on purpose.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data_preprocessing"))

from obs_trainReady.missingness import MissingnessIndicators, missingness_report


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "always_there": rng.normal(size=n),
        "sometimes":    rng.normal(size=n),
        "rarely":       rng.normal(size=n),
        "almost_never": rng.normal(size=n),
    })
    df.loc[rng.choice(n, size=int(0.30 * n), replace=False), "sometimes"] = np.nan
    df.loc[rng.choice(n, size=2, replace=False), "rarely"] = np.nan
    df.loc[rng.choice(n, size=int(0.99 * n), replace=False), "almost_never"] = np.nan
    return df


def test_only_informative_columns_get_an_indicator():
    """
    Nothing to learn from a column that is never missing, and an indicator for
    a column that is almost always missing is near-constant.
    """
    ind = MissingnessIndicators(min_rate=0.01, max_rate=0.95).fit(_frame())
    assert "sometimes" in ind.indicator_cols
    assert "always_there" not in ind.indicator_cols
    assert "almost_never" not in ind.indicator_cols


def test_indicator_records_absence_before_imputation():
    df = _frame()
    flags = MissingnessIndicators().fit_transform(df)
    assert flags["sometimes_missing"].tolist() == df["sometimes"].isna().astype(int).tolist()
    assert set(flags["sometimes_missing"].unique()) <= {0, 1}


def test_which_columns_qualify_is_decided_on_train_alone():
    """Val and test must not be able to add a feature."""
    train = _frame(seed=1)
    train["only_missing_later"] = 1.0
    test = train.copy()
    test["only_missing_later"] = np.nan

    ind = MissingnessIndicators().fit(train)
    assert "only_missing_later" not in ind.indicator_cols
    assert "only_missing_later_missing" not in ind.transform(test).columns


def test_transform_before_fit_is_refused():
    with pytest.raises(ValueError, match="not fitted"):
        MissingnessIndicators().transform(_frame())


def test_rate_bounds_are_validated():
    with pytest.raises(ValueError):
        MissingnessIndicators(min_rate=0.9, max_rate=0.1)


# --- the stratified diagnostic AUDIT.md asks for --------------------------

def test_report_separates_a_hospital_effect_from_a_uniform_gap():
    """
    One analyte is simply never reported by one hospital; another is missing at
    the same rate everywhere. The report has to tell them apart.
    """
    n = 300
    df = pd.DataFrame({
        "site_id": (["H1"] * (n // 2)) + (["H2"] * (n // 2)),
        "mortality": [0, 1] * (n // 2),
        "hospital_specific": 1.0,
        "uniformly_sparse": 1.0,
    })
    df.loc[df["site_id"] == "H2", "hospital_specific"] = np.nan
    df.loc[df.index % 5 == 0, "uniformly_sparse"] = np.nan

    table = missingness_report(df, ["hospital_specific", "uniformly_sparse"],
                               by=["site_id"])
    assert table.loc["hospital_specific", "spread"] == pytest.approx(1.0)
    assert table.loc["uniformly_sparse", "spread"] == pytest.approx(0.0, abs=0.05)
    # Sorted by spread, so the hospital-driven gap is the first thing read.
    assert table.index[0] == "hospital_specific"


def test_report_without_grouping_is_just_the_overall_rate():
    table = missingness_report(_frame(), ["sometimes", "always_there"])
    assert list(table.columns) == ["overall"]
    assert table.loc["always_there", "overall"] == 0.0


# --- end to end: the feature count departs from the paper on purpose ------

def _run(tmp_path, indicators: bool):
    from run_obs_trainReady import ObsTrainReadyPipeline
    rng = np.random.default_rng(0)
    n = 160
    df = pd.DataFrame({
        "patientunitstayid": range(n),
        "uniquepid": [f"U{i:05d}" for i in range(n)],
        "age": [str(40 + i % 50) for i in range(n)],
        "gender": [["Male", "Female"][i % 2] for i in range(n)],
        "ethnicity": [["Caucasian", "Hispanic"][i % 2] for i in range(n)],
        "site_id": [f"H{i % 3}" for i in range(n)],
        "hospitaldischargeyear": [2014 + i % 2 for i in range(n)],
        "mortality": [int(i % 7 == 0) for i in range(n)],
        "heartrate_mean": rng.normal(80, 10, n),
        "lab_glucose_mean": rng.normal(100, 20, n),
    })
    df.loc[rng.choice(n, size=int(0.3 * n), replace=False), "lab_glucose_mean"] = np.nan
    cleaned = tmp_path / "obs_cleaned.csv"
    df.to_csv(cleaned, index=False)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "paths": {"input_cleaned": str(cleaned),
                  "output_train_ready": str(tmp_path / f"out_{indicators}")},
        "random_seed": 42, "n_seeds": 5,
        "missingness": {"indicators": indicators, "min_rate": 0.01, "max_rate": 0.95},
    }))
    p = ObsTrainReadyPipeline(str(cfg))
    p.run()
    return p


def test_indicators_add_features_and_can_be_switched_off(tmp_path):
    with_ind = _run(tmp_path, indicators=True)
    without = _run(tmp_path, indicators=False)
    assert with_ind._fitted["indicator_cols"] == ["lab_glucose_mean_missing"]
    assert without._fitted["indicator_cols"] == []

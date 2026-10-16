"""Regression tests for AUDIT.md EICU-10, R-4 and R-5."""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data_preprocessing"))

from obs_trainReady.data_splitting import SiteHoldoutSplitter
from obs_trainReady.train_ready_exporter import TrainReadyExporter


# --- R-4: split membership is stored as labels, so read it with .loc -------

def test_split_ids_survive_a_non_trivial_index(tmp_path):
    """
    The exporter read label-based indices with .iloc, which agreed with .loc
    only while the frame carried a clean RangeIndex. Give it a frame whose
    labels are not positions and the two disagree.
    """
    patient_ids = pd.Series(["pA", "pB", "pC", "pD"], index=[10, 11, 12, 13])
    exporter = TrainReadyExporter(base_dir=str(tmp_path))
    exporter.save_split_indices(
        protocol="patient_stratified",
        split_indices={"train": [10, 12], "test": [11, 13]},
        patient_ids=patient_ids,
    )
    train = pd.read_csv(tmp_path / "SPLITS" / "patient_stratified" / "train_ids.csv")
    assert train["patient_id"].tolist() == ["pA", "pC"]


# --- R-5: site holdout groups by patient and keeps to its budget -----------

def _multi_site_cohort(n_per_site=40, n_sites=6):
    rows = []
    for s in range(n_sites):
        for i in range(n_per_site):
            pid = f"P{s:02d}{i:03d}"
            for stay in range(2):          # every patient has two stays
                rows.append({
                    "patient_id": pid,
                    "site_id": f"H{s}",
                    "mortality": int(i % 7 == 0),
                    "heartrate_mean": 70.0 + i % 19,
                })
    return pd.DataFrame(rows)


def test_site_holdout_keeps_a_patient_in_one_split():
    splits = SiteHoldoutSplitter(random_state=42).split(_multi_site_cohort())
    seen = {}
    for name, frame in splits.items():
        if len(frame) == 0:
            continue
        for pid in frame["patient_id"].unique():
            seen.setdefault(pid, set()).add(name)
    straddling = {p: s for p, s in seen.items() if len(s) > 1}
    assert not straddling, f"{len(straddling)} patients span two splits"


def test_site_holdout_does_not_overshoot_by_a_whole_hospital():
    df = _multi_site_cohort()
    splits = SiteHoldoutSplitter(test_size=0.2, random_state=42).split(df)
    share = len(splits["test"]) / len(df)
    assert 0.10 <= share <= 0.30, f"test share {share:.2%} is far from the 20% budget"


def test_test_sites_never_appear_in_training():
    splitter = SiteHoldoutSplitter(random_state=42)
    splits = splitter.split(_multi_site_cohort())
    held_out = set(splitter.selected_test_sites)
    assert not held_out & set(splits["train"]["site_id"].unique())


# --- EICU-10: three categorical features, reserved <UNK> ------------------

def _pipeline(tmp_path):
    import yaml
    from run_obs_trainReady import ObsTrainReadyPipeline
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "paths": {"input_cleaned": "x.csv", "output_train_ready": str(tmp_path / "out")},
        "random_seed": 42, "n_seeds": 5,
    }))
    return ObsTrainReadyPipeline(str(cfg))


def _frame(ethnicities):
    n = len(ethnicities)
    return pd.DataFrame({
        "patient_id": [f"p{i}" for i in range(n)],
        "age": ["60"] * n,
        "gender": ["Male"] * n,
        "ethnicity": ethnicities,
        "site_id": ["H1"] * n,
        "mortality": [i % 2 for i in range(n)],
        "heartrate_mean": [80.0 + i for i in range(n)],
    })


def test_site_id_is_not_a_model_feature(tmp_path):
    """The paper's left tower has three features; a fourth cannot give R^22."""
    p = _pipeline(tmp_path)
    train, _, _ = p._dual_tower_preprocess(
        _frame(["Caucasian", "African American"] * 2),
        _frame(["Caucasian", "African American"]),
        _frame(["Caucasian", "African American"]),
    )
    assert "site_id" not in train.columns
    assert p._fitted["cat_cols"] == ["age", "gender", "ethnicity"]


def test_unseen_category_gets_its_own_index(tmp_path):
    """
    An ethnicity absent from training used to become classes_[0], a real
    category. It must land on a reserved index instead.
    """
    from run_obs_trainReady import UNKNOWN_CATEGORY
    p = _pipeline(tmp_path)
    train, _, test = p._dual_tower_preprocess(
        _frame(["Caucasian", "African American"] * 2),
        _frame(["Caucasian", "Caucasian"]),
        _frame(["Asian", "Caucasian"]),          # Asian never seen in train
    )
    le = p._fitted["encoders"]["ethnicity"]
    unknown_idx = int(le.transform([UNKNOWN_CATEGORY])[0])
    assert test["ethnicity"].iloc[0] == unknown_idx
    assert unknown_idx not in set(train["ethnicity"])

"""
Regression tests for AUDIT.md EICU-13 and EICU-14.

These run the stage-2 pipeline end to end on a synthetic obs_cleaned.csv, so
they exercise the real entry point rather than a reconstruction of it.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data_preprocessing"))

from run_obs_trainReady import ObsTrainReadyPipeline


def _cleaned_csv(path: Path, n_patients=160, n_sites=4):
    rows = []
    for i in range(n_patients):
        rows.append({
            "patientunitstayid": 100000 + i,
            "uniquepid": f"U{i:05d}",
            "age": str(40 + i % 50),
            "gender": ["Male", "Female"][i % 2],
            "ethnicity": ["Caucasian", "African American", "Hispanic"][i % 3],
            "site_id": f"H{i % n_sites}",
            "hospitaldischargeyear": 2014 + (i % 2),
            "mortality": int(i % 7 == 0),
            "heartrate_mean": 70.0 + i % 30,
            "heartrate_slope": (i % 11) * 0.01,
            "lab_glucose_mean": 90.0 + i % 40,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _run(tmp_path, n_seeds=5):
    cleaned = tmp_path / "obs_cleaned.csv"
    _cleaned_csv(cleaned)
    out = tmp_path / "train_ready"
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "paths": {"input_cleaned": str(cleaned), "output_train_ready": str(out)},
        "random_seed": 42, "n_seeds": n_seeds,
    }))
    ObsTrainReadyPipeline(str(cfg)).run()
    return out


def test_every_configured_seed_produces_a_split(tmp_path):
    """run() used to hardcode seed=42, so n_seeds had no effect."""
    out = _run(tmp_path, n_seeds=6)
    seed_dirs = sorted(d.name for d in (out / "PACKS" / "patient_stratified").iterdir()
                       if d.is_dir())
    assert len(seed_dirs) == 6, f"expected 6 seeds, found {seed_dirs}"
    for d in seed_dirs:
        for split in ("train", "val", "test"):
            assert (out / "PACKS" / "patient_stratified" / d / f"{split}.csv").exists()


def test_different_seeds_give_different_splits(tmp_path):
    out = _run(tmp_path, n_seeds=5)
    packs = sorted(d for d in (out / "PACKS" / "patient_stratified").iterdir() if d.is_dir())
    a = pd.read_csv(packs[0] / "test.csv")["patient_id"]
    b = pd.read_csv(packs[1] / "test.csv")["patient_id"]
    assert set(a) != set(b), "two seeds produced an identical test split"


def test_fitted_transforms_are_persisted(tmp_path):
    """Encoders, imputer and scaler used to be discarded when the call returned."""
    import pickle
    out = _run(tmp_path)
    pkl = next((out / "PACKS" / "patient_stratified").rglob("transforms.pkl"))
    with open(pkl, "rb") as f:
        fitted = pickle.load(f)
    # Subset, not equality: a branch may persist more than this (mmda_enhance
    # adds the fitted missingness indicators), but never less.
    assert {"encoders", "imputer", "scaler", "cat_cols", "num_cols"} <= set(fitted)
    assert fitted["cat_cols"] == ["age", "gender", "ethnicity"]
    assert hasattr(fitted["scaler"], "mean_")


def test_metadata_directories_are_no_longer_empty(tmp_path):
    """DATA/ and REPROD/ were created by the exporter and never written to."""
    out = _run(tmp_path, n_seeds=5)
    assert (out / "DATA" / "feature_list.json").exists()
    assert (out / "DATA" / "data_version.json").exists()
    seeds = (out / "REPROD" / "seeds.txt").read_text().split()
    assert len(seeds) == 5


def test_temporal_protocol_actually_runs(tmp_path):
    """
    Stage 1 dropped the only time column, so TemporalHoldoutSplitter could
    never execute and PACKS/temporal stayed permanently empty.
    """
    out = _run(tmp_path)
    assert list((out / "PACKS" / "temporal").rglob("train.csv")), \
        "temporal protocol produced nothing"


def test_site_protocol_actually_runs(tmp_path):
    out = _run(tmp_path)
    assert list((out / "PACKS" / "site_holdout").rglob("train.csv"))

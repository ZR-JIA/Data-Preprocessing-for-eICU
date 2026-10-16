#!/usr/bin/env python3
"""
Simple data cleaning pipeline for stroke patients.
7 steps: load -> cohort -> timealign -> outliers -> aggregate -> impute -> save
"""
import argparse
import yaml
from pathlib import Path
from obs_cleaning import (
    load_data,
    select_cohort,
    align_time_window,
    apply_outlier_nan,
    aggregate_vitals,
    aggregate_labs,
    drop_sparse_and_impute,
    save_data,
    extract_mortality_label
)

# Fallback column whitelist and QC limits for the ad-hoc CLI path (no --config).
# They mirror config/obs_cleaning.yaml: both entry points must agree, because
# previously they named different vital columns and only one of the two sets
# ever took effect (AUDIT.md EICU-05, EICU-06). No entry point may read a raw
# table unrestricted -- that is what admitted vitalperiodicid into the feature
# matrix (AUDIT.md EICU-01).
DEFAULT_TABLES = {
    "patient": [
        "patientunitstayid", "age", "gender", "ethnicity", "hospitalid",
        "admissionheight", "admissionweight", "hospitaldischargestatus",
        "unitadmitoffset",
    ],
    "diagnosis": ["patientunitstayid", "icd9code", "diagnosisstring"],
    "vitalPeriodic": [
        "patientunitstayid", "observationoffset", "heartrate", "respiration",
        "sao2", "temperature", "systemicsystolic", "systemicdiastolic",
        "systemicmean",
    ],
    "lab": ["patientunitstayid", "labname", "labresult", "labresultoffset"],
}

DEFAULT_OUTLIER_LIMITS = {
    "heartrate": (20, 220),
    "respiration": (4, 60),
    "sao2": (50, 100),
    "temperature": (32, 42),
    "systemicsystolic": (40, 250),
    "systemicdiastolic": (20, 180),
    "systemicmean": (30, 200),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML config file (optional)")
    parser.add_argument("--input_dir", help="Raw data directory")
    parser.add_argument("--output_file", help="Output CSV file")
    parser.add_argument("--time_window", type=int, help="Time window in hours")
    parser.add_argument("--missing_threshold", type=float, help="Missing threshold")
    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        input_dir = args.input_dir or cfg["paths"]["input_raw"]
        output_file = args.output_file or cfg["paths"]["output_clean"]
        time_window = args.time_window or cfg.get("time_window_hours", 72)
        missing_threshold = args.missing_threshold or cfg["qc"].get("missing_threshold", 0.7)
        outlier_limits = {k: tuple(v) for k, v in cfg["qc"].get("outlier_limits", {}).items()}
        tables = cfg.get("tables")
        if not tables:
            raise ValueError(
                f"{args.config}: missing the 'tables' block. It declares which "
                "columns are read from each raw CSV and is required -- without it "
                "every column is loaded, including database row identifiers."
            )
    else:
        if not args.input_dir or not args.output_file:
            parser.error("--input_dir and --output_file required when not using --config")
        input_dir = args.input_dir
        output_file = args.output_file
        time_window = args.time_window or 72  # keep in step with the YAML default
        missing_threshold = args.missing_threshold or 0.7
        outlier_limits = dict(DEFAULT_OUTLIER_LIMITS)
        tables = DEFAULT_TABLES

    print("=" * 60)
    print("STROKE DATA CLEANING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/7] Loading data...")
    patient, diagnosis, vital, lab = load_data(input_dir, tables=tables)
    print(f"  Patient: {patient.shape}, Diagnosis: {diagnosis.shape}")
    if 'hospitalid' in patient.columns:
        print(f"  Found real hospital IDs: {patient['hospitalid'].nunique()} distinct hospitals")
        patient = patient.rename(columns={'hospitalid': 'site_id'})
    else:
        raise ValueError("CRITICAL: no 'hospitalid' column in the raw data; check patient.csv")
    print(f"  Vital: {vital.shape}, Lab: {lab.shape}")

    # Step 2: Cohort selection
    print("\n[2/7] Selecting stroke cohort...")
    cohort_pat = select_cohort(patient, diagnosis)
    keep_ids = set(cohort_pat["patientunitstayid"].unique())
    print(f"  Selected {len(keep_ids)} stroke patients")
    
    vital = vital[vital["patientunitstayid"].isin(keep_ids)].copy()
    lab = lab[lab["patientunitstayid"].isin(keep_ids)].copy()

    # Step 3: Time alignment
    print(f"\n[3/7] Aligning time window (first {time_window} hours)...")
    vital, lab = align_time_window(vital, lab, time_window)
    print(f"  Vital: {vital.shape}, Lab: {lab.shape}")

    # Step 4: Outlier handling
    print("\n[4/7] Handling outliers...")
    vital, outlier_counts = apply_outlier_nan(vital, outlier_limits)
    total_outliers = sum(outlier_counts.values())
    print(f"  Replaced {total_outliers} outlier values with NaN")

    # Step 5: Feature aggregation
    print("\n[5/7] Aggregating features...")
    agg_v = aggregate_vitals(vital)
    agg_l = aggregate_labs(lab) if not lab.empty else None
    
    agg = agg_v if agg_l is None else agg_v.join(agg_l, how="left")
    print(f"  Aggregated features: {agg.shape[1]} columns")

   # =========================================================================
    # Step 6: Merge & Final Polish
    # =========================================================================
    print(f"\n[6/7] Merging features and generating targets...")
    
    # 1. Build the label and drop leakage columns.
    #    patient_labeled now carries 'mortality' and no discharge status.
    patient_labeled = extract_mortality_label(cohort_pat)
    
    # 2. Select the static features the dual-tower model needs.
    #    Feature scan: age, gender and site_id are categorical; admission_time is noise.
    static_cols = [
        "age", 
        "gender", 
        "ethnicity",
        "admissionheight",  # numeric
        "admissionweight",  # numeric
        "site_id",          # categorical, from hospitalid
        "mortality"         # target
    ]
    
    # Skip anything the patient table does not actually have.
    existing_static = [c for c in static_cols if c in patient_labeled.columns]
    
    # 3. Static table, keyed by stay ID
    demo = patient_labeled.set_index("patientunitstayid")[existing_static]
    
    # 4. Join static demographics with the aggregated dynamic features.
    #    how='left' keeps only patients in the cohort.
    merged = demo.join(agg, how="left")
    
    # 5. Drop noise columns.
    #    'admission_time' has ~1460 distinct values: a raw timestamp, not a feature.
    if 'admission_time' in merged.columns:
        print("  Removing noise column: 'admission_time'...")
        merged = merged.drop(columns=['admission_time'])
    
    # 6. Drop sparse columns
    print(f"  Dropping sparse columns (threshold: {missing_threshold})...")
    # NaNs are deliberately preserved; imputation is fitted on the train split only.
    cleaned, report = drop_sparse_and_impute(merged, missing_threshold)
    
    # 7. Final safety check
    if 'hospitaldischargestatus' in cleaned.columns:
        raise RuntimeError("CRITICAL ALERT: a leakage column survived cleaning")
    if 'mortality' not in cleaned.columns:
        raise RuntimeError("CRITICAL ALERT: the mortality target column was lost")

    print(f"  Dropped {len(report['dropped_cols'])} sparse columns")
    print(f"  Final shape: {cleaned.shape}")
    print(f"  Target Distribution: {cleaned['mortality'].value_counts().to_dict()}")

    # Step 7: Save
    print(f"\n[7/7] Saving to {output_file}...")
    cleaned_reset = cleaned.reset_index()
    save_data(cleaned_reset, output_file)
    print(f"  Saved {cleaned_reset.shape[0]} patients, {cleaned_reset.shape[1]} features")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    main()


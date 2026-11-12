#!/usr/bin/env python3
"""
Data cleaning pipeline for eICU mortality cohorts.
7 steps: load -> cohort -> timealign -> outliers -> aggregate -> impute -> save

The cohort is chosen by the `cohort` key of the config:
    stroke  (default) the published stroke cohort, unchanged
    hf_cs   heart failure-related cardiogenic shock (obs_cleaning/cohort_hfcs.py)
"""
import argparse
import json
import yaml
from pathlib import Path

import pandas as pd

from obs_cleaning import (
    load_data,
    load_extra_tables,
    select_hfcs_cohort,
    filter_adults,
    keep_first_stay,
    apache_iv_prediction,
    treatment_flags,
    select_cohort,
    align_time_window,
    apply_outlier_nan,
    apply_lab_outlier_nan,
    aggregate_vitals,
    aggregate_labs,
    harmonise_lab_units,
    drop_sparse_and_impute,
    save_data,
    extract_mortality_label,
    apply_observation_window,
)

# Fallback column whitelist and QC limits for the ad-hoc CLI path (no --config).
# They mirror config/obs_cleaning.yaml: both entry points must agree, because
# previously they named different vital columns and only one of the two sets
# ever took effect (AUDIT.md EICU-05, EICU-06). No entry point may read a raw
# table unrestricted -- that is what admitted vitalperiodicid into the feature
# matrix (AUDIT.md EICU-01).
DEFAULT_TABLES = {
    "patient": [
        "patientunitstayid", "uniquepid", "age", "gender", "ethnicity", "hospitalid",
        "admissionheight", "admissionweight", "hospitaldischargestatus",
        "unitadmitoffset", "unitdischargeoffset", "hospitaldischargeyear",
    ],
    "diagnosis": ["patientunitstayid", "icd9code", "diagnosisstring",
                  "diagnosisoffset"],
    "vitalPeriodic": [
        "patientunitstayid", "observationoffset", "heartrate", "respiration",
        "sao2", "temperature", "systemicsystolic", "systemicdiastolic",
        "systemicmean",
    ],
    "lab": ["patientunitstayid", "labname", "labresult", "labresultoffset",
            "labmeasurenamesystem"],
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
        lab_outlier_limits = {k: tuple(v)
                              for k, v in cfg["qc"].get("lab_outlier_limits", {}).items()}
        tables = cfg.get("tables")
        cohort_name = cfg.get("cohort", "stroke")
        hfcs_cfg = cfg.get("hf_cs", {})
        vital_stats = cfg.get("features", {}).get("vital_stats")
        lab_stats = cfg.get("features", {}).get("lab_stats")
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
        lab_outlier_limits = {}
        tables = DEFAULT_TABLES
        cohort_name, hfcs_cfg, vital_stats, lab_stats = "stroke", {}, None, None

    if cohort_name not in ("stroke", "hf_cs"):
        raise ValueError(f"Unknown cohort '{cohort_name}'; expected 'stroke' or 'hf_cs'")
    flow = {"cohort": cohort_name, "time_window_hours": time_window}

    print("=" * 60)
    print(f"DATA CLEANING PIPELINE - cohort: {cohort_name}")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/7] Loading data...")
    patient, diagnosis, vital, lab = load_data(input_dir, tables=tables)
    extra = {}
    if cohort_name == "hf_cs":
        wanted = ["admissionDx", "pastHistory", "vitalAperiodic", "infusionDrug",
                  "treatment", "apachePatientResult"]
        extra = load_extra_tables(input_dir, tables, [t for t in wanted if t in tables])
        print("  Extra tables: " + ", ".join(f"{k} {v.shape}" for k, v in extra.items()))
    print(f"  Patient: {patient.shape}, Diagnosis: {diagnosis.shape}")
    if 'hospitalid' in patient.columns:
        print(f"  Found real hospital IDs: {patient['hospitalid'].nunique()} distinct hospitals")
        patient = patient.rename(columns={'hospitalid': 'site_id'})
    else:
        raise ValueError("CRITICAL: no 'hospitalid' column in the raw data; check patient.csv")
    print(f"  Vital: {vital.shape}, Lab: {lab.shape}")

    # Step 2: Cohort selection
    print(f"\n[2/7] Selecting {cohort_name} cohort...")
    window_minutes = time_window * 60
    flow["stays_in_database"] = len(patient)
    if cohort_name == "stroke":
        cohort_pat, cohort_report = select_cohort(patient, diagnosis, window_minutes)
        print(f"  Diagnoses considered:   {cohort_report['diagnoses_total']}")
        if "diagnoses_within_window" in cohort_report:
            print(f"  Within the window:      {cohort_report['diagnoses_within_window']}")
        print(f"  Stays by ICD-9:         {cohort_report['by_icd9']}")
        print(f"  Stays by keyword only:  {cohort_report['by_keyword_only']}")
        print(f"  Stroke stays selected:  {cohort_report['stays_selected']}")
    else:
        adults, adult_report = filter_adults(patient, hfcs_cfg.get("min_age", 18))
        flow["adult_filter"] = adult_report
        print(f"  Adults (age >= {hfcs_cfg.get('min_age', 18)}): {adult_report['stays_after']}")
        cohort_pat, cohort_report = select_hfcs_cohort(
            adults, diagnosis, window_minutes,
            admissiondx=extra.get("admissionDx"),
            pasthistory=extra.get("pastHistory"),
            exclude_operative=hfcs_cfg.get("exclude_operative_admissions", True),
        )
        for k, v in cohort_report.items():
            print(f"  {k}: {v}")
    flow["cohort_selection"] = cohort_report

    # Require the stay to span the observation window, so the length of the
    # record is not itself a predictor of the outcome.
    cohort_pat, window_report = apply_observation_window(cohort_pat, window_minutes)
    flow["observation_window"] = window_report
    print(f"  Shorter than {time_window}h, dropped: {window_report['stays_shorter_than_window']}")
    print(f"  Stays remaining:        {window_report['stays_after']}")

    if cohort_name == "hf_cs":
        # One stay per person: the earliest eligible one.
        cohort_pat, first_report = keep_first_stay(cohort_pat)
        flow["first_stay"] = first_report
        print(f"  Repeat stays dropped:   {first_report['repeat_stays_dropped']}")

    keep_ids = set(cohort_pat["patientunitstayid"].unique())

    vital = vital[vital["patientunitstayid"].isin(keep_ids)].copy()
    lab = lab[lab["patientunitstayid"].isin(keep_ids)].copy()
    if "vitalAperiodic" in extra:
        # Cuff blood pressure lives in vitalAperiodic; stack it under the
        # periodic vitals so it is aligned, cleaned and aggregated the same way.
        aper = extra["vitalAperiodic"]
        aper = aper[aper["patientunitstayid"].isin(keep_ids)]
        vital = pd.concat([vital, aper], ignore_index=True, sort=False)

    # Step 3: Time alignment
    print(f"\n[3/7] Aligning time window (first {time_window} hours)...")
    vital, lab = align_time_window(vital, lab, time_window)
    print(f"  Vital: {vital.shape}, Lab: {lab.shape}")

    # Step 4: Outlier handling
    print("\n[4/7] Handling outliers...")
    vital, outlier_counts = apply_outlier_nan(vital, outlier_limits)
    print(f"  Vitals: blanked {sum(outlier_counts.values())} implausible values")

    # Labs are long-format, so the wide helper above never reached them.
    lab, lab_outlier_counts = apply_lab_outlier_nan(lab, lab_outlier_limits)
    print(f"  Labs:   blanked {sum(lab_outlier_counts.values())} implausible values")

    # One unit per analyte, otherwise mg/dL and mmol/L are averaged together.
    lab, unit_report = harmonise_lab_units(lab)
    if unit_report.get("unit_column_present"):
        mixed = unit_report["analytes_with_mixed_units"]
        print(f"  Labs:   {len(mixed)} of {unit_report['analytes']} analytes had "
              f"mixed units; dropped {unit_report['rows_dropped']} rows")
        if mixed:
            print(f"          mixed-unit analytes: {mixed}")

    # Step 5: Feature aggregation
    print("\n[5/7] Aggregating features...")
    agg_v = aggregate_vitals(vital, stats=vital_stats)
    agg_l = aggregate_labs(lab, stats=lab_stats) if not lab.empty else None

    agg = agg_v if agg_l is None else agg_v.join(agg_l, how="left")

    if cohort_name == "hf_cs":
        flags = treatment_flags(keep_ids, window_minutes,
                                infusion=extra.get("infusionDrug"),
                                treatment=extra.get("treatment"))
        agg = agg.join(flags, how="outer").fillna({c: 0 for c in flags.columns})
        print("  Treatment exposure (stays): "
              + ", ".join(f"{c}={int((flags[c] > 0).sum())}" for c in flags.columns))
        if "apachePatientResult" in extra:
            # Baseline only: run_obs_trainReady.py keeps it out of the features.
            agg = agg.join(apache_iv_prediction(extra["apachePatientResult"]), how="left")
            flow["apache_iv_available"] = int(agg["apache_iv_pred"].notna().sum())
    print(f"  Aggregated features: {agg.shape[1]} columns")

   # =========================================================================
    # Step 6: Merge & Final Polish
    # =========================================================================
    print(f"\n[6/7] Merging features and generating targets...")
    
    # 1. Build the label and drop leakage columns.
    #    patient_labeled now carries 'mortality' and no discharge status.
    patient_labeled, label_report = extract_mortality_label(cohort_pat)
    print(f"  Discharge status counts: {label_report['status_counts']}")
    if label_report["dropped_unusable_status"]:
        print(f"  Dropped (status missing or unrecognised): "
              f"{label_report['dropped_unusable_status']}")
    print(f"  Labelled stays: {label_report['stays_after']}, "
          f"deaths: {label_report['deaths']}, "
          f"prevalence: {label_report['prevalence']}")
    
    # 2. Select the static features the dual-tower model needs.
    #    Feature scan: age, gender and site_id are categorical; admission_time is noise.
    static_cols = [
        "uniquepid",        # person-level key, required for leakage-free splits
        "age", 
        "gender", 
        "ethnicity",
        "admissionheight",  # numeric
        "admissionweight",  # numeric
        "site_id",          # categorical, from hospitalid
        "hospitaldischargeyear",  # time axis for the temporal protocol
        "mortality"         # target
    ]
    
    # Skip anything the patient table does not actually have.
    existing_static = [c for c in static_cols if c in patient_labeled.columns]
    
    # 3. Static table, keyed by stay ID
    demo = patient_labeled.set_index("patientunitstayid")[existing_static]
    
    # 4. Join static demographics with the aggregated dynamic features.
    #    how='left' keeps only patients in the cohort.
    merged = demo.join(agg, how="left")
    
    # 5. Drop noise columns. eICU is de-identified and carries no absolute
    #    timestamp, so 'admission_time' never existed here; the temporal
    #    protocol uses hospitaldischargeyear instead (AUDIT.md EICU-13).
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

    # Participant flow for the paper (TRIPOD+AI asks for one).
    flow["final"] = {
        "stays": int(cleaned_reset.shape[0]),
        "deaths": int(cleaned_reset["mortality"].sum()),
        "columns": int(cleaned_reset.shape[1]),
        "dropped_sparse_columns": report["dropped_cols"],
    }
    flow_path = Path(output_file).with_name(Path(output_file).stem + "_flow.json")
    with open(flow_path, "w") as f:
        json.dump(flow, f, indent=2, default=str)
    print(f"  Cohort flow written to {flow_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    main()


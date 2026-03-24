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
    else:
        if not args.input_dir or not args.output_file:
            parser.error("--input_dir and --output_file required when not using --config")
        input_dir = args.input_dir
        output_file = args.output_file
        time_window = args.time_window or 72  # 更新默认值为72h，与配置文件保持一致
        missing_threshold = args.missing_threshold or 0.7
        outlier_limits = {
            "heartrate": (20, 220),
            "temperature": (32, 42),
            "respiration": (4, 60),
            "systemicsystolic": (40, 250),
            "systemicdiastolic": (20, 180),
            "systemicmean": (30, 200),
            "sao2": (50, 100),
        }

    print("=" * 60)
    print("STROKE DATA CLEANING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/7] Loading data...")
    patient, diagnosis, vital, lab = load_data(input_dir)
    print(f"  Patient: {patient.shape}, Diagnosis: {diagnosis.shape}")
    if 'hospitalid' in patient.columns:
        print(f"  ✓ 捕获真实医院ID: {patient['hospitalid'].nunique()} 个独立医院")
        patient = patient.rename(columns={'hospitalid': 'site_id'})
    else:
        raise ValueError("CRITICAL: 原始数据中找不到 'hospitalid'，请检查 patient.csv！")
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
    # Step 6: Merge & Final Polish (重构版)
    # =========================================================================
    print(f"\n[6/7] Merging features and generating targets...")
    
    # 1. 生成标签 & 剔除泄露 (调用 cohort.py 的新函数)
    # 此时 patient_labeled 包含 'mortality'，且不含 discharge status
    patient_labeled = extract_mortality_label(cohort_pat)
    
    # 2. 明确筛选【双塔模型】需要的静态特征 (根据你的扫描结果)
    # 扫描显示: age, gender, site_id 是类别特征；admission_time 是噪音
    static_cols = [
        "age", 
        "gender", 
        "ethnicity",        # 如果扫描结果有，就加上
        "admissionheight",  # 如果有，作为数值
        "admissionweight",  # 如果有，作为数值
        "site_id",          # 【关键】扫描发现的类别特征
        "mortality"         # 【关键】目标列
    ]
    
    # 自动过滤掉 patient 表里不存在的列，防止报错
    existing_static = [c for c in static_cols if c in patient_labeled.columns]
    
    # 3. 准备静态表 (带 ID)
    demo = patient_labeled.set_index("patientunitstayid")[existing_static]
    
    # 4. 合并 (Static Demo + Dynamic Aggregated Features)
    # how='left' 确保只保留 cohort 里的病人
    merged = demo.join(agg, how="left")
    
    # 5. 剔除噪音 (Noise Removal)
    # 扫描显示 'admission_time' 有 1460 类，这是绝对的时间戳噪音，删掉
    if 'admission_time' in merged.columns:
        print("  Removing noise column: 'admission_time'...")
        merged = merged.drop(columns=['admission_time'])
    
    # 6. 处理稀疏列 (Drop Sparse)
    print(f"  Dropping sparse columns (threshold: {missing_threshold})...")
    # 注意: 这里保留 NaN 是正确的 (Imputation 留给 Dataset 类做)
    cleaned, report = drop_sparse_and_impute(merged, missing_threshold)
    
    # 7. 最终安全检查 (Safety Check) - 这步能救命
    if 'hospitaldischargestatus' in cleaned.columns:
        raise RuntimeError("CRITICAL ALERT: 泄露列依然存在！")
    if 'mortality' not in cleaned.columns:
        raise RuntimeError("CRITICAL ALERT: 目标列 mortality 丢失！")

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


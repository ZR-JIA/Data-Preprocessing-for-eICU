#!/usr/bin/env python3
"""
=============================================================================
eICU/MIMIC data health check
=============================================================================
Reports whether the table is static or time-series, and screens for obvious
leakage risks.

Usage:
    1. Set file_path and target_col below
    2. Run: python data_health_check.py
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple, List, Dict, Optional
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration - edit these
# ============================================================================
file_path = "outputs/obs_cleaned.csv"  # path to the data file
target_col = "mortality"     # name of the target column
id_col = 'patientunitstayid' 
# ============================================================================


def print_section(title: str, symbol: str = "="):
    """Print a section heading."""
    print(f"\n{symbol * 80}")
    print(f"  {title}")
    print(f"{symbol * 80}\n")


def find_patient_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the patient ID column, trying the usual names first.
    """
    # uniquepid first: patientunitstayid identifies one ICU *stay*, so
    # counting it reports stays as if they were people and understates how
    # often one patient recurs (AUDIT.md EICU-15, EICU-02).
    possible_id_cols = [
        'uniquepid', 'subject_id',
        'patient_id',
        'patientunitstayid', 'patient_unit_stay_id',
        'hadm_id', 'icustay_id', 'stay_id', 'id', 'ID'
    ]
    
    for col in possible_id_cols:
        if col in df.columns:
            return col
    
    # Fall back to any column whose name contains 'id'
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    if id_cols:
        return id_cols[0]
    
    return None


def check_id_structure(df: pd.DataFrame, id_col: str) -> Tuple[str, Dict]:
    """
    Check 1: ID structure (static vs dynamic).
    
    Returns:
        data_type: "STATIC" or "DYNAMIC"
        stats: summary statistics
    """
    print_section("Check 1: ID structure (static vs dynamic)", "=")
    
    total_rows = len(df)
    unique_patients = df[id_col].nunique()
    rows_per_patient = total_rows / unique_patients
    
    stats = {
        'total_rows': total_rows,
        'unique_patients': unique_patients,
        'rows_per_patient': rows_per_patient
    }
    
    print(f"  Total rows:            {total_rows:,}")
    print(f"  Unique patients:       {unique_patients:,}")
    print(f"  Rows per patient:      {rows_per_patient:.2f}")
    
    # Classification
    if rows_per_patient <= 1.1:  # small tolerance
        data_type = "STATIC"
        print(f"\n{'='*80}")
        print(f"  Verdict: STATIC table")
        print(f"   ➜ One row per patient")
        print(f"    Suggested models: MLP, ResNet, XGBoost, Random Forest")
        print(f"{'='*80}")
    else:
        data_type = "DYNAMIC"
        print(f"\n{'='*80}")
        print(f"  Verdict: DYNAMIC time series")
        print(f"    Multiple rows per patient")
        print(f"    Suggested models: LSTM, GRU, Transformer, Temporal CNN")
        print(f"{'='*80}")
    
    return data_type, stats


def detect_data_leakage(df: pd.DataFrame, target_col: str) -> Dict:
    """
    Check 2: leakage screening.
    
    Returns:
        leakage_report: what the screen found
    """
    print_section("Check 2: leakage screening", "=")
    
    leakage_report = {
        'high_correlation': [],
        'suspicious_keywords': [],
        'duplicates': 0
    }
    
    # -------------------------------------------------------------------------
    # 2.1 High correlation with the target
    # -------------------------------------------------------------------------
    print("[2.1] Correlation screen (|r| > 0.9)")
    print("-" * 80)
    
    if target_col not in df.columns:
        print(f"  WARNING: target column '{target_col}' not found")
        numeric_cols = []
    else:
        # Numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) == 0:
            print("  No numeric features to screen")
        else:
            correlations = []
            for col in numeric_cols:
                try:
                    corr = df[col].corr(df[target_col])
                    if not np.isnan(corr):
                        correlations.append((col, corr))
                except:
                    pass
            
            # Rank by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Flag anything above the threshold
            high_corr_threshold = 0.9
            found_leakage = False
            
            for col, corr in correlations:
                if abs(corr) > high_corr_threshold:
                    found_leakage = True
                    leakage_report['high_correlation'].append((col, corr))
                    print(f"  SUSPECTED LEAKAGE: '{col}' | r = {corr:.4f}")
            
            if not found_leakage:
                print(f"  No feature exceeds |r| = {high_corr_threshold}")
            
            # Always show the strongest five
            if correlations:
                print(f"\n  Top 5 correlations:")
                for i, (col, corr) in enumerate(correlations[:5], 1):
                    print(f"      {i}. {col:40s} → {corr:+.4f}")
    
    # -------------------------------------------------------------------------
    # 2.2 Suspicious column names
    # -------------------------------------------------------------------------
    print(f"\n[2.2] Column-name screen")
    print("-" * 80)
    
    # Keywords worth a second look, tuned to the prediction task
    suspicious_keywords = [
        'death', 'dead', 'died', 'expire', 'expired', 'expiration',
        'mortality', 'mortal', 'deceased',
        'discharge', 'discharged', 'disposition',
        'outcome', 'result', 'final',
        'diagnosis', 'diagnose', 'icd',  # relevant when predicting a diagnosis
        'future', 'post', 'after',
        'label', 'target', 'ground_truth'
    ]
    
    # Database identifiers are the case this screen was blind to: a column
    # like vitalperiodicid_mean correlates with the target nowhere near 0.9 and
    # matches none of the keywords above, yet it is a row counter rather than a
    # measurement (AUDIT.md EICU-01, EICU-15).
    identifier_markers = ['periodicid', 'labid', '_id_', 'rowid', 'recordid']

    all_columns = [col for col in df.columns if col != target_col]
    for col in all_columns:
        low = col.lower()
        if any(m in low for m in identifier_markers) or low.endswith('id'):
            leakage_report['suspicious_keywords'].append(col)
            print(f"  NON-CLINICAL: '{col}' looks like a database identifier, "
                  f"not a measurement")

    found_suspicious = bool(leakage_report['suspicious_keywords'])
    
    for col in all_columns:
        col_lower = col.lower()
        for keyword in suspicious_keywords:
            if keyword in col_lower:
                found_suspicious = True
                leakage_report['suspicious_keywords'].append(col)
                print(f"  SUSPECT: '{col}' (matched keyword '{keyword}')")
                break
    
    if not found_suspicious:
        print("  No suspicious column names")
    
    # -------------------------------------------------------------------------
    # 2.3 Duplicate rows
    # -------------------------------------------------------------------------
    print(f"\n[2.3] Duplicate rows")
    print("-" * 80)
    
    duplicates = df.duplicated().sum()
    leakage_report['duplicates'] = duplicates
    
    if duplicates > 0:
        dup_percentage = (duplicates / len(df)) * 100
        print(f"  Found {duplicates:,} fully duplicated rows ({dup_percentage:.2f}%)")
        print(f"    Consider df.drop_duplicates()")
    else:
        print(f"  No fully duplicated rows")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*80}")
    total_issues = len(leakage_report['high_correlation']) + \
                   len(leakage_report['suspicious_keywords']) + \
                   (1 if duplicates > 0 else 0)
    
    if total_issues == 0:
        print("Leakage screen: nothing obvious found")
    else:
        print(f"Leakage screen: {total_issues} item(s) to review")
        print("  Review the features flagged above")
    print(f"{'='*80}")
    
    return leakage_report


def sample_temporal_structure(df: pd.DataFrame, id_col: str, data_type: str):
    """
    Check 3: sample the temporal structure. Dynamic tables only.
    """
    if data_type != "DYNAMIC":
        return
    
    print_section("Check 3: temporal structure", "=")
    
    # The patient with the most rows
    patient_counts = df[id_col].value_counts()
    max_patient_id = patient_counts.index[0]
    max_count = patient_counts.iloc[0]
    
    print(f"  Patient with most rows: {max_patient_id}")
    print(f"  Row count: {max_count}")
    
    # Pull that patient's rows
    patient_data = df[df[id_col] == max_patient_id].head(5)
    
    # Look for time-like columns
    time_cols = [col for col in df.columns if any(
        keyword in col.lower() 
        for keyword in ['time', 'offset', 'hour', 'minute', 'datetime', 'timestamp', 'date']
    )]
    
    if time_cols:
        print(f"\n  Time-like columns: {time_cols}")
        display_cols = [id_col] + time_cols + \
                      [col for col in patient_data.columns if col not in [id_col] + time_cols][:5]
    else:
        print("\n  No obvious time column (offset/time)")
        display_cols = patient_data.columns[:8].tolist()
    
    print(f"\n  First 5 rows for that patient:")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(patient_data[display_cols].to_string(index=False))
    print("-" * 80)


def basic_overview(df: pd.DataFrame, target_col: str, id_col: str):
    """
    Check 4: basic overview.
    """
    print_section("Check 4: basic overview", "=")
    
    # Feature count, excluding the ID and target columns
    exclude_cols = [target_col, id_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Split numeric from categorical
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"  Shape:                {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print(f"  Features:             {len(feature_cols):,} (excluding ID and target)")
    print(f"    numeric:            {len(numeric_features):,}")
    print(f"    categorical:        {len(categorical_features):,}")
    
    # Target distribution
    if target_col in df.columns:
        print(f"\n  Target ('{target_col}') distribution:")
        value_counts = df[target_col].value_counts()
        for val, count in value_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   ├─ {val}: {count:,} ({percentage:.2f}%)")
    
    # Missing values
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(cols_with_missing) > 0:
        print(f"\n  Missing values:")
        print(f"    columns affected:   {len(cols_with_missing)}")
        print(f"    worst 5:")
        for i, (col, miss_count) in enumerate(cols_with_missing.head(5).items(), 1):
            miss_pct = (miss_count / len(df)) * 100
            print(f"      {i}. {col:40s} → {miss_count:,} ({miss_pct:.2f}%)")
    else:
        print(f"\n  Missing values: none")
    
    # Memory
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\n  Memory:               {memory_mb:.2f} MB")


def main():
    """Entry point."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 21 + "eICU/MIMIC Data Health Check Report" + " " * 22 + "║")
    print("║" + " " * 15 + "Data Health Check & Leakage Detection" + " " * 26 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # =========================================================================
    # Load
    # =========================================================================
    print_section("Loading data", "=")
    
    try:
        print(f"  File:   {file_path}")
        print(f"  Target: {target_col}")
        print(f"\n  Loading...")
        
        df = pd.read_csv(file_path)
        print(f"  Loaded. Shape: {df.shape}")
        
    except FileNotFoundError:
        print(f"ERROR: file not found - {file_path}")
        print(f"  Check the file_path setting")
        return
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return
    
    # =========================================================================
    # Locate the patient ID column
    # =========================================================================
    id_col = find_patient_id_column(df)
    
    if id_col is None:
        print("\nWARNING: no patient ID column found; some checks will be skipped")
        print("  Expected 'patientunitstayid' or a similarly named column")
        id_col = df.columns[0]  # fall back to the first column
        print(f"  Falling back to the first column as ID: '{id_col}'")
    else:
        print(f"\n  Patient ID column: '{id_col}'")
    
    # =========================================================================
    # Run the checks
    # =========================================================================
    
    # Check 1: ID structure
    data_type, id_stats = check_id_structure(df, id_col)
    
    # Check 2: leakage
    leakage_report = detect_data_leakage(df, target_col)
    
    # Check 3: temporal structure (dynamic tables only)
    sample_temporal_structure(df, id_col, data_type)
    
    # Check 4: basic overview
    basic_overview(df, target_col, id_col)
    
    # =========================================================================
    # Final summary
    # =========================================================================
    print_section("Final summary", "=")
    
    print("Q1: which kind of model fits this table?")
    print("-" * 80)
    if data_type == "STATIC":
        print("A1: a STATIC model")
        print("  Suggested:")
        print("   • MLP (Multi-Layer Perceptron)")
        print("   • ResNet (Residual Network)")
        print("   • XGBoost / LightGBM")
        print("   • Random Forest")
    else:
        print("A1: a TIME-SERIES model")
        print("  Suggested:")
        print("  - LSTM / GRU")
        print("  - Transformer, for longer-range dependencies")
        print("  - Temporal CNN, for local patterns")
        print("  - or aggregate over time first, then use a static model")
    
    print("\nQ2: is there leakage in the features?")
    print("-" * 80)
    
    total_leakage_flags = len(leakage_report['high_correlation']) + \
                         len(leakage_report['suspicious_keywords'])
    
    if total_leakage_flags == 0:
        print("A2: no obviously leaking feature was found")
        print("  NOTE: this screen only catches |r| > 0.9, identifier-shaped")
        print("  names and keyword matches. It cannot see a feature that leaks")
        print("  weakly, nor one that is non-clinical without saying so in its")
        print("  name. A clean result here is not evidence of absence.")
    else:
        print(f"A2: {total_leakage_flags} suspicious feature(s)")
        print("  Suggested:")
        print("  - review the features flagged above")
        print("  - confirm each is actually available at prediction time")
        print("  - drop any that turn out to leak")
    
    if leakage_report['duplicates'] > 0:
        print(f"\nAlso: {leakage_report['duplicates']:,} duplicated rows")
        print("  Consider df = df.drop_duplicates()")
    
    print("\n" + "=" * 80)
    print("Data health check complete.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


import pandas as pd

ICD9_PREFIXES = ("430", "431", "433", "434", "436", "437")  # 430 SAH and 431 ICH included for cohort representativeness
KW = ("stroke", "cva", "ich", "sah")

def _icd9_hit(icd: str) -> bool:
    if not isinstance(icd, str):
        return False
    icd = icd.strip().replace(".", "")
    return any(icd.startswith(p) for p in ICD9_PREFIXES)

def _kw_hit(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(k in t for k in KW)

def select_cohort(patient: pd.DataFrame, diagnosis: pd.DataFrame) -> pd.DataFrame:
    """Return patient rows whose diagnoses match ICD-9 filter or stroke keywords."""
    if "icd9code" not in diagnosis.columns and "diagnosisstring" not in diagnosis.columns:
        raise KeyError("diagnosis must contain 'icd9code' and/or 'diagnosisstring'")
    cond_icd = diagnosis["icd9code"].apply(_icd9_hit) if "icd9code" in diagnosis.columns else False
    cond_kw = diagnosis["diagnosisstring"].apply(_kw_hit) if "diagnosisstring" in diagnosis.columns else False
    diag_keep = diagnosis[cond_icd | cond_kw]
    keep_ids = diag_keep["patientunitstayid"].unique()
    return patient[patient["patientunitstayid"].isin(keep_ids)].copy()

def extract_mortality_label(patient: pd.DataFrame) -> pd.DataFrame:
    """
    1. Build the 'mortality' target.
    2. Drop the discharge columns that contain the answer.
    """
    df = patient.copy()
    
    # --- 1. Build the 0/1 label ---
    # Any status containing 'expired' (case-insensitive) counts as a death.
    if 'hospitaldischargestatus' in df.columns:
        df['mortality'] = df['hospitaldischargestatus'].astype(str).str.lower().apply(
            lambda x: 1 if 'expired' in x else 0
        )
    else:
        # Without this column the target cannot be built at all.
        raise ValueError("Error: 'hospitaldischargestatus' column missing in cohort data!")
    
    # --- 2. Remove leakage columns ---
    # These describe the outcome; keeping any of them lets the model cheat.
    leak_cols = [
        'hospitaldischargestatus', 
        'unitdischargestatus', 
        'hospitaldischargelocation',
        'unitdischargelocation'
    ]
    df = df.drop(columns=leak_cols, errors='ignore')
    
    return df
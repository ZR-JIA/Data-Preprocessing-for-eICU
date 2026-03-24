
import pandas as pd

ICD9_PREFIXES = ("430", "431", "433", "434", "436", "437")  # 包含SAH(430)和ICH(431)，提高样本代表性
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
    1. 生成 'mortality' 标签 (Target)。
    2. 删除包含答案的泄露列。
    """
    df = patient.copy()
    
    # --- 1. 生成 Label (0/1) ---
    # 逻辑: 只要 status 包含 'expired' (不区分大小写) 即为死亡
    if 'hospitaldischargestatus' in df.columns:
        df['mortality'] = df['hospitaldischargestatus'].astype(str).str.lower().apply(
            lambda x: 1 if 'expired' in x else 0
        )
    else:
        # 如果原始数据里没有这一列，可能需要报错或检查逻辑
        raise ValueError("Error: 'hospitaldischargestatus' column missing in cohort data!")
    
    # --- 2. 删除泄露列 (Leakage Removal) ---
    # 这些列必须死，否则模型会作弊
    leak_cols = [
        'hospitaldischargestatus', 
        'unitdischargestatus', 
        'hospitaldischargelocation',
        'unitdischargelocation'
    ]
    df = df.drop(columns=leak_cols, errors='ignore')
    
    return df
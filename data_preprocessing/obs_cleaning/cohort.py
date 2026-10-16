import re

import pandas as pd

# --- ICD-9 selection -------------------------------------------------------
# Matching on three digits alone admits codes that are not acute stroke
# (AUDIT.md EICU-07), so the groups are separated by what the 4th/5th digit
# means.

# Acute by definition at the 3-digit level.
#   430 subarachnoid haemorrhage
#   431 intracerebral haemorrhage
#   436 acute but ill-defined cerebrovascular disease
ICD9_ACUTE_PREFIXES = ("430", "431", "436")

# Occlusion/stenosis of pre-cerebral (433) and cerebral (434) arteries. The
# 5th digit says whether there was an infarction: 1 = with, 0 = without.
# 433.x0 / 434.x0 are chronic stenosis and must not enter a stroke cohort.
ICD9_INFARCTION_PREFIXES = ("433", "434")

# 437 ("other and ill-defined cerebrovascular disease") is deliberately absent.
# It covers cerebral atherosclerosis, non-ruptured aneurysm, transient global
# amnesia and more, none of which is an acute stroke. It was previously
# included; see AUDIT.md EICU-07.
ICD9_EXCLUDED_PREFIXES = ("437",)

# --- Free-text selection ---------------------------------------------------
# Word boundaries matter: a bare substring test for "stroke" also matches
# "heat stroke", a genuine ICU diagnosis and not a cerebrovascular event.
KEYWORD_PATTERNS = tuple(
    re.compile(p) for p in (r"\bstroke\b", r"\bcva\b", r"\bich\b", r"\bsah\b")
)
KEYWORD_EXCLUSIONS = tuple(
    re.compile(p) for p in (r"\bheat\s*stroke\b", r"\bheatstroke\b",
                            r"\bsun\s*stroke\b", r"\bsunstroke\b")
)

# Discharge statuses eICU actually uses. Anything else, including a null, is
# not evidence of survival and must not be labelled as such (AUDIT.md EICU-03).
VALID_DISCHARGE_STATUS = {"alive", "expired"}

LEAK_COLS = [
    "hospitaldischargestatus",
    "unitdischargestatus",
    "hospitaldischargelocation",
    "unitdischargelocation",
]


def _normalise_icd(icd: str) -> str:
    return icd.strip().replace(".", "")


def _icd9_hit(icd) -> bool:
    """True when the code denotes an acute cerebrovascular event."""
    if not isinstance(icd, str):
        return False
    code = _normalise_icd(icd)
    if code.startswith(ICD9_EXCLUDED_PREFIXES):
        return False
    if code.startswith(ICD9_ACUTE_PREFIXES):
        return True
    if code.startswith(ICD9_INFARCTION_PREFIXES):
        # Require the 5th digit to be 1 ("with infarction"). A truncated code
        # carries no such claim, so it does not qualify.
        return len(code) >= 5 and code[4] == "1"
    return False


def _kw_hit(text) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    if any(p.search(t) for p in KEYWORD_EXCLUSIONS):
        return False
    return any(p.search(t) for p in KEYWORD_PATTERNS)


def select_cohort(
    patient: pd.DataFrame,
    diagnosis: pd.DataFrame,
    window_minutes: int = None,
) -> tuple:
    """
    Select stroke patients.

    Args:
        patient: the patient table
        diagnosis: the diagnosis table
        window_minutes: when given, only diagnoses recorded at or before this
            offset qualify a patient. A stroke first recorded on day 9 cannot
            define a cohort whose features stop at hour 72 (AUDIT.md EICU-07).

    Returns:
        (cohort_rows, report)
    """
    if "icd9code" not in diagnosis.columns and "diagnosisstring" not in diagnosis.columns:
        raise KeyError("diagnosis must contain 'icd9code' and/or 'diagnosisstring'")

    diag = diagnosis
    report = {"diagnoses_total": len(diag)}

    if window_minutes is not None and "diagnosisoffset" in diag.columns:
        diag = diag[diag["diagnosisoffset"] <= window_minutes]
        report["diagnoses_within_window"] = len(diag)
    elif window_minutes is not None:
        report["diagnoses_within_window"] = (
            "diagnosisoffset not available; diagnosis timing not enforced"
        )

    cond_icd = (
        diag["icd9code"].apply(_icd9_hit)
        if "icd9code" in diag.columns
        else pd.Series(False, index=diag.index)
    )
    cond_kw = (
        diag["diagnosisstring"].apply(_kw_hit)
        if "diagnosisstring" in diag.columns
        else pd.Series(False, index=diag.index)
    )

    keep_ids = diag.loc[cond_icd | cond_kw, "patientunitstayid"].unique()
    report["by_icd9"] = int(diag.loc[cond_icd, "patientunitstayid"].nunique())
    report["by_keyword_only"] = int(
        diag.loc[cond_kw & ~cond_icd, "patientunitstayid"].nunique()
    )
    report["stays_selected"] = len(keep_ids)

    return patient[patient["patientunitstayid"].isin(keep_ids)].copy(), report


def apply_observation_window(
    patient: pd.DataFrame,
    window_minutes: int,
    stay_length_col: str = "unitdischargeoffset",
) -> tuple:
    """
    Keep only stays that lasted at least as long as the observation window.

    Without this, a patient who died at hour 6 contributes six hours of data to
    a "72-hour" window, so the length of the record is itself a predictor and
    the task stops being prediction (AUDIT.md EICU-04).

    Returns:
        (filtered_rows, report)
    """
    if stay_length_col not in patient.columns:
        raise KeyError(
            f"'{stay_length_col}' is required to enforce the observation window. "
            "Add it to the patient column whitelist in obs_cleaning.yaml."
        )

    length = pd.to_numeric(patient[stay_length_col], errors="coerce")
    keep = length >= window_minutes
    report = {
        "stays_before": len(patient),
        "stays_shorter_than_window": int((~keep).sum()),
        "stays_after": int(keep.sum()),
    }
    return patient[keep].copy(), report


def extract_mortality_label(patient: pd.DataFrame) -> tuple:
    """
    Build the 'mortality' target and drop the columns that contain the answer.

    Stays whose discharge status is missing or unrecognised are removed rather
    than labelled as survivors: absence of the word "expired" is not evidence
    that the patient lived (AUDIT.md EICU-03).

    Returns:
        (labelled_rows, report)
    """
    df = patient.copy()

    if "hospitaldischargestatus" not in df.columns:
        raise ValueError("'hospitaldischargestatus' column missing in cohort data")

    status = df["hospitaldischargestatus"].astype("string").str.strip().str.lower()
    usable = status.isin(VALID_DISCHARGE_STATUS)

    report = {
        "stays_before": len(df),
        "status_counts": status.value_counts(dropna=False).to_dict(),
        "dropped_unusable_status": int((~usable).sum()),
    }

    df = df[usable].copy()
    df["mortality"] = (status[usable] == "expired").astype(int)

    report["stays_after"] = len(df)
    report["deaths"] = int(df["mortality"].sum())
    report["prevalence"] = round(float(df["mortality"].mean()), 4) if len(df) else None

    return df.drop(columns=LEAK_COLS, errors="ignore"), report

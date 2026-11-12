"""
Cohort selection for heart failure-related cardiogenic shock (HF-CS).

A stay enters the cohort when, within the observation window, it carries
evidence of cardiogenic shock AND evidence of heart failure, and it carries no
diagnosis of acute myocardial infarction (AMI) at any point of the stay. This
follows the clinical separation of HF-CS from AMI-CS (Abraham et al., 2021;
Berg et al., 2024). Surgical admissions can be excluded so that
post-cardiotomy shock, a different population, does not enter the cohort.

Evidence sources:
    diagnosis.icd9code         ICD-9 and/or ICD-10 codes, comma separated
    diagnosis.diagnosisstring  eICU hierarchical problem-list text
    admissionDx.admitdxpath    APACHE admission diagnosis (optional table)
    pastHistory.pasthistorypath  documented history of heart failure (optional)
"""
import re

import numpy as np
import pandas as pd

# --- Codes ----------------------------------------------------------------
# eICU stores ICD-9 and ICD-10 codes in the same field, e.g. "785.51, R57.0".
# Codes are compared after removing dots and upper-casing.
CS_CODE_PREFIXES = ("78551", "R570")         # cardiogenic shock
HF_CODE_PREFIXES = ("428", "I50")            # heart failure
AMI_CODE_PREFIXES = ("410", "I21", "I22")    # acute / subsequent MI

# --- Free text --------------------------------------------------------------
# eICU diagnosis strings are pipe-separated paths, for example
# "cardiovascular|shock / hypotension|shock|cardiogenic", so "cardiogenic" and
# "shock" are matched independently rather than as one phrase.
CS_TEXT_ALL = (re.compile(r"\bcardiogenic\b"), re.compile(r"\bshock\b"))
HF_TEXT_ANY = (
    re.compile(r"heart failure"),
    re.compile(r"\bchf\b"),
    re.compile(r"\bcongestive\b"),
    re.compile(r"\bcardiomyopathy\b"),
)
AMI_TEXT_ANY = (
    re.compile(r"myocardial infarction"),
    re.compile(r"\bstemi\b"),
    re.compile(r"\bnstemi\b"),
)

# APACHE admission diagnosis paths mark surgical admissions with "|Operative|".
OPERATIVE_PATH = re.compile(r"\|operative\|")


def _codes(field) -> list:
    if not isinstance(field, str):
        return []
    return [c.strip().replace(".", "").upper() for c in field.split(",") if c.strip()]


def _code_hit(field, prefixes) -> bool:
    return any(code.startswith(prefixes) for code in _codes(field))


def _text_all(text, patterns) -> bool:
    return isinstance(text, str) and all(p.search(text.lower()) for p in patterns)


def _text_any(text, patterns) -> bool:
    return isinstance(text, str) and any(p.search(text.lower()) for p in patterns)


def _stays(df: pd.DataFrame, mask: pd.Series) -> set:
    return set(df.loc[mask, "patientunitstayid"].unique())


def parse_age(age: pd.Series) -> pd.Series:
    """eICU records ages above 89 as the string '> 89'; map it to 90."""
    s = age.astype("string").str.strip()
    s = s.replace({"> 89": "90"})
    return pd.to_numeric(s, errors="coerce")


def filter_adults(patient: pd.DataFrame, min_age: int = 18) -> tuple:
    """Keep stays with a parseable age >= min_age; age becomes numeric."""
    df = patient.copy()
    df["age"] = parse_age(df["age"])
    keep = df["age"] >= min_age
    report = {
        "stays_before": len(df),
        "age_missing": int(df["age"].isna().sum()),
        "under_min_age": int((df["age"] < min_age).sum()),
        "stays_after": int(keep.sum()),
    }
    return df[keep].copy(), report


def select_hfcs_cohort(
    patient: pd.DataFrame,
    diagnosis: pd.DataFrame,
    window_minutes: int,
    admissiondx: pd.DataFrame = None,
    pasthistory: pd.DataFrame = None,
    exclude_operative: bool = True,
) -> tuple:
    """
    Select HF-CS stays.

    Cardiogenic shock and heart failure evidence must be recorded at or before
    `window_minutes`, so the cohort is defined only by information available at
    prediction time. AMI is excluded if diagnosed at any time in the stay, which
    keeps the cohort clinically homogeneous; this is a retrospective phenotype
    choice and is reported as such.

    Returns:
        (cohort_rows, report)
    """
    for col in ("patientunitstayid", "icd9code", "diagnosisstring", "diagnosisoffset"):
        if col not in diagnosis.columns:
            raise KeyError(f"diagnosis must contain '{col}'")

    # Restrict every evidence table to the stays still in play, so that the
    # counts in the report describe the same population that is returned.
    in_play = set(patient["patientunitstayid"])
    diag_all = diagnosis[diagnosis["patientunitstayid"].isin(in_play)]
    diag_win = diag_all[diag_all["diagnosisoffset"] <= window_minutes]
    if admissiondx is not None:
        admissiondx = admissiondx[admissiondx["patientunitstayid"].isin(in_play)]
    if pasthistory is not None:
        pasthistory = pasthistory[pasthistory["patientunitstayid"].isin(in_play)]

    cs = _stays(diag_win, diag_win["icd9code"].apply(_code_hit, args=(CS_CODE_PREFIXES,))
                | diag_win["diagnosisstring"].apply(_text_all, args=(CS_TEXT_ALL,)))
    hf = _stays(diag_win, diag_win["icd9code"].apply(_code_hit, args=(HF_CODE_PREFIXES,))
                | diag_win["diagnosisstring"].apply(_text_any, args=(HF_TEXT_ANY,)))
    ami = _stays(diag_all, diag_all["icd9code"].apply(_code_hit, args=(AMI_CODE_PREFIXES,))
                 | diag_all["diagnosisstring"].apply(_text_any, args=(AMI_TEXT_ANY,)))

    report = {
        "diagnoses_total": len(diag_all),
        "diagnoses_within_window": len(diag_win),
        "cs_from_diagnosis": len(cs),
        "hf_from_diagnosis": len(hf),
    }

    operative = set()
    if admissiondx is not None and not admissiondx.empty:
        path = admissiondx["admitdxpath"]
        cs_adx = _stays(admissiondx, path.apply(_text_all, args=(CS_TEXT_ALL,)))
        hf_adx = _stays(admissiondx, path.apply(_text_any, args=(HF_TEXT_ANY,)))
        operative = _stays(admissiondx, path.apply(
            lambda t: isinstance(t, str) and bool(OPERATIVE_PATH.search(t.lower()))))
        report["cs_added_by_admissiondx"] = len(cs_adx - cs)
        report["hf_added_by_admissiondx"] = len(hf_adx - hf)
        cs |= cs_adx
        hf |= hf_adx

    if pasthistory is not None and not pasthistory.empty:
        hf_hist = _stays(pasthistory, pasthistory["pasthistorypath"].apply(
            _text_any, args=(HF_TEXT_ANY,)))
        report["hf_added_by_pasthistory"] = len(hf_hist - hf)
        hf |= hf_hist

    both = cs & hf
    report["cs_total"] = len(cs)
    report["cs_and_hf"] = len(both)
    report["excluded_ami"] = len(both & ami)
    selected = both - ami
    if exclude_operative:
        report["excluded_operative_admission"] = len(selected & operative)
        selected -= operative
    report["stays_selected"] = len(selected)

    return patient[patient["patientunitstayid"].isin(selected)].copy(), report


def keep_first_stay(patient: pd.DataFrame) -> tuple:
    """
    Keep one ICU stay per patient (uniquepid): the earliest eligible one.

    eICU has no absolute timestamps. Stays are ordered by discharge year, then
    hospital stay id, then the unit visit number within that hospital stay.
    """
    order = [c for c in ("hospitaldischargeyear", "patienthealthsystemstayid",
                         "unitvisitnumber") if c in patient.columns]
    df = patient.sort_values(["uniquepid"] + order, kind="mergesort")
    first = df.drop_duplicates(subset="uniquepid", keep="first")
    report = {
        "stays_before": len(patient),
        "patients": int(first["uniquepid"].nunique()),
        "repeat_stays_dropped": len(patient) - len(first),
        "ordering_columns": order,
    }
    return first.copy(), report


def apache_iv_prediction(apache: pd.DataFrame) -> pd.Series:
    """
    APACHE IV predicted hospital mortality per stay, for use as a BASELINE only.

    eICU stores one row per APACHE version; IVa is preferred over IV. A value of
    -1 means the score could not be computed and is returned as NaN.
    """
    df = apache.copy()
    df["predictedhospitalmortality"] = pd.to_numeric(
        df["predictedhospitalmortality"], errors="coerce")
    df.loc[df["predictedhospitalmortality"] < 0, "predictedhospitalmortality"] = np.nan
    rank = df["apacheversion"].astype("string").str.strip().map({"IVa": 0, "IV": 1})
    # Drop uncomputable scores first, so IV is used when IVa is unavailable.
    df = (df.assign(_rank=rank)
            .dropna(subset=["_rank", "predictedhospitalmortality"])
            .sort_values("_rank", kind="mergesort"))
    first = df.drop_duplicates(subset="patientunitstayid", keep="first")
    return first.set_index("patientunitstayid")["predictedhospitalmortality"].rename(
        "apache_iv_pred")

"""
Binary treatment exposures within the observation window, for the HF-CS cohort.

Sources:
    infusionDrug.drugname / infusionoffset         continuous infusions
    treatment.treatmentstring / treatmentoffset    documented treatments

Records with offset <= window are used, including negative offsets (treatment
started before unit admission), since all of them precede prediction time.
A stay with no matching record is coded 0: the absence of a documented
treatment is read as no treatment. Hospitals that do not chart infusions in
eICU therefore look untreated; this is a known limitation of the database.
"""
import re

import pandas as pd

# Agent -> pattern. Dopamine is grouped with vasopressors, as eICU's own
# treatment hierarchy does. "\bepinephrine" cannot match inside
# "norepinephrine" because there is no word boundary before its "e".
VASOPRESSORS = {
    "norepinephrine": r"norepinephrine|levophed|noradrenaline",
    "epinephrine": r"\bepinephrine|adrenaline",
    "vasopressin": r"vasopressin",
    "phenylephrine": r"phenylephrine|neo-?synephrine",
    "dopamine": r"dopamine",
}
INOTROPES = {
    "dobutamine": r"dobutamine",
    "milrinone": r"milrinone",
    "levosimendan": r"levosimendan",
}
MECH_VENT = re.compile(r"mechanical ventilation|\bventilator\b|intubat|endotracheal")
# eICU files NIV under the same branch, e.g.
# "pulmonary|ventilation and oxygenation|mechanical ventilation|non-invasive
# ventilation"; it is not invasive ventilation.
NON_INVASIVE = re.compile(r"non-?invasive")
MCS = re.compile(r"balloon pump|\biabp\b|impella|ventricular assist|\bvad\b|ecmo|"
                 r"extracorporeal membrane")

FEATURES = ["vasopressor_24h", "inotrope_24h", "n_vasoactive_24h",
            "mech_vent_24h", "mcs_24h"]


def _agents(text: str, table: dict) -> set:
    t = text.lower()
    return {name for name, pat in table.items() if re.search(pat, t)}


def _long_events(df: pd.DataFrame, text_col: str, offset_col: str,
                 window_minutes: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["patientunitstayid", "text"])
    d = df[pd.to_numeric(df[offset_col], errors="coerce") <= window_minutes]
    d = d[["patientunitstayid", text_col]].dropna()
    return d.rename(columns={text_col: "text"})


def treatment_flags(stay_ids, window_minutes: int, infusion: pd.DataFrame = None,
                    treatment: pd.DataFrame = None) -> pd.DataFrame:
    """
    Returns a frame indexed by patientunitstayid with FEATURES as 0/1 columns
    (n_vasoactive_24h counts distinct vasoactive agents).
    """
    events = pd.concat([
        _long_events(infusion, "drugname", "infusionoffset", window_minutes),
        _long_events(treatment, "treatmentstring", "treatmentoffset", window_minutes),
    ], ignore_index=True)
    events = events[events["patientunitstayid"].isin(set(stay_ids))]

    out = pd.DataFrame(0, index=pd.Index(sorted(set(stay_ids)), name="patientunitstayid"),
                       columns=FEATURES)
    if events.empty:
        return out

    grouped = events.groupby("patientunitstayid")["text"].apply(list)
    for sid, texts in grouped.items():
        vaso, ino = set(), set()
        vent = mcs = False
        for t in texts:
            vaso |= _agents(t, VASOPRESSORS)
            ino |= _agents(t, INOTROPES)
            low = t.lower()
            vent = vent or (bool(MECH_VENT.search(low)) and not NON_INVASIVE.search(low))
            mcs = mcs or bool(MCS.search(low))
        out.loc[sid] = [int(bool(vaso)), int(bool(ino)), len(vaso | ino), int(vent), int(mcs)]
    return out

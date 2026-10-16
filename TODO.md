# TODO

Verification work that needs the eICU tables, which this repository does not
and cannot ship. Each item names the number to produce and what it settles.

None of this is urgent. The data from the original experiment still exists and
can be supplied if the work is ever questioned; these items would settle the
open findings in [AUDIT.md](AUDIT.md) in advance of that rather than in
response to it.

This list lives on `fix` because it closes audit findings rather than adding
method. `mmda_enhance` descends from `fix` and inherits it.

---

## 1. Feature manifest from the original run

**Closed for the tower assignment. Open for the numeric contents.**

The config snapshots and `dualtower.py` settle which columns went where: the
left tower embedded gender, ethnicity and site_id, and age was a right-tower
feature. AUDIT.md EICU-10 records the evidence and the consequences -- that
the embedding widths hold but the paper's feature counts do not (94 inputs in
all, 91 of them in the right tower), and that standardising before the
embedding lookup collapsed each categorical column onto a small number of
indices.

**What is still open.** The right tower took every column that was not at
position 1, 2 or 3, so whether the three `vitalperiodicid_*` columns were among
its 91 inputs still depends on the February 2026 train-ready CSVs, which were
not retained under the Data Use Agreement.

Two ways to close it:

- Re-run stage 1 and stage 2 at the `ccai2026-published` tag against the eICU
  tables and inspect the column list. This reproduces the inputs without
  re-running the experiment.
- Or find the column names in the original logs. The logs checked so far
  record only the width (94 in all, so 91 in the right tower) and the first
  five names.

The corrected pipeline writes `train_ready/DATA/feature_list.json` on every
run, so the question cannot recur.

## 2. Patient overlap between splits

**Settles:** EICU-02.

In the stroke cohort, count `uniquepid` values with more than one
`patientunitstayid`, then intersect the train and test ID sets by `uniquepid`.

A zero intersection means the stay-level key happened to be harmless here. A
non-zero one gives the size of the optimism in the published test metrics.

## 3. Discharge status distribution

**Settles:** EICU-03.

Value counts of `hospitaldischargestatus` in the cohort, nulls included. Any
row that is neither "Alive" nor "Expired" was labelled a survivor.

## 4. Deaths inside the observation window

**Settles:** EICU-04.

Count stays with `unitdischargeoffset` below 72 hours, split by outcome. That
is how many patients contributed a truncated window, and how strongly record
length tracked the outcome.

## 5. Cohort composition

**Settles:** EICU-07.

Three counts: stays entering by keyword only, stays entering on 433.x0 /
434.x0 / 437 alone, and stays whose qualifying diagnosis was recorded after
hour 72. Plus the ischaemic / haemorrhagic / SAH breakdown.

## 6. Lab unit consistency

**Settles:** EICU-08.

Distinct `labmeasurenamesystem` values per `labname` across hospitals. Any
analyte with more than one unit was averaged across units in the published
features.

`harmonise_lab_units` already reports this when stage 1 runs.

## 7. Missingness mechanism

**Settles:** EICU-09. **Branch:** already implemented on `mmda_enhance`.

Run `missingness_report` with `by=["site_id"]` and `by=["mortality"]`. A gap
uniform across hospitals is plausibly about patients; one that is not is about
sites, and only the second threatens a multi-centre model's transfer.

---

## Not planned

- Re-running the published experiment. The paper is final.
- Multiple imputation, lab unit conversion, lowering the 70% filter. Reasons
  in [MMDA.md](MMDA.md).

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

**Settles:** EICU-01, EICU-10, and the paper's own numbers.

Count the categorical and numeric features the published run actually used.

- If categorical is 3 and numeric is 94, the paper's `h_cat` in R^22 and
  "all 97 features" hold, and `site_id` was not a model feature at the time.
- If categorical is 4, then `h_cat` was not 22 and the totals in the paper do
  not describe the run.
- Separately, check whether `vitalperiodicid_mean`, `_std` or `_slope` appear
  among the numeric features. If they do, three of the "94-dimensional vector
  of continuous physiological vitals" were not physiological.

The corrected pipeline now writes this to `train_ready/DATA/feature_list.json`
on every run, so the question cannot recur.

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

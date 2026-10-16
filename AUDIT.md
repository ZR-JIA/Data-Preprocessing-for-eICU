# Preprocessing Audit

Independent audit of the preprocessing pipeline that produced the data for:

> **Deep Learning for Stroke Mortality Prediction in eICU: A Dual-Tower
> Transformer Framework**, CCAI 2026 (IEEE).

The paper is published and is not being revised. This audit exists because the
repository is its supplementary material, and supplementary material should be
correct even when the paper it accompanies is final.

## How to read this

Every finding carries one of three states:

| State | Meaning |
|---|---|
| **Confirmed** | The source alone proves the behaviour exists. |
| **Unverified** | A plausible risk, but no evidence yet that it changed results. |
| **Design question** | Not necessarily wrong; depends on the clinical definition. |

A defect in this repository does **not** by itself prove the published results
are affected. Whether the published run used exactly this code path is stated
per finding where it can be established.

Several findings can only be closed with the eICU tables in hand. Those are
marked **needs data** and list the exact statistic that would settle them.

## Branches

| Branch | Contents |
|---|---|
| `main` | The state corresponding to publication. Tagged `ccai2026-published`. |
| `fix` | Corrections to confirmed defects. Changes results where a defect inflated them. |
| `mmda_enhance` | Methodological additions that deliberately depart from the paper's feature schema. |

`main` carries the defects described below. It is kept that way on purpose.

## Reference values fixed by the paper

The paper pins four numbers that any corrected pipeline can be measured
against. They are internally consistent (16+2+4 = 22, 22+64 = 86, 94+3 = 97).

| Quantity | Source | Value |
|---|---|---|
| Categorical features | Left tower: Age 16-d, Gender 2-d, Ethnicity 4-d | 3, h_cat in R^22 |
| Numeric features | Right tower: "94-dimensional vector of continuous vitals" | 94 |
| Total features | Discussion: "all 97 features in a single stream" | 97 |
| Mortality prevalence | Fig. 5: "Baseline (Prevalence = 0.1403)" | 14.03% |
| Seeds | Table III footnote: "averaged over 5 random seeds" | 5 |

## Findings

### EICU-01 - Database row identifiers reached the feature matrix

**Confirmed.** `load_data()` read every column of every raw CSV, so the
`tables` whitelist in `obs_cleaning.yaml` was inert. `aggregate_vitals()` then
aggregated every numeric column it found, including `vitalperiodicid` - the
auto-increment primary key of `vitalPeriodic`. That produced three features:
`vitalperiodicid_mean`, `_std` and `_slope`.

The chain is closed end to end:

1. the loader reads the column (source);
2. aggregation emits the three features (demonstrated by
   `tests/test_eicu01_identifier_columns.py`, which fails on the pre-fix code
   naming all three);
3. a primary key has no missing values, so the >70% sparsity filter could never
   drop it;
4. stage 2 places anything not in `exclude_cols` or `cat_cols` into `num_cols`,
   so all three entered the right tower.

`vitalperiodicid_slope` is of particular concern: with the two-point slope used
here it reduces to the rate at which monitoring rows were written for that
patient, which plausibly tracks acuity. It could raise AUPRC for a reason that
is not clinical.

Neither entry point applied a whitelist -- `load_data()` took no column list at
all -- so whichever path a run took, `vitalperiodicid` was loaded.

Fixed on `fix`. **Needs data** to quantify: a feature manifest from a completed
stage-1 run.

### EICU-02 - "Patient-level" splits were keyed on the ICU stay

**Confirmed.** `patient_id` was derived from `patientunitstayid`, one ICU stay.
eICU gives one `uniquepid` several stays on readmission, so one person could be
in train and test at once. `uniquepid` appeared nowhere in the repository.

`group_cols` defaulted to `['patient_id', 'site_id']`, so a patient seen at two
hospitals became two groups and the grouping did not bind them.

Fixed on `fix`. On the pre-fix code, 58 of 120 synthetic patients straddle two
splits; that cohort gives every patient two stays, so it bounds the mechanism,
not the real rate. **Needs data**: the number of `uniquepid` values with more
than one stay in the stroke cohort, and the train/test intersection by
`uniquepid`.

### EICU-03 - Missing discharge status is silently labelled "survived"

**Confirmed as behaviour.** The label is built with
`1 if 'expired' in str(status).lower() else 0`. A null `hospitaldischargestatus`
stringifies to `"nan"`, does not contain `expired`, and becomes 0. Any
unrecognised status also becomes 0.

On a cohort with ~14% mortality, misclassified positives move AUPRC, recall and
calibration. **Needs data**: the value counts of `hospitaldischargestatus` in
the cohort, including nulls.

### EICU-04 - The prediction task is not pinned down in time

**Confirmed as behaviour; design question as to intent.** `align_time_window`
keeps records in [0, 72h] and nothing requires the patient to survive the
window or to have a minimum length of stay. Three different tasks are
compatible with the code, and they are not the same clinical question:

1. predict eventual death from admission;
2. predict death *after* 72h using the first 72h;
3. classify the whole admission retrospectively using whatever was recorded.

A patient who dies at hour 6 contributes six hours of data to a "72-hour"
window, so observation length itself correlates with the outcome. The paper
positions the work as early warning, which is reading (2).

**Needs data**: how many deaths occur inside the window.

### EICU-05 - The declared feature scope was never enforced

**Confirmed.** Covered by EICU-01: the `tables` block had no effect. Fixed on
`fix`, where a column named in config but absent from the CSV now raises rather
than being dropped silently.

### EICU-06 - Two outlier rules never executed

**Confirmed.** The config limited `map` and `spo2`. Those columns do not exist
in eICU `vitalPeriodic`; the real names are `systemicmean` and `sao2`.
`apply_outlier_nan()` skips unknown columns without warning, so both rules were
dead. The no-config CLI path used the correct names, so which path a run took
determined whether the limits applied at all.

Fixed on `fix`: names corrected, and the two entry points now share one
definition.

### EICU-07 - The stroke cohort is drawn too broadly

**Design question.** Selection is an ICD-9 prefix match on
`430/431/433/434/436/437` **or** a substring match on
`stroke`, `cva`, `ich`, `sah` in `diagnosisstring`.

- `433.x0` and `434.x0` denote occlusion or stenosis *without* infarction;
  `437` is "other and ill-defined cerebrovascular disease". Matching three
  digits admits all of them.
- Substring matching has no word boundary, so `stroke` also matches
  **heat stroke**, a real ICU diagnosis.
- `diagnosis` rows carry `diagnosisoffset` and the code ignores it, so a stroke
  recorded on day 9 still qualifies a patient whose 72-hour window long
  preceded the diagnosis.

**Needs data**: the count entering by keyword only, the subtype breakdown, and
the number diagnosed after the observation window.

### EICU-08 - Labs receive no outlier handling and no unit reconciliation

**Confirmed.** `apply_outlier_nan` is called once, on vitals. Labs get nothing.
`aggregate_labs` pivots on `labname` alone and ignores
`labmeasurenamesystem`, although eICU documents `labname` as hospital-specific
with units in a separate column. Same-named analytes recorded in different
units are averaged together.

**Needs data**: distinct units per `labname` across hospitals.

### EICU-09 - Missingness is treated purely as a gap to fill

**Design question.** Columns above 70% missing are dropped, the rest get
train-fitted median imputation. No missingness indicators are retained. In
eICU, whether a lab was drawn reflects clinical suspicion and which interfaces
a hospital deployed, so missingness is unlikely to be random. Median-imputing a
`*_std` column is the sharpest case: a patient with one measurement has no
variability, and the cohort median asserts a typical one.

Addressed on `mmda_enhance`, not on `fix`: adding indicators changes the feature
schema the paper describes.

### EICU-10 - The left tower does not hold the features the paper names

**Resolved by evidence, and not in the direction first assumed.**

The config snapshots kept with the original experiment name three categorical
features -- `age`, `gender`, `ethnicity`, with vocabularies 150/10/50 and
embedding widths 16/2/4. Those names are labels. The model selects columns by
**position**:

```python
cat_idx_set = set(f['index'] for f in self.cat_features_config)   # {1, 2, 3}
self.num_indices = [i for i in range(input_dim) if i not in cat_idx_set]
...
col_input = torch.clamp(x[:, idx], min=0, max=feat['num_classes'] - 1).long()
```

Stage 2 emits `patient_id, mortality, age, gender, ethnicity, site_id, <numeric>`.
The model drops the target and `drop_cols = [patient_id, hospital_id, unit_id]`;
`hospital_id` and `unit_id` do not exist under those names, so nothing else
goes. What remains is

| position | column | receives |
|---|---|---|
| 0 | age | **right tower** |
| 1 | gender | vocab 150, emb 16 (configured for "age") |
| 2 | ethnicity | vocab 10, emb 2 (configured for "gender") |
| 3 | site_id | vocab 50, emb 4 (configured for "ethnicity") |

So the left tower embedded **gender, ethnicity and site_id**, and age was
processed as a numeric feature. The paper's Section III-B-1 and Table I
describe the intended assignment, not the one that ran.

**The embedding arithmetic holds; the feature counts do not.** Embedding
widths follow position too, so `h_cat` is still 16+2+4 = 22 and `h_joint` is
still 22+64 = 86. But the published runs had 94 input columns in all, not 97:
an Optuna log written the same day from the same train-ready files records
`Feature count: 94`, with the columns beginning `age, gender, ethnicity,
site_id, admissionheight`, and the model repository hard-codes
`input_dim = 94` for its NN baseline. The left tower took three of them and
the right tower the other 91, each column used once. The paper's 94 for the
right tower is the total, and its 97 adds the three categorical columns a
second time. No reported performance number changes; what differs is which
feature sits in which tower, and the two counts.

**A second consequence.** `preprocessor.py` fits `StandardScaler` on the whole
feature matrix. Stage 2 had already label-encoded the categorical columns to
integers, so `select_dtypes(['object', 'category'])` finds nothing, no one-hot
encoding happens, and the scaler standardises the categorical columns along
with everything else. The embedding layers therefore receive standardised
floats, clamped to the vocabulary range and then truncated by `.long()`.

Standardised values cluster around zero. Every category whose code lies below
its column mean is clamped to index 0, and the remaining categories reach only
the lowest few indices; exactly how many depends on each column's distribution,
which the surviving records do not include. Most of each embedding table
(150x16, 10x2, 50x4) was therefore never used, and distinct categories -- the
site_id values in particular -- share indices.

**Still confirmed as a repository defect.** Unseen categories are mapped to
`encoder.classes_[0]`, a real hospital or ethnicity, rather than to a reserved
index. Fixed on `fix`.

### EICU-11 - Sparsity filtering happens before the split

**Design question, and in our view acceptable.** The missing rate is computed on
the full cohort. It is a structural property rather than a learned statistic,
and 70% is a conservative threshold. Worth documenting with the list of dropped
columns rather than changing.

### EICU-12 - A column entirely missing in train is unguarded

**Confirmed, low severity.** `SimpleImputer` defaults to
`keep_empty_features=False` and drops such columns at fit, after which
`pd.DataFrame(arr, columns=num_cols)` raises on the shape mismatch. The failure
mode is a loud crash, not silent dimension loss. The upstream 70% filter makes
it very unlikely in practice.

### EICU-13 - Two split protocols are advertised but unreachable

**Confirmed.** The temporal protocol cannot run at all: stage 1 drops
`admission_time` as noise and never carries `unitadmitoffset` into the output,
so `TemporalHoldoutSplitter` has no time column to use. The exporter still
creates permanently empty `SPLITS/temporal/` and `PACKS/temporal/`.

The synthetic `site_id` fallback in stage 2 (`patient_id % 10`) is **not** a
live risk: stage 1 raises when `hospitalid` is absent, so `site_id` is always
present. The original concern does not hold.

### EICU-14 - A run cannot be reproduced from what it writes

**Confirmed.** `n_seeds: 5` is configured but `run()` hardcodes `seed=42` and
executes only the patient-level protocol. Of the exporter's five writers only
`save_split_indices` and `save_split_data` are ever called, so `DATA/` and
`REPROD/` are created and left empty: no feature manifest, no data version, no
seed record. The `LabelEncoder`, `SimpleImputer` and `StandardScaler` are local
variables, discarded at the end of the function, so the fitted transform cannot
be reapplied to new data.

The missing feature manifest is why EICU-01 cannot be quantified from the
repository alone.

### EICU-15 - The QC script cannot detect this repository's main defect

**Confirmed.** `data_health_check.py` counts unique patients with
`patientunitstayid`, repeating EICU-02's error. Its leakage screen is a
correlation threshold of 0.9 plus a list of suspicious column-name keywords.
`vitalperiodicid_mean` correlates with mortality nowhere near 0.9 and matches no
keyword, so the script reports a clean bill of health while EICU-01 passes
straight through it. A clean result from this script is not evidence of absence.

## Additional findings from the code review

| # | Location | Issue |
|---|---|---|
| R-1 | `config/obs_cleaning.yaml`, `Makefile` | `input_raw: "inputs/Observative"` resolves to `data_preprocessing/inputs/Observative`, which does not exist; the data sits at `Input/Observative` in the repository root. Following the README verbatim fails immediately. |
| R-2 | `config/environment.yaml` | scikit-learn is missing, yet stage 2 imports `LabelEncoder`, `StandardScaler`, `SimpleImputer` and `train_test_split`. The conda path produces an environment that cannot run the pipeline; the README's pip fallback does list it. |
| R-3 | `obs_cleaning/aggregate.py` | `_trend_slope` is a two-point secant, `(last - first) / dt`. With ~860 samples over 72h it discards everything between and is maximally sensitive to two noisy endpoints. |
| R-4 | `obs_trainReady/train_ready_exporter.py` | `save_split_indices` indexes a label-based `split_indices` with positional `.iloc[]`. Correct only while the frame carries a clean `RangeIndex`. |
| R-5 | `obs_trainReady/data_splitting.py` | `SiteHoldoutSplitter._select_test_sites` appends before testing the budget, overshooting by up to one hospital; it then splits rows directly, with no patient grouping. |
| R-6 | `Makefile` | `clean` removes `outputs/` and `train_ready/` at the repository root, but both stages write beneath `data_preprocessing/`. |
| R-7 | `obs_trainReady/preprocessing.py` | `TrainOnlyPreprocessor` (415 lines) is never imported. Stage 2 does its own imputation and scaling inline. The inline code is correctly fit-on-train, so results are unaffected, but the audited-looking class is not what runs. |
| R-8 | `README.md` | The BibTeX entry says `note = {Accepted}` and the citation section promises BibTeX "once the proceedings are published". The paper is on IEEE Xplore. |

## Note on history

The commit history was rewritten once, after publication, to replace Chinese
comments, docstrings and log messages with English. Every rewritten file was
verified to parse and to yield an identical AST under a comparison that masks
string constants and f-string prose while preserving interpolated expressions.
The single intentional exception is two padding integers in the
`qc/data_health_check.py` banner, widened because the English title has a
different display width.

Commit identifiers from before that rewrite are therefore no longer resolvable.
The code at `ccai2026-published` is functionally the published code, but is not
byte-identical to what was originally committed.

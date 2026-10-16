# eICU Stroke Data Preprocessing

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

Data preprocessing pipeline for the paper:
**[Dual-Tower Transformer for ICU Stroke Mortality Prediction](https://github.com/ZR-JIA/Dual-Tower-Transformer-eICU-Stroke)**

This repository takes raw eICU CSVs to train-ready splits. The model training
code lives in the paper repository above.

---

## Start here: which branch do you want?

The paper is published and is not being revised. This repository is its
supplementary material, and supplementary material should be correct even when
the paper it accompanies is final. That is why there are three branches.

| Branch | What it is | Use it when |
|---|---|---|
| **`main`** | The state corresponding to publication. Tagged `ccai2026-published`. | You want to see what produced the published numbers. |
| **`fix`** | Corrections to the defects found in audit. | You want to build on this work, or run the pipeline yourself. |
| **`mmda_enhance`** | Methodological additions that deliberately depart from the paper's feature schema. See **[MMDA.md](MMDA.md)**. | You want the missing-data treatment. |

> **You are on `mmda_enhance`.** It builds on `fix` and adds missingness
> indicators, which change the feature count on purpose. [MMDA.md](MMDA.md)
> explains what changes and what it costs; set `missingness.indicators: false`
> to get `fix`'s behaviour back.

`main` carries known defects, on purpose. They are catalogued in
**[AUDIT.md](AUDIT.md)**, each with its evidence and its status, and the
checks still outstanding are listed in **[TODO.md](TODO.md)**. Do not read a
difference between `main` and `fix` as a correction to the paper's conclusions:
several findings cannot be quantified without the eICU tables, and AUDIT.md
says so for each one.

Corrections to the paper, including the model's categorical tower, are recorded
in the model repository: see [Corrections to the CCAI 2026 paper](https://github.com/ZR-JIA/Dual-Tower-Transformer-eICU-Stroke#corrections-to-the-ccai-2026-paper-october-2026).

**Reproducing the paper**: start from the tag `ccai2026-published`.
**Doing new work**: start from `fix`.

### Why `fix` does not reproduce the published numbers

It is not meant to. Three corrections change the cohort itself:

- stays shorter than the observation window are excluded;
- stays whose discharge status is missing are dropped rather than labelled as
  survivors;
- the stroke definition no longer admits chronic stenosis, ill-defined
  cerebrovascular disease, or heat stroke.

A fourth keeps three aggregates of a database row identifier out of the feature
matrix (AUDIT.md EICU-01).

---

## What this does

**Stage 1, cleaning** (`run_obs_clean.py`): raw eICU CSVs to `obs_cleaned.csv`

- selects stroke patients by ICD-9 and by diagnosis text, requiring the
  diagnosis to fall inside the observation window
- keeps the first 72 hours of vitals and labs per ICU stay
- requires the stay to span that window
- blanks implausible values in vitals and in labs, and reduces each analyte to
  a single unit
- aggregates vitals (mean, std, least-squares slope) and labs (mean, std)
- builds the mortality label, then deletes the columns it came from

**Stage 2, train-ready** (`run_obs_trainReady.py`): `obs_cleaned.csv` to splits

- three protocols: patient-stratified, site holdout, temporal
- splits are grouped on `uniquepid`, so one person never spans two splits
- dual-tower encoding: three categorical features to label indices with a
  reserved `<UNK>` slot, numeric features to median imputation and standard
  scaling
- every statistic is fitted on train only
- runs each configured seed and writes the fitted transforms with the data

---

## Requirements

Python 3.10.

```bash
conda env create -f data_preprocessing/config/environment.yaml
conda activate Preprocessing
```

Or with pip:

```bash
pip install pandas numpy scikit-learn pyyaml tqdm pyarrow
```

---

## Data

You need the [eICU Collaborative Research Database](https://eicu-crd.mit.edu/),
which requires PhysioNet credentialing. Place these four files in
`Input/Observative/`:

```
patient.csv
diagnosis.csv
vitalPeriodic.csv
lab.csv
```

Only the columns listed in `config/obs_cleaning.yaml` are read, and that list
is enforced: a column named there but absent from the CSV raises rather than
being skipped. This is deliberate. The whitelist used to be inert, which is how
the `vitalPeriodic` primary key became three model features.

---

## Usage

```bash
make all          # both stages
make stage1       # raw eICU    -> outputs/obs_cleaned.csv
make stage2       # obs_cleaned -> train_ready/
make test         # the test suite
make check        # optional QC on the cleaned data
```

Or directly, from `data_preprocessing/`:

```bash
python run_obs_clean.py --config config/obs_cleaning.yaml
python run_obs_trainReady.py --config config/obs_trainReady.yaml
```

Paths in both configs are written relative to `data_preprocessing/`, which is
where the Makefile runs them from.

### Output

```
outputs/obs_cleaned.csv

train_ready/
  DATA/      feature_list.json, data_version.json
  SPLITS/    <protocol>/seed_<n>/{train,val,test}_ids.csv
  PACKS/     <protocol>/seed_<n>/{train,val,test}.csv
                                 transforms.pkl
  REPROD/    seeds.txt
```

Protocols are `patient_stratified`, `site_holdout` and `temporal`.
`transforms.pkl` holds the fitted encoders, imputer and scaler, so the same
transform can be reapplied to new data.

---

## Project structure

```
data_preprocessing/
  config/
    obs_cleaning.yaml       column whitelist, window, outlier limits
    obs_trainReady.yaml     split config, seeds
    environment.yaml        conda dependencies
  obs_cleaning/             Stage 1
    load_data.py            reads the whitelisted columns, strictly
    cohort.py               ICD-9 and text selection, window, label
    timealign.py            the 72h window
    outliers.py             range checks for vitals and for labs
    aggregate.py            per-patient mean / std / slope, unit handling
    impute.py               drops sparse columns; does not impute
    save_data.py
  obs_trainReady/           Stage 2
    data_splitting.py       patient / temporal / site protocols
    preprocessing.py        library utilities, NOT used by the pipeline
    train_ready_exporter.py
    reproducibility.py      seeds, hashes, experiment records
    utils.py
  qc/                       optional checks
    scan_features.py
    data_health_check.py
  run_obs_clean.py          Stage 1 entry point
  run_obs_trainReady.py     Stage 2 entry point
tests/                      regression tests, one file per audit finding
```

Each test file is named for the findings it pins. The tests that can run
against both versions fail on `main` and pass on `fix`.

---

## Notes on leakage

- `hospitaldischargestatus` builds the label and is deleted immediately, along
  with the other discharge columns.
- Imputation, scaling and encoding are fitted on the training split only.
- Splits are grouped on `uniquepid`, the person, not on `patientunitstayid`,
  the stay.
- Sparse-column dropping (>70% missing) happens before the split. The missing
  rate is a structural property rather than a learned statistic, and the
  threshold is conservative. AUDIT.md EICU-11 explains why we treat this as
  acceptable rather than as a defect.
- `qc/data_health_check.py` screens for leakage, but a clean result from it is
  not evidence of absence. AUDIT.md EICU-15 shows it missing this repository's
  own worst case.

---

## A note on the history

The commit history was rewritten after publication to translate comments,
docstrings and log messages into English and to normalise commit metadata. Every rewritten file was
verified to parse and to yield an identical AST under a comparison that masks
string constants and f-string prose while preserving interpolated expressions.

Commit identifiers from before these rewrites no longer resolve. The code at
`ccai2026-published` is functionally the published code, but is not
byte-identical to what was originally committed.

---

## Paper

> **Deep Learning for Stroke Mortality Prediction in eICU: A Dual-Tower
> Transformer Framework**
>
> Zhengrong Jia\* (Asia AI Education and Future Technology Association,
> Hong Kong SAR, China)
> Kwong-Cheong Wong\* (School of Governance and Policy Science, The Chinese
> University of Hong Kong, Hong Kong SAR, China)
>
> \*Corresponding authors
>
> In *2026 the 6th International Conference on Computer Communication and
> Artificial Intelligence (CCAI)*, IEEE.
>
> 📄 **[IEEE Xplore](https://ieeexplore.ieee.org/document/11641950/)**

```bibtex
@INPROCEEDINGS{11641950,
  author={Jia, Zhengrong and Wong, Kwong-Cheong},
  booktitle={2026 6th International Conference on Computer Communication and Artificial Intelligence (CCAI)},
  title={Deep Learning for Stroke Mortality Prediction in eICU: A Dual-Tower Transformer Framework},
  year={2026},
  volume={},
  number={},
  pages={76-81},
  doi={10.1109/CCAI69603.2026.11641950}}
```

To cite this repository, reference the tag rather than a branch. Branches move;
tags do not.

---

## Acknowledgment

We would like to express our gratitude to the eICU Collaborative Research
Database team for making the multi-center critical care data publicly
available, which was essential for this study. We also thank the Macau
University of Science and Technology for providing access to the academic
databases and literature resources that supported this research.

---

## License

This project is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
See the [LICENSE](LICENSE) file for full terms, including the Medical AI
Disclaimer.

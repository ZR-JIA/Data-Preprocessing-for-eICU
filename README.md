# eICU Stroke Data Preprocessing

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

Data preprocessing pipeline for the paper:
**[Dual-Tower Transformer for ICU Stroke Mortality Prediction](https://github.com/ZR-JIA/Dual-Tower-Transformer-eICU-Stroke)**

This repo handles everything from raw eICU CSVs to train-ready splits. The model training code lives in the paper repo above.

---

## What this does

Two stages:

1. **Cleaning** (`run_obs_clean.py`) — raw eICU → `obs_cleaned.csv`
   - Selects stroke patients via ICD-9 codes (430/431/433/434/436/437) + keyword match
   - Keeps only the first 72 hours of vitals/labs per ICU stay
   - Replaces physiological outliers with NaN
   - Aggregates vitals (mean, std, slope) and labs (mean, std)
   - Generates mortality label from `hospitaldischargestatus`, then drops it

2. **Train-ready** (`run_obs_trainReady.py`) — `obs_cleaned.csv` → `train/val/test.csv`
   - Patient-level stratified split (70/10/20)
   - Dual-tower encoding: categorical features → LabelEncoder, numeric → Median impute + StandardScaler
   - All stats (median, scaler params) are fit on train only

---

## Requirements

```
python 3.10
```

Create the conda environment:

```bash
conda env create -f data_preprocessing/config/environment.yaml
conda activate Preprocessing
```

Or just pip:

```bash
pip install pandas numpy scikit-learn pyyaml tqdm pyarrow
```

---

## Data

You need access to the [eICU Collaborative Research Database](https://eicu-crd.mit.edu/). It requires PhysioNet credentialing.

Put these four files in `Input/Observative/`:

```
patient.csv
diagnosis.csv
vitalPeriodic.csv
lab.csv
```

---

## Usage

### Step 1 — Clean

```bash
cd data_preprocessing
python run_obs_clean.py --config config/obs_cleaning.yaml
```

Output: `outputs/obs_cleaned.csv`

### Step 2 — Train-ready splits

```bash
python run_obs_trainReady.py --config config/obs_trainReady.yaml
```

Output: `train_ready/PACKS/patient_stratified/train.csv` (+ val.csv, test.csv)

Or just run both with make:

```bash
make all
```

---

## Project structure

```
data_preprocessing/
├── config/
│   ├── obs_cleaning.yaml       # paths, time window, outlier limits
│   ├── obs_trainReady.yaml     # split config, random seed
│   └── environment.yaml        # conda deps
├── obs_cleaning/               # Stage 1 modules
│   ├── load_data.py
│   ├── cohort.py               # ICD-9 filter + mortality label
│   ├── timealign.py            # 72h window
│   ├── outliers.py
│   ├── aggregate.py            # mean/std/slope per patient
│   ├── impute.py               # drop sparse cols (no imputation here)
│   └── save_data.py
├── obs_trainReady/             # Stage 2 modules
│   ├── data_splitting.py       # patient / temporal / site splits
│   ├── preprocessing.py        # leakage-free scaler/imputer
│   ├── train_ready_exporter.py
│   └── utils.py
├── qc/                         # data quality checks (optional)
│   ├── scan_features.py
│   └── data_health_check.py
├── run_obs_clean.py            # Stage 1 entry point
└── run_obs_trainReady.py       # Stage 2 entry point
```

---

## Notes on leakage prevention

- `hospitaldischargestatus` is used to build the label and then immediately deleted
- Imputation (median/mode) is fit only on the training split
- Sparse column dropping (>70% missing) happens before splitting — this is a structural decision, not a learned statistic, and is standard in eICU preprocessing literature

---

## Citation

If you use this pipeline, please cite the paper (link above). BibTeX will be added once the proceedings are published.

## Paper

> **Deep Learning for Stroke Mortality Prediction in eICU: A Dual-Tower Transformer Framework**
>
> Zhengrong Jia\* (Asia AI Education and Future Technology Association, Hong Kong SAR, China)
> Kwong-Cheong Wong\* (School of Governance and Policy Science, The Chinese University of Hong Kong, Hong Kong SAR, China)
>
> \*Corresponding authors
>
> 📄 **[Paper Link](https://drive.google.com/file/d/1yl0XyzkV9QuHWtkpBu8RyXFypU3Ptb8b/view)**

If you find this work useful, please cite:

```bibtex
@inproceedings{Jia2026DualTower,
  title     = {Deep Learning for Stroke Mortality Prediction in eICU: A Dual-Tower Transformer Framework},
  author    = {Jia, Zhengrong and Wong, Kwong-Cheong},
  booktitle = {Proceedings of the CCAI Conference},
  year      = {2026},
  note      = {Accepted}
}
```

---

## Acknowledgment

We would like to express our gratitude to the eICU Collaborative Research Database team for making the multi-center critical care data publicly available, which was essential for this study. We also thank the Macau University of Science and Technology for providing access to the academic databases and literature resources that supported this research.

---

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/). See the [LICENSE](LICENSE) file for full terms, including the Medical AI Disclaimer.

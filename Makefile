PYTHON = python
CONFIG_CLEAN   = config/obs_cleaning.yaml
CONFIG_READY   = config/obs_trainReady.yaml
WORKDIR        = data_preprocessing

.PHONY: all clean stage1 stage2 env check

## Run both stages
all: stage1 stage2

## Stage 1: raw eICU -> obs_cleaned.csv
stage1:
	cd $(WORKDIR) && $(PYTHON) run_obs_clean.py --config $(CONFIG_CLEAN)

## Stage 2: obs_cleaned.csv -> train/val/test splits
stage2:
	cd $(WORKDIR) && $(PYTHON) run_obs_trainReady.py --config $(CONFIG_READY)

## Create conda environment
env:
	conda env create -f $(WORKDIR)/config/environment.yaml
	@echo "Run: conda activate Preprocessing"

## Run QC checks on cleaned data
check:
	cd $(WORKDIR) && $(PYTHON) qc/scan_features.py
	cd $(WORKDIR) && $(PYTHON) qc/data_health_check.py

## Delete generated outputs (keeps raw input)
clean:
	rm -f outputs/obs_cleaned.csv
	rm -rf train_ready/
	@echo "Cleaned outputs. Raw input untouched."

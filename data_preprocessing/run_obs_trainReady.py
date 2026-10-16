"""
Train-ready preparation for the dual-tower model.

1. Read obs_cleaned.csv
2. Apply the patient-level / temporal / site split
3. Dual-tower preprocessing:
   - left tower  -> label encoding (0, 1, 2 ... for the embedding)
   - right tower -> median imputation + standard scaling
4. Write train.csv, val.csv and test.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Make the package importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core utilities
from obs_trainReady import (
    PatientLevelSplitter,
    TemporalHoldoutSplitter,
    SiteHoldoutSplitter,
    SeedManager,
    TrainReadyExporter,
    DataVersionManager,
)
from obs_trainReady.utils import (
    load_config,
    identify_feature_types,
    print_data_summary
)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reserved index for categories absent from the training split.
UNKNOWN_CATEGORY = "<UNK>"



class ObsTrainReadyPipeline:
    """
    Pipeline that turns cleaned data into dual-tower training input.
    """
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.seed_manager = SeedManager(
            base_seed=self.config.get('random_seed', 42),
            n_seeds=self.config.get('n_seeds', 5)
        )
        
        # Paths
        paths = self.config.get('paths', {})
        self.output_dir = paths.get('output_train_ready', '../../train_ready')
        self.exporter = TrainReadyExporter(base_dir=self.output_dir)
        
        # Protocol name mapping
        self.protocol_name_map = {
            'patient_level': 'patient_stratified',
            'temporal': 'temporal',
            'site': 'site_holdout'
        }
        
    def load_data(self) -> pd.DataFrame:
        """Load the cleaned data and adapt it for the dual-tower model."""
        paths = self.config.get('paths', {})
        # Must point at the cleaned output, not a raw table
        data_path = paths.get('input_cleaned', '../../outputs/obs_cleaned.csv')
        logger.info(f"Loading data: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}. Run run_obs_clean.py first.")

        df = pd.read_csv(data_path)
        
        # --- Required columns ---
        required = ['patientunitstayid', 'mortality']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Required column missing from the data: {col}")

        # patient_id must identify the PERSON, not the ICU stay. eICU gives one
        # uniquepid several patientunitstayid values when a patient is admitted
        # more than once; keying splits on the stay let the same person appear in
        # both train and test (AUDIT.md EICU-02).
        if 'uniquepid' not in df.columns:
            raise ValueError(
                "obs_cleaned.csv has no 'uniquepid' column. Regenerate it with "
                "run_obs_clean.py from this branch: output produced by the "
                "previous cleaner cannot support a patient-level split."
            )
        df['patient_id'] = df['uniquepid'].astype(str)
        
        # site_id is needed by the left tower
        if 'site_id' not in df.columns:
            logger.warning("site_id is missing; falling back to a synthetic value")
            # Synthetic stand-in so the run completes. Stage 1 raises when
            # hospitalid is absent, so this should be unreachable in practice.
            df['site_id'] = (df['patient_id'].astype(int) % 10).astype(str)
        else:
            df['site_id'] = df['site_id'].astype(str)

        print_data_summary(df, "Train-ready raw input")
        return df

    def _dual_tower_preprocess(self, train_df, val_df, test_df):
        """
        Dual-tower preprocessing.

        Left tower  -> label encoding, with unseen categories handled.
        Right tower -> median imputation + standard scaling.
        Everything is fitted on train only and applied to val/test.
        """
        logger.info("Starting dual-tower preprocessing (fit on train)...")
        encoders = {}
        
        target_col = 'mortality'
        exclude_cols = ['patient_id', 'patientunitstayid', 'uniquepid',
                        'site_id', 'admission_time', '_group_key', target_col]
        
        # 1. Left tower: categorical features, encoded to 0, 1, 2 ... indices.
        # The paper's left tower has three: Age (16-d), Gender (2-d) and
        # Ethnicity (4-d), concatenated to h_cat in R^22. site_id used to be a
        # fourth, which cannot produce 22 dimensions, so it is no longer a
        # model feature -- it stays on the frame only so the site protocol can
        # split on it (AUDIT.md EICU-10).
        cat_cols = ['age', 'gender', 'ethnicity']
        # Keep only the ones that exist
        cat_cols = [c for c in cat_cols if c in train_df.columns]
        
        # 2. Right tower: everything that is not an ID, the label, or categorical
        all_cols = train_df.columns.tolist()
        num_cols = [c for c in all_cols if c not in exclude_cols + cat_cols]
        
        logger.info(f"Feature split: left tower (cat)={len(cat_cols)}, right tower (num)={len(num_cols)}")
        
        # Copy so the caller's frames are untouched
        X_train_cat, X_val_cat, X_test_cat = train_df[cat_cols].copy(), val_df[cat_cols].copy(), test_df[cat_cols].copy()
        X_train_num, X_val_num, X_test_num = train_df[num_cols].copy(), val_df[num_cols].copy(), test_df[num_cols].copy()
        
        # --- Left tower: label encoding ---
        for col in cat_cols:
            # Missing categories become 'Unknown'
            X_train_cat[col] = X_train_cat[col].fillna('Unknown').astype(str)
            X_val_cat[col] = X_val_cat[col].fillna('Unknown').astype(str)
            X_test_cat[col] = X_test_cat[col].fillna('Unknown').astype(str)
            
            # Fit on the training split, plus a reserved slot for categories
            # that only appear later. Previously an unseen value was folded
            # into encoder.classes_[0] -- a real hospital or ethnicity -- which
            # silently merged two different things (AUDIT.md EICU-10).
            le = LabelEncoder()
            le.fit(list(X_train_cat[col].unique()) + [UNKNOWN_CATEGORY])

            def safe_transform(encoder, series):
                known = set(encoder.classes_)
                return encoder.transform(
                    series.where(series.isin(known), UNKNOWN_CATEGORY)
                )

            X_train_cat[col] = safe_transform(le, X_train_cat[col])
            n_unseen = (~X_val_cat[col].isin(set(le.classes_))).sum() + \
                       (~X_test_cat[col].isin(set(le.classes_))).sum()
            if n_unseen:
                logger.info(f"  {col}: {n_unseen} value(s) unseen in train -> {UNKNOWN_CATEGORY}")
            X_val_cat[col] = safe_transform(le, X_val_cat[col])
            X_test_cat[col] = safe_transform(le, X_test_cat[col])
            encoders[col] = le
            
        # --- Right tower: impute + scale ---
        # 1. Median imputation
        imputer = SimpleImputer(strategy='median')
        X_train_num = imputer.fit_transform(X_train_num)
        X_val_num = imputer.transform(X_val_num)
        X_test_num = imputer.transform(X_test_num)
        
        # 2. Standardisation
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train_num)
        X_val_num = scaler.transform(X_val_num)
        X_test_num = scaler.transform(X_test_num)
        
        # Back to DataFrames for the join
        X_train_num = pd.DataFrame(X_train_num, columns=num_cols, index=train_df.index)
        X_val_num = pd.DataFrame(X_val_num, columns=num_cols, index=val_df.index)
        X_test_num = pd.DataFrame(X_test_num, columns=num_cols, index=test_df.index)
        
        # --- Assemble ---
        # Concatenate categorical + numeric, and attach the label
        def merge_pack(x_cat, x_num, original_df):
            # Reset indices so the pieces align
            x_cat = x_cat.reset_index(drop=True)
            x_num = x_num.reset_index(drop=True)
            y = original_df[target_col].reset_index(drop=True)
            # Keep patient_id for traceability
            pids = original_df['patient_id'].reset_index(drop=True)
            
            # Join side by side
            return pd.concat([pids, y, x_cat, x_num], axis=1)

        train_pack = merge_pack(X_train_cat, X_train_num, train_df)
        val_pack = merge_pack(X_val_cat, X_val_num, val_df)
        test_pack = merge_pack(X_test_cat, X_test_num, test_df)
        
        self._fitted = {"encoders": encoders, "imputer": imputer,
                        "scaler": scaler, "cat_cols": cat_cols,
                        "num_cols": num_cols}
        return train_pack, val_pack, test_pack

    def run_protocol(self, protocol_name: str, df: pd.DataFrame, seed: int):
        """Run one protocol: split -> preprocess -> save."""
        protocol_std = self.protocol_name_map.get(protocol_name, protocol_name)
        
        logger.info(f"\n>>> Running protocol: {protocol_std} (seed: {seed})")
        self.seed_manager.set_seed(seed)
        
        # 1. Split
        if protocol_name == 'patient_level':
            splitter = PatientLevelSplitter(stratify_col='mortality', random_state=seed)
        elif protocol_name == 'temporal':
             # Only possible when a time axis survived stage 1
             if 'hospitaldischargeyear' not in df.columns:
                 logger.warning("Skipping temporal split: no time column")
                 return
             splitter = TemporalHoldoutSplitter(stratify_col='mortality', random_state=seed)
        elif protocol_name == 'site':
            splitter = SiteHoldoutSplitter(site_col='site_id', stratify_col='mortality', random_state=seed)
        else:
            return

        splits = splitter.split(df)
        
        # Persist split membership
        self.exporter.save_split_indices(
            protocol=protocol_std,
            split_indices=splitter.split_indices,
            patient_ids=df['patient_id'],
            seed=seed,
        )
        
        # 2. Preprocess for the dual-tower model
        train_ready, val_ready, test_ready = self._dual_tower_preprocess(
            splits['train'], splits['val'], splits['test']
        )
        
        # 3. Write the final CSVs
        self.exporter.save_split_data(
            protocol=protocol_std,
            train_df=train_ready,
            val_df=val_ready,
            test_df=test_ready,
            seed=seed,
        )

        # Without these the run cannot be reproduced from what it writes
        # (AUDIT.md EICU-14).
        self.exporter.save_fitted_transforms(protocol_std, seed, self._fitted)
        self.exporter.save_feature_list(
            feature_names=self._fitted['cat_cols'] + self._fitted['num_cols'],
            feature_types={'numeric': self._fitted['num_cols'],
                           'categorical': self._fitted['cat_cols']},
        )
        logger.info(f"Protocol {protocol_std} complete. Data written to PACKS/{protocol_std}/")

    def run(self):
        """Entry point."""
        # Load
        df = self.load_data()
        
        seeds = self.seed_manager.get_seeds()
        logger.info(f"Seeds: {seeds}")

        # The paper reports results averaged over 5 random seeds, but run()
        # used to hardcode seed=42 and produce exactly one split, so n_seeds
        # had no effect (AUDIT.md EICU-14). Every configured seed now runs.
        for seed in seeds:
            # 1. Patient level (the primary protocol)
            self.run_protocol('patient_level', df, seed=seed)

            # 2. Site split: cross-hospital generalisation
            if df['site_id'].nunique() > 1:
                self.run_protocol('site', df, seed=seed)

            # 3. Temporal split: only possible when a time axis survived
            # stage 1. eICU exposes hospitaldischargeyear and nothing finer.
            if 'hospitaldischargeyear' in df.columns:
                self.run_protocol('temporal', df, seed=seed)

        self.exporter.save_seeds(seeds)
        self.exporter.save_data_version(
            df, data_hash=DataVersionManager.compute_data_hash(df), config=self.config
        )
        logger.info("\nAll train-ready data prepared.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/obs_trainReady.yaml')
    args = parser.parse_args()
    
    pipeline = ObsTrainReadyPipeline(args.config)
    pipeline.run()

if __name__ == '__main__':
    main()
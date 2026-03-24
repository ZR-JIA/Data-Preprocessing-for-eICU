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
)
from obs_trainReady.utils import (
    load_config,
    identify_feature_types,
    print_data_summary
)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

        # Map the identifier
        df['patient_id'] = df['patientunitstayid'].astype(str)
        
        # site_id is needed by the left tower
        if 'site_id' not in df.columns:
            logger.warning("site_id is missing; falling back to a synthetic value")
            # Synthetic stand-in so the run completes; should not happen in practice.
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
        
        target_col = 'mortality'
        exclude_cols = ['patient_id', 'patientunitstayid', 'admission_time', '_group_key', target_col]
        
        # 1. Left tower: categorical features, encoded to 0, 1, 2 ... indices
        cat_cols = ['age', 'gender', 'ethnicity', 'site_id']
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
            
            # Fit the encoder
            le = LabelEncoder()
            # on the training split only
            le.fit(X_train_cat[col])
            
            # Transform, tolerating categories unseen during fit
            def safe_transform(encoder, series):
                # Unseen categories are mapped to the first known class.
                # A reserved <UNK> index would be stricter.
                import numpy as np
                valid_classes = set(encoder.classes_)
                # Replace anything not seen in training
                safe_series = series.apply(lambda x: x if x in valid_classes else encoder.classes_[0])
                return encoder.transform(safe_series)

            X_train_cat[col] = le.transform(X_train_cat[col])
            X_val_cat[col] = safe_transform(le, X_val_cat[col])
            X_test_cat[col] = safe_transform(le, X_test_cat[col])
            
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
             # Only possible when the data carries admission_time
             if 'admission_time' not in df.columns:
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
            patient_ids=df['patient_id']
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
            test_df=test_ready
        )
        logger.info(f"Protocol {protocol_std} complete. Data written to PACKS/{protocol_std}/")

    def run(self):
        """Entry point."""
        # Load
        df = self.load_data()
        
        # Run the protocols
        # 1. Patient level (the primary protocol)
        self.run_protocol('patient_level', df, seed=42)
        
        # 2. Site split (optional, tests cross-hospital generalisation)
        #if 'site_id' in df.columns:
        #    self.run_protocol('site', df, seed=42)
            
        # 3. Temporal split (optional)
        #if 'admission_time' in df.columns:
        #    self.run_protocol('temporal', df, seed=42)
            
        logger.info("\nAll train-ready data prepared.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/obs_trainReady.yaml')
    args = parser.parse_args()
    
    pipeline = ObsTrainReadyPipeline(args.config)
    pipeline.run()

if __name__ == '__main__':
    main()
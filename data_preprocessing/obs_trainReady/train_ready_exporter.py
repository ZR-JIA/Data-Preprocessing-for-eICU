"""
Structured writer for train-ready output.
Responsible only for persisting data (CSV) and metadata (JSON).
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class TrainReadyExporter:
    """
    Train-ready output manager.
    
    Layout:
    train_ready/
    |- DATA/           (feature list, data version)
    |- SPLITS/         (per-protocol IDs)
    |- PACKS/          (per-protocol train/val/test.csv)
    |- REPROD/         (seeds)
    └─ README.md
    """
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create the output directory tree."""
        dirs = [
            self.base_dir / "DATA",
            self.base_dir / "SPLITS",
            self.base_dir / "PACKS",
            self.base_dir / "REPROD",
        ]
        
        # One subdirectory per split protocol
        protocols = ['patient_stratified', 'temporal', 'site_holdout']
        for protocol in protocols:
            dirs.extend([
                self.base_dir / "SPLITS" / protocol,
                self.base_dir / "PACKS" / protocol,
            ])
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            
    # ========== DATA ==========
    
    def save_feature_list(self, feature_names: List[str], feature_types: Dict[str, List[str]]):
        """Write the feature list."""
        feature_list = {
            'total_features': len(feature_names),
            'feature_names': feature_names,
            'numeric_features': feature_types.get('numeric', []),
            'categorical_features': feature_types.get('categorical', []),
            'generated_at': datetime.now().isoformat()
        }
        output_path = self.base_dir / "DATA" / "feature_list.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(feature_list, f, indent=2)
            
    def save_data_version(self, df: pd.DataFrame, data_hash: str, config: Dict):
        """Write the data version record."""
        data_version = {
            'data_hash': data_hash,
            'shape': list(df.shape),
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        output_path = self.base_dir / "DATA" / "data_version.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_version, f, indent=2)

    # ========== SPLITS ==========
    
    def save_split_indices(self, protocol: str, split_indices: Dict[str, List], patient_ids: pd.Series):
        """Write split membership as IDs."""
        protocol_dir = self.base_dir / "SPLITS" / protocol
        for split_name, indices in split_indices.items():
            if len(indices) == 0: continue
            
            ids = patient_ids.iloc[indices].values
            df_ids = pd.DataFrame({'patient_id': ids, 'original_index': indices})
            df_ids.to_csv(protocol_dir / f"{split_name}_ids.csv", index=False)

    # ========== PACKS (the actual datasets) ==========
    
    def save_split_data(self, protocol: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Write train/val/test CSVs."""
        protocol_dir = self.base_dir / "PACKS" / protocol
        train_df.to_csv(protocol_dir / "train.csv", index=False)
        val_df.to_csv(protocol_dir / "val.csv", index=False)
        test_df.to_csv(protocol_dir / "test.csv", index=False)
        logger.info(f"Wrote data pack: {protocol_dir}")

    # ========== REPROD ==========
    
    def save_seeds(self, seeds: List[int]):
        output_path = self.base_dir / "REPROD" / "seeds.txt"
        with open(output_path, 'w') as f:
            for seed in seeds:
                f.write(f"{seed}\n")
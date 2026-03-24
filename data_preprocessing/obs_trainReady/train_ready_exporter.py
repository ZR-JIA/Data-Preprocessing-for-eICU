"""
Train-Ready 结构化输出管理器 (纯净版)
只负责保存数据 (CSV) 和元数据 (JSON)
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
    Train-Ready 输出管理器
    
    输出结构：
    train_ready/
    ├─ DATA/           (特征列表、版本信息)
    ├─ SPLITS/         (三协议的 IDs)
    ├─ PACKS/          (三协议的 train/val/test.csv)
    ├─ REPROD/         (种子)
    └─ README.md
    """
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """创建精简后的目录结构"""
        dirs = [
            self.base_dir / "DATA",
            self.base_dir / "SPLITS",
            self.base_dir / "PACKS",
            self.base_dir / "REPROD",
        ]
        
        # 三种协议的子目录
        protocols = ['patient_stratified', 'temporal', 'site_holdout']
        for protocol in protocols:
            dirs.extend([
                self.base_dir / "SPLITS" / protocol,
                self.base_dir / "PACKS" / protocol,
            ])
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            
    # ========== DATA 目录输出 ==========
    
    def save_feature_list(self, feature_names: List[str], feature_types: Dict[str, List[str]]):
        """保存特征列表"""
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
        """保存数据版本信息"""
        data_version = {
            'data_hash': data_hash,
            'shape': list(df.shape),
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        output_path = self.base_dir / "DATA" / "data_version.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_version, f, indent=2)

    # ========== SPLITS 目录输出 ==========
    
    def save_split_indices(self, protocol: str, split_indices: Dict[str, List], patient_ids: pd.Series):
        """保存切分索引 (IDs)"""
        protocol_dir = self.base_dir / "SPLITS" / protocol
        for split_name, indices in split_indices.items():
            if len(indices) == 0: continue
            
            ids = patient_ids.iloc[indices].values
            df_ids = pd.DataFrame({'patient_id': ids, 'original_index': indices})
            df_ids.to_csv(protocol_dir / f"{split_name}_ids.csv", index=False)

    # ========== PACKS 目录输出 (最重要!) ==========
    
    def save_split_data(self, protocol: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """保存切分后的 CSV"""
        protocol_dir = self.base_dir / "PACKS" / protocol
        train_df.to_csv(protocol_dir / "train.csv", index=False)
        val_df.to_csv(protocol_dir / "val.csv", index=False)
        test_df.to_csv(protocol_dir / "test.csv", index=False)
        logger.info(f"数据包已保存: {protocol_dir}")

    # ========== REPROD 目录输出 ==========
    
    def save_seeds(self, seeds: List[int]):
        output_path = self.base_dir / "REPROD" / "seeds.txt"
        with open(output_path, 'w') as f:
            for seed in seeds:
                f.write(f"{seed}\n")
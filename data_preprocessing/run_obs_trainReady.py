"""
观察数据训练准备主程序 (Dual-Tower Deep Learning 专用版)
功能：
1. 读取 obs_cleaned.csv
2. 执行 Patient-level / Temporal / Site 切分
3. 双塔预处理：
   - 左塔特征 -> Label Encoding (转为 0,1,2... 供 Embedding 使用)
   - 右塔特征 -> StandardScaler (转为 均值0方差1)
4. 保存 train.csv, val.csv, test.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer  # <--- SimpleImputer 在这里

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入核心工具 (只保留你没删的那几个)
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ObsTrainReadyPipeline:
    """
    专门为 Deep Learning 双塔模型准备数据的流水线
    """
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.seed_manager = SeedManager(
            base_seed=self.config.get('random_seed', 42),
            n_seeds=self.config.get('n_seeds', 5)
        )
        
        # 路径配置
        paths = self.config.get('paths', {})
        self.output_dir = paths.get('output_train_ready', '../../train_ready')
        self.exporter = TrainReadyExporter(base_dir=self.output_dir)
        
        # 协议名称映射
        self.protocol_name_map = {
            'patient_level': 'patient_stratified',
            'temporal': 'temporal',
            'site': 'site_holdout'
        }
        
    def load_data(self) -> pd.DataFrame:
        """加载数据并进行双塔适配"""
        paths = self.config.get('paths', {})
        # 指向 obs_cleaned.csv (必须是清洗后的)
        data_path = paths.get('input_cleaned', '../../outputs/obs_cleaned.csv')
        logger.info(f"正在加载数据: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ 找不到文件: {data_path}，请先运行 run_obs_clean.py")

        df = pd.read_csv(data_path)
        
        # --- 核心列检查 ---
        required = ['patientunitstayid', 'mortality']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"❌ 严重错误: 数据中缺少 {col} 列！")

        # 映射 ID
        df['patient_id'] = df['patientunitstayid'].astype(str)
        
        # 确保 site_id 存在 (用于左塔)
        if 'site_id' not in df.columns:
            logger.warning("⚠️ site_id 缺失！正在尝试修复...")
            # 如果真的没有，临时用 patient_id 模拟 (仅为了跑通流程，实际上不应该发生)
            df['site_id'] = (df['patient_id'].astype(int) % 10).astype(str)
        else:
            df['site_id'] = df['site_id'].astype(str)

        print_data_summary(df, "Train-Ready 原始输入")
        return df

    def _dual_tower_preprocess(self, train_df, val_df, test_df):
        """
        ⚡️ 双塔核心预处理逻辑 ⚡️
        左塔特征 -> Label Encode (Unknown处理)
        右塔特征 -> Median Impute + Standard Scale
        关键原则: Fit on Train, Transform All
        """
        logger.info("开始双塔预处理 (Fit on Train)...")
        
        target_col = 'mortality'
        exclude_cols = ['patient_id', 'patientunitstayid', 'admission_time', '_group_key', target_col]
        
        # 1. 定义左塔特征 (Categorical)
        # 这些必须变成 0, 1, 2... 的整数索引
        cat_cols = ['age', 'gender', 'ethnicity', 'site_id']
        # 确保列存在
        cat_cols = [c for c in cat_cols if c in train_df.columns]
        
        # 2. 定义右塔特征 (Numerical)
        # 排除掉 ID, Label 和 左塔特征，剩下的全是数值
        all_cols = train_df.columns.tolist()
        num_cols = [c for c in all_cols if c not in exclude_cols + cat_cols]
        
        logger.info(f"特征分配: 左塔(Cat)={len(cat_cols)}列, 右塔(Num)={len(num_cols)}列")
        
        # 复制数据以免修改原始 DF
        X_train_cat, X_val_cat, X_test_cat = train_df[cat_cols].copy(), val_df[cat_cols].copy(), test_df[cat_cols].copy()
        X_train_num, X_val_num, X_test_num = train_df[num_cols].copy(), val_df[num_cols].copy(), test_df[num_cols].copy()
        
        # --- 处理左塔 (Label Encoding) ---
        for col in cat_cols:
            # 填充缺失值为 'Unknown'
            X_train_cat[col] = X_train_cat[col].fillna('Unknown').astype(str)
            X_val_cat[col] = X_val_cat[col].fillna('Unknown').astype(str)
            X_test_cat[col] = X_test_cat[col].fillna('Unknown').astype(str)
            
            # 初始化 LabelEncoder
            le = LabelEncoder()
            # 拟合训练集
            le.fit(X_train_cat[col])
            
            # Transform (处理未见过的类别)
            # 定义一个安全的 transform 函数
            def safe_transform(encoder, series):
                # 将未见过的类别映射为 'Unknown' 或者 众数，这里简化为映射到第一个类别(通常是0)
                # 更严谨的做法是把所有 Unknown 映射到一个特定索引，这里为了兼容简单处理
                import numpy as np
                valid_classes = set(encoder.classes_)
                # 将不在训练集中的类别替换为训练集中的第一个类别
                safe_series = series.apply(lambda x: x if x in valid_classes else encoder.classes_[0])
                return encoder.transform(safe_series)

            X_train_cat[col] = le.transform(X_train_cat[col])
            X_val_cat[col] = safe_transform(le, X_val_cat[col])
            X_test_cat[col] = safe_transform(le, X_test_cat[col])
            
        # --- 处理右塔 (Impute + Scale) ---
        # 1. 填补缺失值 (Median)
        imputer = SimpleImputer(strategy='median')
        X_train_num = imputer.fit_transform(X_train_num)
        X_val_num = imputer.transform(X_val_num)
        X_test_num = imputer.transform(X_test_num)
        
        # 2. 标准化 (StandardScaler)
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train_num)
        X_val_num = scaler.transform(X_val_num)
        X_test_num = scaler.transform(X_test_num)
        
        # 转回 DataFrame 方便合并
        X_train_num = pd.DataFrame(X_train_num, columns=num_cols, index=train_df.index)
        X_val_num = pd.DataFrame(X_val_num, columns=num_cols, index=val_df.index)
        X_test_num = pd.DataFrame(X_test_num, columns=num_cols, index=test_df.index)
        
        # --- 合并结果 ---
        # 将处理好的 Cat 和 Num 拼起来，加上 Label
        def merge_pack(x_cat, x_num, original_df):
            # 重置索引以确保对齐
            x_cat = x_cat.reset_index(drop=True)
            x_num = x_num.reset_index(drop=True)
            y = original_df[target_col].reset_index(drop=True)
            # 还需要保留 patient_id 方便追踪
            pids = original_df['patient_id'].reset_index(drop=True)
            
            # 横向拼接
            return pd.concat([pids, y, x_cat, x_num], axis=1)

        train_pack = merge_pack(X_train_cat, X_train_num, train_df)
        val_pack = merge_pack(X_val_cat, X_val_num, val_df)
        test_pack = merge_pack(X_test_cat, X_test_num, test_df)
        
        return train_pack, val_pack, test_pack

    def run_protocol(self, protocol_name: str, df: pd.DataFrame, seed: int):
        """执行单个协议：切分 -> 预处理 -> 保存"""
        protocol_std = self.protocol_name_map.get(protocol_name, protocol_name)
        
        logger.info(f"\n>>> 执行协议: {protocol_std} (Seed: {seed})")
        self.seed_manager.set_seed(seed)
        
        # 1. 切分 (使用你现有的 data_splitting 模块)
        if protocol_name == 'patient_level':
            splitter = PatientLevelSplitter(stratify_col='mortality', random_state=seed)
        elif protocol_name == 'temporal':
             # 只有当数据里有 admission_time 时才跑这个
             if 'admission_time' not in df.columns:
                 logger.warning("跳过 Temporal Split (无时间列)")
                 return
             splitter = TemporalHoldoutSplitter(stratify_col='mortality', random_state=seed)
        elif protocol_name == 'site':
            splitter = SiteHoldoutSplitter(site_col='site_id', stratify_col='mortality', random_state=seed)
        else:
            return

        splits = splitter.split(df)
        
        # 保存 ID 索引
        self.exporter.save_split_indices(
            protocol=protocol_std, 
            split_indices=splitter.split_indices, 
            patient_ids=df['patient_id']
        )
        
        # 2. 预处理 (双塔适配)
        train_ready, val_ready, test_ready = self._dual_tower_preprocess(
            splits['train'], splits['val'], splits['test']
        )
        
        # 3. 保存最终 CSV (标准金条)
        self.exporter.save_split_data(
            protocol=protocol_std,
            train_df=train_ready,
            val_df=val_ready,
            test_df=test_ready
        )
        logger.info(f"✅ 协议 {protocol_std} 完成。数据保存在 PACKS/{protocol_std}/")

    def run(self):
        """主入口"""
        # 加载
        df = self.load_data()
        
        # 运行三个协议
        # 1. Patient Level (最常用)
        self.run_protocol('patient_level', df, seed=42)
        
        # 2. Site Split (选做，验证泛化性)
        #if 'site_id' in df.columns:
        #    self.run_protocol('site', df, seed=42)
            
        # 3. Temporal Split (选做)
        #if 'admission_time' in df.columns:
        #    self.run_protocol('temporal', df, seed=42)
            
        logger.info("\n🎉 所有 Train-Ready 数据准备完成！")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/obs_trainReady.yaml')
    args = parser.parse_args()
    
    pipeline = ObsTrainReadyPipeline(args.config)
    pipeline.run()

if __name__ == '__main__':
    main()
"""
数据分割模块
实现三种分割策略：Patient-level、Temporal Holdout、Site Holdout
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatientLevelSplitter:
    """
    主线：Patient-level 分层切分
    比例：70/10/20 = train/val/test
    按 mortality 分层，按医院/站点分组再分层
    """
    
    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.1,
        test_size: float = 0.2,
        stratify_col: str = 'mortality',
        group_cols: Optional[List[str]] = None,
        random_state: int = 42
    ):
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "比例之和必须为1"
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.stratify_col = stratify_col
        self.group_cols = group_cols or ['patient_id', 'site_id']
        self.random_state = random_state
        self.split_indices = {}
        
    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        执行分层分组切分
        
        Args:
            df: 输入数据框，必须包含 stratify_col 和 group_cols
            
        Returns:
            包含 'train', 'val', 'test' 的字典
        """
        logger.info("开始 Patient-level 分层分组切分...")
        
        # 验证必需列
        required_cols = [self.stratify_col] + self.group_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需列: {missing_cols}")
        
        # 按组聚合，确保同一组不跨集合
        group_key = '_'.join(self.group_cols)
        df['_group_key'] = df[self.group_cols].astype(str).agg('_'.join, axis=1)
        
        # 创建组级别的数据
        group_df = df.groupby('_group_key').agg({
            self.stratify_col: 'first',  # 假设同组内标签一致
        }).reset_index()
        
        # 第一次分割：分出 test (20%)
        train_val_groups, test_groups = train_test_split(
            group_df,
            test_size=self.test_size,
            stratify=group_df[self.stratify_col],
            random_state=self.random_state
        )
        
        # 第二次分割：从 train_val 中分出 val
        # val_size_adjusted = val / (train + val) = 0.1 / 0.8 = 0.125
        val_size_adjusted = self.val_size / (self.train_size + self.val_size)
        
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=val_size_adjusted,
            stratify=train_val_groups[self.stratify_col],
            random_state=self.random_state
        )
        
        # 根据组映射回原始数据
        train_df = df[df['_group_key'].isin(train_groups['_group_key'])].copy()
        val_df = df[df['_group_key'].isin(val_groups['_group_key'])].copy()
        test_df = df[df['_group_key'].isin(test_groups['_group_key'])].copy()
        
        # 清理临时列
        train_df = train_df.drop(columns=['_group_key'])
        val_df = val_df.drop(columns=['_group_key'])
        test_df = test_df.drop(columns=['_group_key'])
        
        # 保存索引
        self.split_indices = {
            'train': train_df.index.tolist(),
            'val': val_df.index.tolist(),
            'test': test_df.index.tolist()
        }
        
        logger.info(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
                   f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
                   f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        # 输出分层统计
        self._log_stratification_stats(train_df, val_df, test_df)
        
        return {'train': train_df, 'val': val_df, 'test': test_df}
    
    def _log_stratification_stats(self, train_df, val_df, test_df):
        """输出分层统计信息"""
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if self.stratify_col in split_df.columns:
                mortality_rate = split_df[self.stratify_col].mean()
                logger.info(f"{split_name} mortality rate: {mortality_rate:.3f}")


class TemporalHoldoutSplitter:
    """
    时间外推：Temporal Holdout
    以入科时间排序，最晚的 20% 作为 test
    其余做 train/val=80/20，并保持分层
    """
    
    def __init__(
        self,
        time_col: str = 'admission_time',
        test_size: float = 0.2,
        val_size: float = 0.2,  # 从 train+val 中分出 val 的比例
        stratify_col: str = 'mortality',
        group_cols: Optional[List[str]] = None,
        random_state: int = 42
    ):
        self.time_col = time_col
        self.test_size = test_size
        self.val_size = val_size
        self.stratify_col = stratify_col
        self.group_cols = group_cols or ['patient_id', 'site_id']
        self.random_state = random_state
        self.split_indices = {}
        self.time_cutoff = None
        
    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        执行时间切分
        
        Args:
            df: 输入数据框，必须包含时间列
            
        Returns:
            包含 'train', 'val', 'test' 的字典
        """
        logger.info("开始 Temporal Holdout 切分...")
        
        # 验证时间列
        if self.time_col not in df.columns:
            raise ValueError(f"缺少时间列: {self.time_col}")
        
        # 转换为 datetime
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        
        # 按组聚合（取组内最早时间）
        df['_group_key'] = df[self.group_cols].astype(str).agg('_'.join, axis=1)
        group_df = df.groupby('_group_key').agg({
            self.time_col: 'min',
            self.stratify_col: 'first'
        }).reset_index()
        
        # 按时间排序
        group_df = group_df.sort_values(self.time_col)
        
        # 计算时间切分点（最晚20%作为test）
        n_groups = len(group_df)
        test_start_idx = int(n_groups * (1 - self.test_size))
        
        train_val_groups = group_df.iloc[:test_start_idx]
        test_groups = group_df.iloc[test_start_idx:]
        
        self.time_cutoff = test_groups[self.time_col].min()
        logger.info(f"时间切分点: {self.time_cutoff}")
        
        # 从 train_val 中分出 val（分层）
        if len(train_val_groups) > 0 and self.val_size > 0:
            train_groups, val_groups = train_test_split(
                train_val_groups,
                test_size=self.val_size,
                stratify=train_val_groups[self.stratify_col],
                random_state=self.random_state
            )
        else:
            train_groups = train_val_groups
            val_groups = pd.DataFrame()
        
        # 映射回原始数据
        train_df = df[df['_group_key'].isin(train_groups['_group_key'])].copy()
        val_df = df[df['_group_key'].isin(val_groups['_group_key'])].copy() if len(val_groups) > 0 else pd.DataFrame()
        test_df = df[df['_group_key'].isin(test_groups['_group_key'])].copy()
        
        # 清理临时列
        train_df = train_df.drop(columns=['_group_key'])
        if len(val_df) > 0:
            val_df = val_df.drop(columns=['_group_key'])
        test_df = test_df.drop(columns=['_group_key'])
        
        # 保存索引
        self.split_indices = {
            'train': train_df.index.tolist(),
            'val': val_df.index.tolist() if len(val_df) > 0 else [],
            'test': test_df.index.tolist()
        }
        
        logger.info(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
                   f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
                   f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return {'train': train_df, 'val': val_df, 'test': test_df}


class SiteHoldoutSplitter:
    """
    站点外推：Site Holdout
    选择若干医院作为 site-test（≈20% 样本），完全隔离
    其余做 train/val=80/20（分层）
    """
    
    def __init__(
        self,
        site_col: str = 'site_id',
        test_size: float = 0.2,
        val_size: float = 0.2,
        stratify_col: str = 'mortality',
        test_sites: Optional[List] = None,
        random_state: int = 42
    ):
        self.site_col = site_col
        self.test_size = test_size
        self.val_size = val_size
        self.stratify_col = stratify_col
        self.test_sites = test_sites
        self.random_state = random_state
        self.split_indices = {}
        self.selected_test_sites = None
        
    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        执行站点切分
        
        Args:
            df: 输入数据框，必须包含站点列
            
        Returns:
            包含 'train', 'val', 'test' 的字典
        """
        logger.info("开始 Site Holdout 切分...")
        
        # 验证站点列
        if self.site_col not in df.columns:
            raise ValueError(f"缺少站点列: {self.site_col}")
        
        df = df.copy()
        
        # 如果未指定test站点，自动选择
        if self.test_sites is None:
            self.test_sites = self._select_test_sites(df)
        
        self.selected_test_sites = self.test_sites
        logger.info(f"Test 站点: {self.selected_test_sites}")
        
        # 按站点切分
        test_df = df[df[self.site_col].isin(self.test_sites)].copy()
        train_val_df = df[~df[self.site_col].isin(self.test_sites)].copy()
        
        # 从 train_val 中分出 val（分层）
        if len(train_val_df) > 0 and self.val_size > 0:
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.val_size,
                stratify=train_val_df[self.stratify_col],
                random_state=self.random_state
            )
        else:
            train_df = train_val_df
            val_df = pd.DataFrame()
        
        # 保存索引
        self.split_indices = {
            'train': train_df.index.tolist(),
            'val': val_df.index.tolist() if len(val_df) > 0 else [],
            'test': test_df.index.tolist()
        }
        
        logger.info(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
                   f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
                   f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        # 输出站点统计
        self._log_site_stats(train_df, val_df, test_df)
        
        return {'train': train_df, 'val': val_df, 'test': test_df}
    
    def _select_test_sites(self, df: pd.DataFrame) -> List:
        """
        自动选择test站点，使其样本量接近test_size
        """
        site_counts = df[self.site_col].value_counts().sort_values(ascending=False)
        total_samples = len(df)
        target_samples = total_samples * self.test_size
        
        # 贪心选择站点
        selected_sites = []
        cumsum = 0
        
        np.random.seed(self.random_state)
        site_list = site_counts.index.tolist()
        np.random.shuffle(site_list)
        
        for site in site_list:
            if cumsum < target_samples:
                selected_sites.append(site)
                cumsum += site_counts[site]
            else:
                break
        
        # 确保至少选择一个站点
        if len(selected_sites) == 0:
            selected_sites = [site_counts.index[0]]
        
        return selected_sites
    
    def _log_site_stats(self, train_df, val_df, test_df):
        """输出站点统计信息"""
        logger.info(f"Train 站点数: {train_df[self.site_col].nunique()}")
        if len(val_df) > 0:
            logger.info(f"Val 站点数: {val_df[self.site_col].nunique()}")
        logger.info(f"Test 站点数: {test_df[self.site_col].nunique()}")


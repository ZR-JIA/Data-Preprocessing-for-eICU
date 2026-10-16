"""
Data splitting.
Three protocols: Patient-level, Temporal Holdout, Site Holdout.
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
    Primary protocol: patient-level stratified split, 70/10/20 train/val/test.
    Stratified on mortality, grouped so one patient never spans two splits.
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
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "split sizes must sum to 1"
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.stratify_col = stratify_col
        # Group on the person alone. Adding site_id would split a patient who
        # was admitted to two hospitals into two groups, reintroducing the very
        # overlap this splitter exists to prevent (AUDIT.md EICU-02).
        self.group_cols = group_cols or ['patient_id']
        self.random_state = random_state
        self.split_indices = {}
        
    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run the stratified grouped split.
        
        Args:
            df: input frame; must contain stratify_col and group_cols
            
        Returns:
            dict with 'train', 'val' and 'test'
        """
        logger.info("Starting patient-level stratified grouped split...")
        
        # Required columns
        required_cols = [self.stratify_col] + self.group_cols
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"missing required columns: {missing_cols}")
        
        # Work on a copy: writing _group_key onto the caller's frame leaked a
        # stale column into every later protocol run.
        df = df.copy()
        df['_group_key'] = df[self.group_cols].astype(str).agg('_'.join, axis=1)
        
        # Collapse to one row per group
        group_df = df.groupby('_group_key').agg({
            self.stratify_col: 'first',  # label is constant within a group
        }).reset_index()
        
        # First split: hold out test (20%)
        train_val_groups, test_groups = train_test_split(
            group_df,
            test_size=self.test_size,
            stratify=group_df[self.stratify_col],
            random_state=self.random_state
        )
        
        # Second split: carve val out of train_val
        # val_size_adjusted = val / (train + val) = 0.1 / 0.8 = 0.125
        val_size_adjusted = self.val_size / (self.train_size + self.val_size)
        
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=val_size_adjusted,
            stratify=train_val_groups[self.stratify_col],
            random_state=self.random_state
        )
        
        # Map groups back to rows
        train_df = df[df['_group_key'].isin(train_groups['_group_key'])].copy()
        val_df = df[df['_group_key'].isin(val_groups['_group_key'])].copy()
        test_df = df[df['_group_key'].isin(test_groups['_group_key'])].copy()
        
        # Drop the temporary key
        train_df = train_df.drop(columns=['_group_key'])
        val_df = val_df.drop(columns=['_group_key'])
        test_df = test_df.drop(columns=['_group_key'])
        
        # Record split membership
        self.split_indices = {
            'train': train_df.index.tolist(),
            'val': val_df.index.tolist(),
            'test': test_df.index.tolist()
        }
        
        logger.info(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
                   f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
                   f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        # Report stratification
        self._log_stratification_stats(train_df, val_df, test_df)
        
        return {'train': train_df, 'val': val_df, 'test': test_df}
    
    def _log_stratification_stats(self, train_df, val_df, test_df):
        """Log the mortality rate of each split."""
        for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if self.stratify_col in split_df.columns:
                mortality_rate = split_df[self.stratify_col].mean()
                logger.info(f"{split_name} mortality rate: {mortality_rate:.3f}")


class TemporalHoldoutSplitter:
    """
    Temporal extrapolation: sort by admission time, hold out the latest 20%
    as test, then split the remainder 80/20 with stratification.
    """
    
    def __init__(
        self,
        time_col: str = 'admission_time',
        test_size: float = 0.2,
        val_size: float = 0.2,  # fraction of train+val held out as val
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
        Run the temporal split.
        
        Args:
            df: input frame; must contain the time column
            
        Returns:
            dict with 'train', 'val' and 'test'
        """
        logger.info("Starting temporal holdout split...")
        
        # Required time column
        if self.time_col not in df.columns:
            raise ValueError(f"missing time column: {self.time_col}")
        
        # Parse to datetime
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        
        # One row per group, using its earliest time
        df['_group_key'] = df[self.group_cols].astype(str).agg('_'.join, axis=1)
        group_df = df.groupby('_group_key').agg({
            self.time_col: 'min',
            self.stratify_col: 'first'
        }).reset_index()
        
        # Order by time
        group_df = group_df.sort_values(self.time_col)
        
        # Cut point: the latest 20% become test
        n_groups = len(group_df)
        test_start_idx = int(n_groups * (1 - self.test_size))
        
        train_val_groups = group_df.iloc[:test_start_idx]
        test_groups = group_df.iloc[test_start_idx:]
        
        self.time_cutoff = test_groups[self.time_col].min()
        logger.info(f"Temporal cut point: {self.time_cutoff}")
        
        # Carve val out of train_val, stratified
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
        
        # Map groups back to rows
        train_df = df[df['_group_key'].isin(train_groups['_group_key'])].copy()
        val_df = df[df['_group_key'].isin(val_groups['_group_key'])].copy() if len(val_groups) > 0 else pd.DataFrame()
        test_df = df[df['_group_key'].isin(test_groups['_group_key'])].copy()
        
        # Drop the temporary key
        train_df = train_df.drop(columns=['_group_key'])
        if len(val_df) > 0:
            val_df = val_df.drop(columns=['_group_key'])
        test_df = test_df.drop(columns=['_group_key'])
        
        # Record split membership
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
    Site extrapolation: hold out whole hospitals as test (~20% of rows),
    fully isolated, then split the remainder 80/20 with stratification.
    """
    
    def __init__(
        self,
        site_col: str = 'site_id',
        test_size: float = 0.2,
        val_size: float = 0.2,
        stratify_col: str = 'mortality',
        test_sites: Optional[List] = None,
        group_col: str = 'patient_id',
        random_state: int = 42
    ):
        self.site_col = site_col
        self.group_col = group_col
        self.test_size = test_size
        self.val_size = val_size
        self.stratify_col = stratify_col
        self.test_sites = test_sites
        self.random_state = random_state
        self.split_indices = {}
        self.selected_test_sites = None
        
    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run the site holdout split.
        
        Args:
            df: input frame; must contain the site column
            
        Returns:
            dict with 'train', 'val' and 'test'
        """
        logger.info("Starting site holdout split...")
        
        # Required site column
        if self.site_col not in df.columns:
            raise ValueError(f"missing site column: {self.site_col}")
        
        df = df.copy()
        
        # Pick test sites automatically when none were given
        if self.test_sites is None:
            self.test_sites = self._select_test_sites(df)
        
        self.selected_test_sites = self.test_sites
        logger.info(f"Test sites: {self.selected_test_sites}")
        
        # Partition by site
        test_df = df[df[self.site_col].isin(self.test_sites)].copy()
        train_val_df = df[~df[self.site_col].isin(self.test_sites)].copy()
        
        # Carve val out of train_val, stratified and grouped by patient.
        # This split used to run on rows, so a patient with two stays inside
        # the training hospitals could land in both train and val, which is the
        # same defect the patient-level protocol exists to avoid
        # (AUDIT.md R-5, EICU-02).
        if len(train_val_df) > 0 and self.val_size > 0:
            groups = train_val_df.groupby(self.group_col).agg(
                {self.stratify_col: 'first'}
            ).reset_index()
            train_groups, val_groups = train_test_split(
                groups,
                test_size=self.val_size,
                stratify=groups[self.stratify_col],
                random_state=self.random_state
            )
            train_df = train_val_df[
                train_val_df[self.group_col].isin(train_groups[self.group_col])
            ].copy()
            val_df = train_val_df[
                train_val_df[self.group_col].isin(val_groups[self.group_col])
            ].copy()
        else:
            train_df = train_val_df
            val_df = pd.DataFrame()
        
        # Record split membership
        self.split_indices = {
            'train': train_df.index.tolist(),
            'val': val_df.index.tolist() if len(val_df) > 0 else [],
            'test': test_df.index.tolist()
        }
        
        logger.info(f"Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%), "
                   f"Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%), "
                   f"Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        # Report site counts
        self._log_site_stats(train_df, val_df, test_df)
        
        return {'train': train_df, 'val': val_df, 'test': test_df}
    
    def _select_test_sites(self, df: pd.DataFrame) -> List:
        """
        Choose test sites until their combined size approaches test_size.
        """
        site_counts = df[self.site_col].value_counts().sort_values(ascending=False)
        total_samples = len(df)
        target_samples = total_samples * self.test_size
        
        # Greedy selection
        selected_sites = []
        cumsum = 0
        
        np.random.seed(self.random_state)
        site_list = site_counts.index.tolist()
        np.random.shuffle(site_list)
        
        # Test the budget before adding, and skip a hospital that would take
        # the total further from the target than stopping would. The previous
        # loop appended first and checked afterwards, so it could overshoot by
        # a whole hospital (AUDIT.md R-5).
        for site in site_list:
            if cumsum >= target_samples:
                break
            n = site_counts[site]
            if selected_sites and abs(cumsum + n - target_samples) > abs(cumsum - target_samples):
                continue
            selected_sites.append(site)
            cumsum += n
        
        # Always keep at least one site
        if len(selected_sites) == 0:
            selected_sites = [site_counts.index[0]]
        
        return selected_sites
    
    def _log_site_stats(self, train_df, val_df, test_df):
        """Log the number of distinct sites per split."""
        logger.info(f"Train sites: {train_df[self.site_col].nunique()}")
        if len(val_df) > 0:
            logger.info(f"Val sites: {val_df[self.site_col].nunique()}")
        logger.info(f"Test sites: {test_df[self.site_col].nunique()}")


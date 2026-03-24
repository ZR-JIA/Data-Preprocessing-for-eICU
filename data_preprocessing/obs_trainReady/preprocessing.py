"""
Preprocessing.

Imputation, scaling and feature selection, all fitted on the training split
only and then applied to val/test.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import Dict, List, Optional, Tuple
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeakageFreePreprocessor:
    """
    Base class for preprocessors that must never see val/test during fit.
    """
    
    def __init__(self):
        self.is_fitted = False
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = None):
        """Fit on the training split."""
        raise NotImplementedError
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transform."""
        raise NotImplementedError
        
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series = None) -> pd.DataFrame:
        """Fit on the training split and transform it."""
        self.fit(X_train, y_train)
        return self.transform(X_train)
    
    def save(self, path: str):
        """Persist a fitted preprocessor."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str):
        """Load a fitted preprocessor."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class TrainOnlyPreprocessor(LeakageFreePreprocessor):
    """
    Full preprocessing pipeline: imputation, scaling and feature selection.
    """
    
    def __init__(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        impute_strategy: str = 'median',
        scale_method: str = 'standard',
        feature_selection: Optional[str] = None,
        n_features: Optional[int] = None,
        target_encode_features: Optional[List[str]] = None
    ):
        super().__init__()
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.impute_strategy = impute_strategy
        self.scale_method = scale_method
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.target_encode_features = target_encode_features or []
        
        # Fitted components
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.selector = None
        self.target_encoders = {}
        self.selected_features = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = None):
        """
        Fit every preprocessing step on the training split.
        
        ⚠️ CRITICAL: All statistics (median, mean, std, etc.) are computed 
        ONLY on X_train to prevent data leakage. Val/test sets will be 
        transformed using these train-only statistics.
        
        Academic Standard: This ensures test set independence.
        """
        logger.info("=" * 60)
        logger.info("Fitting preprocessor (training-split statistics only)")
        logger.info("=" * 60)
        
        X = X_train.copy()
        
        # Log initial state
        n_train_samples = len(X)
        logger.info(f"Training rows: {n_train_samples}")
        
        # 1. Imputation (FIT ON TRAIN ONLY)
        if len(self.numeric_features) > 0:
            logger.info(f"\n[1/4] Numeric imputation (strategy={self.impute_strategy})")
            logger.info(f"  - features: {len(self.numeric_features)}")
            
            # Count missing values BEFORE imputation
            missing_before = X[self.numeric_features].isna().sum().sum()
            logger.info(f"  - missing before: {missing_before}")
            
            # FIT imputer on TRAIN ONLY
            self.numeric_imputer = SimpleImputer(strategy=self.impute_strategy)
            X[self.numeric_features] = self.numeric_imputer.fit_transform(
                X[self.numeric_features]
            )
            
            # Verify no missing values remain
            missing_after = X[self.numeric_features].isna().sum().sum()
            logger.info(f"  - missing after: {missing_after}")
            logger.info(f"  - statistics from {n_train_samples} training rows only")
        
        if len(self.categorical_features) > 0:
            logger.info(f"\n[1/4] Categorical imputation (strategy=most_frequent)")
            logger.info(f"  - features: {len(self.categorical_features)}")
            
            missing_before = X[self.categorical_features].isna().sum().sum()
            logger.info(f"  - missing before: {missing_before}")
            
            # FIT imputer on TRAIN ONLY
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[self.categorical_features] = self.categorical_imputer.fit_transform(
                X[self.categorical_features]
            )
            
            missing_after = X[self.categorical_features].isna().sum().sum()
            logger.info(f"  - missing after: {missing_after}")
            logger.info(f"  - statistics from {n_train_samples} training rows only")
        
        # 2. Target encoding for the named categorical features (FIT ON TRAIN ONLY)
        if y_train is not None and len(self.target_encode_features) > 0:
            logger.info(f"\n[2/4] Target encoding")
            logger.info(f"  - features: {len(self.target_encode_features)}")
            
            for feat in self.target_encode_features:
                if feat in X.columns:
                    # Compute encoding map from TRAIN ONLY
                    encoding_map = X.groupby(feat)[y_train.name].mean().to_dict()
                    self.target_encoders[feat] = encoding_map
                    X[feat] = X[feat].map(encoding_map).fillna(y_train.mean())
                    logger.info(f"  - {feat}: {len(encoding_map)} categories encoded")
            
            logger.info(f"  - mapping derived from {n_train_samples} training rows only")
        
        # 3. Scaling, numeric features only (FIT ON TRAIN ONLY)
        if self.scale_method and len(self.numeric_features) > 0:
            logger.info(f"\n[3/4] Numeric scaling (method={self.scale_method})")
            logger.info(f"  - features: {len(self.numeric_features)}")
            
            if self.scale_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scale_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scale_method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"unknown scaling method: {self.scale_method}")
            
            # FIT scaler on TRAIN ONLY
            X[self.numeric_features] = self.scaler.fit_transform(
                X[self.numeric_features]
            )
            
            # Log train statistics for transparency
            if self.scale_method == 'standard' and hasattr(self.scaler, 'mean_'):
                logger.info(f"  - training mean range: [{self.scaler.mean_.min():.3f}, {self.scaler.mean_.max():.3f}]")
                logger.info(f"  - training std range: [{self.scaler.scale_.min():.3f}, {self.scaler.scale_.max():.3f}]")
            
            logger.info(f"  - scaling parameters from {n_train_samples} training rows only")
        
        # 4. Feature selection (FIT ON TRAIN ONLY)
        if self.feature_selection and y_train is not None:
            logger.info(f"\n[4/4] Feature selection (method={self.feature_selection})")
            
            if self.feature_selection == 'f_classif':
                score_func = f_classif
            elif self.feature_selection == 'mutual_info':
                score_func = mutual_info_classif
            else:
                raise ValueError(f"unknown feature selection method: {self.feature_selection}")
            
            k = self.n_features if self.n_features else 'all'
            self.selector = SelectKBest(score_func=score_func, k=k)
            
            # Numeric features only, scored on the training split
            X_numeric = X[self.numeric_features]
            self.selector.fit(X_numeric, y_train)
            
            # Which features survived
            selected_mask = self.selector.get_support()
            self.selected_features = [
                feat for feat, selected in zip(self.numeric_features, selected_mask)
                if selected
            ]
            logger.info(f"  - from {len(self.numeric_features)} features, selected {len(self.selected_features)}")
            logger.info(f"  - importances from {n_train_samples} training rows only")
        
        self.is_fitted = True
        logger.info("\n" + "=" * 60)
        logger.info("Preprocessor fitted (all statistics from the training split)")
        logger.info("=" * 60)
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted preprocessing.
        
        ⚠️ CRITICAL: This method applies train-fitted statistics to new data
        (validation/test sets). NO new statistics are computed from this data.
        
        Academic Standard: Ensures test set independence.
        """
        if not self.is_fitted:
            raise ValueError("preprocessor is not fitted; call fit() first")
        
        X = X.copy()
        n_samples = len(X)
        logger.info(f"Transforming {n_samples} rows using training-split statistics")
        
        # 1. Imputation (USING TRAIN-FITTED IMPUTER)
        if self.numeric_imputer and len(self.numeric_features) > 0:
            missing_before = X[self.numeric_features].isna().sum().sum()
            
            # TRANSFORM using train-fitted imputer (NO new fit)
            X[self.numeric_features] = self.numeric_imputer.transform(
                X[self.numeric_features]
            )
            
            missing_after = X[self.numeric_features].isna().sum().sum()
            logger.info(f"  - numeric imputation: {missing_before} NaNs -> {missing_after} NaNs")
            
            if missing_after > 0:
                logger.warning(f"  - WARNING: {missing_after} missing values remain")
        
        if self.categorical_imputer and len(self.categorical_features) > 0:
            missing_before = X[self.categorical_features].isna().sum().sum()
            
            # TRANSFORM using train-fitted imputer (NO new fit)
            X[self.categorical_features] = self.categorical_imputer.transform(
                X[self.categorical_features]
            )
            
            missing_after = X[self.categorical_features].isna().sum().sum()
            logger.info(f"  - categorical imputation: {missing_before} NaNs -> {missing_after} NaNs")
        
        # 2. Target encoding (USING TRAIN-COMPUTED MAP)
        for feat, encoding_map in self.target_encoders.items():
            if feat in X.columns:
                # Unseen categories fall back to the training mean
                global_mean = np.mean(list(encoding_map.values()))
                n_unseen = X[feat].map(encoding_map).isna().sum()
                X[feat] = X[feat].map(encoding_map).fillna(global_mean)
                
                if n_unseen > 0:
                    logger.info(f"  - {feat}: {n_unseen} unseen categories filled with the training mean")
        
        # 3. Scaling (USING TRAIN-FITTED SCALER)
        if self.scaler and len(self.numeric_features) > 0:
            # TRANSFORM using train-fitted scaler (NO new fit)
            X[self.numeric_features] = self.scaler.transform(
                X[self.numeric_features]
            )
            logger.info(f"  - scaling: training mean/variance")
        
        # 4. Feature selection (USING TRAIN-SELECTED FEATURES)
        if self.selected_features:
            # Selected numeric features plus every categorical feature
            features_to_keep = self.selected_features + self.categorical_features
            X = X[features_to_keep]
            logger.info(f"  - feature selection: kept the {len(features_to_keep)} chosen at fit time")
        
        # FINAL VALIDATION: Check for remaining NaNs
        total_nans = X.isna().sum().sum()
        if total_nans > 0:
            logger.error(f"ERROR: {total_nans} NaNs remain after preprocessing")
            nan_cols = X.columns[X.isna().any()].tolist()
            logger.error(f"   affected columns: {nan_cols}")
            raise ValueError(f"{total_nans} NaNs remain after preprocessing, in columns: {nan_cols}")
        
        logger.info(f"Preprocessing complete: {X.shape[0]} rows x {X.shape[1]} features, no NaNs")
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """Return the final feature names."""
        if self.selected_features:
            return self.selected_features + self.categorical_features
        else:
            return self.numeric_features + self.categorical_features


class ClassWeightCalculator:
    """
    Class weights for imbalanced targets.
    """
    
    @staticmethod
    def compute_class_weight(y_train: pd.Series, method: str = 'balanced') -> Dict:
        """
        Compute class weights.
        
        Args:
            y_train: training labels
            method: 'balanced' or 'inverse'
            
        Returns:
            {class: weight}
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y_train)
        
        if method == 'balanced':
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
        elif method == 'inverse':
            class_counts = pd.Series(y_train).value_counts()
            total = len(y_train)
            weights = np.array([total / class_counts[c] for c in classes])
        else:
            raise ValueError(f"unknown weighting method: {method}")
        
        class_weights = {c: w for c, w in zip(classes, weights)}
        
        logger.info(f"Class weights: {class_weights}")
        return class_weights
    
    @staticmethod
    def compute_sample_weight(y_train: pd.Series, method: str = 'balanced') -> np.ndarray:
        """
        Compute per-sample weights.
        
        Returns:
            array of sample weights
        """
        class_weights = ClassWeightCalculator.compute_class_weight(y_train, method)
        sample_weights = np.array([class_weights[y] for y in y_train])
        return sample_weights


class FeatureEngineer:
    """
    Feature engineering driven by training-split statistics only.
    """
    
    def __init__(self):
        self.is_fitted = False
        self.train_stats = {}
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = None):
        """
        Compute the training-split statistics.
        """
        logger.info("Computing training-split feature statistics...")
        
        # Mean, spread and quantiles per feature
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.train_stats[col] = {
                'mean': X_train[col].mean(),
                'std': X_train[col].std(),
                'median': X_train[col].median(),
                'q25': X_train[col].quantile(0.25),
                'q75': X_train[col].quantile(0.75),
                'min': X_train[col].min(),
                'max': X_train[col].max()
            }
        
        self.is_fitted = True
        logger.info(f"Statistics computed for {len(self.train_stats)} features")
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Derive new features from the training-split statistics.
        """
        if not self.is_fitted:
            raise ValueError("feature engineer is not fitted")
        
        X = X.copy()
        
        # Example: standardised deviation features
        for col, stats in self.train_stats.items():
            if col in X.columns:
                # Deviation from the training mean
                X[f'{col}_deviation_from_train_mean'] = (
                    (X[col] - stats['mean']) / (stats['std'] + 1e-8)
                )
                
                # Whether the value sits inside the training IQR
                X[f'{col}_within_train_iqr'] = (
                    (X[col] >= stats['q25']) & (X[col] <= stats['q75'])
                ).astype(int)
        
        return X
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series = None) -> pd.DataFrame:
        """Fit and transform."""
        self.fit(X_train, y_train)
        return self.transform(X_train)


"""
预处理模块
防泄露的标准化、缩放、填补、特征选择等
仅在 train 拟合，应用于 val/test
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
    防泄露预处理器基类
    确保所有拟合仅在train上进行
    """
    
    def __init__(self):
        self.is_fitted = False
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = None):
        """在训练集上拟合"""
        raise NotImplementedError
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用转换"""
        raise NotImplementedError
        
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series = None) -> pd.DataFrame:
        """拟合并转换训练集"""
        self.fit(X_train, y_train)
        return self.transform(X_train)
    
    def save(self, path: str):
        """保存拟合的预处理器"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str):
        """加载拟合的预处理器"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class TrainOnlyPreprocessor(LeakageFreePreprocessor):
    """
    完整的训练预处理流程
    包括：缺失填补、标准化、特征选择
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
        
        # 预处理器
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.selector = None
        self.target_encoders = {}
        self.selected_features = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = None):
        """
        在训练集上拟合所有预处理步骤
        
        ⚠️ CRITICAL: All statistics (median, mean, std, etc.) are computed 
        ONLY on X_train to prevent data leakage. Val/test sets will be 
        transformed using these train-only statistics.
        
        Academic Standard: This ensures test set independence.
        """
        logger.info("=" * 60)
        logger.info("拟合预处理器 (仅使用训练集统计量)")
        logger.info("=" * 60)
        
        X = X_train.copy()
        
        # Log initial state
        n_train_samples = len(X)
        logger.info(f"训练集样本数: {n_train_samples}")
        
        # 1. 缺失填补 (FIT ON TRAIN ONLY)
        if len(self.numeric_features) > 0:
            logger.info(f"\n[1/4] 数值特征缺失填补 (strategy={self.impute_strategy})")
            logger.info(f"  - 特征数: {len(self.numeric_features)}")
            
            # Count missing values BEFORE imputation
            missing_before = X[self.numeric_features].isna().sum().sum()
            logger.info(f"  - 缺失值数 (填补前): {missing_before}")
            
            # FIT imputer on TRAIN ONLY
            self.numeric_imputer = SimpleImputer(strategy=self.impute_strategy)
            X[self.numeric_features] = self.numeric_imputer.fit_transform(
                X[self.numeric_features]
            )
            
            # Verify no missing values remain
            missing_after = X[self.numeric_features].isna().sum().sum()
            logger.info(f"  - 缺失值数 (填补后): {missing_after}")
            logger.info(f"  ✓ 填补统计量仅基于 {n_train_samples} 个训练样本")
        
        if len(self.categorical_features) > 0:
            logger.info(f"\n[1/4] 类别特征缺失填补 (strategy=most_frequent)")
            logger.info(f"  - 特征数: {len(self.categorical_features)}")
            
            missing_before = X[self.categorical_features].isna().sum().sum()
            logger.info(f"  - 缺失值数 (填补前): {missing_before}")
            
            # FIT imputer on TRAIN ONLY
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            X[self.categorical_features] = self.categorical_imputer.fit_transform(
                X[self.categorical_features]
            )
            
            missing_after = X[self.categorical_features].isna().sum().sum()
            logger.info(f"  - 缺失值数 (填补后): {missing_after}")
            logger.info(f"  ✓ 填补统计量仅基于 {n_train_samples} 个训练样本")
        
        # 2. 目标编码（仅针对指定的类别特征） (FIT ON TRAIN ONLY)
        if y_train is not None and len(self.target_encode_features) > 0:
            logger.info(f"\n[2/4] 目标编码 (Target Encoding)")
            logger.info(f"  - 特征数: {len(self.target_encode_features)}")
            
            for feat in self.target_encode_features:
                if feat in X.columns:
                    # Compute encoding map from TRAIN ONLY
                    encoding_map = X.groupby(feat)[y_train.name].mean().to_dict()
                    self.target_encoders[feat] = encoding_map
                    X[feat] = X[feat].map(encoding_map).fillna(y_train.mean())
                    logger.info(f"  - {feat}: {len(encoding_map)} 个类别编码")
            
            logger.info(f"  ✓ 编码映射仅基于 {n_train_samples} 个训练样本")
        
        # 3. 标准化（仅数值特征） (FIT ON TRAIN ONLY)
        if self.scale_method and len(self.numeric_features) > 0:
            logger.info(f"\n[3/4] 数值特征标准化 (method={self.scale_method})")
            logger.info(f"  - 特征数: {len(self.numeric_features)}")
            
            if self.scale_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scale_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scale_method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"未知的缩放方法: {self.scale_method}")
            
            # FIT scaler on TRAIN ONLY
            X[self.numeric_features] = self.scaler.fit_transform(
                X[self.numeric_features]
            )
            
            # Log train statistics for transparency
            if self.scale_method == 'standard' and hasattr(self.scaler, 'mean_'):
                logger.info(f"  - 训练集均值范围: [{self.scaler.mean_.min():.3f}, {self.scaler.mean_.max():.3f}]")
                logger.info(f"  - 训练集标准差范围: [{self.scaler.scale_.min():.3f}, {self.scaler.scale_.max():.3f}]")
            
            logger.info(f"  ✓ 缩放参数仅基于 {n_train_samples} 个训练样本")
        
        # 4. 特征选择 (FIT ON TRAIN ONLY)
        if self.feature_selection and y_train is not None:
            logger.info(f"\n[4/4] 特征选择 (method={self.feature_selection})")
            
            if self.feature_selection == 'f_classif':
                score_func = f_classif
            elif self.feature_selection == 'mutual_info':
                score_func = mutual_info_classif
            else:
                raise ValueError(f"未知的特征选择方法: {self.feature_selection}")
            
            k = self.n_features if self.n_features else 'all'
            self.selector = SelectKBest(score_func=score_func, k=k)
            
            # 仅对数值特征进行特征选择 (using TRAIN ONLY)
            X_numeric = X[self.numeric_features]
            self.selector.fit(X_numeric, y_train)
            
            # 获取选中的特征
            selected_mask = self.selector.get_support()
            self.selected_features = [
                feat for feat, selected in zip(self.numeric_features, selected_mask)
                if selected
            ]
            logger.info(f"  - 从 {len(self.numeric_features)} 个特征中选择了 {len(self.selected_features)} 个")
            logger.info(f"  ✓ 特征重要性仅基于 {n_train_samples} 个训练样本")
        
        self.is_fitted = True
        logger.info("\n" + "=" * 60)
        logger.info("✓ 预处理器拟合完成 (所有统计量仅基于训练集)")
        logger.info("=" * 60)
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        应用拟合好的预处理
        
        ⚠️ CRITICAL: This method applies train-fitted statistics to new data
        (validation/test sets). NO new statistics are computed from this data.
        
        Academic Standard: Ensures test set independence.
        """
        if not self.is_fitted:
            raise ValueError("预处理器尚未拟合，请先调用 fit()")
        
        X = X.copy()
        n_samples = len(X)
        logger.info(f"应用预处理变换到 {n_samples} 个样本 (使用训练集统计量)")
        
        # 1. 缺失填补 (USING TRAIN-FITTED IMPUTER)
        if self.numeric_imputer and len(self.numeric_features) > 0:
            missing_before = X[self.numeric_features].isna().sum().sum()
            
            # TRANSFORM using train-fitted imputer (NO new fit)
            X[self.numeric_features] = self.numeric_imputer.transform(
                X[self.numeric_features]
            )
            
            missing_after = X[self.numeric_features].isna().sum().sum()
            logger.info(f"  - 数值填补: {missing_before} NaNs → {missing_after} NaNs")
            
            if missing_after > 0:
                logger.warning(f"  ⚠️  警告: 仍有 {missing_after} 个缺失值未填补")
        
        if self.categorical_imputer and len(self.categorical_features) > 0:
            missing_before = X[self.categorical_features].isna().sum().sum()
            
            # TRANSFORM using train-fitted imputer (NO new fit)
            X[self.categorical_features] = self.categorical_imputer.transform(
                X[self.categorical_features]
            )
            
            missing_after = X[self.categorical_features].isna().sum().sum()
            logger.info(f"  - 类别填补: {missing_before} NaNs → {missing_after} NaNs")
        
        # 2. 目标编码 (USING TRAIN-COMPUTED ENCODING MAP)
        for feat, encoding_map in self.target_encoders.items():
            if feat in X.columns:
                # 对未见过的类别使用训练集的全局均值（防止泄露）
                global_mean = np.mean(list(encoding_map.values()))
                n_unseen = X[feat].map(encoding_map).isna().sum()
                X[feat] = X[feat].map(encoding_map).fillna(global_mean)
                
                if n_unseen > 0:
                    logger.info(f"  - {feat}: {n_unseen} 个未见类别使用训练集均值填充")
        
        # 3. 标准化 (USING TRAIN-FITTED SCALER)
        if self.scaler and len(self.numeric_features) > 0:
            # TRANSFORM using train-fitted scaler (NO new fit)
            X[self.numeric_features] = self.scaler.transform(
                X[self.numeric_features]
            )
            logger.info(f"  - 标准化: 使用训练集的均值/方差")
        
        # 4. 特征选择 (USING TRAIN-SELECTED FEATURES)
        if self.selected_features:
            # 保留选中的数值特征 + 所有类别特征
            features_to_keep = self.selected_features + self.categorical_features
            X = X[features_to_keep]
            logger.info(f"  - 特征选择: 保留训练时选中的 {len(features_to_keep)} 个特征")
        
        # FINAL VALIDATION: Check for remaining NaNs
        total_nans = X.isna().sum().sum()
        if total_nans > 0:
            logger.error(f"❌ 错误: 预处理后仍有 {total_nans} 个NaN值!")
            nan_cols = X.columns[X.isna().any()].tolist()
            logger.error(f"   受影响的列: {nan_cols}")
            raise ValueError(f"预处理后仍有 {total_nans} 个NaN值，列: {nan_cols}")
        
        logger.info(f"✓ 预处理完成: {X.shape[0]} 样本 × {X.shape[1]} 特征 (无NaN, 使用训练集统计量)")
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """获取最终特征名称"""
        if self.selected_features:
            return self.selected_features + self.categorical_features
        else:
            return self.numeric_features + self.categorical_features


class ClassWeightCalculator:
    """
    计算类别权重
    用于处理类不平衡问题
    """
    
    @staticmethod
    def compute_class_weight(y_train: pd.Series, method: str = 'balanced') -> Dict:
        """
        计算类别权重
        
        Args:
            y_train: 训练集标签
            method: 'balanced' 或 'inverse'
            
        Returns:
            类别权重字典 {class: weight}
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
            raise ValueError(f"未知的权重计算方法: {method}")
        
        class_weights = {c: w for c, w in zip(classes, weights)}
        
        logger.info(f"类别权重: {class_weights}")
        return class_weights
    
    @staticmethod
    def compute_sample_weight(y_train: pd.Series, method: str = 'balanced') -> np.ndarray:
        """
        计算样本权重
        
        Returns:
            样本权重数组
        """
        class_weights = ClassWeightCalculator.compute_class_weight(y_train, method)
        sample_weights = np.array([class_weights[y] for y in y_train])
        return sample_weights


class FeatureEngineer:
    """
    特征工程
    仅基于训练集统计量
    """
    
    def __init__(self):
        self.is_fitted = False
        self.train_stats = {}
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series = None):
        """
        计算训练集统计量
        """
        logger.info("计算训练集特征统计量...")
        
        # 计算各类统计量（均值、方差、分位数等）
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
        logger.info(f"已计算 {len(self.train_stats)} 个特征的统计量")
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        基于训练集统计量创建新特征
        """
        if not self.is_fitted:
            raise ValueError("特征工程器尚未拟合")
        
        X = X.copy()
        
        # 示例：创建标准化偏离度特征
        for col, stats in self.train_stats.items():
            if col in X.columns:
                # 与训练集均值的偏离度
                X[f'{col}_deviation_from_train_mean'] = (
                    (X[col] - stats['mean']) / (stats['std'] + 1e-8)
                )
                
                # 是否在训练集的IQR范围内
                X[f'{col}_within_train_iqr'] = (
                    (X[col] >= stats['q25']) & (X[col] <= stats['q75'])
                ).astype(int)
        
        return X
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series = None) -> pd.DataFrame:
        """拟合并转换"""
        self.fit(X_train, y_train)
        return self.transform(X_train)


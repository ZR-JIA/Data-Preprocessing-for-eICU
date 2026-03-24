"""
工具函数模块
"""

import numpy as np
import pandas as pd
import os
import yaml
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"已加载配置文件: {config_path}")
    return config


def save_config(config: Dict, save_path: str):
    """
    保存配置到 YAML 文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"配置已保存: {save_path}")


def identify_feature_types(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    自动识别特征类型（数值型 vs 类别型）
    
    Args:
        df: 数据框
        exclude_cols: 排除的列（如 ID、时间等）
        
    Returns:
        {'numeric': [...], 'categorical': [...]}
    """
    exclude_cols = exclude_cols or []
    
    numeric_features = []
    categorical_features = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # 数值型
            # 如果唯一值很少，可能是类别型
            if df[col].nunique() <= 10:
                categorical_features.append(col)
            else:
                numeric_features.append(col)
        else:
            # 非数值型
            categorical_features.append(col)
    
    logger.info(f"数值特征: {len(numeric_features)} 个")
    logger.info(f"类别特征: {len(categorical_features)} 个")
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }


def create_directory_structure(base_dir: str, protocol_names: List[str]):
    """
    创建目录结构（仅results相关）
    
    Args:
        base_dir: 基础目录
        protocol_names: 协议名称列表
    """
    dirs_to_create = [
        base_dir,
        os.path.join(base_dir, 'results'),
    ]
    
    for protocol in protocol_names:
        dirs_to_create.extend([
            os.path.join(base_dir, 'results', protocol),
            os.path.join(base_dir, 'results', protocol, 'curves'),
            os.path.join(base_dir, 'results', protocol, 'interpretability'),
        ])
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    logger.info(f"结果目录已创建: {base_dir}")


def validate_dataframe(df: pd.DataFrame, required_cols: List[str]):
    """
    验证数据框是否包含必需列
    
    Args:
        df: 数据框
        required_cols: 必需列列表
        
    Raises:
        ValueError: 如果缺少必需列
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据框缺少必需列: {missing_cols}")
    
    logger.info("数据框验证通过")


def merge_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_class: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
    indices: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    合并预测结果为 DataFrame
    
    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        y_pred_class: 预测类别
        patient_ids: 患者ID
        indices: 样本索引
        
    Returns:
        预测结果 DataFrame
    """
    predictions = pd.DataFrame({
        'y_true': y_true,
        'y_pred_proba': y_pred_proba,
        'y_pred_class': y_pred_class
    })
    
    if patient_ids is not None:
        predictions['patient_id'] = patient_ids
    
    if indices is not None:
        predictions['index'] = indices
    
    return predictions


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = 'youden'
) -> float:
    """
    计算最优分类阈值
    
    Args:
        y_true: 真实标签
        y_pred_proba: 预测概率
        metric: 'youden' (Youden's J), 'f1', 'balanced_accuracy'
        
    Returns:
        最优阈值
    """
    from sklearn.metrics import roc_curve, f1_score
    
    if metric == 'youden':
        # Youden's J = Sensitivity + Specificity - 1
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
    elif metric == 'f1':
        # 最大化 F1 score
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        for threshold in thresholds:
            y_pred_class = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_class)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
    elif metric == 'balanced_accuracy':
        # 最大化 balanced accuracy
        from sklearn.metrics import balanced_accuracy_score
        
        thresholds = np.linspace(0, 1, 100)
        ba_scores = []
        for threshold in thresholds:
            y_pred_class = (y_pred_proba >= threshold).astype(int)
            ba = balanced_accuracy_score(y_true, y_pred_class)
            ba_scores.append(ba)
        
        optimal_idx = np.argmax(ba_scores)
        optimal_threshold = thresholds[optimal_idx]
        
    else:
        raise ValueError(f"未知的阈值选择方法: {metric}")
    
    logger.info(f"最优阈值 ({metric}): {optimal_threshold:.4f}")
    return optimal_threshold


def detect_class_imbalance(y: pd.Series, threshold: float = 0.2) -> bool:
    """
    检测类不平衡
    
    Args:
        y: 标签
        threshold: 不平衡阈值（少数类比例）
        
    Returns:
        是否存在类不平衡
    """
    value_counts = y.value_counts()
    minority_ratio = value_counts.min() / len(y)
    
    is_imbalanced = minority_ratio < threshold
    
    logger.info(f"类别分布: {value_counts.to_dict()}")
    logger.info(f"少数类比例: {minority_ratio:.3f}")
    logger.info(f"类不平衡: {'是' if is_imbalanced else '否'}")
    
    return is_imbalanced


def print_data_summary(df: pd.DataFrame, name: str = "数据"):
    """
    打印数据摘要
    
    Args:
        df: 数据框
        name: 数据名称
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"{name} 摘要")
    logger.info(f"{'='*50}")
    logger.info(f"形状: {df.shape}")
    logger.info(f"列数: {len(df.columns)}")
    logger.info(f"样本数: {len(df)}")
    logger.info(f"缺失值比例: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]):.2%}")
    
    # 数值型特征统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        logger.info(f"\n数值特征统计:")
        logger.info(df[numeric_cols].describe())
    
    logger.info(f"{'='*50}\n")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 除零时的默认值
        
    Returns:
        商
    """
    return numerator / denominator if denominator != 0 else default


def encode_categorical_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: List[str],
    method: str = 'onehot'
) -> tuple:
    """
    编码类别特征
    
    Args:
        X_train: 训练集
        X_val: 验证集
        X_test: 测试集
        categorical_features: 类别特征列表
        method: 'onehot', 'label', 'ordinal'
        
    Returns:
        (X_train_encoded, X_val_encoded, X_test_encoded)
    """
    if method == 'onehot':
        from sklearn.preprocessing import OneHotEncoder
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # 拟合在训练集
        X_train_cat_encoded = encoder.fit_transform(X_train[categorical_features])
        X_val_cat_encoded = encoder.transform(X_val[categorical_features])
        X_test_cat_encoded = encoder.transform(X_test[categorical_features])
        
        # 获取编码后的特征名
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        
        # 创建新的DataFrame
        X_train_encoded = X_train.drop(columns=categorical_features).copy()
        X_val_encoded = X_val.drop(columns=categorical_features).copy()
        X_test_encoded = X_test.drop(columns=categorical_features).copy()
        
        # 添加编码后的特征
        X_train_encoded = pd.concat([
            X_train_encoded.reset_index(drop=True),
            pd.DataFrame(X_train_cat_encoded, columns=encoded_feature_names)
        ], axis=1)
        
        X_val_encoded = pd.concat([
            X_val_encoded.reset_index(drop=True),
            pd.DataFrame(X_val_cat_encoded, columns=encoded_feature_names)
        ], axis=1)
        
        X_test_encoded = pd.concat([
            X_test_encoded.reset_index(drop=True),
            pd.DataFrame(X_test_cat_encoded, columns=encoded_feature_names)
        ], axis=1)
        
        return X_train_encoded, X_val_encoded, X_test_encoded
        
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        
        X_train_encoded = X_train.copy()
        X_val_encoded = X_val.copy()
        X_test_encoded = X_test.copy()
        
        for feat in categorical_features:
            encoder = LabelEncoder()
            X_train_encoded[feat] = encoder.fit_transform(X_train[feat].astype(str))
            X_val_encoded[feat] = encoder.transform(X_val[feat].astype(str))
            X_test_encoded[feat] = encoder.transform(X_test[feat].astype(str))
        
        return X_train_encoded, X_val_encoded, X_test_encoded
    
    else:
        raise ValueError(f"未知的编码方法: {method}")


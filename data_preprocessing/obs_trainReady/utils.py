"""
Shared helper functions.
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
    Load a YAML config file.
    
    Args:
        config_path: path to the config file
        
    Returns:
        the config as a dict
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config: {config_path}")
    return config


def save_config(config: Dict, save_path: str):
    """
    Write a config back to YAML.
    
    Args:
        config: the config dict
        save_path: destination path
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Config saved: {save_path}")


def identify_feature_types(df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Classify features as numeric or categorical.
    
    Args:
        df: the dataframe
        exclude_cols: columns to skip, such as IDs and timestamps
        
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
            # Numeric, unless it has very few distinct values
            if df[col].nunique() <= 10:
                categorical_features.append(col)
            else:
                numeric_features.append(col)
        else:
            # Non-numeric
            categorical_features.append(col)
    
    logger.info(f"Numeric features: {len(numeric_features)}")
    logger.info(f"Categorical features: {len(categorical_features)}")
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }


def create_directory_structure(base_dir: str, protocol_names: List[str]):
    """
    Create the results directory tree.
    
    Args:
        base_dir: root directory
        protocol_names: split protocol names
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
    
    logger.info(f"Results directories created: {base_dir}")


def validate_dataframe(df: pd.DataFrame, required_cols: List[str]):
    """
    Check that a dataframe carries the required columns.
    
    Args:
        df: the dataframe
        required_cols: columns that must be present
        
    Raises:
        ValueError: if any required column is missing
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"dataframe is missing required columns: {missing_cols}")
    
    logger.info("Dataframe validation passed")


def merge_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_class: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
    indices: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Collect predictions into a DataFrame.
    
    Args:
        y_true: ground-truth labels
        y_pred_proba: predicted probabilities
        y_pred_class: predicted classes
        patient_ids: patient identifiers
        indices: row indices
        
    Returns:
        a DataFrame of predictions
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
    Find the best classification threshold.
    
    Args:
        y_true: ground-truth labels
        y_pred_proba: predicted probabilities
        metric: 'youden' (Youden's J), 'f1', 'balanced_accuracy'
        
    Returns:
        the chosen threshold
    """
    from sklearn.metrics import roc_curve, f1_score
    
    if metric == 'youden':
        # Youden's J = Sensitivity + Specificity - 1
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
    elif metric == 'f1':
        # Maximise F1
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        for threshold in thresholds:
            y_pred_class = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_class)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
    elif metric == 'balanced_accuracy':
        # Maximise balanced accuracy
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
        raise ValueError(f"unknown threshold metric: {metric}")
    
    logger.info(f"Best threshold ({metric}): {optimal_threshold:.4f}")
    return optimal_threshold


def detect_class_imbalance(y: pd.Series, threshold: float = 0.2) -> bool:
    """
    Detect class imbalance.
    
    Args:
        y: labels
        threshold: minority-class ratio below which data counts as imbalanced
        
    Returns:
        True when the data is imbalanced
    """
    value_counts = y.value_counts()
    minority_ratio = value_counts.min() / len(y)
    
    is_imbalanced = minority_ratio < threshold
    
    logger.info(f"Class distribution: {value_counts.to_dict()}")
    logger.info(f"Minority class ratio: {minority_ratio:.3f}")
    logger.info(f"Imbalanced: {'yes' if is_imbalanced else 'no'}")
    
    return is_imbalanced


def print_data_summary(df: pd.DataFrame, name: str = "data"):
    """
    Log a short summary of a dataframe.
    
    Args:
        df: the dataframe
        name: label to print alongside the summary
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"{name} summary")
    logger.info(f"{'='*50}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info(f"Rows: {len(df)}")
    logger.info(f"Missing rate: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]):.2%}")
    
    # Numeric feature statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        logger.info(f"\nNumeric feature statistics:")
        logger.info(df[numeric_cols].describe())
    
    logger.info(f"{'='*50}\n")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Division that tolerates a zero denominator.
    
    Args:
        numerator: dividend
        denominator: divisor
        default: value returned when the denominator is zero
        
    Returns:
        the quotient
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
    Encode categorical features.
    
    Args:
        X_train: training split
        X_val: validation split
        X_test: test split
        categorical_features: columns to encode
        method: 'onehot', 'label', 'ordinal'
        
    Returns:
        (X_train_encoded, X_val_encoded, X_test_encoded)
    """
    if method == 'onehot':
        from sklearn.preprocessing import OneHotEncoder
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Fit on the training split
        X_train_cat_encoded = encoder.fit_transform(X_train[categorical_features])
        X_val_cat_encoded = encoder.transform(X_val[categorical_features])
        X_test_cat_encoded = encoder.transform(X_test[categorical_features])
        
        # Names of the encoded features
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        
        # Build the output frames
        X_train_encoded = X_train.drop(columns=categorical_features).copy()
        X_val_encoded = X_val.drop(columns=categorical_features).copy()
        X_test_encoded = X_test.drop(columns=categorical_features).copy()
        
        # Attach the encoded features
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
        raise ValueError(f"unknown encoding method: {method}")


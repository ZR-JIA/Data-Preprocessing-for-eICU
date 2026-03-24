
import pandas as pd

def drop_sparse_and_impute(df: pd.DataFrame, missing_threshold: float):
    """
    Drop columns with missing ratio > threshold.
    
    ⚠️ IMPORTANT: This function NO LONGER performs imputation to prevent data leakage.
    Missing values (NaN) are preserved for downstream train-only imputation.
    
    Academic Standard: Statistical operations (median, mode, mean) must ONLY be 
    computed on the training set after data splitting. This function is called 
    BEFORE splitting, so it must NOT perform any statistical imputation.
    
    Args:
        df: Input dataframe
        missing_threshold: Columns with missing rate > this threshold will be dropped
        
    Returns:
        Tuple of (df_cleaned, report_dict)
        - df_cleaned: Dataframe with sparse columns removed, NaNs preserved
        - report_dict: Contains 'dropped_cols' and 'kept_cols_with_missing'
    """
    report = {"dropped_cols": [], "kept_cols_with_missing": []}
    
    # Calculate missing ratio for each column
    miss_ratio = df.isna().mean()
    
    # Identify and drop columns exceeding missing threshold
    drop_cols = miss_ratio[miss_ratio > missing_threshold].index.tolist()
    df_cleaned = df.drop(columns=drop_cols)
    report["dropped_cols"] = drop_cols
    
    # Track columns that still contain missing values (for transparency)
    remaining_cols_with_missing = df_cleaned.columns[df_cleaned.isna().any()].tolist()
    report["kept_cols_with_missing"] = remaining_cols_with_missing
    
    # NO IMPUTATION - NaNs are preserved for train-only preprocessing
    
    return df_cleaned, report

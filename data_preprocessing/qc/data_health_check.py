#!/usr/bin/env python3
"""
=============================================================================
eICU/MIMIC 数据健康检查脚本 (Data Health Check)
=============================================================================
作者: Senior Health Data Scientist
目的: 检查数据结构（静态 vs 动态）+ 排查数据泄露风险

使用方法:
    1. 修改下面的配置变量 (file_path, target_col)
    2. 运行: python data_health_check.py
=============================================================================
"""

import pandas as pd
import numpy as np
import warnings
from typing import Tuple, List, Dict, Optional
warnings.filterwarnings('ignore')


# ============================================================================
# 配置区域 (Configuration) - 请修改这里
# ============================================================================
file_path = "outputs/obs_cleaned.csv"  # 数据文件路径
target_col = "mortality"     # 预测目标列名
id_col = 'patientunitstayid' 
# ============================================================================


def print_section(title: str, symbol: str = "="):
    """打印分节标题"""
    print(f"\n{symbol * 80}")
    print(f"  {title}")
    print(f"{symbol * 80}\n")


def find_patient_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    智能查找病人ID列
    尝试常见的ID列名
    """
    possible_id_cols = [
        'patientunitstayid', 'patient_unit_stay_id',
        'subject_id', 'hadm_id', 'icustay_id',
        'patient_id', 'id', 'ID', 'stay_id'
    ]
    
    for col in possible_id_cols:
        if col in df.columns:
            return col
    
    # 如果都找不到，查找包含'id'的列
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    if id_cols:
        return id_cols[0]
    
    return None


def check_id_structure(df: pd.DataFrame, id_col: str) -> Tuple[str, Dict]:
    """
    检查1: ID结构分析 (Static vs Dynamic)
    
    Returns:
        data_type: "STATIC" or "DYNAMIC"
        stats: 统计信息字典
    """
    print_section("📊 检查 1: ID 结构分析 (Static vs Dynamic)", "=")
    
    total_rows = len(df)
    unique_patients = df[id_col].nunique()
    rows_per_patient = total_rows / unique_patients
    
    stats = {
        'total_rows': total_rows,
        'unique_patients': unique_patients,
        'rows_per_patient': rows_per_patient
    }
    
    print(f"✓ 总行数 (Total Rows):           {total_rows:,}")
    print(f"✓ 唯一病人数 (Unique Patients):   {unique_patients:,}")
    print(f"✓ 平均每病人行数 (Rows/Patient):  {rows_per_patient:.2f}")
    
    # 判定逻辑
    if rows_per_patient <= 1.1:  # 允许少量容错
        data_type = "STATIC"
        print(f"\n{'='*80}")
        print(f"🎯 判定结论: 【静态表格 (STATIC)】")
        print(f"   ➜ One row per patient")
        print(f"   ➜ 推荐模型: MLP, ResNet, XGBoost, Random Forest")
        print(f"{'='*80}")
    else:
        data_type = "DYNAMIC"
        print(f"\n{'='*80}")
        print(f"🎯 判定结论: 【动态时序 (DYNAMIC)】")
        print(f"   ➜ Multiple rows per patient (时序数据)")
        print(f"   ➜ 推荐模型: LSTM, GRU, Transformer, Temporal CNN")
        print(f"{'='*80}")
    
    return data_type, stats


def detect_data_leakage(df: pd.DataFrame, target_col: str) -> Dict:
    """
    检查2: 数据泄露侦探 (Data Leakage Detection)
    
    Returns:
        leakage_report: 泄露检测报告
    """
    print_section("🔍 检查 2: 数据泄露侦探 (Data Leakage Detection)", "=")
    
    leakage_report = {
        'high_correlation': [],
        'suspicious_keywords': [],
        'duplicates': 0
    }
    
    # -------------------------------------------------------------------------
    # 2.1 高相关性排查 (High Correlation Check)
    # -------------------------------------------------------------------------
    print("🔸 [2.1] 高相关性排查 (检测相关系数 > 0.9)")
    print("-" * 80)
    
    if target_col not in df.columns:
        print(f"⚠️  警告: 目标列 '{target_col}' 不存在!")
        numeric_cols = []
    else:
        # 只选择数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) == 0:
            print("   ℹ️  没有数值特征可供检查")
        else:
            correlations = []
            for col in numeric_cols:
                try:
                    corr = df[col].corr(df[target_col])
                    if not np.isnan(corr):
                        correlations.append((col, corr))
                except:
                    pass
            
            # 排序
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # 检查高相关性
            high_corr_threshold = 0.9
            found_leakage = False
            
            for col, corr in correlations:
                if abs(corr) > high_corr_threshold:
                    found_leakage = True
                    leakage_report['high_correlation'].append((col, corr))
                    print(f"   🚨 【高度疑似泄露】 列名: '{col}' | 相关系数: {corr:.4f}")
            
            if not found_leakage:
                print(f"   ✓ 未发现高相关性泄露 (所有特征相关系数 < {high_corr_threshold})")
            
            # 显示前5个最高相关性（即使不超过阈值）
            if correlations:
                print(f"\n   📋 相关性 TOP 5:")
                for i, (col, corr) in enumerate(correlations[:5], 1):
                    print(f"      {i}. {col:40s} → {corr:+.4f}")
    
    # -------------------------------------------------------------------------
    # 2.2 关键词排查 (Keyword Scanning)
    # -------------------------------------------------------------------------
    print(f"\n🔸 [2.2] 关键词排查 (扫描可疑列名)")
    print("-" * 80)
    
    # 定义可疑关键词（根据预测任务定制）
    suspicious_keywords = [
        'death', 'dead', 'died', 'expire', 'expired', 'expiration',
        'mortality', 'mortal', 'deceased',
        'discharge', 'discharged', 'disposition',
        'outcome', 'result', 'final',
        'diagnosis', 'diagnose', 'icd',  # 如果是预测诊断任务
        'future', 'post', 'after',
        'label', 'target', 'ground_truth'
    ]
    
    all_columns = [col for col in df.columns if col != target_col]
    found_suspicious = False
    
    for col in all_columns:
        col_lower = col.lower()
        for keyword in suspicious_keywords:
            if keyword in col_lower:
                found_suspicious = True
                leakage_report['suspicious_keywords'].append(col)
                print(f"   ⚠️  【嫌疑特征】 列名: '{col}' (包含关键词: '{keyword}')")
                break
    
    if not found_suspicious:
        print("   ✓ 未发现可疑列名")
    
    # -------------------------------------------------------------------------
    # 2.3 重复数据检查 (Duplicate Check)
    # -------------------------------------------------------------------------
    print(f"\n🔸 [2.3] 重复数据检查")
    print("-" * 80)
    
    duplicates = df.duplicated().sum()
    leakage_report['duplicates'] = duplicates
    
    if duplicates > 0:
        dup_percentage = (duplicates / len(df)) * 100
        print(f"   ⚠️  发现 {duplicates:,} 行完全重复 ({dup_percentage:.2f}%)")
        print(f"   ➜ 建议: 使用 df.drop_duplicates() 去重")
    else:
        print(f"   ✓ 未发现完全重复的行")
    
    # -------------------------------------------------------------------------
    # 泄露总结
    # -------------------------------------------------------------------------
    print(f"\n{'='*80}")
    total_issues = len(leakage_report['high_correlation']) + \
                   len(leakage_report['suspicious_keywords']) + \
                   (1 if duplicates > 0 else 0)
    
    if total_issues == 0:
        print("✅ 数据泄露检查: 未发现明显泄露风险")
    else:
        print(f"⚠️  数据泄露检查: 发现 {total_issues} 个潜在问题")
        print("   ➜ 建议仔细审查上述标记的特征")
    print(f"{'='*80}")
    
    return leakage_report


def sample_temporal_structure(df: pd.DataFrame, id_col: str, data_type: str):
    """
    检查3: 时序结构抽样 (Temporal Structure Sampling)
    仅在动态数据时执行
    """
    if data_type != "DYNAMIC":
        return
    
    print_section("⏰ 检查 3: 时序结构抽样 (Temporal Structure)", "=")
    
    # 找到记录数最多的病人
    patient_counts = df[id_col].value_counts()
    max_patient_id = patient_counts.index[0]
    max_count = patient_counts.iloc[0]
    
    print(f"✓ 记录数最多的病人 ID: {max_patient_id}")
    print(f"✓ 该病人的记录数: {max_count}")
    
    # 提取该病人的数据
    patient_data = df[df[id_col] == max_patient_id].head(5)
    
    # 尝试找时间相关列
    time_cols = [col for col in df.columns if any(
        keyword in col.lower() 
        for keyword in ['time', 'offset', 'hour', 'minute', 'datetime', 'timestamp', 'date']
    )]
    
    if time_cols:
        print(f"\n✓ 检测到时间相关列: {time_cols}")
        display_cols = [id_col] + time_cols + \
                      [col for col in patient_data.columns if col not in [id_col] + time_cols][:5]
    else:
        print("\n⚠️  未检测到明显的时间列 (offset/time 等)")
        display_cols = patient_data.columns[:8].tolist()
    
    print(f"\n📋 该病人的前 5 行数据预览:")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(patient_data[display_cols].to_string(index=False))
    print("-" * 80)


def basic_overview(df: pd.DataFrame, target_col: str, id_col: str):
    """
    检查4: 基础概览 (Basic Overview)
    """
    print_section("📈 检查 4: 基础数据概览 (Basic Overview)", "=")
    
    # 特征总数（排除ID和目标列）
    exclude_cols = [target_col, id_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 区分数值列和类别列
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"✓ 数据维度:           {df.shape[0]:,} 行 × {df.shape[1]:,} 列")
    print(f"✓ 特征总数:           {len(feature_cols):,} (排除ID和目标列)")
    print(f"   ├─ 数值特征:       {len(numeric_features):,}")
    print(f"   └─ 类别特征:       {len(categorical_features):,}")
    
    # 目标列分布
    if target_col in df.columns:
        print(f"\n✓ 目标列 ('{target_col}') 分布:")
        value_counts = df[target_col].value_counts()
        for val, count in value_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   ├─ {val}: {count:,} ({percentage:.2f}%)")
    
    # 缺失值统计
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(cols_with_missing) > 0:
        print(f"\n✓ 缺失值概况:")
        print(f"   ├─ 含缺失值的列数: {len(cols_with_missing)}")
        print(f"   └─ 缺失值最多的 TOP 5:")
        for i, (col, miss_count) in enumerate(cols_with_missing.head(5).items(), 1):
            miss_pct = (miss_count / len(df)) * 100
            print(f"      {i}. {col:40s} → {miss_count:,} ({miss_pct:.2f}%)")
    else:
        print(f"\n✓ 缺失值: 无")
    
    # 内存使用
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\n✓ 内存占用:           {memory_mb:.2f} MB")


def main():
    """主函数"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "eICU/MIMIC 数据健康检查报告" + " " * 30 + "║")
    print("║" + " " * 15 + "Data Health Check & Leakage Detection" + " " * 26 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # =========================================================================
    # 加载数据
    # =========================================================================
    print_section("📂 加载数据 (Loading Data)", "=")
    
    try:
        print(f"✓ 文件路径: {file_path}")
        print(f"✓ 目标列名: {target_col}")
        print(f"\n正在加载...")
        
        df = pd.read_csv(file_path)
        print(f"✓ 数据加载成功! 维度: {df.shape}")
        
    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在 - {file_path}")
        print(f"   请检查 file_path 配置是否正确")
        return
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return
    
    # =========================================================================
    # 查找病人ID列
    # =========================================================================
    id_col = find_patient_id_column(df)
    
    if id_col is None:
        print("\n⚠️  警告: 未找到病人ID列，部分检查将跳过")
        print("   提示: 请确认数据中是否包含 'patientunitstayid' 或类似的ID列")
        id_col = df.columns[0]  # 使用第一列作为备选
        print(f"   临时使用第一列作为ID: '{id_col}'")
    else:
        print(f"\n✓ 识别到病人ID列: '{id_col}'")
    
    # =========================================================================
    # 执行各项检查
    # =========================================================================
    
    # 检查1: ID结构
    data_type, id_stats = check_id_structure(df, id_col)
    
    # 检查2: 数据泄露
    leakage_report = detect_data_leakage(df, target_col)
    
    # 检查3: 时序结构抽样（仅动态数据）
    sample_temporal_structure(df, id_col, data_type)
    
    # 检查4: 基础概览
    basic_overview(df, target_col, id_col)
    
    # =========================================================================
    # 终极总结
    # =========================================================================
    print_section("🎯 终极总结 (Final Summary)", "=")
    
    print("问题 1️⃣: 我该用什么模型?")
    print("-" * 80)
    if data_type == "STATIC":
        print("✅ 答案: 使用 【静态模型】")
        print("   推荐:")
        print("   • MLP (Multi-Layer Perceptron)")
        print("   • ResNet (Residual Network)")
        print("   • XGBoost / LightGBM")
        print("   • Random Forest")
    else:
        print("✅ 答案: 使用 【时序模型】")
        print("   推荐:")
        print("   • LSTM / GRU (适合长短期记忆)")
        print("   • Transformer (适合复杂时序依赖)")
        print("   • Temporal CNN (适合局部时序模式)")
        print("   • 可考虑先做时序聚合 (Aggregation) 后再用静态模型")
    
    print("\n问题 2️⃣: 我的特征里有数据泄露吗?")
    print("-" * 80)
    
    total_leakage_flags = len(leakage_report['high_correlation']) + \
                         len(leakage_report['suspicious_keywords'])
    
    if total_leakage_flags == 0:
        print("✅ 答案: 未发现明显的数据泄露特征")
        print("   状态: 可以安心训练模型")
    else:
        print(f"⚠️  答案: 发现 {total_leakage_flags} 个可疑特征")
        print("   建议:")
        print("   • 仔细审查上述标记的特征")
        print("   • 确认这些特征在预测时是否真的可用")
        print("   • 如果确认是泄露，请从训练数据中移除")
    
    if leakage_report['duplicates'] > 0:
        print(f"\n⚠️  额外提醒: 发现 {leakage_report['duplicates']:,} 行重复数据")
        print("   建议先去重: df = df.drop_duplicates()")
    
    print("\n" + "=" * 80)
    print("✅ 数据健康检查完成!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

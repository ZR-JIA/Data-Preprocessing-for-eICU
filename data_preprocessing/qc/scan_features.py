import pandas as pd
import numpy as np
import gc

# ================= 配置区 =================
# 这里填你 run_obs_clean.py 生成的那个最终文件
# 如果你想扫原始文件也可以，但为了设计模型，通常是扫 clean_static.csv
FILE_PATH = 'outputs/obs_cleaned.csv' 
CHUNK_SIZE = 50000  # 每次只读 5万行，内存占用极低
# ==========================================

def scan_large_file(file_path):
    print(f"🚀 正在分批扫描超大文件: {file_path}")
    print(f"   (Chunk Size: {CHUNK_SIZE} rows/batch)")
    
    # 1. 先只读表头，获取列名 (不占内存)
    try:
        header_df = pd.read_csv(file_path, nrows=0)
    except FileNotFoundError:
        print("❌ 文件未找到！")
        return

    all_cols = header_df.columns.tolist()
    
    # 排除 ID 和 Target
    target_col = 'mortality'
    id_col = 'patientunitstayid'
    feature_cols = [c for c in all_cols if c not in [target_col, id_col]]
    
    # 初始化统计容器
    # 用 set 来存储类别特征的唯一值 (自动去重)
    cat_stats = {col: set() for col in feature_cols} 
    num_cols = []
    
    # 2. 只有第一次迭代用于判断是 数值 还是 类别
    is_first_chunk = True
    processed_rows = 0
    
    # 3. 开始分批读取 (流水线作业)
    # usecols: 只读取特征列，进一步省内存
    with pd.read_csv(file_path, usecols=feature_cols, chunksize=CHUNK_SIZE) as reader:
        for chunk in reader:
            processed_rows += len(chunk)
            print(f"\r   -> 已扫描行数: {processed_rows:,} ...", end="", flush=True)
            
            # --- 第一批次：决定谁是数值，谁是类别 ---
            if is_first_chunk:
                potential_cats = []
                for col in feature_cols:
                    # 如果是字符串，或者虽然是数字但看起来像分类(比如全是整数且范围小)
                    # 这里简化逻辑：字符串 -> 类别；数字 -> 数值
                    # 如果你想把 'gender'(1/2) 这种数字也当类别，可以在这里判断
                    if pd.api.types.is_numeric_dtype(chunk[col]):
                        # 暂时先归为数值，后面如果发现 unique 值很少，你可以手动调整
                        # 但为了双塔，通常只有明确的 ID 类/字符串才进 Embedding
                        # 除非你能确定这个数字字段只有 <20 个取值
                        num_cols.append(col)
                        # 如果是数值列，就不需要统计 unique 值了，释放 cat_stats
                        del cat_stats[col]
                    else:
                        potential_cats.append(col)
                is_first_chunk = False
            
            # --- 所有批次：统计类别特征的基数 (Cardinality) ---
            # 只有还在 cat_stats 里的列（即被判定为类别的）才需要统计
            for col in list(cat_stats.keys()):
                # 把这一批的唯一值加入集合
                unique_vals = chunk[col].dropna().unique()
                cat_stats[col].update(unique_vals)
                
                # 防爆机制：如果某个本来以为是类别的列，唯一值超过 10000 个，
                # 说明它可能不是类别（或者是高基数特征），防止 set 撑爆内存
                if len(cat_stats[col]) > 10000:
                    print(f"\n⚠️  警告: 列 [{col}] 的唯一值过多 (>10k)，可能不适合做普通 Embedding。")
            
            # 垃圾回收
            del chunk
            gc.collect()

    print(f"\n✅ 扫描完成！总行数: {processed_rows:,}")
    
    # --- 整理输出 ---
    final_cat_cols = []
    for col, unique_set in cat_stats.items():
        final_cat_cols.append((col, len(unique_set)))
    
    print("-" * 40)
    print("📊 双塔模型设计参考 (Metadata)")
    print("-" * 40)
    
    print(f"👉 [Left Tower: Categorical / Embedding]")
    print(f"   共 {len(final_cat_cols)} 个特征")
    print(f"   格式: [列名] (Embedding Input Dimension)")
    for name, count in final_cat_cols:
        # Embedding 维度通常建议: min(50, (count + 1) // 2)
        print(f"   - {name}: {count} classes")

    print(f"\n👉 [Right Tower: Numerical / Dense]")
    print(f"   共 {len(num_cols)} 个特征")
    # print(f"   列表: {num_cols}") # 如果太多就不打印了

    print("-" * 40)
    print("💡 报告完成。")

if __name__ == "__main__":
    scan_large_file(FILE_PATH)
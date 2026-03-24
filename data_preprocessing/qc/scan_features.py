import pandas as pd
import numpy as np
import gc

# ================= Configuration =================
# Point this at the file produced by run_obs_clean.py. Scanning a raw table
# also works, but model design normally starts from the cleaned output.
FILE_PATH = 'outputs/obs_cleaned.csv' 
CHUNK_SIZE = 50000  # 50k rows at a time keeps memory flat
# ==========================================

def scan_large_file(file_path):
    print(f"Scanning large file in chunks: {file_path}")
    print(f"   (Chunk Size: {CHUNK_SIZE} rows/batch)")
    
    # 1. Read the header alone to get column names
    try:
        header_df = pd.read_csv(file_path, nrows=0)
    except FileNotFoundError:
        print("File not found.")
        return

    all_cols = header_df.columns.tolist()
    
    # Exclude IDs and the target
    target_col = 'mortality'
    id_col = 'patientunitstayid'
    feature_cols = [c for c in all_cols if c not in [target_col, id_col]]
    
    # Accumulators. Categorical levels go in a set, which dedupes for us.
    cat_stats = {col: set() for col in feature_cols} 
    num_cols = []
    
    # 2. The first chunk decides numeric vs categorical
    is_first_chunk = True
    processed_rows = 0
    
    # 3. Stream the file
    # usecols: read feature columns only, to save more memory
    with pd.read_csv(file_path, usecols=feature_cols, chunksize=CHUNK_SIZE) as reader:
        for chunk in reader:
            processed_rows += len(chunk)
            print(f"\r   -> rows scanned: {processed_rows:,} ...", end="", flush=True)
            
            # --- First chunk: classify each column ---
            if is_first_chunk:
                potential_cats = []
                for col in feature_cols:
                    # Simple rule: string -> categorical, numeric -> numeric.
                    # A numeric column with few distinct values (e.g. gender
                    # coded 1/2) would also be categorical; adjust here.
                    if pd.api.types.is_numeric_dtype(chunk[col]):
                        # Treated as numeric for now. For the dual-tower model
                        # only string/ID-like columns normally reach an embedding,
                        # unless the field is known to have <20 distinct values.
                        num_cols.append(col)
                        # Numeric: no level counting needed, free the slot
                        del cat_stats[col]
                    else:
                        potential_cats.append(col)
                is_first_chunk = False
            
            # --- Every chunk: accumulate categorical cardinality ---
            # Only columns still in cat_stats are treated as categorical.
            for col in list(cat_stats.keys()):
                # Fold this chunk's levels into the set
                unique_vals = chunk[col].dropna().unique()
                cat_stats[col].update(unique_vals)
                
                # Guard: past 10k levels the column is not really categorical
                # (or is very high cardinality); stop before the set blows up.
                if len(cat_stats[col]) > 10000:
                    print(f"\n  WARNING: column [{col}] has >10k distinct values; not a good fit for a plain embedding.")
            
            # Reclaim memory
            del chunk
            gc.collect()

    print(f"\nScan complete. Total rows: {processed_rows:,}")
    
    # --- Report ---
    final_cat_cols = []
    for col, unique_set in cat_stats.items():
        final_cat_cols.append((col, len(unique_set)))
    
    print("-" * 40)
    print("Dual-tower design reference (metadata)")
    print("-" * 40)
    
    print(f"👉 [Left Tower: Categorical / Embedding]")
    print(f"   {len(final_cat_cols)} features")
    print(f"   format: [column] (embedding input dimension)")
    for name, count in final_cat_cols:
        # Rule of thumb for embedding width: min(50, (count + 1) // 2)
        print(f"   - {name}: {count} classes")

    print(f"\n👉 [Right Tower: Numerical / Dense]")
    print(f"   {len(num_cols)} features")
    # print(f"   columns: {num_cols}")  # usually too many to print

    print("-" * 40)
    print("Report complete.")

if __name__ == "__main__":
    scan_large_file(FILE_PATH)
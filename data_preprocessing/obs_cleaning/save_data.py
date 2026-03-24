import pandas as pd
from pathlib import Path

def save_data(df, output_path):
    """Save cleaned data to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


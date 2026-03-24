import pandas as pd
from pathlib import Path

def load_data(data_dir):
    """Load raw CSV files."""
    data_dir = Path(data_dir)
    patient = pd.read_csv(data_dir / "patient.csv")
    diagnosis = pd.read_csv(data_dir / "diagnosis.csv")
    vital = pd.read_csv(data_dir / "vitalPeriodic.csv")
    lab = pd.read_csv(data_dir / "lab.csv")
    return patient, diagnosis, vital, lab


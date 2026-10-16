import pandas as pd
from pathlib import Path

# Tables the pipeline expects, in the order load_data returns them.
_EXPECTED_TABLES = ("patient", "diagnosis", "vitalPeriodic", "lab")


def _read_table(data_dir: Path, name: str, columns=None) -> pd.DataFrame:
    """
    Read one raw eICU CSV, restricted to `columns` when a whitelist is given.

    The whitelist is enforced strictly: a column named in the config but absent
    from the CSV raises instead of being silently dropped. Silent drift here is
    what previously allowed non-clinical columns (e.g. the vitalPeriodic primary
    key `vitalperiodicid`) to reach feature aggregation -- see AUDIT.md EICU-01.
    """
    path = data_dir / f"{name}.csv"
    if columns is None:
        return pd.read_csv(path)

    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in columns if c not in header]
    if missing:
        raise KeyError(
            f"{path.name}: columns declared in config but not present in the CSV: "
            f"{missing}. Available columns: {sorted(header)}"
        )
    return pd.read_csv(path, usecols=columns)


def load_data(data_dir, tables=None):
    """
    Load the four raw eICU CSV files.

    Args:
        data_dir: directory holding patient.csv, diagnosis.csv,
                  vitalPeriodic.csv and lab.csv
        tables:   optional {table_name: [column, ...]} whitelist, normally the
                  `tables` block of obs_cleaning.yaml. When omitted every column
                  is read, which is NOT the documented behaviour of the pipeline
                  and is kept only for ad-hoc exploration.

    Returns:
        (patient, diagnosis, vital, lab)
    """
    data_dir = Path(data_dir)
    tables = tables or {}
    return tuple(
        _read_table(data_dir, name, tables.get(name))
        for name in _EXPECTED_TABLES
    )

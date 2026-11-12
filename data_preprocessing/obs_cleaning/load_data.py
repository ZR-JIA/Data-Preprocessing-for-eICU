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
    if not path.exists() and (data_dir / f"{name}.csv.gz").exists():
        # PhysioNet distributes eICU as .csv.gz; pandas reads it directly.
        path = data_dir / f"{name}.csv.gz"
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


def load_table_for_stays(data_dir, name, columns, stay_ids, offset_col=None,
                         max_offset=None, chunksize=2_000_000):
    """
    Read a large table in chunks, keeping only rows of `stay_ids`.

    vitalPeriodic holds about 146 million rows; reading it whole needs more
    memory than a 16 GB machine has. Filtering each chunk to the cohort keeps
    the result small. When `offset_col` is given, rows outside
    [0, max_offset] are dropped too -- exactly the range align_time_window()
    keeps afterwards, so the output is unchanged, only cheaper.
    """
    data_dir = Path(data_dir)
    path = data_dir / f"{name}.csv"
    if not path.exists() and (data_dir / f"{name}.csv.gz").exists():
        path = data_dir / f"{name}.csv.gz"
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in columns if c not in header]
    if missing:
        raise KeyError(f"{path.name}: columns declared in config but not present: {missing}")

    keep = set(stay_ids)
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize):
        mask = chunk["patientunitstayid"].isin(keep)
        if offset_col is not None:
            mask &= chunk[offset_col].between(0, max_offset)
        if mask.any():
            parts.append(chunk[mask])
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def load_extra_tables(data_dir, tables, names):
    """
    Load additional eICU tables by name, with the same strict whitelist.

    Only tables that are both requested in `names` and declared in the
    `tables` block are read; requesting an undeclared table raises, so a
    cohort definition cannot silently run without the table it relies on.

    Returns:
        {table_name: DataFrame}
    """
    data_dir = Path(data_dir)
    tables = tables or {}
    undeclared = [n for n in names if n not in tables]
    if undeclared:
        raise KeyError(
            f"Tables requested but not declared in the config 'tables' block: "
            f"{undeclared}"
        )
    return {name: _read_table(data_dir, name, tables[name]) for name in names}

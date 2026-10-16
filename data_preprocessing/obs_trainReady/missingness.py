"""
Missing-data treatment that keeps the fact of absence as a signal.

AUDIT.md EICU-09. The published pipeline dropped columns above 70% missing and
median-imputed the rest, which treats a gap purely as something to fill. In
eICU that is hard to defend: whether a lab was drawn reflects clinical
suspicion and which interfaces a hospital deployed, so missingness is unlikely
to be random. Median-imputing a `*_std` column is the sharpest case -- a
patient with a single measurement has no variability at all, and the cohort
median asserts a typical one.

This module adds two things:

* `MissingnessIndicators`, which records, per feature, whether a value was
  absent before imputation. Fitted on train, like every other statistic here.
* `missingness_report`, the stratified diagnostic AUDIT.md asks for: missing
  rate by hospital and by outcome, which is what distinguishes "not measured
  because nobody was worried" from "not measured because this hospital never
  reports it".

Enabling indicators changes the feature count and therefore departs from the
paper's 94 numeric + 3 categorical = 97. That is why this lives on
`mmda_enhance` and not on `fix`.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SUFFIX = "_missing"


class MissingnessIndicators:
    """
    Binary indicators for values absent before imputation.

    A column earns an indicator when its training missing rate falls between
    `min_rate` and `max_rate`. Below the floor there is nothing to learn from;
    above the ceiling the indicator is almost constant and the column itself is
    nearly empty.

    Which columns qualify is decided on the training split alone, so val and
    test cannot influence the feature set.
    """

    def __init__(
        self,
        min_rate: float = 0.01,
        max_rate: float = 0.95,
        suffix: str = DEFAULT_SUFFIX,
    ):
        if not 0.0 <= min_rate < max_rate <= 1.0:
            raise ValueError("expected 0 <= min_rate < max_rate <= 1")
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.suffix = suffix
        self.indicator_cols: List[str] = []
        self.train_rates: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, columns: Optional[List[str]] = None):
        columns = list(columns) if columns is not None else list(X_train.columns)
        rates = X_train[columns].isna().mean()
        keep = rates[(rates >= self.min_rate) & (rates <= self.max_rate)]
        self.indicator_cols = list(keep.index)
        self.train_rates = {c: float(rates[c]) for c in columns}
        self.is_fitted = True
        logger.info(
            "Missingness indicators: %d of %d columns qualify "
            "(train missing rate in [%.2f, %.2f])",
            len(self.indicator_cols), len(columns), self.min_rate, self.max_rate,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Append one 0/1 column per qualifying feature.

        Must be called before imputation: afterwards there is nothing left to
        record.
        """
        if not self.is_fitted:
            raise ValueError("MissingnessIndicators is not fitted; call fit() first")
        if not self.indicator_cols:
            return pd.DataFrame(index=X.index)
        flags = X[self.indicator_cols].isna().astype(np.int8)
        flags.columns = [f"{c}{self.suffix}" for c in self.indicator_cols]
        return flags

    def fit_transform(self, X_train: pd.DataFrame,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
        return self.fit(X_train, columns).transform(X_train)

    @property
    def feature_names(self) -> List[str]:
        return [f"{c}{self.suffix}" for c in self.indicator_cols]


def missingness_report(
    df: pd.DataFrame,
    columns: List[str],
    by: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Missing rate per feature, optionally split by one or more grouping columns.

    Passing `by=['site_id']` answers whether a gap tracks the hospital, and
    `by=['mortality']` whether it tracks the outcome. Both are the questions
    AUDIT.md EICU-09 says have to be answered before median imputation can be
    called harmless.
    """
    present = [c for c in columns if c in df.columns]
    overall = df[present].isna().mean().rename("overall")

    if not by:
        return overall.to_frame().sort_values("overall", ascending=False)

    missing_by = df.groupby(by)[present].apply(lambda g: g.isna().mean())
    table = missing_by.T
    table.columns = [f"{'_'.join(by)}={c}" for c in table.columns]
    table.insert(0, "overall", overall)
    # How far apart the groups are is the part that matters: a gap that is
    # uniform across hospitals is a different problem from one that is not.
    table["spread"] = table.drop(columns=["overall"]).max(axis=1) - \
                      table.drop(columns=["overall"]).min(axis=1)
    return table.sort_values("spread", ascending=False)

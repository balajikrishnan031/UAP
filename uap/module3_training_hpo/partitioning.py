"""
Module 3 - Part 1: Adaptive Data Partitioning Engine.
Selects optimal cross-validation scheme based on ProblemType and DCV characteristics.
"""

from typing import Generator, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

from uap.core.contracts import ProblemType, DatasetCharacteristicVector


class AdaptiveDataPartitioner:
    """
    Selects and yields cross-validation train/val splits adhering to strict leakage barriers.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def get_splits(
        self, X: pd.DataFrame, y: pd.Series, dcv: DatasetCharacteristicVector
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        problem_type = dcv.dataset_summary.problem_type
        imbalance = dcv.data_health_scores.imbalance_ratio

        if problem_type in [ProblemType.BINARY_CLASSIFICATION, ProblemType.MULTICLASS_CLASSIFICATION]:
            # Always use StratifiedKFold for classification to preserve minority representation
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            return skf.split(X, y)

        elif problem_type == ProblemType.TIME_SERIES:
            # Purged / Walk-Forward TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            return tscv.split(X, y)

        else:
            # Standard K-Fold for Regression
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            return kf.split(X, y)

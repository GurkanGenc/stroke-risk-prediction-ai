from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_PATH, FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def load_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw stroke dataset."""
    return pd.read_csv(filepath or DATA_PATH)


def split_features_and_target(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return raw model features and target without manual preprocessing."""
    X = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data while preserving the target class balance."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


class DataProcessor:
    """Compatibility wrapper around the module-level data helpers."""

    load_data = staticmethod(load_data)
    split_features_and_target = staticmethod(split_features_and_target)
    split_train_test = staticmethod(split_train_test)

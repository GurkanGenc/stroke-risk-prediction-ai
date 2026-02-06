import pandas as pd
from typing import Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def split_data_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.

    Ensures stratification to maintain class distribution.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test

def create_logistic_regression_model() -> LogisticRegression:
    """Initialize the model with specified parameters."""
    return LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )

def train_logistic_regression_model(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """Train the model using the training data."""
    return model.fit(X_train, y_train)
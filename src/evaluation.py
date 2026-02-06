import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

def get_predictions(model: LogisticRegression, X_test: pd.DataFrame) -> np.ndarray:
    """Generate predictions from the trained model."""
    return model.predict(X_test)

def evaluate_model(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Evaluate model performance using standard classification metrics."""
    print("Accuracy:")
    print(accuracy_score(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
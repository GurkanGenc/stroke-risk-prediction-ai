from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PIPELINE_PATH,
    RANDOM_STATE,
)


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing for raw UI/training inputs."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("bin", binary_pipeline, BINARY_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline() -> Pipeline:
    """Build the single end-to-end stroke prediction pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def save_pipeline(pipeline: Pipeline, path: Optional[Path] = None) -> Path:
    """Persist a trained pipeline and return the saved path."""
    output_path = path or PIPELINE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return output_path


def load_pipeline(path: Optional[Path] = None) -> Pipeline:
    """Load the trained preprocessing-plus-model pipeline."""
    return joblib.load(path or PIPELINE_PATH)


def predict_stroke_risk(
    pipeline: Pipeline,
    patient_data: pd.DataFrame,
) -> Tuple[int, float]:
    """Return class prediction and stroke probability for one patient row."""
    prediction = int(pipeline.predict(patient_data)[0])
    probability = float(pipeline.predict_proba(patient_data)[0][1])
    return prediction, probability

from pathlib import Path
from typing import Optional

from sklearn.pipeline import Pipeline

from config import PIPELINE_PATH
from data_processing import load_data, split_features_and_target, split_train_test
from pipeline import build_pipeline, save_pipeline


def train_pipeline() -> Pipeline:
    """Train the end-to-end stroke prediction pipeline."""
    data = load_data()
    X, y = split_features_and_target(data)
    X_train, _, y_train, _ = split_train_test(X, y)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def train_and_save_pipeline(output_path: Optional[Path] = None) -> Path:
    """Train and persist the single production pipeline."""
    trained_pipeline = train_pipeline()
    return save_pipeline(trained_pipeline, output_path or PIPELINE_PATH)


def main() -> None:
    output_path = train_and_save_pipeline()
    print(f"Saved trained pipeline to {output_path}")


if __name__ == "__main__":
    main()

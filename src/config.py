from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "healthcare-dataset-stroke-data.csv"
MODELS_DIR = PROJECT_ROOT / "src" / "models"
PIPELINE_PATH = MODELS_DIR / "stroke_prediction_pipeline.joblib"

TARGET_COLUMN = "stroke"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERICAL_FEATURES = ["age", "avg_glucose_level", "bmi"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

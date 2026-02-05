import pandas as pd
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV data into a pandas DataFrame."""
    return pd.read_csv(filepath)

def fill_missing_values_median(data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing BMI values using the median."""
    return data.assign(bmi=data['bmi'].fillna(data['bmi'].median()))

def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to numerical format (one-hot encoding)."""
    return pd.get_dummies(data,
                          columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'],
                          drop_first=True)

def scale_numerical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Standardize numerical features."""
    numerical_columns = ['age', 'avg_glucose_level', 'bmi']
    copy_data = data.copy()

    scaled = scale.fit_transform(copy_data[numerical_columns])
    copy_data[numerical_columns] = scaled
    return copy_data

def split_features_and_target(data: pd.DataFrame):
    """Split dataset into features (X) and target (y)."""
    X = data.drop(columns=['stroke'])
    y = data['stroke']
    return X, y
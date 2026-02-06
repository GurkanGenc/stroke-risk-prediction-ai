import data_processing as dp
import model_training as mt
import evaluation as ev

def main():
    # Load raw data
    data = dp.load_data("./data/healthcare-dataset-stroke-data.csv")
    # print(data.head())

    # Handle missing values
    data = dp.fill_missing_values_median(data)
    # print(data.isnull().sum())

    # Encode categorical features
    data = dp.encode_categorical_features(data)
    # print(data.head())

    # Scale numerical features
    data = dp.scale_numerical_features(data)
    
    # Split features and target
    X, y = dp.split_features_and_target(data)
    
    # Split train and test sets
    data_splits = mt.split_data_train_test(X, y)
    X_train, X_test, y_train, y_test = data_splits
    
    # Create model
    model = mt.create_logistic_regression_model()
    
    # Train model
    model = mt.train_logistic_regression_model(model, X_train, y_train)
    
    # Generate predictions
    y_pred = ev.get_predictions(model, X_test)

    # Evaluate model
    ev.evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    main()
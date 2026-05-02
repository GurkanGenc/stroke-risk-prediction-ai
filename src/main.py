from data_processing import load_data, split_features_and_target, split_train_test
from evaluation import evaluate_pipeline, print_evaluation
from pipeline import build_pipeline, save_pipeline


def main() -> None:
    data = load_data()
    X, y = split_features_and_target(data)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    saved_path = save_pipeline(pipeline)
    print(f"Saved trained pipeline to {saved_path}")

    metrics = evaluate_pipeline(pipeline, X_test, y_test)
    print_evaluation(metrics)


if __name__ == "__main__":
    main()

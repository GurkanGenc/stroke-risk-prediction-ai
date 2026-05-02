# Stroke Risk Prediction AI

This project trains and serves a stroke risk prediction model using one shared
scikit-learn pipeline. The same saved pipeline handles preprocessing and
prediction in both the training/evaluation scripts and the Streamlit UI.

## Tech Stack

- Python
- pandas
- scikit-learn
- Streamlit
- joblib
- Jupyter Notebook

## Dataset

The project uses the Kaggle Stroke Prediction Dataset by fedesoriano:
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data

Target variable: `stroke`

## Project Structure

```text
stroke-risk-prediction-ai/
|-- data/
|   `-- healthcare-dataset-stroke-data.csv
|-- notebooks/
|   `-- analysis.ipynb
|-- src/
|   |-- app.py
|   |-- config.py
|   |-- data_processing.py
|   |-- evaluation.py
|   |-- main.py
|   |-- model_training.py
|   |-- pipeline.py
|   `-- models/
|       `-- stroke_prediction_pipeline.joblib
|-- requirements.txt
|-- requirements-dev.txt
|-- README.md
`-- report.pdf
```

## Module Responsibilities

- `src/config.py`: shared paths, feature lists, target name, split settings.
- `src/data_processing.py`: raw data loading and train/test splitting.
- `src/pipeline.py`: single preprocessing-plus-model pipeline, save/load helpers,
  and prediction helper.
- `src/model_training.py`: train and save the production pipeline.
- `src/evaluation.py`: evaluation metrics for a trained pipeline.
- `src/main.py`: full train, save, and evaluate entry point.
- `src/app.py`: Streamlit UI that loads and uses the saved pipeline.

## Run Locally

```powershell
venv\Scripts\activate
pip install -r requirements.txt
python src\main.py
streamlit run src\app.py
```

## Single Pipeline Flow

```text
raw CSV data
-> data_processing.load_data
-> data_processing.split_features_and_target
-> pipeline.build_pipeline
-> pipeline.fit
-> pipeline.save_pipeline
-> app.load_pipeline
-> pipeline.predict_stroke_risk
```

The UI should not perform manual encoding, imputation, or scaling. Those steps
belong inside `src/pipeline.py` so training and inference always use the same
transformations.

## Online Demo

Live App:
https://stroke-risk-prediction-ai.streamlit.app

## Disclaimer

This application is for educational and demonstration purposes only. It is not
a medical diagnostic tool.

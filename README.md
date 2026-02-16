# Stroke Prediction – ML Project

This project explores a medical dataset to analyze factors related to stroke
and builds a machine learning system for stroke risk prediction, including:

- Data preprocessing pipeline

- Machine learning model training

- Model evaluation

- Interactive web interface (UI) for predictions.

## Tech Stack
- Python
- pandas
- scikit-learn
- matplotlib
- Streamlit
- Jupyter Notebook

## Dataset
Medical dataset containing features such as:
- age
- gender
- health indicators
- lifestyle actors.

Target variable: `stroke`

## Dataset Source
The dataset was downloaded from Kaggle and is used for educational purposes only.

Source: Stroke Prediction Dataset  
Author: fedesoriano  
Link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data

# Project Structure
```text
stroke-risk-prediction-ai/
│
├── data/
│   └── healthcare-dataset-stroke-data.csv
│
├── src/
│   ├── DataProcessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── app.py
│   ├── main.py
│   └── models/
│       ├── stroke_model.joblib
│       └── feature_schema.joblib
│
├── notebooks/
│   └── analysis.ipynb
│
├── requirements.txt
├── README.md
└── report.pdf
```

# Online Demo (Recommended for Recruiters)
Live App:
    https://your-app-link.streamlit.app

No installation required.
Just open the link and use the interface.

## How to Use the UI
1. Open the application
2. Enter patient information in the form
3. Click Predict Stroke Risk
4. View the prediction result

The model will return a stroke risk prediction based on the trained machine learning model.

⚠️ This application is for educational and demonstration purposes only.
It is not a medical diagnostic tool.

# Local Installation (For Developers / Technical Reviewers)
1️⃣ Activate your virtual environment
### Windows
venv\Scripts\activate

### macOS / Linux
source venv/bin/activate

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
streamlit run src/app.py

A browser window will automatically.

# Machine Learning Pipeline
- Raw Data  
- Preprocessing  
- Encoding  
- Scaling  
- Feature Engineering  
- Model Training  
- Model Evaluation  
- Model Saving  
- Inference via UI  

# Limitations
- Educational dataset
- Class imbalance in target variable
- Simplified medical modeling
- Not clinically validated

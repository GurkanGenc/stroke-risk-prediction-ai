import streamlit as st
import joblib
import pandas as pd
from DataProcessing import DataProcessor


@st.cache_resource
def load_model():
    return joblib.load("src/models/stroke_prediction_model.joblib")

def main():
    st.set_page_config(page_title="Stroke Risk Prediction AI")

    st.title("Stroke Risk Prediction AI")
    st.write("This application predicts stroke risk based on patient data.")

    st.divider()

    st.subheader("Patitent Information")
    st.write("Please enter the following information about the patient:")

    # # Input fields for patient data
    # Create two columns
    col1, col2 = st.columns(2)

    # First row
    with col1:
        age = st.number_input("Age", 0, 100, 25)
    with col2:
        avg_glucose_level = st.number_input(
            "Average Glucose Level", 50.0, 300.0, 100.0)

    # Second row
    with col1:
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    with col2:
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])

    # Third row
    with col1:
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    # Fourth row
    with col1:
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    with col2:
        work_type = st.selectbox(
            "Work Type",
            ["Private", "Self-employed",
             "Govt_job", "children", "Never_worked"])

    # Fifth row
    with col1:
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    with col2:
        smoking_status = st.selectbox(
            "Smoking Status",
            ["never smoked", "formerly smoked", "smokes", "Unknown"])
    
    # Convert inputs to DataFrame for feature alignment
    input_data = pd.DataFrame({
        "age": age,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "gender": gender,
        "ever_married": 1 if ever_married == "Yes" else 0,
        "work_type": work_type,
        "Residence_type": residence_type,
        "smoking_status": smoking_status
    }, index=[0])

    # Process input data (alignment, encoding and scaling)
    processed_input = DataProcessor.encode_categorical_features(input_data)
    processed_input = DataProcessor.scale_numerical_features(processed_input)

    st.divider()

    if st.button("Predict Stroke Risk"):
        load_model()
        st.success("Model loaded successfully!")
        
        st.subheader("Input Summary")
        st.write({
            "age": age,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "gender": gender,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "smoking_status": smoking_status
        })

if __name__ == "__main__":
    main()
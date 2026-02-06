import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("models/stroke_prediction_model.joblib")

def main():
    st.set_page_config(page_title="Stroke Risk Prediction AI")

    st.title("Stroke Risk Prediction AI")
    st.write("This application predicts stroke risk based on patient data.")

    st.divider()

    st.subheader("Patitent Information")
    st.write("Please enter the following information about the patient:")

    # Input fields for patient data
    age =st.number_input("Age", 0, 100, 25)
    
    avg_glucose_level = st.number_input(
        "Average Glucose Level",
        min_value=50.0,
        max_value=300.0,
        value=100.0
    )
    
    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )
    
    hypertension = st.selectbox(
        "Hypertension",
        options=["No", "Yes"]
    )
    
    heart_disease = st.selectbox(
        "Heart Disease",
        options=["No", "Yes"]
    )
    
    gender = st.selectbox(
        "Gender",
        options=["Male", "Female", "Other"]
    )
    
    ever_married = st.selectbox(
        "Ever Married",
        options=["No", "Yes"]
    )
    
    work_type = st.selectbox(
        "Work Type",
        options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    )
    
    residence_type = st.selectbox(
        "Residence Type",
        options=["Urban", "Rural"]
    )
    
    smoking_status = st.selectbox(
        "Smoking Status",
        options=["never smoked", "formerly smoked", "smokes", "Unknown"]
    )
    
    st.divider()

    if st.button("Predict Stroke Risk"):
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
            "residence_type": residence_type,
            "smoking_status": smoking_status
        })

if __name__ == "__main__":
    main()
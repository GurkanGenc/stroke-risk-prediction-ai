import pandas as pd
import streamlit as st

from pipeline import load_pipeline, predict_stroke_risk


GENDER_OPTIONS = ["Male", "Female", "Other"]
YES_NO_OPTIONS = ["No", "Yes"]
WORK_TYPE_OPTIONS = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
RESIDENCE_OPTIONS = ["Urban", "Rural"]
SMOKING_OPTIONS = ["never smoked", "formerly smoked", "smokes", "Unknown"]


@st.cache_resource
def get_pipeline():
    return load_pipeline()


def yes_no_to_int(value: str) -> int:
    return 1 if value == "Yes" else 0


def build_patient_input(
    age: int,
    avg_glucose_level: float,
    bmi: float,
    hypertension: str,
    heart_disease: str,
    gender: str,
    ever_married: str,
    work_type: str,
    residence_type: str,
    smoking_status: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [age],
            "avg_glucose_level": [avg_glucose_level],
            "bmi": [bmi],
            "hypertension": [yes_no_to_int(hypertension)],
            "heart_disease": [yes_no_to_int(heart_disease)],
            "gender": [gender],
            "ever_married": [ever_married],
            "work_type": [work_type],
            "Residence_type": [residence_type],
            "smoking_status": [smoking_status],
        }
    )


def render_patient_form() -> pd.DataFrame:
    st.write("Please enter the following information about the patient:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 100, 25)
        bmi = st.number_input("BMI", 10.0, 60.0, 22.0,
                              help=(
                                    "Body Mass Index (BMI)\n\n"
                                    "Formula:\n"
                                    "BMI = weight (kg) / height² (m²)\n\n"
                                    "Example:\n"
                                    "70 kg and 1.75 m → BMI = 22.9"
                                ))
        heart_disease = st.selectbox("Heart Disease", YES_NO_OPTIONS)
        ever_married = st.selectbox("Ever Married", YES_NO_OPTIONS)
        residence_type = st.selectbox("Residence Type", RESIDENCE_OPTIONS)

    with col2:
        avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0,
                                            help=(
                                                "Dataset uses mg/dL.\n\n"
                                                "Common conversion:\n"
                                                "1 mmol/L = 18 mg/dL\n\n"
                                                "Example:\n"
                                                "5.5 mmol/L * 18 = 99 mg/dL"
                                            ))
        hypertension = st.selectbox("Hypertension", YES_NO_OPTIONS)
        gender = st.selectbox("Gender", GENDER_OPTIONS)
        work_type = st.selectbox("Work Type", WORK_TYPE_OPTIONS)
        smoking_status = st.selectbox("Smoking Status", SMOKING_OPTIONS)

    return build_patient_input(
        age=age,
        avg_glucose_level=avg_glucose_level,
        bmi=bmi,
        hypertension=hypertension,
        heart_disease=heart_disease,
        gender=gender,
        ever_married=ever_married,
        work_type=work_type,
        residence_type=residence_type,
        smoking_status=smoking_status,
    )


def render_prediction_result(patient_input: pd.DataFrame) -> None:
    pipeline = get_pipeline()
    prediction, probability = predict_stroke_risk(pipeline, patient_input)

    st.subheader("Result")
    if prediction == 1:
        st.error(f"High stroke risk ({probability:.2%})")
    else:
        st.success(f"Low stroke risk ({probability:.2%})")

    st.subheader("Input Summary")
    st.dataframe(patient_input, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Stroke Risk Prediction AI")

    st.title("Stroke Risk Prediction AI")
    st.write("Predicts stroke risk based on patient data.")
    st.divider()

    patient_input = render_patient_form()
    st.divider()

    if st.button("Predict"):
        render_prediction_result(patient_input)


if __name__ == "__main__":
    main()

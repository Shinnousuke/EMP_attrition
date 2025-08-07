# app.py

import streamlit as st
import pickle
import numpy as np

# --- Load Trained Model ---
with open("C:/Users/Admin/Desktop/internship_projects/attention_detection_employees/rf_model.pkl", "rb") as f:

    model = pickle.load(f)

with open("C:/Users/Admin/Desktop/internship_projects/attention_detection_employees/trained_columns.pkl", "rb") as f:
    trained_columns = pickle.load(f)

# --- Streamlit App Header ---
st.title("🧠 Employee Attrition Predictor")
st.write("Fill in the employee details to check if they are at risk of leaving.")

# --- Collect User Inputs ---
satisfaction_level = st.slider("Job Satisfaction (1 - Low to 4 - High)", 1, 4, 3)
income = st.number_input("Monthly Income", min_value=1000, value=5000, step=100)
age = st.slider("Age", 18, 60, 30)
years_at_company = st.slider("Years at Company", 0, 40, 5)
overtime = st.selectbox("OverTime", ["Yes", "No"])
job_role = st.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative",
    "Manager", "Sales Representative", "Research Director", "Human Resources"
])
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

# --- Create Input Dictionary with One-Hot Encoded Features ---
input_data = {
    "JobSatisfaction": satisfaction_level,
    "MonthlyIncome": income,
    "Age": age,
    "YearsAtCompany": years_at_company,
    "OverTime_Yes": 1 if overtime == "Yes" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,
    "JobRole_" + job_role: 1,
    "MaritalStatus_" + marital_status: 1
}

# --- Create Full Input Vector Matching Trained Columns ---
input_vector = [input_data.get(col, 0) for col in trained_columns]

# --- Prediction ---
prediction = model.predict([input_vector])[0]
proba = model.predict_proba([input_vector])[0][1]

# --- Display Results ---
if prediction == 1:
    st.error(f"⚠️ High Attrition Risk! (Probability: {proba:.2f})")
else:
    st.success(f"✅ Likely to Stay (Probability: {proba:.2f})")

st.caption("Model trained on IBM HR Analytics dataset (Kaggle)")
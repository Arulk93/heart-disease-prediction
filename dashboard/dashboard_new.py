import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set the correct model directory path
models_dir = os.path.abspath("notebooks/models/")  # Updated path

# Load Pretrained Models & Scaler
scaler = joblib.load(os.path.join(models_dir, "standard_scaler.pkl"))
selected_features = joblib.load(os.path.join(models_dir, "selected_features.pkl"))
model = joblib.load(os.path.join(models_dir, "best_rf_model.pkl"))

# Set Streamlit Page Config
st.set_page_config(layout="wide")

# Title
st.markdown("<h1 style='text-align: center;'>🚀 Heart Disease Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("")

# Sidebar Input Fields
st.sidebar.header("Enter Patient Details")

# Input Fields
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
resting_bp = st.sidebar.number_input("RestingBP", min_value=50, max_value=200, value=120)
cholesterol = st.sidebar.number_input("Cholesterol", min_value=100, max_value=400, value=200)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar", [0, 1])
max_hr = st.sidebar.number_input("Max Heart Rate", min_value=60, max_value=220, value=100)
oldpeak = st.sidebar.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1, value=1.0)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
exercise_angina = st.sidebar.selectbox("Exercise Angina", ["Yes", "No"])
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

# Convert Categorical Inputs
sex_m = 1 if sex == "Male" else 0
exercise_angina_y = 1 if exercise_angina == "Yes" else 0
st_slope_flat = 1 if st_slope == "Flat" else 0
st_slope_up = 1 if st_slope == "Up" else 0

chest_pain_ata = 1 if chest_pain == "ATA" else 0
chest_pain_nap = 1 if chest_pain == "NAP" else 0
chest_pain_ta = 1 if chest_pain == "TA" else 0

resting_ecg_normal = 1 if resting_ecg == "Normal" else 0
resting_ecg_st = 1 if resting_ecg == "ST" else 0

# Create Feature DataFrame
input_data = pd.DataFrame([[age, resting_bp, cholesterol, fasting_bs, max_hr, oldpeak,
                            sex_m, exercise_angina_y, st_slope_flat, st_slope_up,
                            chest_pain_ata, chest_pain_nap, chest_pain_ta,
                            resting_ecg_normal, resting_ecg_st]],
                          columns=["Age", "RestingBP", "Cholesterol", "FastingBS",
                                   "MaxHR", "Oldpeak", "Sex_M", "ExerciseAngina_Y",
                                   "ST_Slope_Flat", "ST_Slope_Up", "ChestPainType_ATA",
                                   "ChestPainType_NAP", "ChestPainType_TA",
                                   "RestingECG_Normal", "RestingECG_ST"])

# Select Only Features Used in Model
input_data = input_data[selected_features]

# Apply Standard Scaling
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)[0]
prediction_prob = model.predict_proba(input_scaled)[0][1] * 100  # Probability for Heart Disease

# Dashboard Prediction Section
st.markdown("<h2 style='text-align: center;'>🚀 Prediction Is</h2>", unsafe_allow_html=True)
st.write("")

# Layout with Two Columns
col1, col2 = st.columns(2)

with col1:
    st.image("heart_patient_image.jpg", width=300)
    st.markdown(f"<h3 style='text-align: center;'>Prediction Probability: {100 - prediction_prob:.1f}%</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: red; text-align: center;'>Heart Patient</h4>", unsafe_allow_html=True)

with col2:
    st.image("healthy_heart_image.jpg", width=300)
    st.markdown(f"<h3 style='text-align: center;'>Prediction Probability: {prediction_prob:.1f}%</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: green; text-align: center;'>No Heart Patient</h4>", unsafe_allow_html=True)")

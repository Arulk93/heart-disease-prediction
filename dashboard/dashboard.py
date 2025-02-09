import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource()
def load_model():
    return joblib.load("notebooks/models/best_rf_model.pkl")  # Make sure the path is correct

model = load_model()

# Define expected features manually
correct_feature_order = [
    'age', 'restingbp', 'cholesterol', 'fastingbs', 'maxhr', 'oldpeak',
    'sex_m', 'chestpaintype_ata', 'chestpaintype_nap', 'chestpaintype_ta',
    'restingecg_normal', 'restingecg_st', 'exerciseangina_y',
    'st_slope_flat', 'st_slope_up'
]

# Streamlit UI
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Prediction Dashboard ❤️")
st.markdown("A simple tool to predict the likelihood of heart disease.")

# Sidebar for user input
st.sidebar.header("User Input Features ⚙️")

# Numerical Inputs (with sliders)
user_input = {}
user_input['age'] = st.sidebar.slider("Age (years)", value=50, min_value=0, max_value=120)
user_input['restingbp'] = st.sidebar.slider("Resting Blood Pressure (mm Hg)", value=130, min_value=0, max_value=200)
user_input['cholesterol'] = st.sidebar.slider("Serum Cholesterol (mg/dL)", value=200, min_value=0, max_value=600)
user_input['fastingbs'] = st.sidebar.number_input("Fasting Blood Sugar (> 120 mg/dL)", value=0, min_value=0, max_value=1)
user_input['maxhr'] = st.sidebar.number_input("Maximum Heart Rate (bpm)", value=150, min_value=0, max_value=300)
user_input['oldpeak'] = st.sidebar.number_input("ST Depression Induced by Exercise", value=1.0, min_value=0.0, max_value=10.0)

# Categorical Inputs
sex = st.sidebar.selectbox("Sex", ["Male ♂️", "Female ♀️"])
user_input['sex_m'] = 1 if sex == "Male ♂️" else 0

chest_pain = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
chest_input_mapping = {"Typical Angina": "ATA", "Atypical Angina": "NAP", "Non-Anginal Pain": "TA", "Asymptomatic": "ASY"}
chest_pain_code = chest_input_mapping[chest_pain]
user_input['chestpaintype_ata'] = 1 if chest_pain_code == "ATA" else 0
user_input['chestpaintype_nap'] = 1 if chest_pain_code == "NAP" else 0
user_input['chestpaintype_ta'] = 1 if chest_pain_code == "TA" else 0

resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
resting_ecg_mapping = {"Normal": "Normal", "ST-T Wave Abnormality": "ST", "Left Ventricular Hypertrophy": "LVH"}
resting_ecg_code = resting_ecg_mapping[resting_ecg]
user_input['restingecg_normal'] = 1 if resting_ecg_code == "Normal" else 0
user_input['restingecg_st'] = 1 if resting_ecg_code == "ST" else 0

exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina", ["Yes", "No"])
user_input['exerciseangina_y'] = 1 if exercise_angina == "Yes" else 0

st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])
st_slope_mapping = {"Up": "Up", "Flat": "Flat", "Down": "Down"}
st_slope_code = st_slope_mapping[st_slope]
user_input['st_slope_up'] = 1 if st_slope_code == "Up" else 0
user_input['st_slope_flat'] = 1 if st_slope_code == "Flat" else 0


# Convert user input into a DataFrame
user_data = pd.DataFrame([user_input])

# Ensure correct feature order and fill missing features
user_data = user_data.reindex(columns=correct_feature_order, fill_value=0)

# No scaling applied here, using the raw user input
user_data_array = np.array(user_data, dtype=np.float64).reshape(1, -1)

# Make prediction when the user clicks the button
if st.sidebar.button("Predict 🩺"):
    prediction_proba = model.predict_proba(user_data_array)
    threshold = 0.45  # Changed threshold to 0.45

    # Adjust threshold dynamically for uncertain cases
    if abs(prediction_proba[0][1] - prediction_proba[0][0]) < 0.05:
        threshold = 0.43  # Further dynamic adjustment

    predicted_class = 1 if prediction_proba[0][1] > threshold else 0

    st.subheader("🩺 Prediction Result 🩺")
    result = "Positive for Heart Disease 💔" if predicted_class == 1 else "No Heart Disease Detected 💚"
    st.markdown(f"<span style='font-size:40px; font-weight:bold;'>{result}</span>", unsafe_allow_html=True)

    st.subheader("📊 Comparison of User Input with Data Distribution 📊")

    # Load your dataset for comparison (assuming you have it loaded somewhere)
    # Example: heart_data = pd.read_csv("path/to/your/heart_disease_dataset.csv")
    # This dataset should contain 'age', 'cholesterol', and 'restingbp' columns for visualization

    # For demonstration, I will generate random data similar to what might be expected
    np.random.seed(42)  # For reproducibility
    heart_data = pd.DataFrame({
        'age': np.random.normal(50, 10, 1000),
        'cholesterol': np.random.normal(200, 50, 1000),
        'restingbp': np.random.normal(130, 20, 1000)
    })

    # Plotting user input against the distribution of the data
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))  # Adjusted figure size to make the chart smaller

    # Age Distribution
    axes[0].hist(heart_data['age'], bins=30, color='lightblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(user_input['age'], color='red', linestyle='dashed', linewidth=2, label="User Input")
    axes[0].set_title("Age Distribution", fontsize=8)
    axes[0].set_xlabel("Age (years)", fontsize=8)
    axes[0].set_ylabel("Frequency", fontsize=8)
    axes[0].tick_params(axis='both', labelsize=8)  # Font size for ticks
    axes[0].legend(fontsize=5)

    # Cholesterol Distribution
    axes[1].hist(heart_data['cholesterol'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(user_input['cholesterol'], color='red', linestyle='dashed', linewidth=2, label="User Input")
    axes[1].set_title("Cholesterol Distribution", fontsize=8)
    axes[1].set_xlabel("Cholesterol (mg/dL)", fontsize=8)
    axes[1].set_ylabel("Frequency", fontsize=8)
    axes[1].tick_params(axis='both', labelsize=8)  # Font size for ticks
    axes[1].legend(fontsize=5)

    # Resting BP Distribution
    axes[2].hist(heart_data['restingbp'], bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[2].axvline(user_input['restingbp'], color='red', linestyle='dashed', linewidth=2, label="User Input")
    axes[2].set_title("Resting Blood Pressure Distribution", fontsize=8)
    axes[2].set_xlabel("Resting BP (mm Hg)", fontsize=8)
    axes[2].set_ylabel("Frequency", fontsize=8)
    axes[2].tick_params(axis='both', labelsize=8)  # Font size for ticks
    axes[2].legend(fontsize=5)

    st.pyplot(fig)



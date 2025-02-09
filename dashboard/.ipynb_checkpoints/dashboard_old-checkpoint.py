import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Load saved models and data
models_dir = os.path.abspath("notebooks/models")
best_rf = joblib.load(os.path.join(models_dir, "best_rf_model.pkl"))
expected_features = joblib.load(os.path.join(models_dir, "selected_features.pkl"))

st.set_page_config(page_title="Heart Failure Prediction", page_icon='🚨')
st.title(':red[Heart] Failure Prediction 🫀🏥')

st.markdown(
    """
    <div style="background-color:#000000; padding:10px; border-radius:5px">
        <h4 style="color:#faf7f7;">A heart failure prediction app uses machine learning and patient data to assess risk, enabling early diagnosis and informed medical decisions.🤖💡
        </h4>
    </div>
    """,
    unsafe_allow_html=True
)

# Define feature selection lists
sex_mapping = {'Male': 1, 'Female': 0}
ChestPainType_mapping = {'TA': [1, 0, 0, 0], 'ATA': [0, 1, 0, 0], 'NAP': [0, 0, 1, 0], 'ASY': [0, 0, 0, 1]}
RestingECG_mapping = {'Normal': [1, 0, 0], 'ST': [0, 1, 0], 'LVH': [0, 0, 1]}
ExerciseAngina_mapping = {'Yes': 1, 'No': 0}
ST_Slope_mapping = {'Up': [1, 0, 0], 'Flat': [0, 1, 0], 'Down': [0, 0, 1]}

# Create three columns layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.header("User Input Features")
    age = st.slider("Age:", 0, 100, 35)
    sex = sex_mapping[st.radio("Sex:", list(sex_mapping.keys()))]
    ChestPainType = st.radio("Chest Pain Type:", list(ChestPainType_mapping.keys()))
    FastingBS = 1 if st.selectbox("Fasting Blood Sugar:", ['Yes', 'No']) == 'Yes' else 0
    RestingECG = st.radio("Resting ECG:", list(RestingECG_mapping.keys()))
    Cholesterol = st.slider("Cholesterol:", 40, 400, 120)
    RestingBP = st.slider("Resting Blood Pressure:", 40, 400, 120)
    Oldpeak = st.slider("Oldpeak:", 0, 10, 0)
    MaxHR = st.slider("Max Heart Rate:", 50, 250, 120)
    ExerciseAngina = st.selectbox("Exercise Induced Angina:", list(ExerciseAngina_mapping.keys()))
    ST_Slope = st.selectbox("ST Slope:", list(ST_Slope_mapping.keys()))
    
    st.title("Start Prediction")
    b = st.button("Start", icon='🚨', use_container_width=True)

# Create DataFrame for input
data = {
    'Age': [age],
    'RestingBP': [RestingBP],
    'Cholesterol': [Cholesterol],
    'MaxHR': [MaxHR],
    'Oldpeak': [Oldpeak],
    'FastingBS': [FastingBS],
    'Sex_M': [sex]
}

# Expand one-hot encoded features
def expand_one_hot_encoding(mapping, selected_key, prefix):
    return {f"{prefix}_{k}".lower(): v for k, v in zip(mapping.keys(), mapping[selected_key])}

data.update(expand_one_hot_encoding(ChestPainType_mapping, ChestPainType, "ChestPainType"))
data.update(expand_one_hot_encoding(RestingECG_mapping, RestingECG, "RestingECG"))
data.update(expand_one_hot_encoding(ST_Slope_mapping, ST_Slope, "ST_Slope"))
data['exerciseangina_y'] = [ExerciseAngina_mapping[ExerciseAngina]]

# Convert to DataFrame
df = pd.DataFrame(data)

# Debugging: Feature Alignment
st.write("### Debugging: Feature Alignment")
st.write("Expected Features:", expected_features)
st.write("DataFrame Columns Before Reindexing:", list(df.columns))
missing_features = [feat for feat in expected_features if feat not in df.columns]
extra_features = [feat for feat in df.columns if feat not in expected_features]
st.write("Missing Features:", missing_features)
st.write("Extra Features:", extra_features)

# Debugging: Check Data Before Reindexing
st.write("### Debugging: Data Before Reindexing")
st.write("Column Types:", df.dtypes)
st.write(df)

# Convert all numerical features to float32
df = df.astype(np.float32)

# Debugging: Feature Values Before Reindexing
st.write("Numerical Feature Values Before Reindexing:")
st.write({feature: df[feature].tolist() for feature in df.columns if df[feature].dtype == np.float32})

# Ensure all model-required features exist
df = df.reindex(columns=expected_features, fill_value=0).astype(np.float32)

# Debugging: Feature Values After Reindexing
st.write("### Debugging: Feature Values After Reindexing")
st.write({feature: df[feature].tolist() for feature in df.columns})
st.write("Check NumPy Conversion After Fix:", df.to_numpy())

df_input_array = df.values.astype(np.float32)

# Debugging: Feature Order and Shape
st.write("Expected Features Length:", len(expected_features))
st.write("Actual DataFrame Shape:", df.shape)
st.write("Feature Order Matches?", list(df.columns) == expected_features)















































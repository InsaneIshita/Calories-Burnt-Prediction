import streamlit as st
import joblib
import numpy as np
import xgboost as xgb

# Load model
model = joblib.load("calories_model.pkl")

# Fix missing GPU attribute issue
if not hasattr(model, 'gpu_id'):
    model.gpu_id = None

# Load scaler
scaler = joblib.load("scaler.pkl")

# -----------------------------------------
# PAGE + CUSTOM FITNESS THEME
# -----------------------------------------
st.set_page_config(page_title="Calories Burnt Prediction", layout="centered")

st.markdown("""
<style>
    body {
        background-color: #000000;
    }
    .main {
        background-color: #000000;
        color: white;
    }
    h1 {
        color: #f7c40f !important;
        font-weight: 900 !important;
        text-transform: uppercase;
    }
    label, .stRadio, .stSelectbox {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    .stButton>button {
        width: 100% !important; 
        background-color: #f7c40f;
        color: black !important;
        border: 1.5px solid #f7c40f !important;
        border-radius: 10px !important;
        height: 45px;
        font-size: 16px !important;
        margin-top: 21px;
    }
    .stButton>button:hover {
        background-color: #ffda2b;
        color: black;
        border: none;
    }
    .result-box {
        background: #1c1c1c;
        border-radius: 15px;
        text-align: center;
        color: #f7c40f;
        font-size: 20px;
        font-weight:900;
        margin-top: 20px;
        border: 2px solid #f7c40f;
        box-shadow: 0px 0px 15px #f7c40f;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# TITLE
# -----------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🔥 Calories Burnt Prediction</h1>",
    unsafe_allow_html=True
)

st.write("")

# -----------------------------------------
# INPUT SECTIONS (Two Columns)
# -----------------------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Select Gender", ['Male', 'Female'])
    age = st.number_input("Enter Age (years):", min_value=1)
    height = st.number_input("Enter Height (cm):", min_value=1.0)
    body_temp = st.number_input("Enter Body Temperature (°C):", min_value=1.0)

with col2:
    weight = st.number_input("Enter Weight (kg):", min_value=1.0)
    duration = st.number_input("Enter Exercise Duration (minutes):", min_value=1.0)
    heart_rate = st.number_input("Enter Heart Rate (bpm):", min_value=1.0)
    # gender encoding
    gender_encoded = 1 if gender == 'Male' else 0
    predict_clicked = st.button("Predict")

# -----------------------------------------
# Prediction Logic
# -----------------------------------------
if predict_clicked:
    input_data = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])
    input_scaled = scaler.transform(input_data)

    try:
        prediction = model.predict(input_scaled)[0]
    except Exception:
        if hasattr(model, '_Booster'):
            dmatrix = xgb.DMatrix(input_scaled)
            prediction = model._Booster.predict(dmatrix)[0]
        else:
            st.error("Prediction failed due to incompatible model.")
            st.stop()

    # Output box
    st.markdown(
        f"<div class='result-box'>Estimated Calories Burnt <br>{prediction:.2f} kcal 🔥</div>",
        unsafe_allow_html=True
    )

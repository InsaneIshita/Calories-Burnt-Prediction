import streamlit as st
import joblib
import numpy as np
import xgboost as xgb

# Load model and fix missing attributes
model = joblib.load("calories_model.pkl")

# Fix the missing gpu_id attribute that causes the error
if not hasattr(model, 'gpu_id'):
    model.gpu_id = None

# Load scaler
scaler = joblib.load("scaler.pkl")

st.title("🔥 Calories Burnt Prediction App")

# Gender encoding
gender = st.selectbox("Select Gender", ['Male', 'Female'])
gender_encoded = 1 if gender == 'Male' else 0

# Numeric inputs
age = st.number_input("Enter Age (years):", min_value=1, value="21")
height = st.number_input("Enter Height (cm):", min_value=1.0, value=178.0)
weight = st.number_input("Enter Weight (kg):", min_value=1.0, value=70.0)
duration = st.number_input("Enter Exercise Duration (minutes):", min_value=1.0, value=110)
heart_rate = st.number_input("Enter Heart Rate (bpm):", min_value=1.0, value=130.0)
body_temp = st.number_input("Enter Body Temperature (°C):", min_value=1.0, value=37.5)
#Calling models and handling errors
if st.button("Predict Calories Burnt"):
    input_data = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])
    
    
    input_scaled = scaler.transform(input_data)
    
    try:
        prediction = model.predict(input_scaled)[0]
        st.success(f"Estimated Calories Burnt: {prediction:.2f} kcal")
            
    except Exception as e:
        if hasattr(model, '_Booster'):
            dmatrix = xgb.DMatrix(input_scaled)
            prediction = model._Booster.predict(dmatrix)[0]
            st.success(f"Estimated Calories Burnt: {prediction:.2f} kcal")
        else:
            st.error(f"Prediction failed: {e}")
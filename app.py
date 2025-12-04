import streamlit as st
import joblib
import numpy as np

model = joblib.load("calories_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔥 Calories Burnt Prediction App")

# Gender encoding
gender = st.selectbox("Select Gender", ['Male', 'Female'])
gender_encoded = 1 if gender == 'Male' else 0

# Numeric inputs
age = st.number_input("Enter Age (years):", min_value=1)
height = st.number_input("Enter Height (cm):", min_value=1.0)
weight = st.number_input("Enter Weight (kg):", min_value=1.0)
duration = st.number_input("Enter Exercise Duration (minutes):", min_value=1.0)
heart_rate = st.number_input("Enter Heart Rate (bpm):", min_value=1.0)
body_temp = st.number_input("Enter Body Temperature (°C):", min_value=1.0)

if st.button("Predict Calories Burnt"):
    input_data = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    st.success(f"Estimated Calories Burnt: {prediction:.2f} kcal")

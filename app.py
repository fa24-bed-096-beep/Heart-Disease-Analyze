import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Patient details enter karein aur risk assess karein.")

# Inputs
age = st.number_input("Age (Years)", min_value=1, max_value=120, value=55)
gender_input = st.selectbox("Gender", ["Male", "Female"])
chol = st.number_input("Cholesterol Level (mg/dl)", min_value=50, max_value=600, value=240)
max_hr = st.number_input("Max Heart Rate (thalach)", min_value=50, max_value=250, value=150)

gender = 1 if gender_input == "Male" else 0

if st.button("Predict Result"):
    try:
        model = pickle.load(open('heart_svc_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))

        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Cholesterol': [chol],
            'MaxHR': [max_hr]
        })

        # Scale input before prediction
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)

        st.markdown("---")
        if prediction[0] == 1:
            st.error("⚠️ **HEART DISEASE DETECTED!**")
        else:
            st.success("✅ **NO HEART DISEASE DETECTED.**")

    except FileNotFoundError:
        st.error("Error: 'heart_svc_model.pkl' ya 'scaler.pkl' file nahi mili.")
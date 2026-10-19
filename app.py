import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Patient ki details enter karein aur Heart Disease risk check karein.")

# Current folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
available_files = os.listdir(BASE_DIR)

# Auto-detect Model File (chahye (2) likha ho ya na likha ho)
model_file = None
scaler_file = None

for file in available_files:
    if file.endswith('.pkl'):
        if 'svc' in file.lower() or 'model' in file.lower():
            model_file = file
        elif 'scaler' in file.lower():
            scaler_file = file

# Agar files na milein to exact screen par dikhaye
if not model_file or not scaler_file:
    st.error("⚠️ **Files Missing in GitHub Repository!**")
    st.write("Aapke Repository folder mein ye files hain:", available_files)
    st.info("Aap check karein ke model aur scaler ki .pkl files upload hain ya nahi.")
else:
    # User Inputs
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=55)
    gender_input = st.selectbox("Gender", ["Male", "Female"])
    chol = st.number_input(
        "Cholesterol Level (mg/dl)", min_value=50, max_value=600, value=240
    )
    max_hr = st.number_input(
        "Max Heart Rate (thalach)", min_value=50, max_value=250, value=150
    )

    gender = 1 if gender_input == "Male" else 0

    if st.button("Predict Result"):
        try:
            # Load files dynamically
            model = pickle.load(open(os.path.join(BASE_DIR, model_file), 'rb'))
            scaler = pickle.load(open(os.path.join(BASE_DIR, scaler_file), 'rb'))

            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Cholesterol': [chol],
                'MaxHR': [max_hr],
            })

            # Scaling before prediction
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)

            st.markdown("---")
            if prediction[0] == 1:
                st.error("⚠️ **HEART DISEASE DETECTED!** (High Risk)")
            else:
                st.success("✅ **NO HEART DISEASE DETECTED.** (Low Risk)")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

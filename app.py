import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')

st.title("Heart Disease Prediction")

# Collect user input
age = st.slider('Age', 20, 80, 30)
sex = st.selectbox('Sex', [0, 1])  # 0: female, 1: male
cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', 80, 200, 120)
chol = st.number_input('Serum Cholestoral in mg/dl (chol)', 100, 600, 200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
restecg = st.selectbox('Resting Electrocardiographic Results (restecg)', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', 60, 220, 150)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
oldpeak = st.slider('ST depression induced by exercise (oldpeak)', 0.0, 6.0, 1.0)
slope = st.selectbox('Slope of the peak exercise ST segment (slope)', [0, 1, 2])
ca = st.selectbox('Number of major vessels (ca)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia (thal)', [0, 1, 2])

# Make prediction
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)
    st.success("Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected")

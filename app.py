
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

st.title("🩺 Patient No-Show Predictor")

st.write("Enter patient information to predict the likelihood of a no-show.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 115, 25)
neighbourhood = st.text_input("Neighbourhood (label encoded)", "50")
scholarship = st.selectbox("Scholarship", ["No", "Yes"])
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
diabetes = st.selectbox("Diabetes", ["No", "Yes"])
alcoholism = st.selectbox("Alcoholism", ["No", "Yes"])
handcap = st.selectbox("Handicap Level", [0, 1, 2, 3, 4])
sms_received = st.selectbox("SMS Received", ["No", "Yes"])

# Convert inputs to numeric
gender = 1 if gender == "Female" else 0
scholarship = 1 if scholarship == "Yes" else 0
hypertension = 1 if hypertension == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
alcoholism = 1 if alcoholism == "Yes" else 0
sms_received = 1 if sms_received == "Yes" else 0

# Placeholder for dates
scheduled_day = 0  # Not used in model here
appointment_day = 0  # Not used in model here

# Final feature vector
features = np.array([[gender, scheduled_day, appointment_day, age, int(neighbourhood),
                      scholarship, hypertension, diabetes, alcoholism, handcap, sms_received]])

# Scale features
scaled_features = scaler.transform(features)

# Predict
if st.button("Predict"):
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]
    if prediction == 1:
        st.error(f"❌ Patient likely to miss the appointment. (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Patient likely to show up. (Probability: {probability:.2f})")

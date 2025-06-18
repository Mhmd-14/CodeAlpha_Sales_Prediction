import streamlit as st
import joblib
import numpy as np
import os

# Load model
model_path = 'models/model.pkl'
if not os.path.exists(model_path):
    st.error("Model file not found at /models/model.pkl")
else:
    model = joblib.load(model_path)

SCALER_PATH = 'models/scaler.pkl'
if not os.path.exists(SCALER_PATH):
    st.warning("⚠️ Scaler not found. Assuming model does NOT require scaled inputs.")
    scaler = None
else:
    scaler = joblib.load(SCALER_PATH)

st.title("Sales Prediction App")

st.subheader("Enter Input Features (in K $)")

feature1 = st.number_input("TV", value=0.0, help="Enter the value (K $)")

feature2 = st.number_input("Radio", value=0.0, help="Enter the value (K $)")

feature3 = st.number_input("Newspaper", value=0.0, help="Enter the value (K $)")


if st.button("Predict Units Sold"):
    input_data = np.array([[feature1, feature2, feature3]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Units Sold: {prediction[0]:,.2f} thousand")

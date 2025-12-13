import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("/mnt/data/LR_model.pkl", "rb") as f:
    model = pickle.load(f)

# App title
st.title("House Price Prediction App")

# Input fields
st.header("Enter the house details:")
sqft_living = st.number_input("Square Footage of Living Area (sqft)", min_value=100, max_value=20000, value=1500)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=20, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, value=2.0)
yr_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)

# Predict button
if st.button("Predict Price"):
    # Prepare input array
    input_data = np.array([[sqft_living, bedrooms, bathrooms, yr_built]])
    # Make prediction
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")

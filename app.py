import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('LR_model.pkl', 'rb') as file:
    model = pickle.load(file)

# App title
st.title("🏠 House Price Prediction App")
st.write("""
This app predicts the **House Price** based on input features like square footage, bedrooms, bathrooms, and year built.
""")

# User inputs
sqft_living = st.number_input("Square Footage of Living Area", min_value=100, max_value=20000, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
yr_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)

# Predict button
if st.button("Predict Price"):
    input_features = np.array([[sqft_living, bedrooms, bathrooms, yr_built]])
    predicted_price = model.predict(input_features)[0]
    st.success(f"Estimated House Price: ₹{predicted_price:,.2f}")

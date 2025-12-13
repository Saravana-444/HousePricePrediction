import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# -------- Load pickle safely --------
with open("saldf.pkl", "rb") as f:
    obj = pickle.load(f)

model = None
scaler = None

# Case 1: sklearn Pipeline
if hasattr(obj, "predict"):
    model = obj

# Case 2: tuple (model, scaler)
elif isinstance(obj, tuple):
    model = obj[0]
    if len(obj) > 1:
        scaler = obj[1]

# Case 3: dictionary
elif isinstance(obj, dict):
    if "model" in obj:
        model = obj["model"]
    if "scaler" in obj:
        scaler = obj["scaler"]

# If still not found
if model is None:
    st.error("❌ Could not extract model from saldf.pkl")
    st.stop()

# -------- UI --------
st.title("🏠 House Price Prediction")
st.write("Predict house prices using Linear Regression")

st.divider()

st.subheader("📊 Enter House Details")

bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, step=0.5)
sqft_living = st.number_input("Living Area (sqft)", min_value=0)

# -------- Prediction --------
if st.button("Predict Price"):
    X = np.array([[bedrooms, bathrooms, sqft_living]])

    if scaler is not None:
        X = scaler.transform(X)

    prediction = model.predict(X)

    st.divider()
    st.subheader("💰 Predicted House Price")
    st.success(f"₹ {prediction[0]:,.2f}")

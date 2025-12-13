import streamlit as st
import pickle
import numpy as np

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# ---------------- Load Model ----------------
with open("saldf.pkl", "rb") as file:
    loaded = pickle.load(file)

# Handle different saved formats
scaler = None

if isinstance(loaded, tuple):
    model = loaded[0]
    if len(loaded) > 1:
        scaler = loaded[1]
elif hasattr(loaded, "predict"):
    model = loaded
else:
    st.error("❌ Unsupported model format in saldf.pkl")
    st.stop()

# ---------------- UI ----------------
st.title("🏠 House Price Prediction App")
st.write("Predict house prices using Linear Regression")

st.divider()

st.subheader("📊 Enter House Details")

# ⚠️ Change feature names ONLY if your dataset is different
bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, step=0.5)
sqft_living = st.number_input("Living Area (sqft)", min_value=0)



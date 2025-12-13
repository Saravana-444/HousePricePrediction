import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="House Prediction App",
    page_icon="📈",
    layout="centered"
)

# Load model
with open("saldf.pkl", "rb") as file:
    model = pickle.load(file)

st.title("📈 Linear Regression Prediction App")
st.write("Enter input values to predict the output")

st.divider()

st.subheader("🔢 Input Features")

# ⚠️ Change feature count if needed
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(input_data)

    st.success(f"Predicted Value: {prediction[0]:.2f}")

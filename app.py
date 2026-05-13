import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("ANN_CW.pkl", "rb"))

# Page title
st.set_page_config(page_title="ANN Prediction App")

st.title("ANN Prediction System")
st.write("Enter input values for prediction")

# Input fields
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")

# Predict button
if st.button("Predict"):

    # Convert inputs into array
    features = np.array([[feature1, feature2, feature3, feature4]])

    # Prediction
    prediction = model.predict(features)

    st.success(f"Prediction Result: {prediction[0]}")
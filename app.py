import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load Model and Scalers
with open("svm_regression_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler_X.pkl", "rb") as file:
    scaler_X = pickle.load(file)

with open("scaler_y.pkl", "rb") as file:
    scaler_y = pickle.load(file)

st.set_page_config(
    page_title="SVM Regression House Price Prediction",
    layout="centered"
)

st.title("🏠 House Price Prediction using SVM Regression")

st.write("Enter house details to predict the estimated house price.")

area = st.number_input("Area (sq.ft)", min_value=500, max_value=10000, value=1500)

bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)

bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

floors = st.number_input("Floors", min_value=1, max_value=5, value=2)

parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)

house_age = st.number_input("House Age", min_value=0, max_value=100, value=10)

distance_city = st.number_input(
    "Distance From City (km)",
    min_value=1.0,
    max_value=100.0,
    value=10.0
)

if st.button("Predict House Price"):

    input_data = np.array([[
        area,
        bedrooms,
        bathrooms,
        floors,
        parking,
        house_age,
        distance_city
    ]])

    input_scaled = scaler_X.transform(input_data)

    prediction_scaled = model.predict(input_scaled)

    prediction = scaler_y.inverse_transform(
        prediction_scaled.reshape(-1,1)
    )

    predicted_price = prediction[0][0]

    st.success(f"Estimated House Price: ₹ {predicted_price:,.2f}")

st.markdown("---")

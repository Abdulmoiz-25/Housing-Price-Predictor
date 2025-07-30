import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained pipeline + model
model = joblib.load("final_model.pkl")

st.title("🏡 California House Price Prediction")
st.markdown("Enter the property details below to estimate the **median house price**.")

# User inputs
MedInc = st.number_input("Median Income (in $10,000s)", min_value=0.0, max_value=20.0, value=3.0)
HouseAge = st.slider("House Age", 1, 52, 20)
AveRooms = st.number_input("Average Rooms", min_value=1.0, max_value=50.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms", min_value=0.5, max_value=10.0, value=1.0)
Population = st.number_input("Population", min_value=1, max_value=50000, value=1000)
AveOccup = st.number_input("Average Occupancy", min_value=1.0, max_value=10.0, value=3.0)
Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
Longitude = st.slider("Longitude", -124.0, -114.0, -120.0)

# Feature engineering
rooms_per_household = AveRooms / (Population / AveOccup)
bedrooms_per_room = AveBedrms / AveRooms
population_per_household = Population / (Population / AveOccup)

# Create input features DataFrame
features = pd.DataFrame([[
    MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude,
    rooms_per_household, bedrooms_per_room, population_per_household
]], columns=[
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
    'Latitude', 'Longitude', 'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
])

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict(features)
    st.success(f"🏠 Estimated Median House Value: **${prediction[0] * 100000:,.2f}**")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# Load pre-trained model and scaler
model = joblib.load('motor_threshold_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title and explanation for the app
st.title('Motor Threshold Prediction App')
st.markdown("""
    This app predicts the threshold based on the motor size entered. The model has been trained on motor sizes and their corresponding threshold values. 
    Enter any motor size, and the app will predict the threshold.
""")

# Input from the user
motor_size = st.number_input("Enter Motor Size (kW):", min_value=0.0, step=0.1)

# Check if the input motor size is valid
if motor_size > 0:
    # Example motor features, assuming motor_size is one of the features
    # Add additional features if necessary, like voltage, RPM, etc.
    motor_features = np.array([motor_size]).reshape(1, -1)  # Reshape to 2D for the scaler

    # Scale the motor features
    motor_features_scaled = scaler.transform(motor_features)

    # Predict the threshold using the model
    predicted_threshold = model.predict(motor_features_scaled)

    # Display the predicted threshold
    st.write(f"The predicted threshold for {motor_size} kW motor size is: {predicted_threshold[0]:.2f} units")
else:
    st.write("Please enter a valid motor size greater than 0.")

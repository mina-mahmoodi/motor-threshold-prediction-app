import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scaler
model = joblib.load('motor_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load your historical motor data (replace with your dataset)
data = {
    'Motor Size (kW)': [5, 10, 15, 20, 25, 30],
    'Threshold (A)': [10, 15, 20, 25, 30, 35]
}
df = pd.DataFrame(data)

# Streamlit App Interface
st.title('Motor Threshold Prediction')

# Motor size input from the user
motor_size = st.number_input('Enter motor size (kW)', min_value=1, max_value=100, step=1)

if motor_size:
    st.write(f'You entered motor size: {motor_size} kW')
    
    # Create input feature array (if model uses more than 1 feature, ensure you have all of them)
    # Assuming the model was trained on 'Motor Size (kW)' as the only feature
    motor_features = np.array([[motor_size]])  # Ensure this is a 2D array
    
    # Scale the input features using the loaded scaler
    try:
        motor_features_scaled = scaler.transform(motor_features)
    except ValueError as e:
        st.error(f"Error scaling features: {e}")
        st.stop()  # Stop execution if scaling fails
    
    # Predict the threshold using the trained model
    prediction = model.predict(motor_features_scaled)
    
    # Output the prediction
    st.write(f"Predicted threshold current for {motor_size} kW motor: {prediction[0]} A")
    
    # Show data for visualization
    st.write(df)

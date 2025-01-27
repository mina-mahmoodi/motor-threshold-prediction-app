import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Title
st.title("Motor Threshold Prediction App")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your CSV file (e.g., Statistical summary.csv)", type="csv")

if uploaded_file is not None:
    # Step 2: Load and Display the Data
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(df.head())

    # Handle missing values (NaNs) - Impute with mean for numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    # Step 3: Aggregate Data by Motor Output Power (kW)
    grouped_data = df.groupby("Motor Output Power (kW)").agg({
        "X Mean": "mean", "X Std Dev": "mean", "X Min": "mean", "X Max": "mean",
        "Y Mean": "mean", "Y Std Dev": "mean", "Y Min": "mean", "Y Max": "mean",
        "Z Mean": "mean", "Z Std Dev": "mean", "Z Min": "mean", "Z Max": "mean",
        "RPM Mean": "mean", "RPM Std Dev": "mean",
        "Torque Mean": "mean", "Torque Std Dev": "mean",
        "Energy Efficiency Mean": "mean", "Energy Efficiency Std Dev": "mean",
        "X Warning Threshold": "mean", "X Error Threshold": "mean",
        "Y Warning Threshold": "mean", "Y Error Threshold": "mean",
        "Z Warning Threshold": "mean", "Z Error Threshold": "mean"
    }).reset_index()

    # Step 4: Separate Features and Labels
    X = grouped_data[[ 
        "Motor Output Power (kW)", "X Mean", "X Std Dev", "X Min", "X Max",
        "Y Mean", "Y Std Dev", "Y Min", "Y Max",
        "Z Mean", "Z Std Dev", "Z Min", "Z Max",
        "RPM Mean", "RPM Std Dev", "Torque Mean", "Torque Std Dev",
        "Energy Efficiency Mean", "Energy Efficiency Std Dev"
    ]].values

    y = grouped_data[[ 
        "X Warning Threshold", "X Error Threshold",
        "Y Warning Threshold", "Y Error Threshold",
        "Z Warning Threshold", "Z Error Threshold"
    ]].values

    # Step 5: Normalize Features (Input data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Rescale target (y) values to [0, 1] for better performance
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(y)

    # Step 6: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Step 7: Build and Train Neural Network
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(6)  # Predict 6 thresholds
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Use early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with early stopping
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=4, verbose=1, callbacks=[early_stopping])

    # Step 8: User Input for Motor Size
    motor_size_input = st.number_input("Enter Motor Output Power (kW)", min_value=0.0, step=1.0)

    # Step 9: Predict Thresholds for the Input Motor Size
    if motor_size_input > 0:
        # Get the corresponding feature values for the input motor size
        motor_features = np.zeros(X.shape[1])  # initialize all features as zeros
        motor_features[0] = motor_size_input  # Set the motor size (the first feature)

        # Normalize input motor features using the same scaler
        motor_features_scaled = scaler.transform([motor_features])

        # Predict thresholds for the given motor size
        predicted_thresholds = model.predict(motor_features_scaled)

        # Rescale the predicted values back to the original range
        predicted_thresholds_rescaled = target_scaler.inverse_transform(predicted_thresholds)

        # Display the prediction
        st.write(f"Predicted Thresholds for Motor Size {motor_size_input} kW:")
        st.write(f"  X Warning: {predicted_thresholds_rescaled[0][0]:.3f}, X Error: {predicted_thresholds_rescaled[0][1]:.3f}")
        st.write(f"  Y Warning: {predicted_thresholds_rescaled[0][2]:.3f}, Y Error: {predicted_thresholds_rescaled[0][3]:.3f}")
        st.write(f"  Z Warning: {predicted_thresholds_rescaled[0][4]:.3f}, Z Error: {predicted_thresholds_rescaled[0][5]:.3f}")

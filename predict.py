import pandas as pd
import numpy as np
import joblib
import os
from preprocess import load_preprocessing_objects, preprocess_input

def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        scaler, label_encoders, model = load_preprocessing_objects()
        if model is None:
            raise ValueError("Model not found. Please train the model first.")
        return model, scaler, label_encoders
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None

def predict_price(input_data, model, scaler, label_encoders):
    """Make price prediction"""
    df_processed = preprocess_input(input_data, scaler, label_encoders)
    prediction = model.predict(df_processed)[0]
    return max(0, round(prediction, 2))  # Ensure price is not negative

def predict_multiple(input_data_list, model, scaler, label_encoders):
    """Make predictions for multiple inputs"""
    predictions = []
    for input_data in input_data_list:
        pred = predict_price(input_data, model, scaler, label_encoders)
        predictions.append(pred)
    return predictions

# Test function
if __name__ == "__main__":
    # Test prediction
    model, scaler, label_encoders = load_model()
    if model is not None:
        test_input = {
            'model': 'Fiesta',
            'year': 2018,
            'transmission': 'Manual',
            'mileage': 20000,
            'fuelType': 'Petrol',
            'tax': 150,
            'mpg': 50.0,
            'engineSize': 1.2
        }
        price = predict_price(test_input, model, scaler, label_encoders)
        print(f"Test Prediction: £{price:,.2f}")
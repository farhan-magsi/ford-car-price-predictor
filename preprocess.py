import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def load_and_preprocess_data(file_path="cars_ford.csv"):
    """Load and preprocess the dataset"""
    df = pd.read_csv(file_path)
    
    # Create a copy for preprocessing
    df_processed = df.copy()
    
    # Handle categorical variables with Label Encoding
    categorical_cols = ['model', 'transmission', 'fuelType']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
    
    # Select features
    X = df_processed.drop('price', axis=1)
    y = df_processed['price']
    
    # Scale numerical features
    numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, scaler, label_encoders, df

def save_preprocessing_objects(scaler, label_encoders, model=None):
    """Save preprocessing objects and model"""
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    if model:
        joblib.dump(model, 'model.pkl')

def load_preprocessing_objects():
    """Load preprocessing objects"""
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    model = joblib.load('model.pkl') if os.path.exists('model.pkl') else None
    return scaler, label_encoders, model

def preprocess_input(input_data, scaler, label_encoders):
    """Preprocess user input for prediction"""
    df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col].astype(str))
            except ValueError:
                # If unknown category, use the most frequent class
                df[col] = le.transform([le.classes_[0]])[0]
    
    # Scale numerical features
    numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
from preprocess import load_and_preprocess_data, save_preprocessing_objects

def train_and_save_model():
    """Train and save the model"""
    print("🚗 Training Ford Car Price Predictor Model...")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists('cars_ford.csv'):
        print("❌ Dataset not found. Please ensure 'cars_ford.csv' is in the current directory.")
        return False
    
    # Load and preprocess data
    X, y, scaler, label_encoders, df = load_and_preprocess_data('cars_ford.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("📊 Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n📈 Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: £{mae:,.2f}")
    print(f"   RMSE: £{rmse:,.2f}")
    
    # Save model and preprocessors
    save_preprocessing_objects(scaler, label_encoders, model)
    
    print("\n✅ Model saved successfully!")
    print(f"   Model saved to: models/model.pkl")
    print(f"   Scaler saved to: models/scaler.pkl")
    print(f"   Label encoders saved to: models/label_encoders.pkl")
    
    return model, scaler, label_encoders, r2, mae, rmse

if __name__ == "__main__":
    train_and_save_model()
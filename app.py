import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from preprocess import load_preprocessing_objects
from predict import load_model, predict_price

# Page configuration
st.set_page_config(
    page_title="Ford Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        border-radius: 15px;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .prediction-price {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffd54f;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a237e;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #757575;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #757575;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_models():
    try:
        model, scaler, label_encoders = load_model()
        if model is not None:
            return model, scaler, label_encoders, True
        else:
            return None, None, None, False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, False

model, scaler, label_encoders, model_loaded = load_models()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/744/744604.png", width=100)
    st.title("🚗 Car Predictor")
    st.markdown("---")
    
    st.markdown("### 📊 Model Info")
    if model_loaded:
        st.success("✅ Model loaded successfully")
    else:
        st.warning("⚠️ Model not loaded")
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Tips")
    st.markdown("""
    - Fill all car details
    - Click Predict Price
    - Get instant estimate
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Dataset")
    try:
        df = pd.read_csv('cars_ford.csv')
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Avg Price", f"£{df['price'].mean():,.2f}")
    except:
        pass

# Main content
st.markdown('<div class="main-header">🚗 Ford Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict the market price of used Ford cars</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Explore Data", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 🚘 Car Details")
        
        # Car model selection
        model_list = ['Fiesta', 'Focus', 'Puma', 'Kuga', 'EcoSport', 'C-MAX', 'Mondeo', 
                      'Ka+', 'Tourneo Custom', 'S-MAX', 'B-MAX', 'Edge', 'Tourneo Connect',
                      'Grand C-MAX', 'KA', 'Galaxy', 'Mustang', 'Grand Tourneo Connect',
                      'Fusion', 'Ranger', 'Streetka', 'Escort', 'Transit Tourneo']
        
        car_model = st.selectbox("Model", model_list)
        
        # Year
        year = st.number_input("Year", min_value=1996, max_value=2024, value=2018, step=1)
        
        # Transmission
        transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])
        
        # Fuel Type
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric", "Other"])
        
        # Numerical inputs
        col1a, col1b = st.columns(2)
        with col1a:
            mileage = st.number_input("Mileage (miles)", min_value=0, max_value=200000, value=20000, step=1000)
            tax = st.number_input("Road Tax (£)", min_value=0, max_value=600, value=150, step=5)
        with col1b:
            mpg = st.number_input("MPG", min_value=10.0, max_value=200.0, value=50.0, step=0.5)
            engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=6.0, value=1.2, step=0.1)
        
        predict_button = st.button("🔮 Predict Price", use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Prediction Result")
        
        if predict_button:
            if not model_loaded:
                st.error("⚠️ Model not loaded. Please train the model first.")
                st.info("Run `python train_model.py` to train the model.")
            else:
                # Prepare input data
                input_data = {
                    'model': car_model,
                    'year': year,
                    'transmission': transmission,
                    'mileage': mileage,
                    'fuelType': fuel_type,
                    'tax': tax,
                    'mpg': mpg,
                    'engineSize': engine_size
                }
                
                try:
                    predicted_price = predict_price(input_data, model, scaler, label_encoders)
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Estimated Price</div>
                        <div class="prediction-price">£{predicted_price:,.2f}</div>
                        <div style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
                            Based on {len(model_list)} car models
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Price range
                    price_range_low = predicted_price * 0.85
                    price_range_high = predicted_price * 1.15
                    
                    st.markdown(f"""
                    <div style="margin-top: 1rem; padding: 1rem; background: #e8f5e9; border-radius: 10px;">
                        <div style="font-size: 0.9rem; color: #2e7d32;">
                            💡 Estimated Range: £{price_range_low:,.2f} - £{price_range_high:,.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        else:
            st.info("👆 Fill in car details and click 'Predict Price'")
            
            st.markdown("""
            <div style="background: #f5f5f5; border-radius: 10px; padding: 2rem; text-align: center; border: 2px dashed #bdbdbd;">
                <div style="color: #757575; font-size: 1.1rem;">
                    <span style="font-size: 3rem;">🔮</span><br>
                    Enter car details to get price estimate
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Data Explorer")
    
    try:
        df = pd.read_csv('cars_ford.csv')
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Avg Price", f"£{df['price'].mean():,.2f}")
        with col3:
            st.metric("Min Price", f"£{df['price'].min():,.2f}")
        with col4:
            st.metric("Max Price", f"£{df['price'].max():,.2f}")
        
        # Price distribution
        st.markdown("#### Price Distribution")
        fig = px.histogram(df, x='price', nbins=50, title='Distribution of Car Prices',
                          labels={'price': 'Price (£)'}, color_discrete_sequence=['#1a237e'])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("#### Feature Correlations")
        numeric_cols = ['year', 'price', 'mileage', 'tax', 'mpg', 'engineSize']
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Correlation Matrix", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plots
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df, x='mileage', y='price', title='Price vs Mileage',
                            labels={'mileage': 'Mileage (miles)', 'price': 'Price (£)'},
                            opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='year', y='price', title='Price vs Year',
                            labels={'year': 'Year', 'price': 'Price (£)'},
                            opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("⚠️ Dataset not found. Please ensure 'cars_ford.csv' is in the correct location.")

with tab3:
    st.markdown("### ℹ️ About This Project")
    
    st.markdown("""
    #### 🎯 Overview
    This web application uses a **Linear Regression** model to predict the price of used Ford cars.
    
    #### 📊 Features Used
    - **Car Specifications**: Model, Year, Transmission, Fuel Type
    - **Usage**: Mileage, Road Tax
    - **Performance**: MPG, Engine Size
    
    #### 🛠️ Technologies
    - Streamlit - Web Framework
    - Scikit-learn - Machine Learning
    - Pandas - Data Processing
    - Plotly - Visualization
    
    #### 📈 Model Performance
    - **R² Score**: ~0.73
    - **MAE**: ~£1,850
    - **RMSE**: ~£2,450
    """)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    🚗 Ford Car Price Predictor | Made with ❤️<br>
    Powered by Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)

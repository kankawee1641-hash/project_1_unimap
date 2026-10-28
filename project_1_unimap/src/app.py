"""
IFJC4002 | MC Practical Artificial Intelligence with Python
Faculty of Intelligent Computing, Universiti Malaysia Perlis (UniMAP)
Case 2: Used Car Price Prediction Dashboard (Streamlit App)
"""

import os
import datetime
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Used Car Price Prediction | UniMAP IFJC4002",
    page_icon="🚗",
    layout="wide"
)

# Constants & Paths
MODEL_PATH = "models/car_price_pipeline.joblib"
LOG_PATH = "outputs/prediction_logs.csv"

# Load Trained Pipeline
@st.cache_resource
def load_pipeline():
    # Check alternate paths
    paths_to_check = [
        MODEL_PATH,
        "car_price_pipeline.joblib",
        "../models/car_price_pipeline.joblib"
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            return joblib.load(p)
    return None

pipeline = load_pipeline()

# Title and Context Header
st.title("🚗 Used Car Resale Price Prediction System")
st.caption("IFJC4002 Practical Artificial Intelligence with Python | Group 1 (UniMAP)")

st.markdown("""
Welcome to the **Used Car Price Prediction Dashboard**. 
This machine learning system estimates the resale value (in **Malaysian Ringgit - RM**) 
of pre-owned vehicles based on vehicle specifications, condition, and market segment.
""")

st.write("---")

# Layout: Two columns (Inputs on the left, Predictions & Diagnostics on the right)
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📋 Enter Vehicle Details")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            vehicle_age = st.number_input(
                "Vehicle Age (Years)",
                min_value=0.0,
                max_value=30.0,
                value=5.0,
                step=0.5,
                help="Age of vehicle since initial registration"
            )
            
            mileage = st.number_input(
                "Mileage (km)",
                min_value=0.0,
                max_value=800000.0,
                value=65000.0,
                step=1000.0,
                help="Total distance traveled in kilometers"
            )
            
            engine_size = st.slider(
                "Engine Size (Liters)",
                min_value=0.8,
                max_value=5.0,
                value=1.8,
                step=0.1,
                help="Engine displacement in liters"
            )
            
            market_segment = st.selectbox(
                "Market Segment",
                options=["Economy", "Mid_Range", "Premium"],
                index=1,
                help="Vehicle classification tier"
            )

        with col2:
            fuel_type = st.selectbox(
                "Fuel Type",
                options=["Petrol", "Diesel", "Hybrid"],
                index=0,
                help="Engine fuel category"
            )
            
            transmission = st.selectbox(
                "Transmission Type",
                options=["Automatic", "Manual"],
                index=0,
                help="Gear transmission mechanism"
            )
            
            condition_rating = st.select_slider(
                "Condition Rating (1-5)",
                options=[1, 2, 3, 4, 5],
                value=4,
                help="Overall vehicle condition (1 = Poor, 5 = Excellent)"
            )
            
            previous_owners = st.number_input(
                "Previous Owners",
                min_value=0,
                max_value=10,
                value=1,
                step=1,
                help="Number of past registered owners"
            )

        submit_btn = st.form_submit_button("Predict Resale Price 🚀", use_container_width=True)

with col_right:
    st.subheader("💡 Prediction & Model Evaluation")
    
    if submit_btn:
        # 1. Input Validation and Boundary Checking
        warnings = []
        if mileage > 450000:
            warnings.append("⚠️ **High Mileage Warning:** Mileage exceeds 450,000 km; predictions may carry wider confidence intervals.")
        if vehicle_age > 18:
            warnings.append("⚠️ **Vehicle Age Warning:** Age is over 18 years, near the dataset's maximum boundary.")
        if vehicle_age == 0 and mileage > 30000:
            warnings.append("⚠️ **Boundary Anomaly:** A new vehicle (Age 0) typically has under 15,000 km.")

        for w in warnings:
            st.warning(w)

        # 2. Prepare Feature Vector matching training column names exactly
        input_data = pd.DataFrame([{
            'vehicle_age_years': float(vehicle_age),
            'mileage_km': float(mileage),
            'engine_size_l': float(engine_size),
            'fuel_type': fuel_type,
            'transmission': transmission,
            'market_segment': market_segment,
            'condition_rating': int(condition_rating),
            'previous_owners': int(previous_owners)
        }])

        if pipeline is not None:
            # Predict
            pred_price = pipeline.predict(input_data)[0]
            # Ensure price is non-negative
            final_price = max(1000.0, float(pred_price))
            
            # Display Prediction Card
            st.success(f"### Estimated Resale Price:\n# **RM {final_price:,.2f}**")
            
            # Contextual Benchmarks
            avg_segment_prices = {
                "Economy": 32938.0,
                "Mid_Range": 56174.0,
                "Premium": 114567.0
            }
            avg_price = avg_segment_prices.get(market_segment, 66797.0)
            diff = final_price - avg_price
            diff_pct = (diff / avg_price) * 100
            
            st.metric(
                label=f"Comparison to {market_segment} Segment Average",
                value=f"RM {final_price:,.2f}",
                delta=f"{diff_pct:+.1f}% vs Segment Mean (RM {avg_price:,.0f})"
            )
            
            # Log Prediction for Monitoring (Task 3 Requirement)
            log_entry = input_data.copy()
            log_entry['predicted_price_rm'] = round(final_price, 2)
            log_entry['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry['anomaly_flag'] = len(warnings) > 0
            
            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
            if not os.path.exists(LOG_PATH):
                log_entry.to_csv(LOG_PATH, index=False)
            else:
                log_entry.to_csv(LOG_PATH, mode='a', header=False, index=False)
                
            st.info("✅ Prediction successfully recorded to local monitoring log.")
            
        else:
            st.error("❌ Model pipeline file not found. Please run `python train.py` first to export `models/car_price_pipeline.joblib`.")
    else:
        st.info("👈 Enter the vehicle specifications on the left and click **Predict Resale Price**.")

st.write("---")

# Monitoring and Audit History Section
st.subheader("📊 Operational Monitoring & Audit Logs")
col_mon1, col_mon2 = st.columns([2, 1])

with col_mon1:
    if os.path.exists(LOG_PATH):
        try:
            logs_df = pd.read_csv(LOG_PATH)
            st.markdown(f"**Recent Logged Inquiries (Total: {len(logs_df)})**")
            st.dataframe(logs_df.tail(5), use_container_width=True)
        except Exception as e:
            st.caption("No log records available yet.")
    else:
        st.caption("No prediction logs recorded yet. Submit a prediction to see live monitoring.")

with col_mon2:
    st.markdown("#### 🔍 Model Review Triggers")
    st.markdown("""
    According to the project rubric, the system monitors for signs that trigger model retraining:
    - **Data Drift:** Sustained inputs outside baseline bounds (e.g. Mileage > 450,000 km or Age > 18 yrs).
    - **Inflation / Policy Shift:** Road tax or market price changes in Malaysia.
    - **Extreme Anomalies:** Persistent flagged inputs exceeding 15% of recent traffic.
    """)

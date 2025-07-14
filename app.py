import streamlit as st
import pandas as pd
import numpy as np
import math

# Configure the page
st.set_page_config(
    page_title="Biogas Production Predictor",
    page_icon="🔬",
    layout="wide"
)

# ---- 1. Load Data ----
@st.cache_data
def load_data():
    # Updated data based on 30g substrate producing 12L in 30 days
    data = pd.DataFrame({
        'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'mL_Produced': [400, 600, 800, 1050, 1300, 1550, 1800, 2050, 2300, 2550, 
                        3600, 3700, 3800, 3750, 3400, 3050, 2700, 2350, 2000, 
                        1650, 1300, 950, 700, 400, 100, 0, 0, 0, 0, 0],
        'Percent': [3.3, 5.0, 6.7, 8.8, 10.8, 12.9, 15.0, 17.1, 19.2, 21.3,
                    30.0, 30.8, 31.7, 31.3, 28.3, 25.4, 22.5, 19.6, 16.7,
                    13.8, 10.8, 7.9, 5.8, 3.3, 0.8, 0, 0, 0, 0, 0]
    })
    return data

# [Previous mathematical functions remain the same...]

# ---- 3. Prediction Functions ----
@st.cache_data
def fit_prediction_models(data):
    """Fit prediction models to the data"""
    days = data['Day'].tolist()
    daily_prod = data['mL_Produced'].tolist()
    cumulative = []
    running_sum = 0
    for val in daily_prod:
        running_sum += val
        cumulative.append(running_sum)
    
    # Fit polynomial models
    try:
        daily_coeffs = polynomial_regression(days, daily_prod, degree=2)
        cumulative_coeffs = polynomial_regression(days, cumulative, degree=2)
    except:
        # Fallback to simple averages
        daily_coeffs = [sum(daily_prod) / len(daily_prod), 0]
        cumulative_coeffs = [0, sum(daily_prod) / len(daily_prod)]
    
    return {
        'daily_coeffs': daily_coeffs,
        'cumulative_coeffs': cumulative_coeffs,
        'actual_daily': daily_prod,
        'actual_cumulative': cumulative,
        'days': days,
        'base_substrate': 30  # 30g substrate in our base data
    }

def predict_daily_production(day, models, data, substrate_amount):
    """Predict daily production scaled by substrate amount"""
    # Get base prediction
    poly_pred = evaluate_polynomial(day, models['daily_coeffs'])
    
    # Linear interpolation for days within data range
    if 1 <= day <= 25:
        interp_pred = linear_interpolation(day, models['days'], models['actual_daily'])
    else:
        interp_pred = poly_pred
    
    # Pattern recognition
    if day <= 10:
        pattern_pred = 200 * day  # Updated growth pattern
    elif day <= 15:
        pattern_pred = 3800 - 100 * (day - 10)  # Updated peak pattern
    else:
        pattern_pred = max(100, 3400 - 250 * (day - 15))  # Updated decline pattern
    
    # Average the predictions
    final_pred = (poly_pred + interp_pred + pattern_pred) / 3
    
    # Scale by substrate amount (30g base -> 12000mL total)
    substrate_factor = substrate_amount / models['base_substrate']
    final_pred *= substrate_factor
    
    return max(0, final_pred)

def predict_cumulative_production(day, models, data, substrate_amount):
    """Predict cumulative production scaled by substrate amount"""
    # Sum of daily predictions
    daily_sum = 0
    for d in range(1, int(day) + 1):
        daily_sum += predict_daily_production(d, models, data, substrate_amount)
    
    return daily_sum

# ---- 4. Main App ----
def main():
    # Header
    st.title('🔬 Biogas Production Predictor')
    st.markdown("*Predict biogas production based on substrate amount*")
    st.markdown("---")
    
    # Load data and fit models
    data = load_data()
    models = fit_prediction_models(data)
    
    # Sidebar - User Inputs
    st.sidebar.header("🌱 Substrate Input")
    substrate_amount = st.sidebar.number_input(
        "Enter substrate amount (grams):",
        min_value=1.0,
        max_value=10000.0,
        value=30.0,
        step=1.0,
        help="Base data used 30g substrate producing 12L in 30 days"
    )
    
    # Display raw data
    with st.expander("📋 View Experimental Data (30g substrate)"):
        st.dataframe(data, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Days", len(data))
        with col2:
            st.metric("Total Production", f"{sum(data['mL_Produced'])/1000:.2f} L")
        with col3:
            st.metric("Average Daily", f"{sum(data['mL_Produced'])/len(data):.0f} mL")
    
    # Main prediction tabs
    tab1, tab2 = st.tabs(["📈 Daily Prediction", "📊 Cumulative Prediction"])
    
    with tab1:
        st.subheader("Daily Biogas Production Prediction")
        
        day = st.slider("Select day for prediction", 1, 30, 11)
        
        # Make prediction
        daily_pred = predict_daily_production(day, models, data, substrate_amount)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Daily Production", 
                     f"{daily_pred:.0f} mL",
                     f"{daily_pred/1000:.2f} L")
        
        with col2:
            if day <= 25:
                actual_val = data[data['Day'] == day]['mL_Produced'].iloc[0]
                scaled_actual = actual_val * (substrate_amount / 30)
                st.metric("Scaled Experimental Value", 
                         f"{scaled_actual:.0f} mL",
                         f"{scaled_actual/1000:.2f} L")
        
        # Daily production chart
        chart_days = list(range(1, 31))
        chart_pred = [predict_daily_production(d, models, data, substrate_amount) for d in chart_days]
        
        chart_data = pd.DataFrame({
            'Day': chart_days,
            'Predicted (mL)': chart_pred,
            'Predicted (L)': [x/1000 for x in chart_pred]
        })
        
        st.line_chart(chart_data.set_index('Day')['Predicted (L)'])
    
    with tab2:
        st.subheader("Cumulative Biogas Production Prediction")
        
        day = st.slider("Select day for cumulative prediction", 1, 30, 15)
        
        cumulative_pred = predict_cumulative_production(day, models, data, substrate_amount)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Cumulative Production",
                     f"{cumulative_pred/1000:.2f} L",
                     f"{cumulative_pred:.0f} mL")
        
        with col2:
            if day <= 25:
                actual_cum = sum(data[data['Day'] <= day]['mL_Produced'])
                scaled_actual = actual_cum * (substrate_amount / 30)
                st.metric("Scaled Experimental Total",
                         f"{scaled_actual/1000:.2f} L",
                         f"{scaled_actual:.0f} mL")
        
        # Cumulative chart
        cum_days = list(range(1, 31))
        cum_pred = [predict_cumulative_production(d, models, data, substrate_amount) for d in cum_days]
        
        cum_data = pd.DataFrame({
            'Day': cum_days,
            'Cumulative (L)': [x/1000 for x in cum_pred]
        })
        
        st.line_chart(cum_data.set_index('Day'))

if __name__ == "__main__":
    main()

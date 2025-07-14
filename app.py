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
    # Experimental data for 30g substrate producing 12L in 30 days
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

# ---- 2. Mathematical Functions ----
def linear_interpolation(x, x_data, y_data):
    """Simple linear interpolation between two points"""
    if x <= x_data[0]:
        return y_data[0]
    if x >= x_data[-1]:
        return y_data[-1]
    
    for i in range(len(x_data) - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            slope = (y_data[i + 1] - y_data[i]) / (x_data[i + 1] - x_data[i])
            return y_data[i] + slope * (x - x_data[i])
    return y_data[-1]

def polynomial_regression(x_data, y_data, degree=2):
    """Polynomial regression using numpy"""
    coeffs = np.polyfit(x_data, y_data, degree)
    return coeffs[::-1]  # Return in ascending order of powers

def evaluate_polynomial(x, coeffs):
    """Evaluate polynomial at point x"""
    return sum(coeff * (x ** i) for i, coeff in enumerate(coeffs))

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
    
    # Only fit models to days 1-25 (active production period)
    active_days = days[:25]
    active_prod = daily_prod[:25]
    active_cumulative = cumulative[:25]
    
    try:
        daily_coeffs = polynomial_regression(active_days, active_prod, degree=2)
        cumulative_coeffs = polynomial_regression(active_days, active_cumulative, degree=2)
    except:
        daily_coeffs = [sum(active_prod) / len(active_prod), 0]
        cumulative_coeffs = [0, sum(active_prod) / len(active_prod)]
    
    return {
        'daily_coeffs': daily_coeffs,
        'cumulative_coeffs': cumulative_coeffs,
        'actual_daily': daily_prod,
        'actual_cumulative': cumulative,
        'days': days,
        'base_substrate': 30,  # 30g substrate in base data
        'total_production': sum(active_prod)  # Total from days 1-25
    }

def predict_daily_production(day, models, data, substrate_amount):
    """Predict daily production scaled by substrate amount"""
    # No production after day 25
    if day > 25:
        return 0
    
    # Get base prediction
    poly_pred = evaluate_polynomial(day, models['daily_coeffs'])
    
    # Linear interpolation for days within data range
    if 1 <= day <= 25:
        interp_pred = linear_interpolation(day, models['days'][:25], models['actual_daily'][:25])
    else:
        interp_pred = poly_pred
    
    # Pattern recognition based on experimental data
    if day <= 10:
        pattern_pred = 200 * day  # Growth phase
    elif day <= 15:
        pattern_pred = 3800 - 100 * (day - 10)  # Peak phase
    else:
        pattern_pred = max(100, 3400 - 250 * (day - 15))  # Decline phase
    
    # Average the predictions and scale by substrate
    final_pred = (poly_pred + interp_pred + pattern_pred) / 3
    substrate_factor = substrate_amount / models['base_substrate']
    return max(0, final_pred * substrate_factor)

def predict_cumulative_production(day, models, data, substrate_amount):
    """Predict cumulative production scaled by substrate amount"""
    if day <= 25:
        # Calculate normally for days 1-25
        daily_sum = 0
        for d in range(1, int(day) + 1):
            daily_sum += predict_daily_production(d, models, data, substrate_amount)
        return daily_sum
    else:
        # Return day 25 cumulative value for days 26-30 (no additional production)
        day25_cumulative = predict_cumulative_production(25, models, data, substrate_amount)
        return day25_cumulative

# ---- 4. Main App ----
def main():
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
        st.dataframe(data)
        
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
        day = st.slider("Select day for prediction", 1, 30, 11, key="daily")
        
        daily_pred = predict_daily_production(day, models, data, substrate_amount)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Daily", 
                     f"{daily_pred:.0f} mL",
                     f"{daily_pred/1000:.2f} L")
        with col2:
            if day <= 25:
                actual = data[data['Day'] == day]['mL_Produced'].iloc[0]
                scaled = actual * (substrate_amount / 30)
                st.metric("Scaled Experimental", 
                         f"{scaled:.0f} mL",
                         f"{scaled/1000:.2f} L")
        
        # Daily chart
        days = list(range(1, 31))
        preds = [predict_daily_production(d, models, data, substrate_amount) for d in days]
        chart_data = pd.DataFrame({
            'Day': days,
            'Predicted (L)': [x/1000 for x in preds]
        })
        st.line_chart(chart_data.set_index('Day'))
    
    with tab2:
        st.subheader("Cumulative Biogas Production Prediction")
        day = st.slider("Select day for prediction", 1, 30, 15, key="cumulative")
        
        cum_pred = predict_cumulative_production(day, models, data, substrate_amount)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Cumulative",
                     f"{cum_pred/1000:.2f} L",
                     f"{cum_pred:.0f} mL")
        with col2:
            if day <= 25:
                actual = sum(data[data['Day'] <= day]['mL_Produced'])
                scaled = actual * (substrate_amount / 30)
                st.metric("Scaled Experimental",
                         f"{scaled/1000:.2f} L",
                         f"{scaled:.0f} mL")
        
        # Cumulative chart
        days = list(range(1, 31))
        cum_preds = [predict_cumulative_production(d, models, data, substrate_amount) for d in days]
        cum_data = pd.DataFrame({
            'Day': days,
            'Cumulative (L)': [x/1000 for x in cum_preds]
        })
        st.line_chart(cum_data.set_index('Day'))

if __name__ == "__main__":
    main()

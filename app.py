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
    data = pd.DataFrame({
        'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'mL_Produced': [96, 144, 192, 252, 312, 372, 432, 492, 552, 612, 864, 888, 912, 900, 816, 732, 648, 564, 480, 396, 312, 228, 168, 96, 24, 00, 00, 00, 00, 00],
        'Percent': [0.8, 1.2, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1, 7.2, 7.4, 7.6, 7.5, 6.8, 6.1, 5.4, 4.7, 4.0, 3.3, 2.6, 1.9, 1.4, 0.8, 0.2, 0, 0, 0, 0, 0]
    })
    return data

# ---- 2. Simple Mathematical Functions ----
def linear_interpolation(x, x_data, y_data):
    """Simple linear interpolation between two points"""
    if x <= x_data[0]:
        return y_data[0]
    if x >= x_data[-1]:
        return y_data[-1]
    
    for i in range(len(x_data) - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            # Linear interpolation formula
            slope = (y_data[i + 1] - y_data[i]) / (x_data[i + 1] - x_data[i])
            return y_data[i] + slope * (x - x_data[i])
    
    return y_data[-1]

def polynomial_regression(x_data, y_data, degree=2):
    """Simple polynomial regression using least squares"""
    n = len(x_data)
    
    # Create matrix A for polynomial fitting
    A = []
    for i in range(n):
        row = []
        for j in range(degree + 1):
            row.append(x_data[i] ** j)
        A.append(row)
    
    # Solve using normal equations: (A^T * A) * coeffs = A^T * b
    # This is a simplified implementation
    if degree == 2:
        # For quadratic: ax^2 + bx + c
        sum_x = sum(x_data)
        sum_x2 = sum(x * x for x in x_data)
        sum_x3 = sum(x * x * x for x in x_data)
        sum_x4 = sum(x * x * x * x for x in x_data)
        sum_y = sum(y_data)
        sum_xy = sum(x_data[i] * y_data[i] for i in range(n))
        sum_x2y = sum(x_data[i] * x_data[i] * y_data[i] for i in range(n))
        
        # Solve 3x3 system for quadratic
        det = n * sum_x2 * sum_x4 + 2 * sum_x * sum_x2 * sum_x3 - sum_x2 * sum_x2 * sum_x2 - n * sum_x3 * sum_x3 - sum_x * sum_x * sum_x4
        
        if abs(det) < 1e-10:
            # Fallback to linear
            if sum_x2 * n - sum_x * sum_x != 0:
                a = (sum_xy * n - sum_x * sum_y) / (sum_x2 * n - sum_x * sum_x)
                b = (sum_y - a * sum_x) / n
                return [b, a, 0]
            else:
                return [sum_y / n, 0, 0]
        
        c = (sum_y * sum_x2 * sum_x4 + sum_x * sum_x3 * sum_x2y + sum_x2 * sum_x3 * sum_xy - sum_x2 * sum_x2 * sum_x2y - sum_y * sum_x3 * sum_x3 - sum_x * sum_x2 * sum_x4) / det
        b = (n * sum_x2 * sum_x2y + sum_x * sum_x3 * sum_y + sum_x2 * sum_x * sum_xy - sum_x2 * sum_x2 * sum_y - n * sum_x3 * sum_xy - sum_x * sum_x * sum_x2y) / det
        a = (n * sum_x4 * sum_xy + sum_x * sum_x2 * sum_y + sum_x2 * sum_x3 * sum_xy - sum_x2 * sum_x2 * sum_xy - n * sum_x3 * sum_x2y - sum_x * sum_x4 * sum_y) / det
        
        return [c, b, a]
    else:
        # Linear regression fallback
        if sum(x * x for x in x_data) * n - sum(x_data) * sum(x_data) != 0:
            slope = (sum(x_data[i] * y_data[i] for i in range(n)) * n - sum(x_data) * sum(y_data)) / (sum(x * x for x in x_data) * n - sum(x_data) * sum(x_data))
            intercept = (sum(y_data) - slope * sum(x_data)) / n
            return [intercept, slope]
        else:
            return [sum(y_data) / n, 0]

def evaluate_polynomial(x, coeffs):
    """Evaluate polynomial at point x"""
    result = 0
    for i, coeff in enumerate(coeffs):
        result += coeff * (x ** i)
    return result

def simple_gompertz(t, Y_max=12000, R_max=500, lag=8):
    """Simplified Gompertz model"""
    try:
        exp_term = math.exp((R_max * math.e / Y_max) * (lag - t) + 1)
        return Y_max * math.exp(-exp_term)
    except (OverflowError, ValueError):
        # Fallback to logistic-like growth
        if t < lag:
            return Y_max * 0.01
        else:
            return Y_max * (1 - math.exp(-(t - lag) / 10))

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
        'days': days
    }

def predict_daily_production(day, models, data):
    """Predict daily production for a given day"""
    # Method 1: Polynomial prediction
    poly_pred = evaluate_polynomial(day, models['daily_coeffs'])
    
    # Method 2: Linear interpolation for days within data range
    if 1 <= day <= 25:
        interp_pred = linear_interpolation(day, models['days'], models['actual_daily'])
    else:
        interp_pred = poly_pred
    
    # Method 3: Simple pattern recognition
    if day <= 10:
        pattern_pred = 60 * day  # Early growth phase
    elif day <= 15:
        pattern_pred = 900 - 20 * (day - 10)  # Peak phase
    else:
        pattern_pred = max(50, 700 - 30 * (day - 15))  # Decline phase
    
    # Average the predictions
    final_pred = (poly_pred + interp_pred + pattern_pred) / 3
    return max(0, final_pred)  # Ensure non-negative

def predict_cumulative_production(day, models, data):
    """Predict cumulative production for a given day"""
    # Method 1: Polynomial prediction
    poly_pred = evaluate_polynomial(day, models['cumulative_coeffs'])
    
    # Method 2: Gompertz model
    gompertz_pred = simple_gompertz(day)
    
    # Method 3: Sum of daily predictions
    daily_sum = 0
    for d in range(1, int(day) + 1):
        daily_sum += predict_daily_production(d, models, data)
    
    # Average the predictions
    final_pred = (poly_pred + gompertz_pred + daily_sum) / 3
    return max(0, final_pred)

# ---- 4. Main App ----
def main():
    # Header
    st.title('🔬 Biogas Production Predictor')
    st.markdown("*Using experimental data from 12L anaerobic digester*")
    st.markdown("---")
    
    # Load data and fit models
    data = load_data()
    models = fit_prediction_models(data)
    
    # Sidebar information
    st.sidebar.header("📊 Model Information")
    st.sidebar.info("This app uses polynomial regression and pattern recognition for predictions.")
    st.sidebar.markdown("**Prediction Methods:**")
    st.sidebar.markdown("- Polynomial regression")
    st.sidebar.markdown("- Linear interpolation")
    st.sidebar.markdown("- Simplified Gompertz model")
    st.sidebar.markdown("- Pattern recognition")
    
    # Display raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(data, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Days", len(data))
        with col2:
            st.metric("Total Production", f"{sum(data['mL_Produced'])} mL")
        with col3:
            st.metric("Average Daily", f"{sum(data['mL_Produced'])/len(data):.0f} mL")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Daily Prediction", "📊 Cumulative Prediction", "📉 Data Analysis"])
    
    with tab1:
        st.subheader("Daily Biogas Production Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            day = st.slider("Select day for prediction", 1, 30, 11, key="daily_slider")
        
        with col2:
            st.markdown("**Prediction Range:**")
            st.markdown("• Days 1-30: Interpolated")
        
        # Make prediction
        daily_pred = predict_daily_production(day, models, data)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Predicted Daily Production", f"{daily_pred:.1f} mL")
        
        with col2:
            if day <= 25:
                actual_val = data[data['Day'] == day]['mL_Produced'].iloc[0] if day in data['Day'].values else "N/A"
                if actual_val != "N/A":
                    error = abs(daily_pred - actual_val)
                    st.metric("❌ Prediction Error", f"{error:.1f} mL")
                else:
                    st.metric("⚠️ Status", "Extrapolated")
            else:
                st.metric("⚠️ Status", "Extrapolated")
        
        with col3:
            cumulative_pred = predict_cumulative_production(day, models, data)
            st.metric("📈 Cumulative by Day", f"{cumulative_pred:.0f} mL")
        
        # Create chart data for daily predictions
        chart_days = list(range(1, 31))
        chart_predictions = [predict_daily_production(d, models, data) for d in chart_days]
        
        chart_data = pd.DataFrame({
            'Day': chart_days,
            'Predicted_Production': chart_predictions
        })
        
        # Add actual data to chart
        actual_data = pd.DataFrame({
            'Day': data['Day'],
            'Actual_Production': data['mL_Produced']
        })
        
        st.subheader("Daily Production Chart")
        
        # Use Streamlit's native charting
        combined_chart = pd.DataFrame({
            'Day': chart_days,
            'Predicted': chart_predictions
        })
        
        st.line_chart(combined_chart.set_index('Day'))
        
        # Show actual vs predicted for first 25 days
        st.subheader("Actual vs Predicted (Days 1-25)")
        comparison_data = pd.DataFrame({
            'Day': data['Day'],
            'Actual': data['mL_Produced'],
            'Predicted': [predict_daily_production(d, models, data) for d in data['Day']]
        })
        
        st.line_chart(comparison_data.set_index('Day'))
    
    with tab2:
        st.subheader("Cumulative Biogas Production Prediction")
        
        day = st.slider("Select day for cumulative prediction", 1, 30, 11, key="cumulative_slider")
        
        cumulative_pred = predict_cumulative_production(day, models, data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📊 Predicted Cumulative Production", f"{cumulative_pred:.0f} mL")
        
        with col2:
            if day <= 25:
                actual_cumulative = sum(data[data['Day'] <= day]['mL_Produced'])
                error = abs(cumulative_pred - actual_cumulative)
                st.metric("❌ Prediction Error", f"{error:.0f} mL")
            else:
                st.metric("⚠️ Status", "Extrapolated")
        
        # Create cumulative chart
        chart_days = list(range(1, 31))
        cumulative_predictions = [predict_cumulative_production(d, models, data) for d in chart_days]
        
        cumulative_chart = pd.DataFrame({
            'Day': chart_days,
            'Predicted_Cumulative': cumulative_predictions
        })
        
        st.subheader("Cumulative Production Chart")
        st.line_chart(cumulative_chart.set_index('Day'))
        
        # Show actual cumulative data
        actual_cumulative_data = []
        running_total = 0
        for val in data['mL_Produced']:
            running_total += val
            actual_cumulative_data.append(running_total)
        
        comparison_cumulative = pd.DataFrame({
            'Day': data['Day'],
            'Actual_Cumulative': actual_cumulative_data,
            'Predicted_Cumulative': [predict_cumulative_production(d, models, data) for d in data['Day']]
        })
        
        st.subheader("Actual vs Predicted Cumulative (Days 1-25)")
        st.line_chart(comparison_cumulative.set_index('Day'))
    
    with tab3:
        st.subheader("Data Analysis & Statistics")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Daily Production Statistics:**")
            st.markdown(f"• Maximum: {max(data['mL_Produced'])} mL (Day {data.loc[data['mL_Produced'].idxmax(), 'Day']})")
            st.markdown(f"• Minimum: {min(data['mL_Produced'])} mL (Day {data.loc[data['mL_Produced'].idxmin(), 'Day']})")
            st.markdown(f"• Average: {sum(data['mL_Produced'])/len(data):.1f} mL")
            st.markdown(f"• Total: {sum(data['mL_Produced'])} mL")
        
        with col2:
            st.markdown("**Production Phases:**")
            st.markdown("• Days 1-10: Growth phase")
            st.markdown("• Days 11-15: Peak production")
            st.markdown("• Days 16-25: Decline phase")
            st.markdown("• Peak day: Day 13 (912 mL)")
        
        # Show data trends
        st.subheader("Production Trends")
        
        trend_data = pd.DataFrame({
            'Day': data['Day'],
            'Daily_Production': data['mL_Produced'],
            'Biogas_Percentage': data['Percent']
        })
        
        st.line_chart(trend_data.set_index('Day'))
        
        # Production phases analysis
        st.subheader("Production Phase Analysis")
        
        phase1 = data[data['Day'] <= 10]['mL_Produced']
        phase2 = data[(data['Day'] > 10) & (data['Day'] <= 15)]['mL_Produced']
        phase3 = data[data['Day'] > 15]['mL_Produced']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Phase 1 (Days 1-10):**")
            st.markdown(f"• Average: {sum(phase1)/len(phase1):.1f} mL")
            st.markdown(f"• Total: {sum(phase1)} mL")
            st.markdown("• Trend: Steady increase")
        
        with col2:
            st.markdown("**Phase 2 (Days 11-15):**")
            st.markdown(f"• Average: {sum(phase2)/len(phase2):.1f} mL")
            st.markdown(f"• Total: {sum(phase2)} mL")
            st.markdown("• Trend: Peak production")
        
        with col3:
            st.markdown("**Phase 3 (Days 16-25):**")
            st.markdown(f"• Average: {sum(phase3)/len(phase3):.1f} mL")
            st.markdown(f"• Total: {sum(phase3)} mL")
            st.markdown("• Trend: Declining")
    
    # Data export
    st.sidebar.header("📥 Export Data")
    
    # Generate prediction data
    export_days = list(range(1, 31))
    daily_predictions = [predict_daily_production(d, models, data) for d in export_days]
    cumulative_predictions = [predict_cumulative_production(d, models, data) for d in export_days]
    
    export_data = pd.DataFrame({
        'Day': export_days,
        'Daily_Predicted_mL': daily_predictions,
        'Cumulative_Predicted_mL': cumulative_predictions
    })
    
    csv_data = export_data.to_csv(index=False)
    
    st.sidebar.download_button(
        label="📊 Download Predictions CSV",
        data=csv_data,
        file_name="biogas_predictions.csv",
        mime="text/csv"
    )
    
    # Model information
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("**Version:** 1.0 (Minimal)")
    st.sidebar.markdown("**Dependencies:** None")
    st.sidebar.markdown("**Compatibility:** All platforms")

if __name__ == "__main__":
    main()

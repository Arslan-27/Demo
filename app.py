import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Try to import scipy, if not available use fallback
try:
    from scipy.optimize import curve_fit
    from scipy import interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ SciPy not available. Using simplified prediction methods.")

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
        'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
        'mL_Produced': [96, 144, 192, 252, 312, 372, 432, 492, 552, 612, 864, 888, 912, 900, 816, 732, 648, 564, 480, 396, 312, 228, 168, 72, 48],
        'Percent': [0.8, 1.2, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1, 7.2, 7.4, 7.6, 7.5, 6.8, 6.1, 5.4, 4.7, 4.0, 3.3, 2.6, 1.9, 1.4, 0.6, 0.4]
    })
    return data

# ---- 2. Fallback Interpolation (without scipy) ----
def linear_interpolation(x, x_data, y_data):
    """Simple linear interpolation"""
    if x <= x_data[0]:
        return y_data[0]
    if x >= x_data[-1]:
        return y_data[-1]
    
    for i in range(len(x_data) - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            # Linear interpolation between two points
            t = (x - x_data[i]) / (x_data[i + 1] - x_data[i])
            return y_data[i] + t * (y_data[i + 1] - y_data[i])
    
    return y_data[-1]

def polynomial_fit(x_data, y_data, degree=3):
    """Polynomial fitting using numpy"""
    coeffs = np.polyfit(x_data, y_data, degree)
    return coeffs

def predict_polynomial(x, coeffs):
    """Predict using polynomial coefficients"""
    return np.polyval(coeffs, x)

# ---- 3. Gompertz Model (with fallback) ----
def gompertz(t, Y_max, R_max, lag):
    """Gompertz growth model"""
    try:
        return Y_max * np.exp(-np.exp((R_max * np.exp(1) / Y_max) * (lag - t) + 1))
    except:
        # Fallback to simpler exponential model
        return Y_max * (1 - np.exp(-(t - lag) / R_max)) if t > lag else 0

# ---- 4. Model Fitting Functions ----
@st.cache_data
def fit_models(data):
    """Fit different models to the data"""
    cumulative = np.cumsum(data['mL_Produced'])
    
    results = {
        'gompertz_params': None,
        'polynomial_coeffs': None,
        'linear_coeffs': None,
        'r_squared': 0.0
    }
    
    if SCIPY_AVAILABLE:
        # Try Gompertz fitting with scipy
        try:
            params, _ = curve_fit(gompertz, data['Day'], cumulative, 
                                p0=[12000, 1000, 10], 
                                maxfev=5000,
                                bounds=([1000, 100, 0], [50000, 5000, 30]))
            results['gompertz_params'] = params
            
            # Calculate R-squared
            y_pred = gompertz(data['Day'], *params)
            ss_res = np.sum((cumulative - y_pred) ** 2)
            ss_tot = np.sum((cumulative - np.mean(cumulative)) ** 2)
            results['r_squared'] = 1 - (ss_res / ss_tot)
            
        except Exception as e:
            st.warning(f"Gompertz fitting failed: {e}")
    
    # Fallback polynomial fitting
    try:
        results['polynomial_coeffs'] = polynomial_fit(data['Day'], cumulative, degree=3)
        results['linear_coeffs'] = polynomial_fit(data['Day'], data['mL_Produced'], degree=2)
    except Exception as e:
        st.error(f"Polynomial fitting failed: {e}")
    
    return results

# ---- 5. Prediction Functions ----
def predict_daily(day, data, model_results):
    """Predict daily production for a given day"""
    if SCIPY_AVAILABLE and model_results['gompertz_params'] is not None:
        # Use scipy interpolation
        try:
            interp_func = interpolate.interp1d(data['Day'], data['mL_Produced'], 
                                             kind='cubic', fill_value='extrapolate')
            return float(interp_func(day))
        except:
            pass
    
    # Fallback to linear interpolation
    if model_results['linear_coeffs'] is not None:
        return predict_polynomial(day, model_results['linear_coeffs'])
    else:
        return linear_interpolation(day, data['Day'].values, data['mL_Produced'].values)

def predict_cumulative(day, data, model_results):
    """Predict cumulative production for a given day"""
    if model_results['gompertz_params'] is not None:
        Y_max, R_max, lag = model_results['gompertz_params']
        return gompertz(day, Y_max, R_max, lag)
    elif model_results['polynomial_coeffs'] is not None:
        return predict_polynomial(day, model_results['polynomial_coeffs'])
    else:
        # Simple linear extrapolation
        cumulative = np.cumsum(data['mL_Produced'])
        return linear_interpolation(day, data['Day'].values, cumulative.values)

# ---- 6. Main App ----
def main():
    # Header
    st.title('🔬 Biogas Production Predictor')
    st.markdown("*Using experimental data from 12L anaerobic digester*")
    
    # Load data and fit models
    data = load_data()
    model_results = fit_models(data)
    
    # Sidebar for model parameters
    st.sidebar.header("📊 Model Information")
    
    if model_results['gompertz_params'] is not None:
        Y_max, R_max, lag = model_results['gompertz_params']
        st.sidebar.markdown("**Gompertz Model Fit:**")
        st.sidebar.metric("Y_max (Total Potential)", f"{Y_max:.0f} mL")
        st.sidebar.metric("R_max (Max Rate)", f"{R_max:.0f} mL/day")
        st.sidebar.metric("Lag Phase", f"{lag:.1f} days")
        st.sidebar.metric("R-squared", f"{model_results['r_squared']:.3f}")
    else:
        st.sidebar.markdown("**Using Polynomial Model:**")
        st.sidebar.info("Advanced Gompertz model not available. Using polynomial approximation.")
    
    # Display raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(data, use_container_width=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Daily Prediction", "📊 Cumulative Prediction", "📉 Data Visualization"])
    
    with tab1:
        st.subheader("Daily Biogas Production Prediction")
        
        day = st.slider("Select day for prediction", 1, 30, 11, key="daily_slider")
        
        col1, col2 = st.columns(2)
        
        with col1:
            daily_pred = predict_daily(day, data, model_results)
            st.metric("Predicted Daily Production", f"{daily_pred:.1f} mL")
        
        with col2:
            if day > 1:
                prev_cumulative = predict_cumulative(day-1, data, model_results)
                curr_cumulative = predict_cumulative(day, data, model_results)
                daily_from_cumulative = curr_cumulative - prev_cumulative
            else:
                daily_from_cumulative = predict_cumulative(day, data, model_results)
            
            st.metric("Daily from Cumulative Model", f"{daily_from_cumulative:.1f} mL")
        
        # Plot daily production
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data['Day'], data['mL_Produced'], color='blue', s=50, label='Actual Data', zorder=5)
        
        # Prediction line
        pred_days = np.arange(1, 31)
        daily_predictions = [predict_daily(d, data, model_results) for d in pred_days]
        ax.plot(pred_days, daily_predictions, 'r-', linewidth=2, label='Prediction', alpha=0.8)
        
        # Highlight selected day
        ax.axvline(x=day, color='green', linestyle='--', alpha=0.7, label=f'Selected Day {day}')
        ax.scatter([day], [daily_pred], color='green', s=100, zorder=10)
        
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Daily Biogas Production (mL)', fontsize=12)
        ax.set_title('Daily Biogas Production Prediction', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Cumulative Biogas Production Prediction")
        
        day = st.slider("Select day for cumulative prediction", 1, 30, 11, key="cumulative_slider")
        
        cumulative_pred = predict_cumulative(day, data, model_results)
        st.metric("Predicted Cumulative Production", f"{cumulative_pred:.0f} mL")
        
        # Plot cumulative production
        fig, ax = plt.subplots(figsize=(10, 6))
        cumulative_actual = np.cumsum(data['mL_Produced'])
        ax.scatter(data['Day'], cumulative_actual, color='blue', s=50, label='Actual Data', zorder=5)
        
        # Prediction line
        pred_days = np.arange(1, 31)
        cumulative_predictions = [predict_cumulative(d, data, model_results) for d in pred_days]
        ax.plot(pred_days, cumulative_predictions, 'g-', linewidth=2, label='Prediction', alpha=0.8)
        
        # Highlight selected day
        ax.axvline(x=day, color='red', linestyle='--', alpha=0.7, label=f'Selected Day {day}')
        ax.scatter([day], [cumulative_pred], color='red', s=100, zorder=10)
        
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Cumulative Biogas Production (mL)', fontsize=12)
        ax.set_title('Cumulative Biogas Production Prediction', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Data Visualization & Analysis")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original daily data
        ax1.bar(data['Day'], data['mL_Produced'], color='skyblue', alpha=0.7)
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Daily Production (mL)')
        ax1.set_title('Daily Biogas Production (Actual)')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative data
        cumulative_actual = np.cumsum(data['mL_Produced'])
        ax2.plot(data['Day'], cumulative_actual, 'bo-', linewidth=2, markersize=6)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Cumulative Production (mL)')
        ax2.set_title('Cumulative Biogas Production (Actual)')
        ax2.grid(True, alpha=0.3)
        
        # Percentage data
        ax3.plot(data['Day'], data['Percent'], 'ro-', linewidth=2, markersize=6)
        ax3.set_xlabel('Day')
        ax3.set_ylabel('Percentage (%)')
        ax3.set_title('Biogas Percentage Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Production rate (derivative)
        production_rate = np.diff(data['mL_Produced'])
        ax4.plot(data['Day'][1:], production_rate, 'go-', linewidth=2, markersize=6)
        ax4.set_xlabel('Day')
        ax4.set_ylabel('Production Rate Change (mL/day)')
        ax4.set_title('Daily Production Rate Change')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Data export
    st.sidebar.header("📥 Export Data")
    
    # Generate prediction data
    export_days = np.arange(1, 31)
    daily_predictions = [predict_daily(d, data, model_results) for d in export_days]
    cumulative_predictions = [predict_cumulative(d, data, model_results) for d in export_days]
    
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

if __name__ == "__main__":
    main()

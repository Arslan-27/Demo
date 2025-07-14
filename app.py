import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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

# ---- 2. Gompertz Model Fitting ----
def gompertz(t, Y_max, R_max, lag):
    """Gompertz growth model"""
    return Y_max * np.exp(-np.exp((R_max * np.exp(1) / Y_max) * (lag - t) + 1))

@st.cache_data
def fit_gompertz_model(data):
    """Fit Gompertz model to cumulative data"""
    cumulative = np.cumsum(data['mL_Produced'])
    try:
        # Fixed the syntax error in curve_fit
        params, covariance = curve_fit(gompertz, data['Day'], cumulative, 
                                     p0=[12000, 1000, 10], 
                                     maxfev=5000,
                                     bounds=([1000, 100, 0], [50000, 5000, 30]))
        Y_max, R_max, lag = params
        
        # Calculate R-squared
        y_pred = gompertz(data['Day'], Y_max, R_max, lag)
        ss_res = np.sum((cumulative - y_pred) ** 2)
        ss_tot = np.sum((cumulative - np.mean(cumulative)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return Y_max, R_max, lag, r_squared
    except Exception as e:
        st.warning(f"Model fitting failed: {e}. Using fallback values.")
        return 12000, 1000, 10, 0.0  # Fallback values

@st.cache_data
def create_interpolation_functions(data):
    """Create interpolation functions for daily and cumulative data"""
    cumulative = np.cumsum(data['mL_Produced'])
    
    try:
        daily_interp = interpolate.interp1d(data['Day'], data['mL_Produced'], 
                                          kind='cubic', fill_value='extrapolate')
        cumulative_interp = interpolate.interp1d(data['Day'], cumulative, 
                                               kind='cubic', fill_value='extrapolate')
        return daily_interp, cumulative_interp
    except Exception as e:
        st.error(f"Interpolation failed: {e}")
        return None, None

# ---- 3. Main App ----
def main():
    # Header
    st.title('🔬 Biogas Production Predictor')
    st.markdown("*Using experimental data from 12L anaerobic digester*")
    
    # Load data and fit models
    data = load_data()
    Y_max, R_max, lag, r_squared = fit_gompertz_model(data)
    daily_interp, cumulative_interp = create_interpolation_functions(data)
    
    if daily_interp is None or cumulative_interp is None:
        st.error("Failed to create interpolation functions. Please check your data.")
        return
    
    # Sidebar for model parameters
    st.sidebar.header("📊 Model Parameters")
    st.sidebar.markdown("**Gompertz Model Fit:**")
    st.sidebar.metric("Y_max (Total Potential)", f"{Y_max:.0f} mL")
    st.sidebar.metric("R_max (Max Rate)", f"{R_max:.0f} mL/day")
    st.sidebar.metric("Lag Phase", f"{lag:.1f} days")
    st.sidebar.metric("R-squared", f"{r_squared:.3f}")
    
    # Display raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(data, use_container_width=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📈 Daily Prediction", "📊 Cumulative Prediction", "📉 Model Comparison"])
    
    with tab1:
        st.subheader("Daily Biogas Production Prediction")
        
        day = st.slider("Select day for prediction", 1, 30, 11, key="daily_slider")
        
        col1, col2 = st.columns(2)
        
        with col1:
            daily_pred = daily_interp(day)
            st.metric("Daily Production (Interpolation)", 
                     f"{daily_pred:.1f} mL",
                     delta=f"{daily_pred - daily_interp(day-1):.1f} mL" if day > 1 else None)
        
        with col2:
            if day > 1:
                gompertz_daily = gompertz(day, Y_max, R_max, lag) - gompertz(day-1, Y_max, R_max, lag)
            else:
                gompertz_daily = gompertz(day, Y_max, R_max, lag)
            
            st.metric("Daily Production (Gompertz Model)", 
                     f"{gompertz_daily:.1f} mL")
        
        # Plot daily production
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data['Day'], data['mL_Produced'], color='blue', s=50, label='Actual Data', zorder=5)
        
        pred_days = np.linspace(1, 30, 100)
        daily_pred_vals = [daily_interp(x) for x in pred_days]
        ax.plot(pred_days, daily_pred_vals, 'r-', linewidth=2, label='Interpolation', alpha=0.8)
        
        # Highlight selected day
        ax.axvline(x=day, color='green', linestyle='--', alpha=0.7, label=f'Selected Day {day}')
        ax.scatter([day], [daily_interp(day)], color='green', s=100, zorder=10)
        
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Daily Biogas Production (mL)', fontsize=12)
        ax.set_title('Daily Biogas Production Prediction', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Cumulative Biogas Production Prediction")
        
        day = st.slider("Select day for cumulative prediction", 1, 30, 11, key="cumulative_slider")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cumulative_pred = cumulative_interp(day)
            st.metric("Cumulative (Interpolation)", 
                     f"{cumulative_pred:.0f} mL")
        
        with col2:
            gompertz_cumulative = gompertz(day, Y_max, R_max, lag)
            st.metric("Cumulative (Gompertz Model)", 
                     f"{gompertz_cumulative:.0f} mL")
        
        # Plot cumulative production
        fig, ax = plt.subplots(figsize=(10, 6))
        cumulative_actual = np.cumsum(data['mL_Produced'])
        ax.scatter(data['Day'], cumulative_actual, color='blue', s=50, label='Actual Data', zorder=5)
        
        pred_days = np.linspace(1, 30, 100)
        gompertz_vals = [gompertz(x, Y_max, R_max, lag) for x in pred_days]
        ax.plot(pred_days, gompertz_vals, 'g-', linewidth=2, label='Gompertz Model', alpha=0.8)
        
        # Highlight selected day
        ax.axvline(x=day, color='red', linestyle='--', alpha=0.7, label=f'Selected Day {day}')
        ax.scatter([day], [gompertz(day, Y_max, R_max, lag)], color='red', s=100, zorder=10)
        
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Cumulative Biogas Production (mL)', fontsize=12)
        ax.set_title('Cumulative Biogas Production Prediction', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Model Comparison")
        
        # Compare both models
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Daily comparison
        pred_days = np.linspace(1, 30, 100)
        daily_pred_vals = [daily_interp(x) for x in pred_days]
        daily_gompertz = [gompertz(x, Y_max, R_max, lag) - gompertz(x-1, Y_max, R_max, lag) if x > 1 else gompertz(x, Y_max, R_max, lag) for x in pred_days]
        
        ax1.scatter(data['Day'], data['mL_Produced'], color='blue', s=50, label='Actual Data', zorder=5)
        ax1.plot(pred_days, daily_pred_vals, 'r-', linewidth=2, label='Interpolation', alpha=0.8)
        ax1.plot(pred_days, daily_gompertz, 'g-', linewidth=2, label='Gompertz (Daily)', alpha=0.8)
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Daily Production (mL)')
        ax1.set_title('Daily Production Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative comparison
        cumulative_actual = np.cumsum(data['mL_Produced'])
        cumulative_pred_vals = [cumulative_interp(x) for x in pred_days]
        gompertz_vals = [gompertz(x, Y_max, R_max, lag) for x in pred_days]
        
        ax2.scatter(data['Day'], cumulative_actual, color='blue', s=50, label='Actual Data', zorder=5)
        ax2.plot(pred_days, cumulative_pred_vals, 'r-', linewidth=2, label='Interpolation', alpha=0.8)
        ax2.plot(pred_days, gompertz_vals, 'g-', linewidth=2, label='Gompertz Model', alpha=0.8)
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Cumulative Production (mL)')
        ax2.set_title('Cumulative Production Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Data export
    st.sidebar.header("📥 Export Data")
    
    # Generate prediction data
    export_days = np.arange(1, 31)
    daily_predictions = [daily_interp(x) for x in export_days]
    cumulative_predictions = [gompertz(x, Y_max, R_max, lag) for x in export_days]
    
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

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
import matplotlib.pyplot as plt

# ---- 1. Load Data ----
data = pd.DataFrame({
    'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    'mL_Produced': [96, 144, 192, 252, 312, 372, 432, 492, 552, 612, 864, 888, 912, 900, 816, 732, 648, 564, 480, 396, 312, 228, 168, 72, 48],
    'Percent': [0.8, 1.2, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1, 7.2, 7.4, 7.6, 7.5, 6.8, 6.1, 5.4, 4.7, 4.0, 3.3, 2.6, 1.9, 1.4, 0.6, 0.4]
})

# ---- 2. Gompertz Model Fitting ----
def gompertz(t, Y_max, R_max, lag):
    return Y_max * np.exp(-np.exp((R_max * np.exp(1) / Y_max) * (lag - t) + 1))

# Fit to cumulative data
cumulative = np.cumsum(data['mL_Produced'])
try:
    params, _ = curve_fit(gompertz, data['Day'], cumulative, p0=[12000, 1, 1])
    Y_max, R_max, lag = params
except:
    Y_max, R_max, lag = 12000, 1, 1  # Fallback values

# ---- 3. Interpolation ----
daily_interp = interpolate.interp1d(data['Day'], data['mL_Produced'], kind='cubic', fill_value='extrapolate')
cumulative_interp = interpolate.interp1d(data['Day'], cumulative, kind='cubic', fill_value='extrapolate')

# ---- 4. Streamlit UI ----
st.title('Biogas Production Predictor')
st.write("Using experimental data from 12L digester")

tab1, tab2 = st.tabs(["Daily Prediction", "Cumulative Prediction"])

with tab1:
    day = st.slider("Select day for prediction", 1, 30, 11)
    
    st.subheader(f"Day {day} Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Daily Production (Interpolation)", f"{daily_interp(day):.1f} mL")
    with col2:
        st.metric("Daily Production (Gompertz Model)", 
                 f"{gompertz(day, Y_max, R_max, lag) - gompertz(day-1, Y_max, R_max, lag) if day > 1 else gompertz(day, Y_max, R_max, lag):.1f} mL")
    
    # Plot
    fig, ax = plt.subplots()
    ax.scatter(data['Day'], data['mL_Produced'], label='Actual Data')
    pred_days = np.linspace(1, 30, 30)
    ax.plot(pred_days, [daily_interp(x) for x in pred_days], 'r-', label='Interpolation')
    ax.set_xlabel('Day')
    ax.set_ylabel('Biogas Production (mL)')
    ax.legend()
    st.pyplot(fig)

with tab2:
    day = st.slider("Select day for cumulative prediction", 1, 30, 11)
    
    st.subheader(f"Cumulative by Day {day}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Cumulative (Interpolation)", f"{cumulative_interp(day):.1f} mL")
    with col2:
        st.metric("Cumulative (Gompertz Model)", f"{gompertz(day, Y_max, R_max, lag):.1f} mL")
    
    # Plot
    fig, ax = plt.subplots()
    ax.scatter(data['Day'], cumulative, label='Actual Data')
    ax.plot(pred_days, [gompertz(x, Y_max, R_max, lag) for x in pred_days], 'g-', label='Gompertz Model')
    ax.set_xlabel('Day')
    ax.set_ylabel('Cumulative Biogas (mL)')
    ax.legend()
    st.pyplot(fig)

# ---- 5. Model Parameters ----
st.sidebar.header("Model Parameters")
st.sidebar.write(f"Gompertz Fit:")
st.sidebar.write(f"- Y_max (Total Potential): {Y_max:.1f} mL")
st.sidebar.write(f"- R_max (Max Rate): {R_max:.1f} mL/day")
st.sidebar.write(f"- Lag Phase: {lag:.1f} days")

# ---- 6. Data Export ----
st.sidebar.download_button(
    label="Download Predictions",
    data=pd.DataFrame({
        'Day': pred_days,
        'Daily_Predicted_mL': [daily_interp(x) for x in pred_days],
        'Cumulative_Predicted_mL': [gompertz(x, Y_max, R_max, lag) for x in pred_days]
    }).to_csv(index=False),
    file_name="biogas_predictions.csv",
    mime="text/csv"
)
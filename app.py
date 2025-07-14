# INSTALLATION INSTRUCTIONS (Run these commands in your terminal first)
"""
pip install streamlit pandas numpy matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Scientific Biogas Predictor Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BASE_SUBSTRATE = 30  # grams
BASE_TOTAL_PRODUCTION = 12000  # mL (12L)

# ---- 1. Scientific Models ----
def gompertz_model(day, substrate_amount, max_gas, lag, rate):
    """
    Modified Gompertz equation for biogas production
    Parameters:
    - max_gas: Maximum biogas potential (mL)
    - lag: Lag phase duration (days)
    - rate: Maximum production rate (1/day)
    """
    return max_gas * np.exp(-np.exp(rate * np.e * (lag - day)/max_gas + 1)) * (substrate_amount/BASE_SUBSTRATE)

def first_order_model(day, substrate_amount, k):
    """
    First-order kinetic model for substrate degradation
    Parameters:
    - k: Degradation rate constant (1/day)
    """
    return BASE_TOTAL_PRODUCTION * (1 - np.exp(-k * min(day,25))) * (substrate_amount/BASE_SUBSTRATE)

# ---- 2. Data Processing ----
@st.cache_data
def calculate_productions(substrate_schedule, params):
    """Calculate production using both models"""
    results = []
    last_amount = BASE_SUBSTRATE
    
    for day in range(1, 31):
        current_amount = substrate_schedule.get(day, last_amount)
        last_amount = current_amount
        
        # Calculate model predictions
        g_prod = gompertz_model(
            day, current_amount,
            params['gompertz_max'],
            params['gompertz_lag'],
            params['gompertz_rate']
        )
        
        k_prod = first_order_model(
            day, current_amount,
            params['kinetic_k']
        )
        
        # Weighted average of models
        combined = 0.7*g_prod + 0.3*k_prod  # 70% Gompertz, 30% Kinetic
        
        results.append({
            'Day': day,
            'Substrate (g)': current_amount,
            'Gompertz (mL)': g_prod,
            'Kinetic (mL)': k_prod,
            'Combined (mL)': combined,
            'Status': 'Active' if day <= 25 else 'Inactive'
        })
    
    df = pd.DataFrame(results)
    # Calculate cumulative productions
    df['Gompertz Cumulative'] = df['Gompertz (mL)'].cumsum()
    df['Kinetic Cumulative'] = df['Kinetic (mL)'].cumsum()
    df['Combined Cumulative'] = df['Combined (mL)'].cumsum()
    return df

# ---- 3. Visualization Functions ----
def plot_model_comparison(df):
    """Create publication-quality comparison plots"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Daily Production Plot
    ax1.plot(df['Day'], df['Gompertz (mL)'], label='Gompertz', color='#1f77b4', linewidth=2)
    ax1.plot(df['Day'], df['Kinetic (mL)'], label='First-Order', color='#ff7f0e', linewidth=2)
    ax1.plot(df['Day'], df['Combined (mL)'], label='Combined', color='#2ca02c', linestyle='--', linewidth=2)
    ax1.set_title('Daily Biogas Production', fontsize=14, pad=20)
    ax1.set_ylabel('Production (mL/day)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 30)
    
    # Cumulative Production Plot
    ax2.plot(df['Day'], df['Gompertz Cumulative'], label='Gompertz', color='#1f77b4', linewidth=2)
    ax2.plot(df['Day'], df['Kinetic Cumulative'], label='First-Order', color='#ff7f0e', linewidth=2)
    ax2.plot(df['Day'], df['Combined Cumulative'], label='Combined', color='#2ca02c', linestyle='--', linewidth=2)
    ax2.set_title('Cumulative Biogas Production', fontsize=14, pad=20)
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('Cumulative Production (mL)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 30)
    
    plt.tight_layout()
    st.pyplot(fig)

# ---- 4. Main App ----
def main():
    st.title('🧪 Scientific Biogas Modeling Pro')
    st.markdown("""
    **Advanced prediction using Gompertz and First-Order Kinetic models**  
    *Base case: 30g substrate produces exactly 12L in 25 days*
    """)
    st.markdown("---")
    
    # ---- Sidebar Controls ----
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Model Selection
        st.subheader("Model Configuration")
        model_weight = st.slider("Gompertz Model Weight", 0.0, 1.0, 0.7, 
                               help="Relative weight of Gompertz model in combined prediction")
        
        # Gompertz Parameters
        st.subheader("Gompertz Parameters")
        g_max = st.slider("Maximum gas (mL)", 10000, 20000, 12000, step=100)
        g_lag = st.slider("Lag phase (days)", 0, 10, 3)
        g_rate = st.slider("Growth rate (1/day)", 0.01, 0.5, 0.15, step=0.01)
        
        # Kinetic Parameters
        st.subheader("Kinetic Parameters")
        k_value = st.slider("Degradation rate (k)", 0.01, 0.3, 0.12, step=0.01,
                           help="First-order rate constant for substrate degradation")
        
        # Substrate Inputs
        st.subheader("🌱 Substrate Schedule")
        num_inputs = st.number_input("Number of substrate additions", 1, 10, 3)
        
        substrate_schedule = {}
        for i in range(num_inputs):
            cols = st.columns(2)
            with cols[0]:
                day = st.number_input(f"Addition Day {i+1}", 1, 30, (i+1)*5, key=f'day_{i}')
            with cols[1]:
                amount = st.number_input(f"Amount (g) {i+1}", 1, 1000, 30*(i+1), key=f'amt_{i}')
            substrate_schedule[day] = amount
    
    # ---- Calculate Productions ----
    params = {
        'gompertz_max': g_max,
        'gompertz_lag': g_lag,
        'gompertz_rate': g_rate,
        'kinetic_k': k_value,
        'model_weight': model_weight
    }
    
    results_df = calculate_productions(substrate_schedule, params)
    
    # ---- Display Results ----
    st.header("📊 Model Results")
    
    # Key Metrics
    cols = st.columns(3)
    with cols[0]:
        st.metric("Gompertz Prediction", 
                 f"{results_df['Gompertz Cumulative'].iloc[24]/1000:.2f} L",
                 "Day 25 Total")
    with cols[1]:
        st.metric("Kinetic Prediction", 
                 f"{results_df['Kinetic Cumulative'].iloc[24]/1000:.2f} L",
                 "Day 25 Total")
    with cols[2]:
        st.metric("Combined Prediction", 
                 f"{results_df['Combined Cumulative'].iloc[24]/1000:.2f} L",
                 "Day 25 Total")
    
    # Model Comparison Plots
    st.markdown("### 📈 Model Comparison")
    plot_model_comparison(results_df)
    
    # Data Table
    st.markdown("### 🔍 Detailed Production Data")
    st.dataframe(
        results_df.style.format({
            'Gompertz (mL)': '{:.0f}',
            'Kinetic (mL)': '{:.0f}', 
            'Combined (mL)': '{:.0f}',
            'Gompertz Cumulative': '{:.0f}',
            'Kinetic Cumulative': '{:.0f}',
            'Combined Cumulative': '{:.0f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Export Data
    st.markdown("---")
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Prediction Data (CSV)",
        data=csv,
        file_name="biogas_model_predictions.csv",
        mime="text/csv",
        help="Download complete prediction data for all days"
    )

if __name__ == "__main__":
    main()

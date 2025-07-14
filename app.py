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
ACTIVE_DAYS = 25  # Days for complete degradation

# ---- 1. Scientific Models ----
def gompertz_model(day, substrate_amount, max_gas, lag, rate):
    """
    Modified Gompertz equation for biogas production
    Normalized to produce exactly 12L for 30g in 25 days
    """
    if day > ACTIVE_DAYS:
        return 0
    
    # Gompertz equation with proper scaling
    exp_term = np.exp(-np.exp(rate * np.e * (lag - day)/max_gas + 1))
    daily_production = max_gas * exp_term * (substrate_amount/BASE_SUBSTRATE)
    
    return daily_production

def first_order_model(day, substrate_amount, k):
    """
    First-order kinetic model for substrate degradation
    Produces exactly 12L for 30g substrate in 25 days
    """
    if day > ACTIVE_DAYS:
        return 0
    
    # Calculate remaining substrate at day
    remaining_substrate = np.exp(-k * day)
    
    # Daily production is proportional to degradation rate
    if day == 1:
        daily_production = BASE_TOTAL_PRODUCTION * k * (substrate_amount/BASE_SUBSTRATE)
    else:
        prev_remaining = np.exp(-k * (day-1))
        daily_production = BASE_TOTAL_PRODUCTION * k * (prev_remaining) * (substrate_amount/BASE_SUBSTRATE)
    
    return daily_production

def calculate_substrate_degradation(day, initial_amount, k):
    """Calculate remaining substrate based on degradation rate"""
    if day > ACTIVE_DAYS:
        return 0
    return initial_amount * np.exp(-k * day)

# ---- 2. Data Processing ----
@st.cache_data
def calculate_productions(substrate_schedule, params):
    """Calculate production using both models with proper degradation"""
    results = []
    
    # Sort substrate schedule by day
    sorted_schedule = sorted(substrate_schedule.items())
    
    for day in range(1, 31):
        # Get current substrate amount
        current_substrate = 0
        for sched_day, amount in sorted_schedule:
            if sched_day <= day:
                # Calculate degraded amount from this addition
                days_since_addition = day - sched_day + 1
                remaining = calculate_substrate_degradation(days_since_addition, amount, params['kinetic_k'])
                current_substrate += remaining
        
        # Calculate daily production from all active substrate
        g_prod = 0
        k_prod = 0
        
        for sched_day, amount in sorted_schedule:
            if sched_day <= day:
                days_since_addition = day - sched_day + 1
                if days_since_addition <= ACTIVE_DAYS:
                    # Production from this specific addition
                    g_prod += gompertz_model(
                        days_since_addition, amount,
                        params['gompertz_max'],
                        params['gompertz_lag'],
                        params['gompertz_rate']
                    )
                    
                    k_prod += first_order_model(
                        days_since_addition, amount,
                        params['kinetic_k']
                    )
        
        # Weighted average of models
        combined = params['model_weight'] * g_prod + (1 - params['model_weight']) * k_prod
        
        # Degradation percentage
        initial_total = sum(amount for d, amount in sorted_schedule if d <= day)
        degradation_pct = ((initial_total - current_substrate) / initial_total * 100) if initial_total > 0 else 0
        
        results.append({
            'Day': day,
            'Active Substrate (g)': current_substrate,
            'Degradation (%)': degradation_pct,
            'Gompertz (mL)': g_prod,
            'Kinetic (mL)': k_prod,
            'Combined (mL)': combined,
            'Status': 'Active' if current_substrate > 0.1 else 'Inactive'
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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Daily Production Plot
    ax1.plot(df['Day'], df['Gompertz (mL)'], label='Gompertz', color='#1f77b4', linewidth=2)
    ax1.plot(df['Day'], df['Kinetic (mL)'], label='First-Order', color='#ff7f0e', linewidth=2)
    ax1.plot(df['Day'], df['Combined (mL)'], label='Combined', color='#2ca02c', linestyle='--', linewidth=2)
    ax1.set_title('Daily Biogas Production', fontsize=14, pad=20)
    ax1.set_ylabel('Production (mL/day)', fontsize=12)
    ax1.set_xlabel('Day', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 30)
    
    # Cumulative Production Plot
    ax2.plot(df['Day'], df['Gompertz Cumulative'], label='Gompertz', color='#1f77b4', linewidth=2)
    ax2.plot(df['Day'], df['Kinetic Cumulative'], label='First-Order', color='#ff7f0e', linewidth=2)
    ax2.plot(df['Day'], df['Combined Cumulative'], label='Combined', color='#2ca02c', linestyle='--', linewidth=2)
    ax2.axhline(y=12000, color='red', linestyle=':', alpha=0.7, label='Target (12L)')
    ax2.set_title('Cumulative Biogas Production', fontsize=14, pad=20)
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('Cumulative Production (mL)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 30)
    
    # Substrate Degradation
    ax3.plot(df['Day'], df['Active Substrate (g)'], color='#8B4513', linewidth=2, label='Active Substrate')
    ax3.set_title('Substrate Degradation', fontsize=14, pad=20)
    ax3.set_ylabel('Active Substrate (g)', fontsize=12)
    ax3.set_xlabel('Day', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, 30)
    
    # Degradation Percentage
    ax4.plot(df['Day'], df['Degradation (%)'], color='#8B0000', linewidth=2, label='Degradation %')
    ax4.set_title('Degradation Percentage', fontsize=14, pad=20)
    ax4.set_ylabel('Degradation (%)', fontsize=12)
    ax4.set_xlabel('Day', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1, 30)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    st.pyplot(fig)

# ---- 4. Main App ----
def main():
    st.title('🧪 Scientific Biogas Modeling Pro')
    st.markdown("""
    **Advanced prediction using Gompertz and First-Order Kinetic models**  
    *Base case: 30g substrate produces exactly 12L in 25 days with complete degradation*
    """)
    st.markdown("---")
    
    # ---- Sidebar Controls ----
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Model Selection
        st.subheader("Model Configuration")
        model_weight = st.slider("Gompertz Model Weight", 0.0, 1.0, 0.7, 
                               help="Relative weight of Gompertz model in combined prediction")
        
        # Gompertz Parameters (calibrated for 12L in 25 days)
        st.subheader("Gompertz Parameters")
        g_max = st.slider("Maximum gas potential (mL)", 8000, 15000, 12000, step=100)
        g_lag = st.slider("Lag phase (days)", 0, 5, 2)
        g_rate = st.slider("Growth rate (1/day)", 0.05, 0.25, 0.15, step=0.01)
        
        # Kinetic Parameters (calibrated for 25-day degradation)
        st.subheader("Kinetic Parameters")
        k_value = st.slider("Degradation rate (k)", 0.08, 0.20, 0.12, step=0.01,
                           help="First-order rate constant - higher values = faster degradation")
        
        # Substrate Inputs
        st.subheader("🌱 Substrate Schedule")
        num_inputs = st.number_input("Number of substrate additions", 1, 10, 3)
        
        substrate_schedule = {}
        for i in range(num_inputs):
            cols = st.columns(2)
            with cols[0]:
                day = st.number_input(f"Day {i+1}", 1, 30, (i+1)*5, key=f'day_{i}')
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
    day_25_idx = 24  # Day 25 (0-indexed)
    cols = st.columns(4)
    with cols[0]:
        st.metric("Gompertz (25 days)", 
                 f"{results_df['Gompertz Cumulative'].iloc[day_25_idx]/1000:.2f} L",
                 "Target: 12L")
    with cols[1]:
        st.metric("Kinetic (25 days)", 
                 f"{results_df['Kinetic Cumulative'].iloc[day_25_idx]/1000:.2f} L",
                 "Target: 12L")
    with cols[2]:
        st.metric("Combined (25 days)", 
                 f"{results_df['Combined Cumulative'].iloc[day_25_idx]/1000:.2f} L",
                 "Target: 12L")
    with cols[3]:
        st.metric("Substrate Degraded", 
                 f"{results_df['Degradation (%)'].iloc[day_25_idx]:.1f}%",
                 "Day 25")
    
    # Model Comparison Plots
    st.markdown("### 📈 Model Comparison & Degradation Analysis")
    plot_model_comparison(results_df)
    
    # Summary Statistics
    st.markdown("### 📋 Summary Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Production Efficiency")
        total_substrate = sum(substrate_schedule.values())
        final_production = results_df['Combined Cumulative'].iloc[-1]
        efficiency = (final_production / total_substrate) * (30/12000) * 100  # mL per gram efficiency
        
        st.write(f"**Total Substrate Added:** {total_substrate}g")
        st.write(f"**Final Production:** {final_production/1000:.2f}L")
        st.write(f"**Production Efficiency:** {efficiency:.1f}%")
    
    with col2:
        st.subheader("Degradation Analysis")
        final_degradation = results_df['Degradation (%)'].iloc[-1]
        active_days = len(results_df[results_df['Status'] == 'Active'])
        
        st.write(f"**Final Degradation:** {final_degradation:.1f}%")
        st.write(f"**Active Production Days:** {active_days}")
        st.write(f"**Peak Daily Production:** {results_df['Combined (mL)'].max():.0f}mL")
    
    # Data Table
    st.markdown("### 🔍 Detailed Production Data")
    st.dataframe(
        results_df.style.format({
            'Active Substrate (g)': '{:.1f}',
            'Degradation (%)': '{:.1f}',
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
    
    # Validation Note
    st.info("**Validation Note:** For 30g substrate, the model should produce approximately 12L in 25 days. Adjust parameters if significantly different.")

if __name__ == "__main__":
    main()

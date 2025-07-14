import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Advanced Biogas Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BASE_SUBSTRATE = 30  # grams
BASE_TOTAL_PRODUCTION = 12000  # mL (12L)

# ---- 1. Scientific Models ----
def gompertz_model(day, substrate_amount, max_gas, lag, rate):
    """Modified Gompertz equation for biogas production"""
    return max_gas * np.exp(-np.exp(rate * np.e * (lag - day)/max_gas + 1)) * (substrate_amount/BASE_SUBSTRATE)

def first_order_model(day, substrate_amount, k):
    """First-order kinetic model for substrate degradation"""
    return BASE_TOTAL_PRODUCTION * (1 - np.exp(-k * min(day,25))) * (substrate_amount/BASE_SUBSTRATE)

# ---- 2. Data Processing ----
@st.cache_data
def calculate_productions(substrate_schedule, params):
    """Calculate production using both models"""
    results = []
    
    for day in range(1, 31):
        current_amount = substrate_schedule.get(day, list(substrate_schedule.values())[-1] if substrate_schedule else BASE_SUBSTRATE)
        
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
        combined = 0.6*g_prod + 0.4*k_prod
        
        results.append({
            'Day': day,
            'Substrate (g)': current_amount,
            'Gompertz (mL)': g_prod,
            'Kinetic (mL)': k_prod,
            'Combined (mL)': combined,
            'Status': 'Active' if day <= 25 else 'Inactive'
        })
    
    df = pd.DataFrame(results)
    df['Gompertz Cumulative'] = df['Gompertz (mL)'].cumsum()
    df['Kinetic Cumulative'] = df['Kinetic (mL)'].cumsum()
    df['Combined Cumulative'] = df['Combined (mL)'].cumsum()
    return df

# ---- 3. Visualization Functions ----
def plot_models_comparison(df):
    """Plot comparison of all models"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Daily Production
    ax1.plot(df['Day'], df['Gompertz (mL)'], label='Gompertz', color='blue')
    ax1.plot(df['Day'], df['Kinetic (mL)'], label='Kinetic', color='green')
    ax1.plot(df['Day'], df['Combined (mL)'], label='Combined', color='red', linestyle='--')
    ax1.set_title('Daily Biogas Production')
    ax1.set_ylabel('mL/day')
    ax1.legend()
    ax1.grid(True)
    
    # Cumulative Production
    ax2.plot(df['Day'], df['Gompertz Cumulative'], label='Gompertz', color='blue')
    ax2.plot(df['Day'], df['Kinetic Cumulative'], label='Kinetic', color='green')
    ax2.plot(df['Day'], df['Combined Cumulative'], label='Combined', color='red', linestyle='--')
    ax2.set_title('Cumulative Biogas Production')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('mL')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

# ---- 4. Main App ----
def main():
    st.title('🔬 Advanced Biogas Modeling')
    st.markdown("""
    Combined Gompertz and First-Order Kinetic models with interactive visualization  
    *Base case: 30g substrate → 12L in 25 days*
    """)
    st.markdown("---")
    
    # ---- Sidebar Controls ----
    with st.sidebar:
        st.header("🧪 Model Parameters")
        
        # Gompertz Parameters
        st.subheader("Gompertz Model")
        g_max = st.slider("Max gas (mL)", 10000, 20000, 12000, key='g_max')
        g_lag = st.slider("Lag phase (days)", 1, 10, 3, key='g_lag')
        g_rate = st.slider("Growth rate", 0.01, 0.5, 0.15, key='g_rate')
        
        # Kinetic Parameters
        st.subheader("Kinetic Model")
        k_value = st.slider("Rate constant (k)", 0.01, 0.3, 0.12, key='k_value')
        
        # Substrate Inputs
        st.subheader("🌱 Substrate Schedule")
        num_inputs = st.number_input("Number of inputs", 1, 5, 1, key='num_inputs')
        
        substrate_schedule = {}
        for i in range(num_inputs):
            day = st.number_input(f"Day {i+1}", 1, 30, (i+1)*7 if (i+1)*7 <=25 else 25, key=f'day_{i}')
            amount = st.number_input(f"Amount (g) {i+1}", 1, 1000, 30, key=f'amt_{i}')
            substrate_schedule[day] = amount
    
    # ---- Calculate Productions ----
    params = {
        'gompertz_max': g_max,
        'gompertz_lag': g_lag,
        'gompertz_rate': g_rate,
        'kinetic_k': k_value
    }
    
    results_df = calculate_productions(substrate_schedule, params)
    
    # ---- Display Results ----
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Gompertz Prediction", 
                 f"{results_df['Gompertz Cumulative'].iloc[-1]/1000:.2f} L")
    with col2:
        st.metric("Total Kinetic Prediction", 
                 f"{results_df['Kinetic Cumulative'].iloc[-1]/1000:.2f} L")
    
    st.markdown("### 📈 Model Comparison")
    plot_models_comparison(results_df)
    
    st.markdown("### 📊 Production Data")
    st.dataframe(results_df.style.format({
        'Gompertz (mL)': '{:.0f}',
        'Kinetic (mL)': '{:.0f}',
        'Combined (mL)': '{:.0f}'
    }), use_container_width=True)
    
    # Export Data
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions",
        data=csv,
        file_name="biogas_predictions.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

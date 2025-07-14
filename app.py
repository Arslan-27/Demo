import streamlit as st
import pandas as pd
import numpy as np
import math

# Configure the page
st.set_page_config(
    page_title="Dynamic Biogas Predictor",
    page_icon="🔬",
    layout="wide"
)

# ---- 1. Load Base Data ----
@st.cache_data
def load_data():
    # Base data for 30g substrate
    return pd.DataFrame({
        'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'mL_Produced': [400, 600, 800, 1050, 1300, 1550, 1800, 2050, 2300, 2550, 
                        3600, 3700, 3800, 3750, 3400, 3050, 2700, 2350, 2000, 
                        1650, 1300, 950, 700, 400, 100, 0, 0, 0, 0, 0],
        'Percent': [3.3, 5.0, 6.7, 8.8, 10.8, 12.9, 15.0, 17.1, 19.2, 21.3,
                    30.0, 30.8, 31.7, 31.3, 28.3, 25.4, 22.5, 19.6, 16.7,
                    13.8, 10.8, 7.9, 5.8, 3.3, 0.8, 0, 0, 0, 0, 0]
    })

# ---- 2. Mathematical Functions ----
def linear_interpolation(x, x_data, y_data):
    for i in range(len(x_data) - 1):
        if x_data[i] <= x <= x_data[i + 1]:
            slope = (y_data[i + 1] - y_data[i]) / (x_data[i + 1] - x_data[i])
            return y_data[i] + slope * (x - x_data[i])
    return y_data[-1] if x >= x_data[-1] else y_data[0]

# ---- 3. Dynamic Prediction Functions ----
def calculate_dynamic_production(substrate_inputs, base_data):
    """Calculate production with varying substrate amounts"""
    daily_prod = []
    cumulative = 0
    
    # Create substrate schedule (day: amount)
    substrate_schedule = {int(day): amount for day, amount in substrate_inputs.items()}
    
    for day in range(1, 31):
        # Get substrate amount for this day (default to last specified amount)
        substrate = substrate_schedule.get(day, substrate_schedule.get(max(k for k in substrate_schedule.keys() if k <= day), 30))
        
        # Get base production for this day (from 30g data)
        if day <= 25:
            base_prod = base_data[base_data['Day'] == day]['mL_Produced'].values[0]
        else:
            base_prod = 0
        
        # Scale by substrate amount
        scaled_prod = base_prod * (substrate / 30)
        daily_prod.append(scaled_prod)
        cumulative += scaled_prod
    
    return daily_prod, cumulative

# ---- 4. Main App ----
def main():
    st.title('🌱 Dynamic Biogas Production Predictor')
    st.markdown("Predict production with varying substrate amounts over time")
    st.markdown("---")
    
    base_data = load_data()
    
    # ---- Substrate Input Section ----
    st.sidebar.header("Substrate Schedule")
    st.sidebar.markdown("Enter substrate amounts (grams) for specific days:")
    
    substrate_inputs = {}
    cols = st.sidebar.columns(2)
    for i in range(5):  # Allow 5 input pairs
        with cols[0]:
            day = st.number_input(f"Day {i+1}", min_value=1, max_value=30, value=(i+1)*5 if (i+1)*5 <= 25 else 25, key=f"day_{i}")
        with cols[1]:
            amount = st.number_input(f"Amount (g) {i+1}", min_value=1, max_value=1000, value=30*(i+1), key=f"amount_{i}")
        substrate_inputs[day] = amount
    
    # Remove zero/empty inputs
    substrate_inputs = {k: v for k, v in substrate_inputs.items() if v > 0}
    
    # ---- Prediction Section ----
    st.header("Production Prediction")
    prediction_day = st.number_input("Enter day to predict production", min_value=1, max_value=30, value=27)
    
    if st.button("Calculate Production"):
        daily_prod, cumulative = calculate_dynamic_production(substrate_inputs, base_data)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"Day {prediction_day} Production", 
                     f"{daily_prod[prediction_day-1]:.0f} mL",
                     f"{daily_prod[prediction_day-1]/1000:.2f} L")
        with col2:
            st.metric(f"Cumulative to Day {prediction_day}",
                     f"{sum(daily_prod[:prediction_day]):.0f} mL",
                     f"{sum(daily_prod[:prediction_day])/1000:.2f} L")
        
        # Create production dataframe
        prod_df = pd.DataFrame({
            'Day': range(1, 31),
            'Substrate (g)': [substrate_inputs.get(day, substrate_inputs.get(max(k for k in substrate_inputs.keys() if k <= day), 30)) 
                             for day in range(1, 31)],
            'Daily Production (mL)': daily_prod,
            'Cumulative (mL)': [sum(daily_prod[:i]) for i in range(1, 31)]
        })
        
        # Show production table
        with st.expander("View Full Production Schedule"):
            st.dataframe(prod_df)
        
        # Show charts
        st.subheader("Production Charts")
        
        # Daily Production Chart
        st.line_chart(prod_df.set_index('Day')[['Daily Production (mL)']])
        
        # Cumulative Production Chart
        st.line_chart(prod_df.set_index('Day')[['Cumulative (mL)']])

if __name__ == "__main__":
    main()

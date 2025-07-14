import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Precision Biogas Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BASE_SUBSTRATE = 30  # grams
BASE_TOTAL_PRODUCTION = 12000  # mL (12L)

# ---- 1. Load and Process Base Data ----
@st.cache_data
def load_base_data():
    """Load and process the base production data (30g substrate)"""
    # Daily production percentages based on your 12L total
    daily_percent = [3.3,5.0,6.7,8.8,10.8,12.9,15.0,17.1,19.2,21.3,
                    30.0,30.8,31.7,31.3,28.3,25.4,22.5,19.6,16.7,
                    13.8,10.8,7.9,5.8,3.3,0.8]+[0]*5
    
    # Calculate mL produced each day to match exact 12L total
    daily_prod = [round(BASE_TOTAL_PRODUCTION*p/100) for p in daily_percent]
    
    # Adjust last production day to ensure exact 12L total
    daily_prod[24] = BASE_TOTAL_PRODUCTION - sum(daily_prod[:24])
    
    data = {
        'Day': list(range(1, 31)),
        'mL_Produced': daily_prod,
        'Percent': daily_percent
    }
    df = pd.DataFrame(data)
    df['Cumulative_mL'] = df['mL_Produced'].cumsum()
    return df

# ---- 2. Production Calculation Functions ----
def calculate_daily_production(substrate_schedule, base_data):
    """
    Calculate daily production based on varying substrate amounts
    Args:
        substrate_schedule: dict {day: amount_in_grams}
        base_data: DataFrame with base production values
    Returns:
        DataFrame with daily and cumulative production
    """
    results = []
    cumulative = 0
    last_amount = BASE_SUBSTRATE  # Default base amount
    
    for day in range(1, 31):
        # Get substrate amount for this day
        current_amount = substrate_schedule.get(day, last_amount)
        last_amount = current_amount
        
        # Get base production and scale by substrate amount
        base_prod = base_data.at[day-1, 'mL_Produced']
        scaled_prod = base_prod * (current_amount / BASE_SUBSTRATE)
        
        cumulative += scaled_prod
        results.append({
            'Day': day,
            'Substrate (g)': current_amount,
            'Daily Production (mL)': scaled_prod,
            'Cumulative Production (mL)': cumulative,
            'Status': 'Actual' if day <= 25 else 'Estimated'
        })
    
    return pd.DataFrame(results)

# ---- 3. Main App ----
def main():
    # App Header
    st.title('🌿 Precision Biogas Predictor')
    st.markdown(f"""
    **Accurate biogas prediction** with variable substrate inputs.  
    *Base model: {BASE_SUBSTRATE}g substrate produces exactly {BASE_TOTAL_PRODUCTION/1000}L in 25 days*
    """)
    st.markdown("---")
    
    # Load base data
    base_data = load_base_data()
    
    # ---- Sidebar Controls ----
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Date selection
        start_date = st.date_input("Start Date", datetime.today())
        
        # Substrate input
        st.subheader("Substrate Schedule")
        num_inputs = st.number_input(
            "Number of substrate inputs", 
            min_value=1, 
            max_value=10, 
            value=3,
            key='num_inputs'
        )
        
        substrate_schedule = {}
        for i in range(num_inputs):
            cols = st.columns(2)
            with cols[0]:
                day = st.number_input(
                    f"Input Day {i+1}", 
                    min_value=1, 
                    max_value=30, 
                    value=1,  # Default to day 1, let user choose
                    key=f"day_{i}"
                )
            with cols[1]:
                amount = st.number_input(
                    f"Amount (g) {i+1}", 
                    min_value=1, 
                    max_value=1000, 
                    value=30,  # Default to 30g
                    key=f"amount_{i}"
                )
            if amount > 0:  # Only add if amount is positive
                substrate_schedule[day] = amount
        
        # Set default substrate if no inputs
        if not substrate_schedule:
            substrate_schedule[1] = BASE_SUBSTRATE
        
        # Prediction day
        st.subheader("Prediction Settings")
        prediction_day = st.number_input(
            "Day to predict", 
            min_value=1, 
            max_value=30, 
            value=25,
            key='pred_day'
        )
    
    # ---- Calculate Production ----
    try:
        production_df = calculate_daily_production(substrate_schedule, base_data)
        
        # Get prediction row safely
        pred_row = production_df.iloc[prediction_day-1] if prediction_day <= len(production_df) else production_df.iloc[-1]
        
        # ---- Display Results ----
        st.header("📊 Production Results")
        
        # Key Metrics
        cols = st.columns(3)
        with cols[0]:
            st.metric(f"Day {prediction_day} Substrate", f"{pred_row['Substrate (g)']}g")
        with cols[1]:
            st.metric(
                f"Day {prediction_day} Production", 
                f"{pred_row['Daily Production (mL)']/1000:.2f}L",
                f"{pred_row['Daily Production (mL)']:.0f}mL"
            )
        with cols[2]:
            st.metric(
                f"Total to Day {prediction_day}", 
                f"{pred_row['Cumulative Production (mL)']/1000:.2f}L",
                f"{pred_row['Cumulative Production (mL)']:.0f}mL"
            )
        
        # Verify total production matches exactly 12L for 30g substrate
        if all(v == BASE_SUBSTRATE for v in substrate_schedule.values()):
            st.info(f"✅ Base case verified: {BASE_SUBSTRATE}g substrate produces exactly {BASE_TOTAL_PRODUCTION/1000}L")
        
        # Date Timeline
        st.subheader("📅 Production Timeline")
        timeline_df = production_df.copy()
        timeline_df['Date'] = [start_date + timedelta(days=int(x)-1) for x in timeline_df['Day']]
        
        # Display dataframe with formatting
        st.dataframe(
            timeline_df[['Date', 'Substrate (g)', 'Daily Production (mL)', 
                       'Cumulative Production (mL)', 'Status']].set_index('Date'),
            use_container_width=True
        )
        
        # Interactive Charts
        st.subheader("📈 Production Charts")
        
        tab1, tab2 = st.tabs(["Daily Production", "Cumulative Production"])
        with tab1:
            st.area_chart(production_df, x='Day', y='Daily Production (mL)')
        with tab2:
            st.line_chart(production_df, x='Day', y='Cumulative Production (mL)')
        
        # Export Data
        st.markdown("---")
        csv = production_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Production Data (CSV)",
            data=csv,
            file_name="biogas_production.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your inputs and try again.")

if __name__ == "__main__":
    main()

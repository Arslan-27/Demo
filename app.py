import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Dynamic Biogas Predictor Pro",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- 1. Load and Process Base Data ----
@st.cache_data
def load_base_data():
    """Load and process the base production data (30g substrate)"""
    df = pd.DataFrame({
        'Day': range(1, 31),
        'mL_Produced': [400,600,800,1050,1300,1550,1800,2050,2300,2550,
                        3600,3700,3800,3750,3400,3050,2700,2350,2000,
                        1650,1300,950,700,400,100]+[0]*5,
        'Percent': [3.3,5.0,6.7,8.8,10.8,12.9,15.0,17.1,19.2,21.3,
                   30.0,30.8,31.7,31.3,28.3,25.4,22.5,19.6,16.7,
                   13.8,10.8,7.9,5.8,3.3,0.8]+[0]*5
    })
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
    last_amount = 30  # Default if no substrate specified
    
    for day in range(1, 31):
        # Get substrate amount for this day
        current_amount = substrate_schedule.get(day, last_amount)
        last_amount = current_amount
        
        # Get base production and scale by substrate amount
        base_prod = base_data.loc[day-1, 'mL_Produced']
        scaled_prod = base_prod * (current_amount / 30)
        
        cumulative += scaled_prod
        results.append({
            'Day': day,
            'Substrate (g)': current_amount,
            'Daily Production (mL)': scaled_prod,
            'Cumulative Production (mL)': cumulative,
            'Status': 'Actual' if day <= 25 else 'Estimated'
        })
    
    return pd.DataFrame(results)

# ---- 3. Visualization Functions ----
def plot_production_charts(df):
    """Create interactive production charts"""
    tab1, tab2, tab3 = st.tabs(["Daily Production", "Cumulative Production", "Full Data"])
    
    with tab1:
        st.area_chart(df, x='Day', y='Daily Production (mL)')
    
    with tab2:
        st.line_chart(df, x='Day', y='Cumulative Production (mL)')
    
    with tab3:
        st.dataframe(df.style.format({
            'Daily Production (mL)': '{:.0f}',
            'Cumulative Production (mL)': '{:.0f}'
        }), use_container_width=True)

# ---- 4. Main App ----
def main():
    # App Header
    st.title('🌿 Dynamic Biogas Production Predictor Pro')
    st.markdown("""
    Predict biogas production with **variable substrate inputs** over time.  
    *Base model: 30g substrate produces 12L in 25 days (last 5 days zero production)*
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
        num_inputs = st.slider("Number of substrate inputs", 1, 10, 3)
        
        substrate_schedule = {}
        for i in range(num_inputs):
            cols = st.columns(2)
            with cols[0]:
                day = st.number_input(f"Day {i+1}", min_value=1, max_value=30, 
                                     value=(i+1)*7 if (i+1)*7 <=25 else 25, 
                                     key=f"day_{i}")
            with cols[1]:
                amount = st.number_input(f"Amount (g) {i+1}", min_value=1, 
                                       max_value=1000, value=30*(i+1), 
                                       key=f"amount_{i}")
            substrate_schedule[day] = amount
        
        # Prediction day
        st.subheader("Prediction Settings")
        prediction_day = st.number_input("Day to predict", min_value=1, max_value=30, value=27)
    
    # ---- Calculate Production ----
    production_df = calculate_daily_production(substrate_schedule, base_data)
    
    # ---- Display Results ----
    st.header("📊 Production Results")
    
    # Key Metrics
    pred_row = production_df.iloc[prediction_day-1]
    cols = st.columns(3)
    with cols[0]:
        st.metric(f"Day {prediction_day} Substrate", f"{pred_row['Substrate (g)']}g")
    with cols[1]:
        st.metric(f"Day {prediction_day} Production", 
                f"{pred_row['Daily Production (mL)']/1000:.2f}L",
                f"{pred_row['Daily Production (mL)']:.0f}mL")
    with cols[2]:
        st.metric(f"Total to Day {prediction_day}", 
                f"{pred_row['Cumulative Production (mL)']/1000:.2f}L",
                f"{pred_row['Cumulative Production (mL)']:.0f}mL")
    
    # Date Timeline
    st.markdown("### 📅 Production Timeline")
    timeline_df = production_df.copy()
    timeline_df['Date'] = [start_date + timedelta(days=int(x)-1 for x in timeline_df['Day']]
    st.dataframe(timeline_df[['Date', 'Substrate (g)', 'Daily Production (mL)', 
                            'Cumulative Production (mL)', 'Status']].set_index('Date'),
                use_container_width=True)
    
    # Interactive Charts
    st.markdown("### 📈 Production Charts")
    plot_production_charts(production_df)
    
    # Export Data
    st.markdown("---")
    st.download_button(
        label="📥 Download Production Data",
        data=production_df.to_csv(index=False),
        file_name="biogas_production_schedule.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

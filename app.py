import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib
from datetime import datetime

# Local imports
from pipeline import load_data, preprocess_data, prepare_data_pipeline
from model import DemandOptimizationModel, retrain_model_pipeline

# -----------------------------------------------------------------------------
# Configuration & Theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MLOps Medicine Demand System",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Metrics Styling */
    div[data-testid="metric-container"] {
        background-color: #f7f9fc;
        border: 1px solid #e0e6ed;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
    }
    
    /* Header Enhancements */
    h1, h2, h3 {
        color: #1E3A8A; /* Dark Blue */
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* Add subtle animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stMarkdown, .stDataFrame {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Stateful Variables
# -----------------------------------------------------------------------------
DEFAULT_DATA_PATH = "data/sample_data.csv"

@st.cache_data
def load_default_data():
    if os.path.exists(DEFAULT_DATA_PATH):
        return pd.read_csv(DEFAULT_DATA_PATH)
    return pd.DataFrame()

df = load_default_data()

# Initialize Model instance
model_instance = DemandOptimizationModel()

# Ensure model is trained initially if data exists
if not model_instance.is_trained() and not df.empty:
    with st.spinner("Initializing MLOps Pipeline: Training first model version..."):
        from pipeline import prepare_data_pipeline # re-import locally to avoid early dependency issue
        data = load_data(DEFAULT_DATA_PATH)
        X, y = preprocess_data(data, is_training=True)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        metrics = retrain_model_pipeline(X_train, y_train, X_test, y_test)
    st.success("✅ Baseline Model Trained and Saved Successfully!")

# -----------------------------------------------------------------------------
# Sidebar: Setup & Data Ingestion
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3024/3024310.png", width=60) # placeholder logo
    st.title("Admin Console")
    
    st.header("1. Data Ingestion")
    st.markdown("Upload new historical medicine usage to trigger automated model retraining.")
    
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load new dataframe
            new_df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded!")
            
            if st.button("Trigger Retraining Pipeline", type="primary"):
                # MLOps Auto-Updating Simulation
                with st.spinner("Running CI/CD simulation: Preprocessing & Retraining..."):
                    # 1. Preprocess
                    X_new, y_new = preprocess_data(new_df, is_training=True)
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
                    
                    # 2. Retrain
                    metrics = retrain_model_pipeline(X_train, y_train, X_test, y_test)
                    
                    # update active dataframe for view
                    df = new_df 
                    df.to_csv(DEFAULT_DATA_PATH, index=False) # update default path reference
                    
                st.success(f"Deployed new model version! MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.2f}")
                
        except Exception as e:
            st.error(f"Failed to process file: {e}")

    st.divider()
    st.header("2. System Status")
    if model_instance.is_trained():
        st.success("🟢 Model: Active")
        metrics_file = os.path.join(model_instance.model_dir, 'metrics.json')
        if os.path.exists(metrics_file):
            try:
                m = pd.read_json(metrics_file, typ='series')
                st.caption(f"Last updated: {m.get('timestamp', 'N/A')[:10]}")
                st.caption(f"R² Score: {m.get('r2', 0):.3f}")
            except:
                pass
    else:
        st.error("🔴 Model: Offline")

# -----------------------------------------------------------------------------
# Main Dashboard
# -----------------------------------------------------------------------------
st.title("💊 Medicine Demand & Supply Dashboard")
st.markdown("Predictive analytics powered by MLOps for optimal inventory health in rural clinics.")

if df.empty:
    st.warning("No data found. Please run `generate_data.py` or upload a dataset in the sidebar.")
    st.stop()

# --- Top Level Metrics ---
col1, col2, col3, col4 = st.columns(4)
total_records = len(df)
total_medtypes = df['Medicine_Name'].nunique()
avg_demand = df['Actual_Demand'].mean()
recent_temp = df['Temperature'].iloc[-1]

col1.metric("Database Records", f"{total_records:,}")
col2.metric("Tracked Medicines", total_medtypes)
col3.metric("Avg Daily Demand", f"{avg_demand:.1f} units")
col4.metric("Recent Est. Temp", f"{recent_temp:.1f} °C")


# --- Charts Layout ---
st.divider()
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Historical Demand Trends")
    # aggregate by date
    trend_df = df.groupby('Date')['Actual_Demand'].sum().reset_index()
    fig1 = px.line(trend_df, x='Date', y='Actual_Demand', 
                   title="Overall Medicine Consumption",
                   color_discrete_sequence=['#2563EB'])
    fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
                       xaxis_title="", yaxis_title="Units Requested")
    st.plotly_chart(fig1, use_container_width=True)

with row1_col2:
    st.subheader("Demand by Medicine Type")
    med_df = df.groupby('Medicine_Name')['Actual_Demand'].sum().reset_index()
    med_df = med_df.sort_values(by='Actual_Demand', ascending=True)
    fig2 = px.bar(med_df, x='Actual_Demand', y='Medicine_Name', 
                  orientation='h',
                  color='Actual_Demand',
                  color_continuous_scale="Blues",
                  title="Total Units Required")
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       xaxis_title="Units", yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)
    
# -----------------------------------------------------------------------------
# Prediction Section
# -----------------------------------------------------------------------------
st.divider()
st.header("🔮 Demand Forecasting Simulator")
st.markdown("Use the active machine learning model to simulate demand for the upcoming week based on predicted conditions.")

with st.expander("Configure Predictor", expanded=True):
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    sel_med = p_col1.selectbox("Medicine", df['Medicine_Name'].unique())
    sel_season = p_col2.selectbox("Expected Season", df['Season'].unique())
    sel_temp = p_col3.slider("Est. Avg Temp (°C)", min_value=0, max_value=45, value=25)
    sel_pop = p_col4.number_input("Est. Catchment Pop.", min_value=1000, max_value=50000, value=15000, step=1000)

if st.button("Run Prediction Inference", use_container_width=True):
    if not model_instance.is_trained():
        st.error("Model must be trained before predicting.")
    else:
        # Create a tiny dataframe for the prediction
        # (Assuming Previous_Demand is roughly average of recent for this demo)
        recent_med_data = df[df['Medicine_Name'] == sel_med]
        est_prev_demand = recent_med_data['Actual_Demand'].mean() if len(recent_med_data) > 0 else 100
        
        input_data = pd.DataFrame([{
            'Medicine_Name': sel_med,
            'Previous_Demand': est_prev_demand,
            'Season': sel_season,
            'Temperature': sel_temp,
            'Population': sel_pop
        }])
        
        with st.spinner("Model inference triggered..."):
            # Preprocess using saved scalers/encoders
            X_infer, _ = preprocess_data(input_data, is_training=False)
            
            # Predict
            pred_val = model_instance.predict(X_infer)[0]
            pred_val = max(0, int(pred_val))
            
        # Display Result Box
        st.markdown(f"""
        <div style='background-color: #DBEAFE; border-left: 5px solid #2563EB; padding: 20px; border-radius: 5px;'>
            <h3 style='margin:0; color: #1E3A8A;'>Predicted Demand: {pred_val} Units</h3>
            <p style='margin:0; color: #3B82F6;'>Ensure supply chain has at least this amount of <b>{sel_med}</b> routed to the clinic for optimal operations during <b>{sel_season}</b>.</p>
        </div>
        """, unsafe_allow_html=True)

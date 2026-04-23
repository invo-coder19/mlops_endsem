import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import time

from pipeline import load_data, preprocess_data_for_model, DISTRICTS, MEDICINES
from model import train_model, predict_demand

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Medicine Inventory Manager",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
def apply_custom_css():
    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: 'Inter', 'Roboto', sans-serif;
        }
        
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #1e293b;
            color: white;
        }
        
        /* Sidebar text */
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
            color: #e2e8f0 !important;
        }
        
        /* Headers and text */
        h1, h2, h3, h4, h5, h6, .stMarkdown p strong {
            color: #f8fafc !important;
            font-weight: 700;
        }
        
        .stMarkdown p, .stMarkdown span {
            color: #cbd5e1;
        }
        
        /* Cards / Metrics */
        [data-testid="metric-container"] {
            background-color: #1e293b;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border-left: 5px solid #3b82f6;
            transition: transform 0.2s ease;
        }
        
        [data-testid="metric-container"]:hover {
            transform: translateY(-2px);
        }
        
        /* Metric value */
        [data-testid="stMetricValue"] {
            color: #f8fafc !important;
        }
        
        /* Metric label */
        [data-testid="metric-container"] label {
            color: #94a3b8 !important;
            font-weight: 600;
        }
        
        /* Button */
        .stButton>button {
            background-color: #3b82f6;
            color: white;
            border-radius: 6px;
            border: none;
            padding: 10px 20px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        .stButton>button:hover {
            background-color: #2563eb;
            color: white;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 10px;
            padding: 10px 16px;
            font-weight: 600;
            color: #94a3b8;
        }
        
        .stTabs [aria-selected="true"] {
            color: #3b82f6 !important;
        }
        
        /* Overrides to remove default Red/Yellow from alerts */
        div[data-testid="stAlert"] {
            background-color: #1e293b !important;
            color: #f8fafc !important;
            border: 1px solid #3b82f6 !important;
        }
        div[data-testid="stAlert"] svg {
            fill: #3b82f6 !important;
        }
        
        /* Dataframes */
        [data-testid="stDataFrame"] {
            background-color: #1e293b;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# --- APP HEADER ---
st.title("💊 Medicine Inventory Manager")
st.markdown("**With Demand Prediction for Maharashtra**")
st.markdown("---")

# --- DATA LOADING ---
@st.cache_data
def get_data(uploaded_file=None):
    return load_data(uploaded_file)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Inventory CSV", type=["csv"])
    if uploaded_file is not None:
        with st.spinner("Processing new data..."):
            df = get_data(uploaded_file)
            st.session_state['data_updated'] = True
            st.success("Data uploaded successfully!")
    else:
        df = get_data()
        
    st.markdown("---")
    st.header("🔍 Filters")
    
    # Filters
    selected_months = st.multiselect("Select Month(s)", options=sorted(df['Month'].unique()), default=sorted(df['Month'].unique()))
    selected_districts = st.multiselect("Select District(s)", options=sorted(df['District'].unique()), default=sorted(df['District'].unique()))
    selected_medicines = st.multiselect("Select Medicine(s)", options=sorted(df['Medicine Name'].unique()), default=sorted(df['Medicine Name'].unique()))

# Apply filters
filtered_df = df[
    (df['Month'].isin(selected_months)) &
    (df['District'].isin(selected_districts)) &
    (df['Medicine Name'].isin(selected_medicines))
]

if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust your selection.")
    st.stop()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Inventory Dashboard", "📈 Demand Prediction", "🗺️ Maharashtra Map & Insights"])

# ==========================================
# TAB 1: INVENTORY DASHBOARD
# ==========================================
with tab1:
    st.subheader("Inventory Overview")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    total_stock = filtered_df['Quantity'].sum()
    low_stock_threshold = 500
    low_stock_items = filtered_df[filtered_df['Quantity'] < low_stock_threshold].shape[0]
    avg_demand = filtered_df['Demand'].mean()
    
    with col1:
        st.metric(label="Total Stock Available", value=f"{total_stock:,}")
    with col2:
        st.metric(label="Low Stock Alerts (<500)", value=f"{low_stock_items}")
    with col3:
        st.metric(label="Avg Demand", value=f"{avg_demand:.0f}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.markdown("**Medicine-wise Quantity**")
        med_qty_df = filtered_df.groupby('Medicine Name')['Quantity'].sum().reset_index()
        fig_bar = px.bar(
            med_qty_df, 
            x='Medicine Name', 
            y='Quantity',
            color='Quantity',
            color_continuous_scale='Blues',
            text_auto='.2s'
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_chart2:
        st.markdown("**Season-wise Stock**")
        season_qty_df = filtered_df.groupby('Season')['Quantity'].sum().reset_index()
        fig_pie = px.pie(
            season_qty_df, 
            names='Season', 
            values='Quantity',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("**Raw Inventory Data**")
    st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# ==========================================
# TAB 2: DEMAND Prediction (ML MODEL)
# ==========================================
with tab2:
    st.subheader("Machine Learning Model: Demand Prediction")
    
    # Model Training
    st.markdown("Train the model on the current dataset to predict future demand.")
    
    if st.button("Train / Retrain Model"):
        with st.spinner("Training RandomForestRegressor..."):
            time.sleep(1) # Fake loading for effect
            X, y = preprocess_data_for_model(df)
            if X is not None and y is not None:
                model, metrics, y_test, predictions = train_model(X, y)
                st.success("Model trained successfully and saved!")
                st.session_state['metrics'] = metrics
                st.session_state['y_test'] = y_test
                st.session_state['predictions'] = predictions
            else:
                st.error("Failed to preprocess data. Please check dataset columns.")
                
    if 'metrics' in st.session_state:
        st.markdown("### Model Evaluation Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Mean Absolute Error (MAE)", st.session_state['metrics']['MAE'])
        m_col2.metric("Root Mean Squared Error (RMSE)", st.session_state['metrics']['RMSE'])
        m_col3.metric("R² Score", st.session_state['metrics']['R2'])
        
        st.markdown("### Actual vs Predicted Demand (Test Set Sample)")
        # Plotting first 50 points of test set for clarity
        y_test_sample = list(st.session_state['y_test'])[:50]
        pred_sample = list(st.session_state['predictions'])[:50]
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=list(range(len(y_test_sample))), y=y_test_sample, mode='lines+markers', name='Actual', line=dict(color='#1e293b')))
        fig_line.add_trace(go.Scatter(x=list(range(len(pred_sample))), y=pred_sample, mode='lines+markers', name='Predicted', line=dict(color='#3b82f6')))
        fig_line.update_layout(
            xaxis_title="Samples",
            yaxis_title="Demand",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 Interactive Demand Forecaster")
    st.markdown("Predict the demand for all medicines in a specific district and month.")
    
    col_inf1, col_inf2, col_inf3 = st.columns(3)
    with col_inf1:
        inf_district = st.selectbox("Select District", options=list(DISTRICTS.keys()))
    with col_inf2:
        inf_month = st.slider("Select Month", min_value=1, max_value=12, value=5)
    with col_inf3:
        inf_temp = st.slider("Average Temperature (°C)", min_value=10, max_value=45, value=30)
        
    if st.button("Predict Future Demand"):
        # Infer season
        if inf_month in [12, 1, 2]: inf_season = "Winter"
        elif inf_month in [3, 4, 5]: inf_season = "Summer"
        elif inf_month in [6, 7, 8, 9]: inf_season = "Monsoon"
        else: inf_season = "Post-Monsoon"
        
        # Build DataFrame for all medicines
        inf_data = []
        for med in MEDICINES:
            inf_data.append({
                "Month": inf_month,
                "District": inf_district,
                "Medicine Name": med,
                "Season": inf_season,
                "Temperature": inf_temp
            })
        
        inf_df = pd.DataFrame(inf_data)
        
        # Preprocess with get_dummies just like training
        inf_X = pd.get_dummies(inf_df, columns=["District", "Medicine Name", "Season"])
        
        # Predict
        preds = predict_demand(inf_X)
        
        if preds is not None:
            inf_df['Predicted Demand'] = np.round(preds).astype(int)
            
            st.success(f"Predicted demand for {inf_district} in Month {inf_month} ({inf_season})")
            
            res_col1, res_col2 = st.columns([1, 2])
            with res_col1:
                st.dataframe(inf_df[['Medicine Name', 'Predicted Demand']], use_container_width=True, hide_index=True)
            with res_col2:
                fig_inf = px.bar(
                    inf_df, 
                    x='Medicine Name', 
                    y='Predicted Demand',
                    color='Predicted Demand',
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                fig_inf.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(fig_inf, use_container_width=True)
        else:
            st.error("Model not found. Please click 'Train / Retrain Model' first to generate the model file.")

# ==========================================
# TAB 3: MAHARASHTRA MAP & INSIGHTS
# ==========================================
with tab3:
    st.subheader("Regional Needs across Maharashtra")
    
    col_map, col_insights = st.columns([2, 1])
    
    with col_map:
        st.markdown("**District-wise Aggregate Demand**")
        
        # Calculate aggregate demand per district
        district_demand = filtered_df.groupby('District')['Demand'].sum().reset_index()
        
        # Create map centered on Maharashtra
        m = folium.Map(location=[19.7515, 75.7139], zoom_start=6, tiles="CartoDB positron")
        
        for _, row in district_demand.iterrows():
            dist_name = row['District']
            demand_val = row['Demand']
            
            if dist_name in DISTRICTS:
                coords = DISTRICTS[dist_name]
                
                # Dynamic circle size based on demand
                radius = min(max(demand_val / 500, 5), 25)
                
                folium.CircleMarker(
                    location=[coords['lat'], coords['lon']],
                    radius=radius,
                    popup=f"<b>{dist_name}</b><br>Total Demand: {demand_val:,}",
                    tooltip=f"{dist_name}: {demand_val:,}",
                    color="#2563eb",
                    fill=True,
                    fill_color="#3b82f6",
                    fill_opacity=0.7
                ).add_to(m)
                
        # Render Folium map in Streamlit
        st_folium(m, width=700, height=500, returned_objects=[])

    with col_insights:
        st.markdown("**Top Medicines Required**")
        top_meds = filtered_df.groupby('Medicine Name')['Demand'].sum().reset_index()
        top_meds = top_meds.sort_values(by='Demand', ascending=False)
        
        fig_pie2 = px.pie(
            top_meds, 
            names='Medicine Name', 
            values='Demand',
            hole=0.5,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig_pie2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie2, use_container_width=True)
        
        if not top_meds.empty:
            most_needed = top_meds.iloc[0]['Medicine Name']
            st.info(f"**Insight:** **{most_needed}** is currently the most in-demand medicine based on selected filters.")

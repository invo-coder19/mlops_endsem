# Medicine Inventory Manager with Demand Prediction for Maharashtra

A modern, interactive Streamlit application that tracks medicine supply, predicts demand using machine learning (Random Forest), and visualizes regional needs across Maharashtra districts.

## Features
- **Tab 1: Inventory Dashboard**: Total stock, low stock alerts, bar charts, and an interactive dataset view.
- **Tab 2: Demand Prediction**: Train a RandomForestRegressor model on the data to predict demand based on Month, Season, Temperature, and Medicine Name. Shows evaluation metrics and actual vs. predicted charts.
- **Tab 3: Maharashtra Map & Insights**: Uses Folium to display district-wise aggregate demand on an interactive map.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `app.py`: Main Streamlit web application.
- `pipeline.py`: Data generation and preprocessing logic.
- `model.py`: Machine learning model training, saving, and prediction.
- `data/sample_data.csv`: Auto-generated dummy dataset.
- `models/`: Directory where the trained model is saved using joblib.

## Technologies Used
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Plotly
- Folium & Streamlit-Folium

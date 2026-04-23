import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "sample_data.csv")

# Maharashtra districts and their approx coordinates
DISTRICTS = {
    "Mumbai": {"lat": 18.9690, "lon": 72.8205},
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Nashik": {"lat": 19.9975, "lon": 73.7898},
    "Aurangabad": {"lat": 19.8762, "lon": 75.3433},
    "Solapur": {"lat": 17.6599, "lon": 75.9064},
    "Amravati": {"lat": 20.9320, "lon": 77.7523},
    "Kolhapur": {"lat": 16.7050, "lon": 74.2433}
}

MEDICINES = ["Paracetamol", "Amoxicillin", "Ibuprofen", "Cetirizine", "Azithromycin", "Vitamin C"]
SEASONS = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]

def generate_sample_data(num_records=500):
    """Generates synthetic dataset for medicine inventory and demand."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    np.random.seed(42)
    
    months = np.random.randint(1, 13, num_records)
    districts = np.random.choice(list(DISTRICTS.keys()), num_records)
    medicines = np.random.choice(MEDICINES, num_records)
    
    # Assign seasons based on month
    seasons = []
    for m in months:
        if m in [12, 1, 2]: seasons.append("Winter")
        elif m in [3, 4, 5]: seasons.append("Summer")
        elif m in [6, 7, 8, 9]: seasons.append("Monsoon")
        else: seasons.append("Post-Monsoon")
        
    # Generate mock temperatures based on season
    temperatures = []
    for s in seasons:
        if s == "Winter": temperatures.append(np.random.uniform(15, 25))
        elif s == "Summer": temperatures.append(np.random.uniform(30, 42))
        elif s == "Monsoon": temperatures.append(np.random.uniform(25, 32))
        else: temperatures.append(np.random.uniform(22, 30))
        
    # Generate quantity and demand based on some arbitrary logic for correlation
    # E.g., Paracetamol demand is high in Monsoon and Winter
    quantities = np.random.randint(100, 5000, num_records)
    demands = []
    
    for m, s, med in zip(months, seasons, medicines):
        base_demand = np.random.randint(50, 1000)
        if med == "Paracetamol" and s in ["Monsoon", "Winter"]:
            base_demand += np.random.randint(500, 1500)
        if med == "Cetirizine" and s == "Winter":
            base_demand += np.random.randint(300, 800)
        demands.append(base_demand)
        
    df = pd.DataFrame({
        "Month": months,
        "District": districts,
        "Medicine Name": medicines,
        "Quantity": quantities,
        "Demand": demands,
        "Season": seasons,
        "Temperature": np.round(temperatures, 1)
    })
    
    # Sort by month
    df = df.sort_values("Month").reset_index(drop=True)
    
    # Save to file
    df.to_csv(DATA_FILE, index=False)
    return df

def load_data(uploaded_file=None):
    """Loads data from uploaded file or generates sample data if none exists."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Save the new data as the standard sample
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
            df.to_csv(DATA_FILE, index=False)
            return df
        except Exception as e:
            return None
    
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    
    return generate_sample_data()

def preprocess_data_for_model(df):
    """Preprocesses data for the ML model."""
    # Create a copy
    model_df = df.copy()
    
    categorical_cols = ["District", "Medicine Name", "Season"]
    
    # Ensure all categorical columns exist
    for col in categorical_cols:
        if col not in model_df.columns:
            return None, None
            
    # Features (X) and Target (y)
    # We predict Demand. We don't use Quantity for prediction.
    X = model_df.drop(["Demand", "Quantity"], axis=1, errors='ignore') 
    y = model_df["Demand"]
    
    # Get dummies
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X, y

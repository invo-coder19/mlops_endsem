import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np

MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "demand_model.pkl")
FEATURES_FILE = os.path.join(MODEL_DIR, "model_features.pkl")

def train_model(X, y):
    """Trains a RandomForestRegressor model and saves it."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 2)
    }
    
    # Save model and features list to ensure consistency during prediction
    joblib.dump(model, MODEL_FILE)
    joblib.dump(list(X.columns), FEATURES_FILE)
    
    return model, metrics, y_test, predictions

def load_model_and_features():
    """Loads the trained model and feature list."""
    if os.path.exists(MODEL_FILE) and os.path.exists(FEATURES_FILE):
        model = joblib.load(MODEL_FILE)
        features = joblib.load(FEATURES_FILE)
        return model, features
    return None, None

def predict_demand(input_data_df):
    """Predicts demand using the trained model."""
    model, features = load_model_and_features()
    if model is None:
        return None
        
    # Ensure input data has all the required features
    # Missing columns will be filled with 0 (e.g. for missing one-hot categories)
    for feature in features:
        if feature not in input_data_df.columns:
            input_data_df[feature] = 0
            
    # Reorder columns to match training features exactly
    input_data_df = input_data_df[features]
    
    predictions = model.predict(input_data_df)
    return predictions

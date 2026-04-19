import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(file_path):
    """
    Load the dataset from the specified file path.
    Supported format: CSV.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data, is_training=True):
    """
    Preprocess the raw data: Handle missing values, encode categoricals, scale numerical features.
    Saves encoders if is_training=True, otherwise loads them.
    """
    df = data.copy()
    
    # Drop rows where target or key features are missing
    if 'Actual_Demand' in df.columns:
        df = df.dropna(subset=['Actual_Demand'])
    
    # Fill remaining missing with median/mode
    for col in ['Temperature', 'Population', 'Previous_Demand']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    # Features and Target
    if 'Actual_Demand' in df.columns:
        X = df.drop(columns=['Date', 'Actual_Demand'], errors='ignore')
        y = df['Actual_Demand']
    else:
        # Prediction mode
        X = df.drop(columns=['Date'], errors='ignore')
        y = None
        
    os.makedirs('models', exist_ok=True)
    
    # Encoding Categorical Variables
    categorical_cols = ['Medicine_Name', 'Season']
    for col in categorical_cols:
        if col in X.columns:
            encoder_path = f'models/{col}_encoder.joblib'
            if is_training:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                joblib.dump(le, encoder_path)
            else:
                if os.path.exists(encoder_path):
                    le = joblib.load(encoder_path)
                    # Handle unseen classes gracefully
                    X[col] = X[col].map(lambda s: s if s in le.classes_ else '<unknown>')
                    if '<unknown>' in X[col].values:
                        # assign to mode for simplicity in this prototype
                        mode_val = le.classes_[0] 
                        X[col] = X[col].replace('<unknown>', mode_val)
                    X[col] = le.transform(X[col])
                else:
                    print(f"Warning: Encoder for {col} not found. Attempting basic fallback.")
                    X[col] = X[col].astype('category').cat.codes

    # Scaling Numerical Features
    numerical_cols = ['Previous_Demand', 'Temperature', 'Population']
    scaler_path = 'models/scaler.joblib'
    if is_training:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        joblib.dump(scaler, scaler_path)
    else:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X[numerical_cols] = scaler.transform(X[numerical_cols])
            
    return X, y

def prepare_data_pipeline(file_path):
    """
    End-to-End data pipeline.
    """
    print(f"Running pipeline for: {file_path}")
    data = load_data(file_path)
    if data is not None:
        X, y = preprocess_data(data, is_training=True)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    return None, None, None, None

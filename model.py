import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from datetime import datetime

class DemandOptimizationModel:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, 'rf_demand_model.joblib')
        self.metrics_path = os.path.join(self.model_dir, 'metrics.json')
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train(self, X_train, y_train):
        """
        Trains the RandomForest model.
        """
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Save model (Versioning Concept)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
        return self.model
        
    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained model on test data.
        """
        if self.model is None:
            self.load_model()
            
        predictions = self.model.predict(X_test)
        
        # Calculate regression metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'r2': float(r2),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log metrics conceptually for MLOps tracking
        pd.Series(metrics).to_json(self.metrics_path)
        return metrics

    def predict(self, X_new):
        """
        Predicts demand for new data entries.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Model not found. Train the model first.")
            
        # load active model version
        clf = joblib.load(self.model_path)
        return clf.predict(X_new)

    def is_trained(self):
        return os.path.exists(self.model_path)

def retrain_model_pipeline(X_train, y_train, X_test, y_test):
    """
    Simulates a CI/CD retrain trigger.
    """
    dom = DemandOptimizationModel()
    dom.train(X_train, y_train)
    metrics = dom.evaluate(X_test, y_test)
    return metrics

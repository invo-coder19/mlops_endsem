import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_sample_data(num_records=500):
    np.random.seed(42)
    
    # Medicines and base demand parameters
    medicines = ['Paracetamol', 'Amoxicillin', 'Ibuprofen', 'Cetirizine', 'Azithromycin']
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    
    # Generate dates over the last 6 months
    start_date = datetime.now() - timedelta(days=180)
    dates = [start_date + timedelta(days=int(i/(len(medicines)))) for i in range(num_records)]
    
    data = []
    
    for i in range(num_records):
        medicine = np.random.choice(medicines)
        date = dates[i]
        
        # Determine season based on month (simple mapping for synthetic purposes)
        month = date.month
        if month in [12, 1, 2]:
            season = 'Winter'
        elif month in [3, 4, 5]:
            season = 'Spring'
        elif month in [6, 7, 8]:
            season = 'Summer'
        else:
            season = 'Autumn'
            
        # Synthetic features
        temperature = np.random.normal(loc=25 if season in ['Summer', 'Spring'] else 15, scale=5)
        population = np.random.normal(loc=15000, scale=1000) # Rural population
        
        # Previous demand with some noise
        base_demand = np.random.randint(50, 300)
        previous_demand = base_demand + np.random.randint(-20, 20)
        
        # Add a seasonal effect or temperature effect for some medicines
        actual_demand = base_demand
        if medicine == 'Paracetamol' and season == 'Winter':
            actual_demand += 50
        if medicine == 'Cetirizine' and season == 'Spring':
            actual_demand += 40
            
        # Introduce some random fluctuation
        actual_demand += np.random.randint(-30, 30)
        
        # Ensure non-negative
        actual_demand = max(0, actual_demand)
        previous_demand = max(0, previous_demand)
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Medicine_Name': medicine,
            'Previous_Demand': previous_demand,
            'Season': season,
            'Temperature': round(temperature, 2),
            'Population': int(population),
            'Actual_Demand': int(actual_demand)
        })
        
    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    file_path = 'data/sample_data.csv'
    df.to_csv(file_path, index=False)
    print(f'Successfully generated {num_records} records to {file_path}')

if __name__ == '__main__':
    generate_sample_data()

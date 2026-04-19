# MLOps-Based Medicine Demand Prediction and Supply Optimization System

## Project Overview

This project is a modern, interactive Streamlit application designed for rural healthcare environments. It demonstrates how Machine Learning Operations (MLOps) principles can be applied to forecast medicine demand and optimize stock levels to prevent shortages.

### Core Features
- **Interactive Dashboard:** Visualizes total medicines in stock, predicted future demand, and alerts for critical low stock. Features a modern blue, white, and grey color theme with readable typography.
- **Data Input & Auto-Updating Model:** Users can upload a CSV containing historical medicine usage, which triggers the system to automatically retrain the underlying Machine Learning model (Random Forest) and update predictions.
- **Model Versioning:** The active model is saved locally (`.joblib`). This simulates the simple, automated registry typical in MLOps architectures.

## Architecture

The project is structured with modularity in mind:
- `app.py`: The Main Streamlit interface containing the dashboard, data upload, and layout logic.
- `model.py`: Handles model initialization, retraining logic, parameter usage, and prediction capabilities.
- `pipeline.py`: Orchestrates loading and preprocessing the data, preparing it for the trained model.
- `data/sample_data.csv`: A synthetic dataset replicating rural healthcare demand factors including season, temp, population, and previous usage.

## CI/CD Pipeline Integration (Simulation)

In a real-world MLOps implementation, we would use Continuous Integration and Continuous Deployment (CI/CD) pipelines (e.g., GitHub Actions, GitLab CI) to automate the entire lifecycle of this model.

**How it works conceptually:**
1. **Trigger (Source Control/Data Change):** When a new model script is pushed to the repository or new training data is added, a GitHub Actions workflow is triggered.
2. **Build and Test (CI):** The pipeline runs automated unit tests to ensure `pipeline.py` preprocesses data correctly and `model.py` initializes without errors.
3. **Training & Validation:** The pipeline spins up a secure runner, trains the new model on the new dataset, validates performance against baseline metrics, and saves the final artifact (e.g., `model.joblib`).
4. **Deploy (CD):** Once the model is validated, it is packed in a Docker container alongside the Streamlit application and deployed to a cloud server (AWS, Azure, or GCP). 

In this prototype, we simulate this feedback loop: the app triggers an automatic background retraining the moment you upload a new dataset. Future iterations would move this retraining out of the Streamlit application memory and into a deployed background worker.

## Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed.

### Installation

1. Clone this repository (if applicable) or download the files.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

1. Execute the following command from the project root:
   ```bash
   streamlit run app.py
   ```
2. The application will initialize and attempt to train the model using `data/sample_data.csv` if it hasn't been trained already.
3. Check the dashboard for the visualizations and metrics.
4. Try uploading a new dataset in the Data Input section on the sidebar to observe the automated retraining!

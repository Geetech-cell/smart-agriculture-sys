"""
Model Training Script for Smart Agriculture System

This script trains a machine learning model to predict crop yields based on various
environmental and agricultural parameters. The trained model is saved to the artifacts
folder for use in the prediction service.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the src directory to the path so we can import from it
sys.path.append(str(Path(__file__).parent))

# Import scikit-learn components
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the training data.
    
    Args:
        data_path: Path to the training data CSV file
        
    Returns:
        Tuple of (X, y) where X is the feature matrix and y is the target vector
    """
    logger.info(f"Loading data from {data_path}")
    
    # Example data loading - replace this with your actual data loading logic
    # df = pd.read_csv(data_path)
    
    # For demonstration, we'll create synthetic data
    # In a real scenario, replace this with your actual data loading code
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data for demonstration
    data = {
        'region': np.random.choice(['North', 'South', 'East', 'West'], size=n_samples),
        'crop': np.random.choice(['Maize', 'Wheat', 'Rice', 'Soybean'], size=n_samples),
        'season': np.random.choice(['Kharif', 'Rabi', 'Summer'], size=n_samples),
        'avg_temp_c': np.random.normal(25, 5, n_samples).clip(10, 40),
        'avg_rainfall_mm': np.random.gamma(2, 50, n_samples).clip(0, 500),
        'soil_moisture_pct': np.random.uniform(30, 90, n_samples),
        'ndvi': np.random.uniform(0.2, 0.9, n_samples),
        'soil_ph': np.random.uniform(4.5, 8.5, n_samples),
        'pest_pressure_idx': np.random.beta(1, 5, n_samples),
        'fertilizer_kg_per_ha': np.random.uniform(50, 300, n_samples).astype(int),
    }
    
    # Create a simple relationship for the target variable (yield in t/ha)
    # In a real scenario, this would come from your actual data
    df = pd.DataFrame(data)
    df['yield'] = (
        2.0 +  # Base yield
        (df['avg_temp_c'] - 25) * 0.1 +  # Temperature effect
        (df['avg_rainfall_mm'] - 100) * 0.01 +  # Rainfall effect
        (df['soil_moisture_pct'] - 60) * 0.05 +  # Soil moisture effect
        (df['ndvi'] - 0.5) * 2.0 +  # Vegetation index effect
        (df['soil_ph'] - 6.5) * 0.2 +  # Soil pH effect
        -df['pest_pressure_idx'] * 1.5 +  # Pest pressure effect
        (df['fertilizer_kg_per_ha'] / 100) * 0.5  # Fertilizer effect
    )
    
    # Add some random noise
    df['yield'] += np.random.normal(0, 0.5, n_samples)
    
    # Ensure yield is positive
    df['yield'] = df['yield'].clip(0.5, 10.0)
    
    return df

def train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """
    Train a machine learning model on the given data.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Fitted scikit-learn pipeline
    """
    logger.info("Training model...")
    
    # Define feature types
    numeric_features = [
        'avg_temp_c', 'avg_rainfall_mm', 'soil_moisture_pct',
        'ndvi', 'soil_ph', 'pest_pressure_idx', 'fertilizer_kg_per_ha'
    ]
    categorical_features = ['region', 'crop', 'season']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Train the model
    pipeline.fit(X, y)
    
    logger.info("Model training completed")
    return pipeline

def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model pipeline
        X_test: Test feature matrix
        y_test: True target values for test set
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'n_samples': len(X_test)
    }
    
    logger.info(f"Test MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
    return metrics

def save_model(model: Pipeline, output_dir: str = 'artifacts') -> None:
    """
    Save the trained model and metadata to disk.
    
    Args:
        model: Trained model pipeline
        output_dir: Directory to save the model and metadata
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(output_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_path': model_path,
        'model_type': 'RandomForestRegressor',
        'features': [
            'region', 'crop', 'season', 'avg_temp_c', 'avg_rainfall_mm',
            'soil_moisture_pct', 'ndvi', 'soil_ph', 'pest_pressure_idx',
            'fertilizer_kg_per_ha'
        ],
        'target': 'yield',
        'metrics': {
            'training_date': datetime.now().isoformat(),
            'description': 'Initial model trained on synthetic data'
        },
        'version': '1.0.0'
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Main function to run the training pipeline."""
    try:
        # 1. Load and prepare data
        # In a real scenario, replace this with your actual data loading code
        # data_path = 'path/to/your/data.csv'
        # df = load_data(data_path)
        
        # For demonstration, we'll use synthetic data
        df = load_data('synthetic_data')
        
        # Split into features and target
        X = df.drop('yield', axis=1)
        y = df['yield']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 2. Train the model
        model = train_model(X_train, y_train)
        
        # 3. Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)
        
        # 4. Save the model and metadata
        save_model(model)
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

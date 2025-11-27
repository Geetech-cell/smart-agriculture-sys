"""
Test script for the prediction service.
Run this from the project root directory with:
    python -m test_prediction
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict_service import PredictionService, PredictionRequest

def test_prediction():
    """Test the prediction service with sample data."""
    try:
        # Initialize the service (this will load the model)
        service = PredictionService("models/crop_yield_model.pkl")
        
        # Create a test prediction request
        request = PredictionRequest(
            region="Test",
            crop="Maize",
            season="Kharif",
            avg_temp_c=25.0,
            avg_rainfall_mm=150.0,
            soil_moisture_pct=60.0,
            ndvi=0.7,
            soil_ph=6.5,
            pest_pressure_idx=0.3,
            fertilizer_kg_per_ha=120.0
        )
        
        # Make a prediction
        result = service.predict(request)
        print("\nPrediction successful!")
        print(f"Predicted yield: {result['prediction_t_per_ha']} t/ha")
        print(f"Status: {result['status']}")
        
        return True
        
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing prediction service...")
    success = test_prediction()
    sys.exit(0 if success else 1)

"""
Prediction Service Module

This module provides a service for making crop yield predictions using a trained model.
It handles model loading, input validation, and prediction logic.

Classes:
    PredictionRequest: Data class for prediction request parameters
    PredictionService: Main service class for making predictions
"""

from __future__ import annotations
import logging
from typing import Tuple, Optional, Dict, Any, List, Union

import json
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import joblib


@dataclass(frozen=True)
class PredictionRequest:
    """
    Data class representing a prediction request with all required parameters.
    
    Attributes:
        region: The agricultural region (e.g., 'North', 'South')
        crop: The type of crop (e.g., 'Maize', 'Wheat')
        season: The growing season (e.g., 'Kharif', 'Rabi')
        avg_temp_c: Average temperature in Celsius
        avg_rainfall_mm: Average rainfall in millimeters
        soil_moisture_pct: Soil moisture percentage
        ndvi: Normalized Difference Vegetation Index (0-1)
        soil_ph: Soil pH level (0-14)
        pest_pressure_idx: Pest pressure index (0-1)
        fertilizer_kg_per_ha: Fertilizer amount in kg per hectare
    """
    region: str
    crop: str
    season: str
    avg_temp_c: float
    avg_rainfall_mm: float
    soil_moisture_pct: float
    ndvi: float
    soil_ph: float
    pest_pressure_idx: float
    fertilizer_kg_per_ha: float

    def to_frame(self) -> pd.DataFrame:
        """Convert the prediction request to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Single-row DataFrame containing all request parameters
        """
        return pd.DataFrame([asdict(self)])


class PredictionService:
    """
    Service class for making crop yield predictions using a trained model.
    
    This class handles model loading, input validation, and prediction logic.
    It provides a simple interface for making predictions with proper error handling.
    
    Args:
        model_path: Path to the trained model file
    """
    
    def __init__(self, model_path: Union[str, Path]) -> None:
        """
        Initialize the prediction service with a trained model.
        
        Args:
            model_path: Path to the trained model file
            
        Raises:
            FileNotFoundError: If the model file is not found
            RuntimeError: If the model fails to load or validate
        """
        self.model_path = Path(model_path)
        self.pipeline: Any = None
        self._load_model()
        logger.info(f"Successfully initialized PredictionService with model: {model_path}")
    
    def _load_model(self) -> None:
        """
        Load and validate the machine learning model pipeline.
        
        This method loads the model from disk and performs validation to ensure
        it's properly fitted and ready for making predictions.
        
        Raises:
            ValueError: If the model is not properly initialized or validated
            RuntimeError: If the model fails to load or validate
        """
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.pipeline = joblib.load(self.model_path)
            logger.debug("Model loaded successfully")
            
            # Check if the pipeline has the predict method
            if not hasattr(self.pipeline, 'predict'):
                error_msg = "Model is not properly initialized - missing 'predict' method"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Test prediction with dummy data
            try:
                logger.debug("Running model validation with test data")
                dummy_data = {
                    'region': ['Test'], 'crop': ['Maize'], 'season': ['Kharif'],
                    'avg_temp_c': [25.0], 'avg_rainfall_mm': [150.0],
                    'soil_moisture_pct': [60.0], 'ndvi': [0.7],
                    'soil_ph': [6.5], 'pest_pressure_idx': [0.3],
                    'fertilizer_kg_per_ha': [120.0]
                }
                dummy_df = pd.DataFrame(dummy_data)
                _ = self.pipeline.predict(dummy_df.head(1))
                logger.debug("Model validation successful")
                
            except Exception as e:
                error_msg = f"Model prediction test failed: {str(e)}. The model may not be properly fitted."
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
                
        except Exception as e:
            error_msg = f"Failed to load or validate model: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    def predict(self, request: PredictionRequest) -> Dict[str, Any]:
        """
        Make a yield prediction for the given input parameters.
        
        Args:
            request: A PredictionRequest object containing all required parameters
            
        Returns:
            Dict containing:
                - prediction_t_per_ha: Predicted yield in tonnes per hectare (if successful)
                - status: Either 'success' or 'error'
                - error: Error message if status is 'error'
                
        Raises:
            RuntimeError: If the model is not loaded or another critical error occurs
        """
        logger.info(f"Making prediction for {request.crop} in {request.region} ({request.season})")
        
        if self.pipeline is None:
            error_msg = "Model not loaded. Please check the model path and try again."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            # Convert request to DataFrame
            df = request.to_frame()
            
            # Validate input columns
            expected_columns = [
                'region', 'crop', 'season', 'avg_temp_c', 'avg_rainfall_mm',
                'soil_moisture_pct', 'ndvi', 'soil_ph', 'pest_pressure_idx',
                'fertilizer_kg_per_ha'
            ]
            
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                error_msg = f"Missing required columns in input data: {missing_cols}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Make prediction
            logger.debug("Calling model.predict()")
            prediction = float(self.pipeline.predict(df)[0])
            result = {
                "prediction_t_per_ha": round(prediction, 3),
                "status": "success"
            }
            
            logger.info(f"Prediction successful: {result['prediction_t_per_ha']} t/ha")
            return result
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "error": error_msg,
                "status": "error"
            }

    @classmethod
    def from_metadata(cls, metadata_path: Union[str, Path]) -> 'PredictionService':
        """
        Create a PredictionService instance from a metadata file.
        
        Args:
            metadata_path: Path to the metadata JSON file
            
        Returns:
            A new instance of PredictionService
            
        Raises:
            FileNotFoundError: If the metadata file is not found
            json.JSONDecodeError: If the metadata file is not valid JSON
            KeyError: If the metadata file is missing required fields
        """
        logger.info(f"Loading model from metadata: {metadata_path}")
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            if "model_path" not in meta:
                error_msg = f"Metadata file is missing required field 'model_path': {metadata_path}"
                logger.error(error_msg)
                raise KeyError(error_msg)
                
            return cls(meta["model_path"])
            
        except FileNotFoundError as e:
            error_msg = f"Metadata file not found: {metadata_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in metadata file: {metadata_path}"
            logger.error(error_msg)
            raise json.JSONDecodeError(error_msg, e.doc, e.pos) from e


"""Shared inference helpers used by CLI scripts and the Streamlit app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

try:
    from .model import load_model
except ImportError:  # pragma: no cover - fallback for direct script execution
    from model import load_model  # type: ignore


@dataclass
class PredictionRequest:
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
        return pd.DataFrame([self.__dict__])


class PredictionService:
    """Wrapper around a persisted pipeline for easy predictions."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.pipeline = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load and validate the model pipeline."""
        try:
            self.pipeline = load_model(self.model_path)
            # Check if the pipeline is fitted by looking for the 'model' step
            if hasattr(self.pipeline, 'steps') and 'model' in dict(self.pipeline.steps):
                model_step = dict(self.pipeline.steps)['model']
                if not hasattr(model_step, 'fit'):
                    raise ValueError("Model is not properly initialized - missing 'fit' method")
            else:
                raise ValueError("Pipeline is missing required 'model' step")
        except Exception as e:
            raise RuntimeError(f"Failed to load or validate model: {str(e)}")

    def predict(self, request: PredictionRequest) -> Dict[str, Any]:
        """Make a prediction using the loaded model."""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Please check the model path and try again.")
            
        try:
            df = request.to_frame()
            # Ensure the input data has the expected columns
            expected_columns = [
                'region', 'crop', 'season', 'avg_temp_c', 'avg_rainfall_mm',
                'soil_moisture_pct', 'ndvi', 'soil_ph', 'pest_pressure_idx',
                'fertilizer_kg_per_ha'
            ]
            missing_cols = [col for col in expected_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in input data: {missing_cols}")
                
            prediction = float(self.pipeline.predict(df)[0])
            return {
                "prediction_t_per_ha": round(prediction, 3),
                "status": "success"
            }
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "status": "error"
            }

    @classmethod
    def from_metadata(cls, metadata_path: str | Path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(meta["model_path"])


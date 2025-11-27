import streamlit as st
import json
from pathlib import Path
import logging
from typing import Any, Optional, Dict

# Configure logging
logger = logging.getLogger(__name__)

@st.cache_resource
def load_service(metadata_path: str) -> Any:
    """
    Load the prediction service from metadata or default model path.
    
    Args:
        metadata_path: Path to the metadata JSON file containing model configuration
        
    Returns:
        An initialized PredictionService instance
        
    Raises:
        FileNotFoundError: If the model file is not found
        RuntimeError: If the service fails to initialize
    """
    try:
        # Load metadata and resolve model path
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            model_path_str = metadata.get("model_path", "artifacts/model.joblib")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(
                f"Could not load metadata from {metadata_path}: {str(e)}. Using default model path."
            )
            model_path_str = "artifacts/model.joblib"

        # Handle Windows-style absolute paths from local training (e.g. "D:\\Smart Agriculture System\\artifacts\\model.joblib")
        # On Streamlit Cloud (Linux), we only care about the filename and expect it under the repo's artifacts/ folder.
        raw = str(model_path_str)
        if ":" in raw or "\\" in raw:
            # Strip drive and directories, keep only the filename
            model_path = Path(raw).name
        else:
            model_path = Path(raw)

        # Resolve relative paths relative to the metadata file location
        if not Path(model_path).is_absolute():
            model_path = Path(metadata_path).parent / model_path
        
        logger.info(f"Loading model from {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please ensure the model file exists or train the model first."
            )
        
        # Import here to avoid circular imports
        from src.predict_service import PredictionService
        
        # Initialize and return the service
        service = PredictionService(model_path)
        logger.info("Successfully initialized prediction service")
        return service
        
    except Exception as e:
        error_msg = f"Failed to initialize prediction service: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

# Smart Agriculture System – Technical Report

## 1. System Overview

This project is a **Streamlit-based decision support tool** for agriculture. It combines a pre-trained scikit‑learn pipeline with an interactive dashboard to provide:

- Crop yield prediction
- Crop health analysis
- Historical prediction tracking and export

The focus of this version is a **clean, production-ready app** rather than model training.

---

## 2. Architecture

### 2.1 High-Level Components

- **UI Layer (`app.py`)**  
  - Defines the main Streamlit app and page layout (tabs / sections).  
  - Uses `Theme` and `UI` helpers from `src/ui_components.py` for consistent styling.

- **Service Layer (`src/predict_service.py`, `src/model_loader.py`)**  
  - `PredictionRequest`: dataclass describing model inputs.  
  - `PredictionService`: responsible for loading the trained pipeline and making predictions.  
  - `load_service(metadata_path)`: reads `artifacts/metadata.json`, resolves `model_path`, and instantiates a cached `PredictionService` via Streamlit's `st.cache_resource`.

- **Data & History (`src/data/history.py`, `data/`)**  
  - `PredictionHistory` handles persistence of past predictions (e.g. SQLite / file-based).  
  - `data/predictions.db` stores recorded runs used in the History tab.

- **Weather Integration (`src/api/weather.py`)**  
  - `WeatherAPI` calls the Open‑Meteo API when available, with a robust fallback to **mock weather data**.  
  - `get_weather_for_region(region)` returns a simple dictionary used to render the dashboard weather card and forecast chart.

- **Model Compatibility (`src/data_pipeline.py`)**  
  - Minimal placeholder classes and functions (e.g. `DataPipeline`, `FeatureEngineer`) exist to satisfy imports referenced inside the pickled model (`artifacts/model.joblib`).  
  - These stubs are intentionally lightweight and are **not** used for new training.

---

## 3. Model Details

- **Type:** scikit‑learn pipeline, serialized with `joblib`.  
- **Location:** `artifacts/model.joblib`.  
- **Loading:**
  - `PredictionService._load_model()` calls `joblib.load(model_path)`.  
  - After loading, a **dummy prediction** is executed on synthetic data to verify that the pipeline is fitted and has a working `.predict` method.  
  - Errors in loading or inference are logged and wrapped in a clear `RuntimeError`.

- **Inputs (conceptual):**
  - Region, crop, season
  - Average temperature, rainfall
  - Soil moisture, NDVI, soil pH
  - Pest pressure, fertilizer rate

- **Output:**
  - Predicted yield (e.g. tonnes per hectare) as a single numeric value.  
  - The UI derives additional labels (e.g. "above average", confidence level, harvest window) for display only.

The original training pipeline is **not** part of this repository snapshot; the model is treated as an external artifact.

---

## 4. User Interface Flow

### 4.1 Dashboard (Home)

- Displays key metrics such as:
  - Average yield
  - Soil health index
  - Number of active fields
  - Next harvest countdown
- Uses the weather service to show:
  - Current temperature, wind and humidity
  - 7‑day forecast (min/max temperatures and precipitation) in a Plotly chart.
- Shows a crop distribution donut chart and a small summary list.

### 4.2 Prediction Studio

Implemented via a combination of `app.py` and `src/modules/prediction_studio.py`:

- Organizes inputs into:
  - Crop information (crop, season, region)
  - Environmental factors (temperature, rainfall)
  - Soil conditions (moisture, pH)
  - Management (fertilizer, irrigation, pest control)
- On **Predict**:
  - Constructs a `PredictionRequest`-compatible dict/DataFrame.  
  - Calls the `PredictionService` to obtain the yield.  
  - Displays metrics for predicted yield, confidence and optimal harvest window.
- Additional sections:
  - Bar chart of factor contributions (conceptual, UI-focused).  
  - Text recommendations to improve yield.  
  - Simple line charts comparing last season vs projected yield.  
  - One‑click CSV and text report export for the current prediction.

### 4.3 Crop Health Analysis

- Tabs: **Health Overview**, **Environmental**, **Trends**.  
- Inputs: soil moisture, NDVI, temperature, pest/disease pressure, growth stage.  
- Computes a scalar **health_score** and displays it in a colour‑coded card.  
- Shows:
  - Key indicators (temperature, moisture, NDVI, threats).  
  - Environmental charts (temperature curve, light intensity, nutrient metrics).  
  - 30‑day trends for health score and temperature.
- Recommendation section with:
  - Growth tips
  - Alerts (e.g. low moisture, high pest pressure)
  - Simple management schedule.

### 4.4 History & Export

- Retrieves recent predictions from `PredictionHistory`.  
- Displays them as a DataFrame with date/time, region, crop, season, prediction and confidence.  
- Generates a CSV on the fly using `pandas.DataFrame(...).to_csv(index=False)` and exposes it via `st.download_button`.

---

## 5. Error Handling & Logging

- `src/predict_service.py` and `src/model_loader.py` use Python `logging` to report:
  - Model load attempts and paths
  - Success or failure of initialization
  - Detailed stack traces when model loading or validation fails
- The dummy prediction step during `_load_model()` catches common issues early (e.g. unfitted pipeline, incompatible columns).

---

## 6. Dependencies

Runtime dependencies are defined in `requirements.txt` and include:

- `pandas`, `numpy`
- `scikit-learn`, `joblib`
- `xgboost`, `lightgbm` (used inside the pickled model)
- `streamlit`, `plotly`, `requests`
- `pytest` (optional, for tests)

These versions are intentionally kept reasonably up‑to‑date while remaining compatible with common Python 3.10+ environments.

---

## 7. Extending the System

To plug in a new model:

1. Train a compatible scikit‑learn pipeline in a separate environment.
2. Save it as `model.joblib` and place it under `artifacts/`.
3. Update `artifacts/metadata.json` to point to the new model path if necessary.
4. Ensure any custom classes referenced by the pipeline are importable from `src/` (or adjust the pipeline to use standard components).

To add new UI pages or KPIs:

- Reuse components from `src/ui_components.py` for consistent styling.  
- Add new sections or tabs to `app.py` or create new modules under `src/modules/` and import them into the main app.

This completes the high-level technical description of the current Smart Agriculture System implementation.

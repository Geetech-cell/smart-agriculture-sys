# 🌾 Smart Agriculture System

Interactive Streamlit dashboard for **crop yield prediction**, **crop health analysis**, and **farm analytics** built on a trained scikit-learn pipeline.

The project is now optimized as a **single-page Streamlit app** with:
- A modern dashboard home page
- An enhanced prediction studio
- A crop health analysis page
- History & export tools

---

## 🎯 Project Overview

This app helps farmers and agronomists:
- Estimate expected yield for different crops and regions
- Monitor crop health using environmental and management signals
- Track historical predictions and export them for reporting

Backend logic is implemented in `src/` and exposed through a clean UI in `app.py`.

---

## 🌟 Main Features

- **📊 Dashboard (Home)**
  - Key metrics (avg yield, soil health, active fields, next harvest)
  - Weather forecast with Open‑Meteo / mock data
  - Crop distribution donut chart and mini summary table

- **🔮 Prediction Studio**
  - Structured inputs for crop, season, region, weather, soil and management
  - Yield prediction using a pre-trained scikit‑learn pipeline (`artifacts/model.joblib`)
  - Result cards (yield, confidence, harvest window)
  - Detailed analysis tabs: factors, recommendations, simple history charts

- **🌱 Crop Health Analysis**
  - Tabs for overview, environmental conditions, and trends
  - Adjustable sliders for moisture, NDVI, temperature, pest/disease pressure
  - Computed health score with colour‑coded card and indicators
  - Recommendations and alerts based on current conditions

- **📈 History & Export**
  - Local SQLite / file‑backed prediction history (`src/data/history.py` + `data/predictions.db`)
  - Table view of recent predictions
  - One‑click CSV export from the app

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Web Framework:** Streamlit
- **ML / Data:** scikit‑learn, pandas, numpy, joblib
- **Visualization:** Plotly
- **HTTP:** requests (for weather API)

The model is loaded from `artifacts/model.joblib` using `src/predict_service.PredictionService` via `src/model_loader.load_service`.

---

## 🚀 Getting Started

### 1. Create virtual environment

```bash
python -m venv venv
.\u005cvenv\u005cScriptsactivate  # Windows
# or: source venv/bin/activate        # macOS / Linux
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the app

```bash
python -m streamlit run app.py
```

Then open: `http://localhost:8501`

---

## 📁 Project Structure (Current)

```text
Smart Agriculture System/
├── app.py                 # Main Streamlit application / UI
├── README.md              # Project overview & usage
├── REPORT.md              # Short technical report (architecture & model)
├── requirements.txt       # Runtime dependencies
├── artifacts/
│   ├── model.joblib       # Trained model pipeline
│   └── metadata.json      # Model configuration & path
├── configs/
│   └── default.json       # (Optional) configuration
├── data/
│   ├── predictions.db     # Saved prediction history
│   └── sample_crop_data.csv
├── src/
│   ├── __init__.py
│   ├── api/
│   │   └── weather.py     # Weather helper using Open‑Meteo or mock
│   ├── data/
│   │   └── history.py     # PredictionHistory storage helper
│   ├── data_pipeline.py   # Minimal compatibility stubs for pickled model
│   ├── model_loader.py    # `load_service` wrapper around PredictionService
│   ├── modules/
│   │   ├── prediction_studio.py  # Advanced prediction tab (used by app)
│   │   └── data_explorer.py      # Optional data exploration helpers
│   ├── predict_service.py  # PredictionRequest/PredictionService
│   └── ui_components.py    # Theme + reusable UI helpers
└── test_prediction.py      # Simple script to validate the model service
```

Note: older modules like `alert_store`, `crop_health`, `train.py`, and `tests/` have been removed to keep this project focused on the Streamlit app.

---

## 🔍 Key Modules

- `src/predict_service.py`
  - `PredictionRequest` dataclass encapsulating model inputs.
  - `PredictionService` handles model loading, dummy prediction check and `.predict`.

- `src/model_loader.py`
  - `load_service(metadata_path)` reads `artifacts/metadata.json`, resolves `model_path`, and returns a cached `PredictionService`.

- `src/ui_components.py`
  - `Theme` for light/dark mode and CSS injection.
  - `UI` helper methods: header, cards, metric grids, bar charts, info boxes.

- `src/api/weather.py`
  - `get_weather_for_region(region)` returning real or mock forecast for the dashboard.

- `src/data/history.py`
  - `PredictionHistory` for recording and retrieving past predictions used in the History tab.

---

## 📊 Using the App

- **Dashboard:**
  - View high‑level farm metrics, 7‑day weather forecast, and crop distribution.

- **Prediction:**
  - Select crop, region, season, weather and soil variables, then click **Predict** to get yield.
  - Review detailed factor breakdown and improvement recommendations.

- **Crop Health:**
  - Adjust moisture/NDVI/pest/disease sliders and view health score and recommendations.

- **History:**
  - Browse previous predictions and **Export CSV** with one click.

---

## 📝 Notes

- The current model is loaded from existing artifacts; training code is intentionally omitted from this repo snapshot.
- The app can still be integrated with new models by updating `artifacts/metadata.json` and the referenced `model.joblib`.

For a short technical summary of the architecture and model, see **REPORT.md**.


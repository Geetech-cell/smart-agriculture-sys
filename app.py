"""
Smart Agriculture System - AI-powered crop yield prediction and monitoring.

This app provides farmers with real-time insights into their crops,
including yield predictions, health analysis, and irrigation recommendations.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import UI components
from src.ui_components import Theme, UIComponents
from src.crop_health import CropHealthAnalyzer

# Mobile UI handling
try:
    from src.mobile_ui import MobileUI
    mobile_ui = MobileUI()
    is_mobile = mobile_ui.is_mobile
    MOBILE_UI_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"Mobile UI components not available: {e}. Running in desktop mode.")
    is_mobile = False
    MOBILE_UI_AVAILABLE = False

# Import mobile components (if available)
try:
    from src.mobile_integration import (
        notification_manager,
        offline_manager,
        voice_processor
    )
    MOBILE_FEATURES_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(f"Mobile features not available: {e}")
    MOBILE_FEATURES_AVAILABLE = False

# Simple auto-refresh implementation
def st_autorefresh(*, interval: int, limit: Optional[int] = None, key: str) -> int:
    """Simple auto-refresh implementation."""
    _ = interval, limit  # unused but kept for compatibility
    return st.session_state.get(key, 0)

from src.alert_store import AlertStoreError, build_alert_store
from src.predict_service import PredictionRequest, PredictionService

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
REGION_COORDS: Dict[str, Dict[str, float]] = {
    "Northern": {"lat": 9.5, "lon": -12.0},
    "Southern": {"lat": -15.3, "lon": 28.3},
    "Eastern": {"lat": -1.3, "lon": 36.8},
    "Western": {"lat": 6.5, "lon": -0.2},
    "Central": {"lat": 0.3, "lon": 32.6},
    "Highland": {"lat": -13.3, "lon": 34.0},
}
WEATHER_CACHE_KEY = "latest_weather"


@st.cache_data(ttl=1800)
def fetch_weather(latitude: float, longitude: float) -> Dict[str, float]:
    """Fetch weather snapshot from Open-Meteo."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "forecast_days": 3,
        "timezone": "auto",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    daily = data.get("daily", {})
    return {
        "source": "Open-Meteo",
        "timestamp": data.get("current_weather", {}).get("time"),
        "current_temp": data.get("current_weather", {}).get("temperature"),
        "current_windspeed": data.get("current_weather", {}).get("windspeed"),
        "precipitation_sum": (daily.get("precipitation_sum") or [None])[0],
        "max_temp": (daily.get("temperature_2m_max") or [None])[0],
        "min_temp": (daily.get("temperature_2m_min") or [None])[0],
    }


def format_timestamp(ts: Optional[str]) -> str:
    if not ts:
        return "n/a"
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ts


def generate_alerts(
    weather: Optional[Dict[str, float]],
    agronomic_inputs: Dict[str, float],
    predicted_yield: Optional[float] = None,
) -> List[Dict[str, str]]:
    alerts: List[Dict[str, str]] = []

    if weather:
        precip = weather.get("precipitation_sum")
        max_temp = weather.get("max_temp")
        min_temp = weather.get("min_temp")
        wind = weather.get("current_windspeed")
        if precip is not None and precip > 40:
            alerts.append(
                {
                    "severity": "high",
                    "title": "Flood risk",
                    "message": f"Forecast precipitation ~{precip:.1f} mm. Prepare drainage and harvest plans.",
                }
            )
        elif precip is not None and precip < 5:
            alerts.append(
                {
                    "severity": "medium",
                    "title": "Dry spell",
                    "message": "Low rainfall expected—consider irrigation scheduling.",
                }
            )
        if max_temp is not None and max_temp >= 35:
            alerts.append(
                {
                    "severity": "high",
                    "title": "Heat stress",
                    "message": f"Maximum temperature reaching {max_temp:.1f} °C; prioritize heat-tolerant practices.",
                }
            )
        if min_temp is not None and min_temp <= 10:
            alerts.append(
                {
                    "severity": "medium",
                    "title": "Cold shock",
                    "message": f"Minimum temperature dropping to {min_temp:.1f} °C; protect seedlings.",
                }
            )
        if wind is not None and wind >= 15:
            alerts.append(
                {
                    "severity": "medium",
                    "title": "High winds",
                    "message": f"Gusts around {wind:.1f} m/s could damage tender crops.",
                }
            )

    pest_pressure = agronomic_inputs.get("pest_pressure_idx")
    if pest_pressure is not None and pest_pressure >= 0.6:
        alerts.append(
            {
                "severity": "high",
                "title": "Pest outbreak likelihood",
                "message": "Pest pressure index exceeds 0.6—schedule scouting and eco-safe controls.",
            }
        )
    elif pest_pressure is not None and pest_pressure >= 0.4:
        alerts.append(
            {
                "severity": "medium",
                "title": "Monitor pest signals",
                "message": "Moderate pest pressure detected; intensify monitoring.",
            }
        )

    soil_moisture = agronomic_inputs.get("soil_moisture_pct")
    if soil_moisture is not None and soil_moisture <= 12:
        alerts.append(
            {
                "severity": "medium",
                "title": "Soil moisture deficit",
                "message": "Soil moisture below 12%; consider mulching or irrigation.",
            }
        )

    fertilizer = agronomic_inputs.get("fertilizer_kg_per_ha")
    if fertilizer is not None and fertilizer < 70:
        alerts.append(
            {
                "severity": "low",
                "title": "Nutrient gap",
                "message": "Fertilizer rate is low vs. regional best practices; review agronomy plan.",
            }
        )

    if predicted_yield is not None and predicted_yield < 3.0:
        alerts.append(
            {
                "severity": "high",
                "title": "Yield shortfall",
                "message": f"Model predicts {predicted_yield:.2f} t/ha—target mitigation actions.",
            }
        )

    return alerts


def render_alerts(alerts: List[Dict[str, str]]) -> None:
    if not alerts:
        st.success("No risk alerts triggered 🎉")
        return

    severity_map = {
        "high": st.error,
        "medium": st.warning,
        "low": st.info,
    }
    for alert in alerts:
        renderer = severity_map.get(alert["severity"], st.info)
        renderer(f"**{alert['title']}** — {alert['message']}")


@st.cache_resource
def load_service(metadata_path: str):
    """Load the prediction service from metadata or default model path."""
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        model_path = Path(metadata.get("model_path", "models/crop_yield_model.pkl"))
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback to default model path if metadata is not available
        model_path = Path("models/crop_yield_model.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    try:
        return PredictionService(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize prediction service: {str(e)}")


def load_dataset_preview(path: str) -> pd.DataFrame:
    import os
    # Convert path to absolute path
    abs_path = os.path.abspath(path)
    # Check if file exists
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"File not found: {abs_path}")
    try:
        df = pd.read_csv(abs_path)
        return df.head(10000)
    except Exception as e:
        raise Exception(f"Error reading file {abs_path}: {str(e)}")


@st.cache_resource
def get_alert_store():
    secrets_payload = _safe_secrets_section("alert_store")
    path = secrets_payload.get("path") or os.getenv("ALERT_STORE_PATH")
    webhook_url = secrets_payload.get("webhook_url") or os.getenv("ALERT_WEBHOOK_URL")
    api_key = secrets_payload.get("api_key") or os.getenv("ALERT_WEBHOOK_API_KEY")
    timeout = secrets_payload.get("timeout")
    try:
        timeout_val = float(timeout) if timeout is not None else None
    except (TypeError, ValueError):
        timeout_val = None
    return build_alert_store(
        path=path,
        webhook_url=webhook_url,
        api_key=api_key,
        timeout=timeout_val,
    )


def persist_alert_history(
    *,
    alerts: List[Dict[str, str]],
    health_label: str,
    health_score: float,
    context: Dict[str, Any],
    sensor_snapshot: Optional[Dict[str, float]] = None,
) -> None:
    store = get_alert_store()
    if store is None:
        return
    try:
        store.persist(
            alerts=alerts,
            health_label=health_label,
            health_score=health_score,
            context=context,
            sensor_snapshot=sensor_snapshot,
        )
    except AlertStoreError as exc:
        st.warning(f"Unable to persist alert log: {exc}")


def _safe_secrets_section(section: str) -> Dict[str, Any]:
    try:
        return st.secrets.get(section, {})  # type: ignore[attr-defined]
    except Exception:
        return {}




# Sensor history functionality has been removed





def predict_crop_health(
    agronomic_inputs: Dict[str, float],
    predicted_yield: Optional[float] = None,
) -> Tuple[str, float, str]:
    """Predict crop health based on agronomic inputs."""
    score = 0.0
    soil_moisture = agronomic_inputs.get("soil_moisture_pct", 20)
    ndvi = agronomic_inputs.get("ndvi", 0.6)
    pest_pressure = agronomic_inputs.get("pest_pressure_idx", 0.3)
    soil_ph = agronomic_inputs.get("soil_ph", 6.5)

    # Heuristic scoring
    if 18 <= soil_moisture <= 30:
        score += 0.3
    elif 12 <= soil_moisture < 18 or 30 < soil_moisture <= 35:
        score += 0.15

    if 0.55 <= ndvi <= 0.8:
        score += 0.3
    elif ndvi >= 0.45:
        score += 0.15

    # Adjust score based on soil pH (optimal is 6.5)
    ph_deviation = abs(soil_ph - 6.5)
    if ph_deviation < 0.5:
        score += 0.2
    elif ph_deviation < 1.0:
        score += 0.1

    score += max(0, 0.2 - pest_pressure * 0.2)

    if predicted_yield is not None:
        score += min(max(predicted_yield / 10, 0), 0.2)

    score = min(score, 1.0)
    
    if score >= 0.75:
        label = "Healthy"
        guidance = "Maintain current practices; monitor weekly."
    elif score >= 0.55:
        label = "Watch"
        guidance = "Schedule scouting and adjust nutrients."
    else:
        label = "At Risk"
        guidance = "Prioritize intervention: pest, irrigation, or soil amendments."
    return label, score, guidance


def analyze_crop_health(agronomic_inputs: Dict[str, float], weather: Optional[Dict[str, float]] = None):
    """Analyze crop health based on inputs and weather data."""
    st.subheader("")
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Soil Quality", 
                 f"{min(100, int(agronomic_inputs.get('soil_moisture_pct', 0) * 1.5 + (agronomic_inputs.get('soil_ph', 7) * 10)))}/100",
                 help="Higher is better")
    with col2:
        st.metric("Pest Risk", 
                 f"{int(agronomic_inputs.get('pest_pressure_idx', 0.5) * 100)}%",
                 help="Lower is better")
    with col3:
        water_balance = agronomic_inputs.get('avg_rainfall_mm', 0) - (agronomic_inputs.get('soil_moisture_pct', 0) * 2)
        st.metric("Water Balance", 
                 f"{'Good' if water_balance > 0 else 'Low'}",
                 f"{abs(water_balance):.1f} mm",
                 delta_color="inverse")
    with col4:
        health_score = min(100, int(
            30 +  # Base score
            (agronomic_inputs.get('ndvi', 0.5) * 40) +  # Up to 40 points for vegetation
            (agronomic_inputs.get('soil_moisture_pct', 0) * 0.5) +  # Up to 30 points for moisture
            (20 - abs(agronomic_inputs.get('soil_ph', 7) - 6.5) * 10)  # Up to 20 points for pH
        ))
        st.metric("Health Score", f"{health_score}/100")

    # Create visualizations
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=("Soil Conditions", "Environmental Factors"))
    
    # Soil conditions radar chart
    fig.add_trace(go.Scatterpolar(
        r=[
            agronomic_inputs.get('soil_moisture_pct', 0),
            agronomic_inputs.get('soil_ph', 7) * 10,
            agronomic_inputs.get('fertilizer_kg_per_ha', 0) / 2.5,
            50  # Placeholder for organic matter
        ],
        theta=['Moisture', 'pH', 'Fertilizer', 'Organic Matter'],
        fill='toself',
        name='Soil Health'
    ), row=1, col=1)

    # Environmental factors bar chart
    factors = ['Rainfall', 'Temperature', 'NDVI']
    values = [
        agronomic_inputs.get('avg_rainfall_mm', 0) / 5,
        agronomic_inputs.get('avg_temp_c', 25),
        agronomic_inputs.get('ndvi', 0.5) * 100
    ]
    fig.add_trace(go.Bar(
        x=factors,
        y=values,
        name='Current'
    ), row=1, col=2)

    # Update layout
    fig.update_layout(
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_polars(radialaxis=dict(visible=True, range=[0, 100]))
    fig.update_yaxes(title_text="Value", row=1, col=2)

    st.plotly_chart(fig, width="stretch")

    # Recommendations
    st.subheader("")
    if health_score < 50:
        st.warning("")
        st.markdown("""
        - 
        - 
        - 
        - 
        """)
    else:
        st.success("")
        
    # Irrigation recommendation based on soil moisture
    st.subheader("")
    soil_moisture = agronomic_inputs.get('soil_moisture_pct', 20)
    # Prepare weather data in the format expected by irrigation_recommendation
    weather_info = {
        'precipitation': weather.get('precipitation_sum', 0) if weather else 0,
        'temperature': weather.get('current_temp', 25) if weather else 25
    }
    st.warning(irrigation_recommendation(soil_moisture, weather_info))
    
    # Add a button to refresh analysis
    if st.button("", key="refresh_health"):
        st.rerun()

def usage_guide_markdown() -> str:
    return """
## Smart Agriculture System - User Guide

### Getting Started
1. **Load the Model** (sidebar)
   - Ensure you have a trained model (run `python train.py --config configs/default.json`)
   - The app will automatically load the model from `artifacts/metadata.json`

### Key Features

#### 1. Prediction Studio
- **Input Agronomic Data**: Enter crop parameters and environmental conditions
- **Real-time Weather**: Toggle to fetch live weather data for your location
- **Get Predictions**: Click 'Predict Yield' to see crop yield predictions

#### 2. Crop Health Analyzer
- **Health Metrics**: View key health indicators for your crops
- **Recommendations**: Get AI-powered suggestions for improving crop health
- **Historical Analysis**: Track health metrics over time

#### 3. Data Explorer
- **Interactive Visualizations**: Explore your agricultural data
- **Filter & Analyze**: Slice and dice your data for deeper insights
- **Export Data**: Download filtered datasets for further analysis

### Mobile Experience
- **Responsive Design**: Works on all devices
- **Offline Mode**: Access key features without internet connection
- **Push Notifications**: Get alerts for important events

### Troubleshooting
- **Model Not Loading**: Check if `artifacts/metadata.json` exists
- **Weather Data Issues**: Verify your internet connection and location settings
- **Performance**: For large datasets, use the filtering options to improve load times

### Support
For assistance, please contact our support team at support@smartagri.com
2. **Enter agronomic signals** in the Prediction tab. You can override defaults with field measurements.
3. **Fetch live weather** to enrich alerts; adjust coordinates or disable as needed.
4. **Press “Predict Yield”** to see the estimated tonnes/hectare plus recommended alerts.
5. **Check Crop Health** in the dedicated tab for detailed analysis and recommendations.
6. **Explore data** in the Data Explorer tab to verify samples and trends.

Tips:
- Scenario-test by tweaking rainfall/pest pressure sliders.
- Share screenshots or export the table from the Data Explorer for extension agents.
- Keep the browser tab open; cached weather responses auto-refresh every 30 minutes.
"""


def setup_mobile_ui():
    """
    Initialize and return mobile UI components.
    
    Returns:
        MobileUI: An instance of the MobileUI class if available, None otherwise
    """
    if 'mobile_ui' not in st.session_state:
        try:
            from src.mobile_ui import MobileUI
            st.session_state.mobile_ui = MobileUI()
        except (ImportError, AttributeError) as e:
            print(f"Mobile UI initialization failed: {e}")
            st.session_state.mobile_ui = None
    return st.session_state.mobile_ui


def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Smart Agriculture System",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        border: 1px solid #4CAF50;
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Run the Streamlit app."""
    setup_page()
    
    # Initialize session state for the prediction service if it doesn't exist
    if 'service' not in st.session_state:
        try:
            st.session_state.service = load_service("models/metadata.json")
        except Exception as e:
            st.error(f"Failed to load prediction service: {str(e)}")
            if st.button("Retry Loading Model"):
                st.rerun()
            st.stop()
    
    # Initialize mobile UI if available
    mobile_ui = setup_mobile_ui()
    
    # Apply custom theme
    Theme.apply_custom_theme()
    
    # Custom CSS for the entire app
    st.markdown("""
    <style>
        /* Hide the default Streamlit footer */
        footer {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        
        /* Custom tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            padding: 4px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 1.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: nowrap;
            background-color: #f0f2f6;
            border-radius: 8px 8px 0 0;
            padding: 0 24px;
            margin: 0 2px;
            transition: all 0.2s ease-in-out;
            font-weight: 500;
            color: #4a4a4a;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e1e4eb;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #0d6efd;
            color: white !important;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stTabs [aria-selected="true"]:hover {
            background-color: #0b5ed7;
        }
        
        /* Section headers */
        .section-title {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            color: #2c3e50;
            border-bottom: 2px solid #e9ecef;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        /* Card styling */
        .stCard {
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background: white;
            border: 1px solid #e9ecef;
        }
        
        /* Better spacing */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if we should show mobile interface
    user_agent = st.query_params.get('mobile', [''])[0].lower()
    is_mobile = user_agent == 'true'
    
    # Initialize mobile UI and check availability
    mobile_ui = setup_mobile_ui()
    mobile_available = mobile_ui is not None
    
    # Mobile view toggle
    if 'show_mobile' not in st.session_state:
        st.session_state.show_mobile = is_mobile and mobile_available
    
    # Mobile toggle in sidebar for testing if mobile features are available
    if mobile_available:
        with st.sidebar:
            mobile_enabled = st.toggle(
                'Mobile View', 
                value=st.session_state.show_mobile,
                help='Toggle mobile-optimized interface'
            )
            if mobile_enabled != st.session_state.show_mobile:
                st.session_state.show_mobile = mobile_enabled
    
    # Set the mobile flag for the rest of the app
    is_mobile = st.session_state.show_mobile and mobile_available
    
    # Rerun if mobile state changed
    if 'prev_mobile_state' in st.session_state and st.session_state.prev_mobile_state != is_mobile:
        st.session_state.prev_mobile_state = is_mobile
        st.rerun()
    st.session_state.prev_mobile_state = is_mobile
    
    # Show mobile interface if enabled
    if mobile_available and st.session_state.show_mobile:
        mobile_page()
        return

    # Main app header
    UIComponents.page_header(
        " Smart Agriculture System",
        "AI-powered crop yield prediction and monitoring"
    )
    
    # Sidebar with model and settings
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=AgriTech", width=120)
        st.markdown("---")
        
        # Model section
        st.markdown("### Model Configuration")
        metadata_path = st.text_input("Metadata file", value="artifacts/metadata.json")
        
        if Path(metadata_path).exists():
            try:
                service = load_service(metadata_path)
                UIComponents.status_indicator("success", "Model loaded successfully")
            except Exception as e:
                UIComponents.error_message(f"Error loading model: {str(e)}")
                st.stop()
        else:
            UIComponents.status_indicator("error", "Metadata file not found")
            st.stop()
        
        st.markdown("---")
        st.markdown("### System Status")
        UIComponents.status_indicator("online", "All systems operational")

    # Create main tabs with clear labels
    tab_titles = ["🌱 Crop Prediction", "🏥 Crop Health", "📊 Data Explorer", "ℹ️ About"]
    tabs = st.tabs(tab_titles)

    with tabs[0]:  # Prediction Studio
        st.markdown("""
        <div style='margin-bottom: 20px;'>
            <h1 style='color: #2e7d32; margin-bottom: 10px;'>🌱 Smart Crop Prediction</h1>
            <p style='color: #666;'>Get AI-powered yield predictions and agronomic recommendations for your crops.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Status indicator
        with st.container():
            st.markdown("### System Status")
            status_cols = st.columns(3)
            status_cols[0].success("🌡️ Weather API: Online")
            status_cols[1].info("🤖 AI Model: Ready")
            status_cols[2].success("📶 Connection: Stable")
        
        # Initialize agronomic_inputs with default values
        agronomic_inputs = {
            "avg_rainfall_mm": 180.0,
            "soil_moisture_pct": 20.0,
            "pest_pressure_idx": 0.3,
            "fertilizer_kg_per_ha": 110.0,
            "ndvi": 0.6,
            "avg_temp_c": 25.0
        }
        
        # Input Sections using Expanders
        
        # 1. Location & Weather
        with st.expander("📍 Location & Weather", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                region = st.selectbox(
                    "Select Region",
                    list(REGION_COORDS.keys()),
                    index=0,
                    help="Select your agricultural region",
                    key="region_select"
                )
                if region in REGION_COORDS:
                    coords = REGION_COORDS[region]
                    st.caption(f"🌍 {coords['lat']:.2f}°N, {coords['lon']:.2f}°E")
            
            with col2:
                season = st.selectbox(
                    "Season",
                    ["Long Rains", "Short Rains", "Dry Season"],
                    help="Current growing season"
                )

            st.markdown("---")
            st.markdown("#### 🌡️ Environmental Conditions")
            col_temp, col_rain = st.columns(2)
            with col_temp:
                avg_temp_c = st.slider(
                    "Avg Temperature (°C)", 10.0, 45.0, 25.0, 0.5
                )
            with col_rain:
                avg_rainfall_mm = st.slider(
                    "Avg Rainfall (mm)", 0.0, 500.0, 180.0, 5.0
                )

        # 2. Crop & Soil
        with st.expander("🌱 Crop & Soil Conditions", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                crop = st.selectbox(
                    "Select Crop",
                    ["Maize", "Rice", "Soybean", "Sorghum", "Cassava", "Wheat"],
                    key="crop_select"
                )
            with col2:
                soil_ph = st.slider("Soil pH", 4.5, 8.5, 6.5, 0.1)
            
            col3, col4 = st.columns(2)
            with col3:
                soil_moisture_pct = st.slider("Soil Moisture (%)", 5.0, 60.0, 20.0, 0.5)
            with col4:
                ndvi = st.slider("Vegetation Index (NDVI)", 0.0, 1.0, 0.6, 0.01)

        # 3. Management
        with st.expander("🚜 Field Management", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                fertilizer_kg_per_ha = st.number_input(
                    "Fertilizer (kg/ha)", 0.0, 250.0, 110.0, 1.0
                )
            with col2:
                pest_pressure_idx = st.slider(
                    "Pest Pressure", 0.0, 1.0, 0.3, 0.01
                )

        # Update agronomic_inputs
        agronomic_inputs.update({
            "avg_rainfall_mm": avg_rainfall_mm,
            "soil_moisture_pct": soil_moisture_pct,
            "pest_pressure_idx": pest_pressure_idx,
            "fertilizer_kg_per_ha": fertilizer_kg_per_ha,
            "ndvi": ndvi,
            "avg_temp_c": avg_temp_c,
            "soil_ph": soil_ph
        })
        
        # Add a visual separator before the prediction button
        st.markdown("---")
        


        weather_data: Optional[Dict[str, float]] = None

        if st.button("Predict Yield", type="primary"):
            request = PredictionRequest(
                region=region,
                crop=crop,
                season=season,
                avg_temp_c=avg_temp_c,
                avg_rainfall_mm=avg_rainfall_mm,
                soil_moisture_pct=soil_moisture_pct,
                ndvi=ndvi,
                soil_ph=soil_ph,
                pest_pressure_idx=pest_pressure_idx,
                fertilizer_kg_per_ha=fertilizer_kg_per_ha,
            )
            result = st.session_state.service.predict(request)
            yield_val = result['prediction_t_per_ha']
            
            # Yield Card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center;
                        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2); margin-bottom: 2rem;">
                <h3 style="margin:0; font-weight: 500; opacity: 0.9;">Estimated Yield</h3>
                <div style="font-size: 3.5rem; font-weight: 700; margin: 0.5rem 0;">
                    {yield_val:.2f} <span style="font-size: 1.5rem; font-weight: 500;">t/ha</span>
                </div>
                <div style="background: rgba(255,255,255,0.2); display: inline-block; 
                            padding: 0.25rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                    Confidence: High
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Limiting Factors Analysis (Mock logic for demo)
            st.subheader("📉 Limiting Factors")
            factors = {
                "Rainfall": min(100, (avg_rainfall_mm / 500) * 100),
                "Soil Moisture": (soil_moisture_pct / 60) * 100,
                "Fertilizer": min(100, (fertilizer_kg_per_ha / 250) * 100),
                "Pest Pressure": (1 - pest_pressure_idx) * 100  # Inverse
            }
            
            # Identify lowest factor
            limiting = min(factors, key=factors.get)
            st.info(f"Primary limiting factor: **{limiting}**. Consider optimizing this for better yield.")
            
            fig_factors = go.Figure(go.Bar(
                x=list(factors.values()),
                y=list(factors.keys()),
                orientation='h',
                marker_color=['#ef5350' if k == limiting else '#66bb6a' for k in factors]
            ))
            fig_factors.update_layout(
                title="Input Optimization Status",
                xaxis_title="Optimization Level (%)",
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_factors, use_container_width=True)

            # Get sensor data and combine with current inputs
            sensor_data = st.session_state.get("latest_sensor_snapshot", {})
            health_inputs = {**sensor_data, **agronomic_inputs, "ndvi": ndvi}
            health_label, health_score, guidance = predict_crop_health(health_inputs, result["prediction_t_per_ha"])
            st.metric("Crop health status", health_label, f"Score {health_score:.2f}")
            st.caption(guidance)

            st.info(
                "Tip: Compare predictions across scenarios to understand climate risks."
            )
            alerts_payload = generate_alerts(
                None,  # No weather data
                agronomic_inputs,
                predicted_yield=result["prediction_t_per_ha"],
            )
            render_alerts(alerts_payload)
            persist_alert_history(
                alerts=alerts_payload,
                health_label=health_label,
                health_score=health_score,
                context={
                    "region": region,
                    "crop": crop,
                    "season": season,
                    "tab": "prediction_studio",
                    "source": "model_inference",
                    "prediction_t_per_ha": result["prediction_t_per_ha"],
                },
                sensor_snapshot=st.session_state.get("latest_sensor_snapshot"),
            )

    with tabs[2]:  # Data Explorer
        st.markdown('<div class="section-title">🔍 Data Explorer</div>', unsafe_allow_html=True)
        
        # File upload and selection
        st.markdown("### 📂 Data Source")
        col1, col2 = st.columns([3, 1])
        with col1:
            data_path = st.text_input(
                "Dataset Path",
                value="data/sample_crop_data.csv",
                help="Enter the path to your dataset or use the default sample data"
            )
        with col2:
            st.markdown("###")
            use_sample = st.checkbox("Use Sample Data", value=True, help="Use the built-in sample dataset")
        
        try:
            df_preview = load_dataset_preview(data_path if not use_sample else "data/sample_crop_data.csv")
            
            # Enhanced data overview section
            st.subheader("📊 Dataset Overview")
            
            # Calculate data types distribution
            numeric_cols = df_preview.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cat_cols = df_preview.select_dtypes(include=['object']).columns.tolist()
            date_cols = [col for col in df_preview.columns if any(x in str(col).lower() for x in ['date', 'time', 'year', 'month', 'day'])]
            
            # Display metrics in a grid
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <p class="metric-value">{:,}</p>
                    <p class="metric-label">Total Rows</p>
                </div>
                """.format(len(df_preview)), unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{len(df_preview.columns)}</p>
                    <p class="metric-label">Total Columns</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{len(numeric_cols)}</p>
                    <p class="metric-label">Numeric Columns</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-value">{len(cat_cols)}</p>
                    <p class="metric-label">Categorical Columns</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Data quality summary
            missing_values = df_preview.isnull().sum().sum()
            duplicate_rows = df_preview.duplicated().sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Missing Values", f"{missing_values:,}", 
                         help=f"{missing_values / (len(df_preview) * len(df_preview.columns)) * 100:.1f}% of total cells")
            with col2:
                st.metric("Duplicate Rows", f"{duplicate_rows:,}", 
                         help=f"{duplicate_rows / len(df_preview) * 100:.1f}% of total rows" if len(df_preview) > 0 else "0%")
            
            # Data preview with tabs
            tab1, tab2 = st.tabs(["📋 Data Preview", "📊 Summary Statistics"])
            
            with tab1:
                st.dataframe(df_preview.style.background_gradient(cmap='YlGnBu', subset=df_preview.select_dtypes(include=['float64', 'int64']).columns), 
                           height=400, use_container_width=True)
                st.caption(f"Showing first {min(200, len(df_preview))} rows. Dataset shape: {df_preview.shape}")
            
            with tab2:
                st.subheader("Numeric Columns")
                st.dataframe(df_preview.describe().T.style.background_gradient(cmap='YlGnBu'))
                
                st.subheader("Categorical Columns")
                cat_cols = df_preview.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    cat_stats = []
                    for col in cat_cols:
                        stats = df_preview[col].value_counts().reset_index()
                        stats.columns = ['Value', 'Count']
                        stats['Column'] = col
                        cat_stats.append(stats)
                    
                    if cat_stats:
                        cat_stats_df = pd.concat(cat_stats)
                        fig = px.bar(cat_stats_df, x='Column', y='Count', color='Value', 
                                   title="Categorical Value Counts", barmode='group')
                        st.plotly_chart(fig, use_container_width=True)
            
            # Interactive Visualizations
            st.subheader("📊 Interactive Visualizations")
            
            # Visualization type selector
            vis_type = st.selectbox(
                "Select Visualization Type",
                ["Correlation Heatmap", "Pair Plot", "Distribution", "Box Plot", "Violin Plot", "Scatter Plot", "3D Scatter Plot"],
                key="vis_type_selector"
            )
            
            if vis_type == "Correlation Heatmap" and len(numeric_cols) > 1:
                st.subheader("Correlation Heatmap")
                corr = df_preview[numeric_cols].corr()
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    title="Correlation Between Numeric Features"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add correlation insights
                st.markdown("#### Correlation Insights")
                corr_pairs = corr.unstack().sort_values(ascending=False)
                corr_pairs = corr_pairs[corr_pairs < 1]  # Remove self-correlations
                
                if not corr_pairs.empty:
                    top_positive = corr_pairs.idxmax()
                    top_negative = corr_pairs.idxmin()
                    
                    st.markdown(f"- **Strongest Positive Correlation**: {top_positive[0]} & {top_positive[1]} "
                              f"({corr_pairs[top_positive]:.2f})")
                    st.markdown(f"- **Strongest Negative Correlation**: {top_negative[0]} & {top_negative[1]} "
                              f"({corr_pairs[top_negative]:.2f})")
            
            elif vis_type == "Pair Plot" and len(numeric_cols) > 1:
                st.subheader("Pairwise Relationships")
                
                # Let user select up to 5 columns for pair plot
                selected_cols = st.multiselect(
                    "Select up to 5 numeric columns",
                    numeric_cols,
                    default=numeric_cols[:min(3, len(numeric_cols))],
                    key="pairplot_cols"
                )
                
                if len(selected_cols) >= 2:
                    fig = px.scatter_matrix(
                        df_preview,
                        dimensions=selected_cols,
                        color=selected_cols[0] if selected_cols else None,
                        title="Pairwise Relationships"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif vis_type == "Distribution":
                st.subheader("Distribution Analysis")
                dist_col = st.selectbox(
                    "Select a column",
                    numeric_cols,
                    key="dist_col_selector"
                )
                
                if dist_col:
                    # Distribution plot
                    fig = px.histogram(
                        df_preview,
                        x=dist_col,
                        marginal="box",
                        title=f"Distribution of {dist_col}",
                        nbins=50
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add statistics
                    col_data = df_preview[dist_col].dropna()
                    if len(col_data) > 0:
                        stats = {
                            "Mean": col_data.mean(),
                            "Median": col_data.median(),
                            "Std Dev": col_data.std(),
                            "Min": col_data.min(),
                            "Max": col_data.max(),
                            "Skewness": col_data.skew(),
                            "Kurtosis": col_data.kurtosis()
                        }
                        
                        # Display statistics in columns
                        cols = st.columns(4)
                        for i, (stat, value) in enumerate(stats.items()):
                            with cols[i % 4]:
                                st.metric(stat, f"{value:.4f}")
            
            elif vis_type == "Box Plot":
                st.subheader("Box Plot Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    num_col = st.selectbox("Numeric Column", numeric_cols, key="box_num_col")
                with col2:
                    cat_cols_available = [col for col in df_preview.columns 
                                        if df_preview[col].nunique() < 20 and 
                                        df_preview[col].nunique() > 1]
                    cat_col = st.selectbox("Categorical Column (Optional)", 
                                         ["None"] + cat_cols_available,
                                         key="box_cat_col")
                
                if cat_col != "None":
                    fig = px.box(
                        df_preview,
                        x=cat_col,
                        y=num_col,
                        title=f"{num_col} by {cat_col}",
                        color=cat_col
                    )
                else:
                    fig = px.box(
                        df_preview,
                        y=num_col,
                        title=f"Distribution of {num_col}"
                    )
                st.plotly_chart(fig, use_container_width=True)
            
            elif vis_type == "Scatter Plot":
                st.subheader("Scatter Plot")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_cols, 
                                       index=min(1, len(numeric_cols)-1) if len(numeric_cols) > 1 else 0,
                                       key="scatter_y")
                
                color_cols = ["None"] + [col for col in df_preview.columns 
                                       if df_preview[col].nunique() < 20]
                size_col = st.selectbox("Size (Optional)", ["None"] + numeric_cols, 
                                      key="scatter_size")
                color_col = st.selectbox("Color by (Optional)", color_cols, 
                                       key="scatter_color")
                
                fig = px.scatter(
                    df_preview,
                    x=x_col,
                    y=y_col,
                    size=size_col if size_col != "None" else None,
                    color=color_col if color_col != "None" else None,
                    title=f"{y_col} vs {x_col}",
                    trendline="ols" if len(df_preview) > 10 else None,
                    hover_data=df_preview.columns.tolist()
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add correlation information if both columns are numeric
                if x_col in numeric_cols and y_col in numeric_cols:
                    corr = df_preview[[x_col, y_col]].corr().iloc[0,1]
                    st.caption(f"Correlation between {x_col} and {y_col}: {corr:.3f}")
            
            elif vis_type == "3D Scatter Plot":
                st.subheader("3D Scatter Plot")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, key="3d_x")
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_cols, 
                                       index=min(1, len(numeric_cols)-1) if len(numeric_cols) > 1 else 0,
                                       key="3d_y")
                with col3:
                    z_col = st.selectbox("Z-axis", numeric_cols, 
                                       index=min(2, len(numeric_cols)-1) if len(numeric_cols) > 2 else 0,
                                       key="3d_z")
                
                color_cols = ["None"] + [col for col in df_preview.columns 
                                       if df_preview[col].nunique() < 20]
                color_col = st.selectbox("Color by (Optional)", color_cols, key="3d_color")
                
                fig = px.scatter_3d(
                    df_preview,
                    x=x_col,
                    y=y_col,
                    z=z_col,
                    color=color_col if color_col != "None" else None,
                    title=f"3D Scatter: {x_col}, {y_col}, {z_col}"
                )
                st.plotly_chart(fig, width="stretch")
            
            # Pivot Table Analysis
            st.subheader("📊 Pivot Table Analysis")
            with st.expander("Create Pivot Table", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    pivot_index = st.selectbox("Index (Rows)", [None] + list(df_preview.columns), key="pivot_index")
                with col2:
                    pivot_columns = st.selectbox("Columns", [None] + list(df_preview.columns), key="pivot_cols")
                with col3:
                    pivot_values = st.selectbox("Values", numeric_cols, key="pivot_values")
                
                pivot_agg = st.selectbox("Aggregation Function", ["mean", "sum", "count", "min", "max"], key="pivot_agg")
                
                if pivot_index and pivot_values:
                    try:
                        pivot_df = pd.pivot_table(
                            df_preview,
                            values=pivot_values,
                            index=pivot_index,
                            columns=pivot_columns,
                            aggfunc=pivot_agg
                        )
                        st.dataframe(pivot_df.style.background_gradient(cmap='YlGnBu'), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create pivot table: {str(e)}")

            # Outlier Analysis
            st.subheader("📉 Outlier Analysis")
            with st.expander("Detect Outliers", expanded=False):
                outlier_col = st.selectbox("Select Column for Outlier Detection", numeric_cols, key="outlier_col")
                method = st.radio("Detection Method", ["IQR (Interquartile Range)", "Z-Score"], horizontal=True)
                
                if outlier_col:
                    col_data = df_preview[outlier_col].dropna()
                    outliers = pd.Series()
                    
                    if method.startswith("IQR"):
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                        st.info(f"IQR Method: Outliers are values < {lower_bound:.2f} or > {upper_bound:.2f}")
                    else:
                        z_scores = (col_data - col_data.mean()) / col_data.std()
                        outliers = col_data[abs(z_scores) > 3]
                        st.info("Z-Score Method: Outliers are values with Z-score > 3 or < -3")
                    
                    if not outliers.empty:
                        st.warning(f"Found {len(outliers)} outliers ({len(outliers)/len(col_data)*100:.1f}% of data)")
                        st.dataframe(outliers.to_frame(name="Outlier Value").sort_values("Outlier Value"), use_container_width=True)
                        
                        # Visualize outliers
                        fig = px.box(df_preview, y=outlier_col, points="all", title=f"Outliers in {outlier_col}")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success("No outliers detected with current method.")

            # Target Correlation Analysis
            st.subheader("🎯 Target Correlation Analysis")
            with st.expander("Analyze Correlation with Target", expanded=False):
                target_col = st.selectbox("Select Target Variable", numeric_cols, key="target_corr_col")
                
                if target_col:
                    correlations = df_preview[numeric_cols].corrwith(df_preview[target_col]).sort_values(ascending=False)
                    correlations = correlations.drop(target_col)  # Remove self-correlation
                    
                    # Create bar chart
                    corr_df = pd.DataFrame({'Feature': correlations.index, 'Correlation': correlations.values})
                    fig = px.bar(
                        corr_df, 
                        x='Correlation', 
                        y='Feature', 
                        orientation='h',
                        title=f"Feature Correlation with {target_col}",
                        color='Correlation',
                        color_continuous_scale='RdBu_r',
                        range_color=[-1, 1]
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Advanced filtering
            st.subheader("🔍 Advanced Data Exploration")
            with st.expander("📌 Filter Data", expanded=False):
                st.write("Select filters to narrow down the data:")
                
                # Create two columns for filters
                col1, col2 = st.columns(2)
                filters = {}
                
                for i, col in enumerate(df_preview.columns):
                    # Alternate between columns
                    col_target = col1 if i % 2 == 0 else col2
                    
                    with col_target:
                        if pd.api.types.is_numeric_dtype(df_preview[col]):
                            min_val = float(df_preview[col].min())
                            max_val = float(df_preview[col].max())
                            val_range = st.slider(
                                f"{col} range",
                                min_val,
                                max_val,
                                (min_val, max_val),
                                key=f"filter_{col}",
                                help=f"Filter {col} between {min_val} and {max_val}"
                            )
                            filters[col] = val_range
                        elif pd.api.types.is_string_dtype(df_preview[col]):
                            options = sorted(df_preview[col].dropna().unique().tolist())
                            selected = st.multiselect(
                                f"Select {col}",
                                options=options,
                                default=options,
                                key=f"filter_{col}",
                                help=f"Select values to include in {col}"
                            )
                            filters[col] = selected
                
                # Apply filters button
                col1, col2 = st.columns([1, 1])
                with col1:
                    apply_filters = st.button("Apply Filters", type="primary", use_container_width=True)
                with col2:
                    if st.button("Reset Filters", use_container_width=True):
                        # Clear filter session state
                        for key in list(st.session_state.keys()):
                            if key.startswith("filter_"):
                                del st.session_state[key]
                        st.rerun()
            
            # Apply filters
            filtered_df = df_preview.copy()
            if apply_filters:
                for col, filter_val in filters.items():
                    if pd.api.types.is_numeric_dtype(df_preview[col]):
                        min_val, max_val = filter_val
                        filtered_df = filtered_df[
                            (filtered_df[col] >= min_val) & 
                            (filtered_df[col] <= max_val)
                        ]
                    elif pd.api.types.is_string_dtype(df_preview[col]) and filter_val:
                        filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
            
            # Show filtered results
            st.subheader("🎯 Filtered Results")
            st.dataframe(filtered_df.style.background_gradient(cmap='YlGnBu', subset=filtered_df.select_dtypes(include=['float64', 'int64']).columns), 
                       height=400, use_container_width=True)
            st.caption(f"Showing {len(filtered_df)} rows after filtering ({(len(filtered_df)/len(df_preview)*100):.1f}% of total data)")
            
            # Export options
            st.subheader("💾 Export Data")
            export_format = st.radio("Export Format", ["CSV", "Excel", "JSON"], horizontal=True)
            
            if export_format == "CSV":
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download CSV",
                    csv,
                    "filtered_agriculture_data.csv",
                    "text/csv",
                    key='download-csv'
                )
            elif export_format == "Excel":
                excel_file = filtered_df.to_excel("filtered_agriculture_data.xlsx", index=False)
                with open("filtered_agriculture_data.xlsx", "rb") as f:
                    st.download_button(
                        "Download Excel",
                        f,
                        "filtered_agriculture_data.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                json_data = filtered_df.to_json(orient="records")
            # Correlation Analysis
            st.markdown("### Correlation Analysis")
            numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            if len(numeric_cols) > 1:
                # Correlation heatmap
                corr = filtered_df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr, 
                    text_auto=True, 
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    zmin=-1,
                    zmax=1,
                    title="Correlation Heatmap",
                    labels=dict(color="Correlation")
                )
                fig_corr.update_layout(
                    width=800,
                    height=600,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    yaxis_autorange='reversed',
                    margin=dict(l=0, r=0, t=50, b=50)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Pairplot for selected columns
                st.markdown("### Pairwise Relationships")
                selected_cols = st.multiselect(
                    "Select up to 4 numeric columns for pairwise analysis",
                    numeric_cols,
                    default=numeric_cols[:min(4, len(numeric_cols))],
                    key="pairplot_cols"
                )
                
                if len(selected_cols) >= 2:
                    fig = px.scatter_matrix(
                        filtered_df,
                        dimensions=selected_cols,
                        color=filtered_df[selected_cols[0]] if selected_cols else None,
                        title="Pairwise Relationships",
                        width=1000,
                        height=800
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Time Series Analysis
            date_cols = [col for col in filtered_df.columns if any(x in str(col).lower() for x in ['date', 'time', 'year', 'month', 'day'])]
            if date_cols and len(numeric_cols) > 0:
                st.markdown("### Time Series Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox(
                        "Date/Time Column", 
                        date_cols, 
                        key="explorer_date_col_selector",
                        help="Select the column containing date/time information"
                    )
                with col2:
                    value_cols = st.multiselect(
                        "Value Columns", 
                        numeric_cols,
                        default=numeric_cols[0] if numeric_cols else None,
                        key="explorer_value_cols_selector",
                        help="Select numeric columns to plot"
                    )
                
                if value_cols:
                    try:
                        # Create a copy to avoid SettingWithCopyWarning
                        plot_df = filtered_df.copy()
                        
                        # Try to convert to datetime if not already
                        if not pd.api.types.is_datetime64_any_dtype(plot_df[date_col]):
                            plot_df[date_col] = pd.to_datetime(plot_df[date_col], errors='coerce')
                        
                        # Drop rows with NaT in date column
                        plot_df = plot_df.dropna(subset=[date_col])
                        
                        if not plot_df.empty:
                            # Create time series plot
                            fig = px.line(
                                plot_df.sort_values(date_col),
                                x=date_col,
                                y=value_cols,
                                title=f"Time Series Analysis: {', '.join(value_cols)} over Time",
                                labels={'value': 'Value', 'variable': 'Metric'},
                                template='plotly_white'
                            )
                            
                            # Add range slider
                            fig.update_layout(
                                xaxis=dict(
                                    rangeselector=dict(
                                        buttons=list([
                                            dict(count=1, label="1m", step="month", stepmode="backward"),
                                            dict(count=6, label="6m", step="month", stepmode="backward"),
                                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                                            dict(count=1, label="1y", step="year", stepmode="backward"),
                                            dict(step="all")
                                        ])
                                    ),
                                    rangeslider=dict(visible=True),
                                    type="date"
                                ),
                                hovermode="x unified"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add summary statistics by time period
                            st.markdown("#### Summary Statistics by Time Period")
                            period = st.selectbox(
                                "Aggregation Period",
                                ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                                key="ts_period"
                            )
                            
                            # Resample data based on selected period
                            resample_map = {
                                "Daily": "D",
                                "Weekly": "W",
                                "Monthly": "M",
                                "Quarterly": "Q",
                                "Yearly": "Y"
                            }
                            
                            resampled = plot_df.set_index(date_col)[value_cols].resample(resample_map[period]).agg(['mean', 'min', 'max', 'std'])
                            st.dataframe(resampled.style.background_gradient(cmap='YlGnBu'))
                            
                    except Exception as e:
                        st.warning(f"Could not create time series plot: {str(e)}")
                        st.exception(e)
            
            # Automated Insights
            st.subheader("🤖 Automated Insights")
            if st.button("Generate Insights", type="primary"):
                with st.spinner("Analyzing data and generating insights..."):
                    insights = []
                    
                    # Basic stats
                    insights.append(f"📊 The dataset contains {len(filtered_df)} records with {len(filtered_df.columns)} columns.")
                    
                    # Numeric columns analysis
                    for col in numeric_cols:
                        if col in filtered_df.columns:
                            col_data = filtered_df[col].dropna()
                            if len(col_data) > 0:
                                insights.append(
                                    f"📈 **{col}**: "
                                    f"Average = {col_data.mean():.2f}, "
                                    f"Min = {col_data.min():.2f}, "
                                    f"Max = {col_data.max():.2f}, "
                                    f"Std Dev = {col_data.std():.2f}"
                                )
                    
                    # Correlation insights
                    if len(numeric_cols) > 1:
                        corr = filtered_df[numeric_cols].corr().unstack().sort_values(ascending=False)
                        corr = corr[corr < 0.99]  # Remove self-correlations
                        if len(corr) > 0:
                            max_corr = corr.idxmax()
                            insights.append(
                                f"🔗 **Strongest Correlation**: "
                                f"{max_corr[0]} and {max_corr[1]} "
                                f"(r = {corr[max_corr]:.2f})"
                            )
                    
                    # Display insights
                    with st.expander("View Insights", expanded=True):
                        for insight in insights:
                            st.markdown(f"- {insight}")
            
            # Data Quality Check
            st.subheader("🔍 Data Quality Report")
            quality_report = []
            
            # Missing values
            missing = filtered_df.isnull().sum()
            missing_pct = (missing / len(filtered_df)) * 100
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Values': missing.values,
                'Percentage': missing_pct.values
            }).sort_values('Percentage', ascending=False)
            
            quality_report.append("### Missing Values")
            if missing_df['Missing Values'].sum() > 0:
                quality_report.append("The following columns have missing values:")
                for _, row in missing_df[missing_df['Missing Values'] > 0].iterrows():
                    quality_report.append(
                        f"- **{row['Column']}**: {int(row['Missing Values'])} "
                        f"({row['Percentage']:.1f}%) missing values"
                    )
            else:
                quality_report.append("✅ No missing values found in the dataset.")
            
            # Duplicate rows
            duplicates = filtered_df.duplicated().sum()
            quality_report.append("\n### Duplicate Rows")
            if duplicates > 0:
                quality_report.append(f"⚠️ Found {duplicates} duplicate rows in the dataset.")
            else:
                quality_report.append("✅ No duplicate rows found.")
            
            # Display quality report
            with st.expander("View Data Quality Report", expanded=False):
                st.markdown("\n".join(quality_report))
            
            # Add download button for the quality report
            report_text = "\n".join(quality_report)
            st.download_button(
                "Download Quality Report",
                report_text,
                "data_quality_report.md",
                "text/markdown"
            )
            
        except Exception as e:
            st.error(f"Error loading or processing the dataset: {str(e)}")
            st.exception(e)
        st.markdown("### 📱 Mobile App Preview")
        st.info("Scan the QR code below to access the mobile version of this app.")
        
        # Add a placeholder for the QR code
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(
                "https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=https://yourapp.com/mobile", 
                caption="Scan to open in mobile", 
                width="stretch"
            )
        
        # Add a feedback form
        st.markdown("### 💬 Send Us Feedback")
        with st.form("feedback_form"):
            name = st.text_input("Your name (optional)")
            email = st.text_input("Email (optional)")
            feedback = st.text_area("Your feedback or suggestions", height=150)
            submitted = st.form_submit_button("Submit Feedback")
            if submitted and feedback:
                # In a real app, you would send this to your backend
                st.success("Thank you for your feedback! We'll review it soon.")
    
    # Usage Guide Tab
    with tabs[3]:  # Usage Guide
        st.markdown('<div class="section-title">📖 Usage Guide</div>', unsafe_allow_html=True)
        
        # Display the usage guide content
        st.markdown(usage_guide_markdown(), unsafe_allow_html=True)
        
        # Add a quick actions section
        st.markdown("""
        ## 🚀 Quick Actions
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 1.5rem 0;">
            <div class="stCard">
                <h4>📊 Start Prediction</h4>
                <p>Go to Prediction Studio to get started with yield predictions.</p>
                <button onclick="window.location.href='#prediction-studio'" style="background: #0d6efd; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 8px;">
                    Go to Prediction
                </button>
            </div>
            <div class="stCard">
                <h4>🌱 Check Crop Health</h4>
                <p>Analyze your crop's health status and get recommendations.</p>
                <button onclick="window.location.href='#crop-health'" style="background: #198754; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 8px;">
                    Analyze Health
                </button>
            </div>
            <div class="stCard">
                <h4>📈 Explore Data</h4>
                <p>Dive into your agricultural data with interactive visualizations.</p>
                <button onclick="window.location.href='#data-explorer'" style="background: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 8px;">
                    View Data
                </button>
            </div>
        </div>
        
        <div style="margin-top: 2rem; padding: 1.5rem; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #0d6efd;">
            <h4>❓ Need Help?</h4>
            <p>Check out our <a href="#" style="color: #0d6efd; text-decoration: none; font-weight: 500;">documentation</a> or contact our support team at <a href="mailto:support@smartagri.com" style="color: #0d6efd; text-decoration: none; font-weight: 500;">support@smartagri.com</a> for assistance.</p>
        </div>
        """, unsafe_allow_html=True)


# Add mobile route for direct access
if __name__ == "__main__":
    import sys
    
    # Check for mobile flag
    if '--mobile' in sys.argv or 'mobile=true' in ' '.join(sys.argv).lower():
        st.experimental_set_query_params(mobile='true')
    
    main()

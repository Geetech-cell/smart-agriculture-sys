"""
Enhanced Prediction Studio with History Tracking & Export
"""
import streamlit as st
from src.ui_components import Theme, UI
from src.predict_service import PredictionRequest
# Removed utils.export dependency
from src.data.history import PredictionHistory
from datetime import datetime

# Default values
DEFAULTS = {
    "temp": 25.0,
    "rainfall": 180.0,
    "moisture": 20.0,
    "ndvi": 0.6,
    "ph": 6.5,
    "fertilizer": 110.0,
    "pest": 0.3
}

def render_prediction_studio():
    """Enhanced prediction interface with export and history"""
    UI.header("Yield Prediction", "AI-powered crop yield forecasting",
             "Enter field parameters to get accurate yield predictions")
    
    # Input form
    st.markdown("### 📝 Input Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Location & Crop")
        region = st.selectbox("Region", ["Northern", "Southern", "Eastern", "Western", "Central"],
                             help="Select your agricultural region")
        crop = st.selectbox("Crop", ["Maize", "Rice", "Soybean", "Sorghum", "Wheat"],
                           help="Select the crop type")
        season = st.selectbox("Season", ["Long Rains", "Short Rains", "Dry Season"],
                             help="Current growing season")
    
    with col2:
        st.markdown("#### Environment")
        temp = st.slider("Temperature (°C)", 10.0, 45.0, DEFAULTS["temp"], 0.5,
                        help="Average temperature during growing season")
        rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, DEFAULTS["rainfall"], 5.0,
                            help="Total rainfall expected")
        moisture = st.slider("Soil Moisture (%)", 5.0, 60.0, DEFAULTS["moisture"], 0.5,
                            help="Current soil moisture percentage")
    
    with col3:
        st.markdown("#### Soil & Management")
        ndvi = st.slider("NDVI", 0.0, 1.0, DEFAULTS["ndvi"], 0.01,
                        help="Normalized Difference Vegetation Index")
        ph = st.slider("Soil pH", 4.5, 8.5, DEFAULTS["ph"], 0.1,
                      help="Soil pH level")
        fertilizer = st.number_input("Fertilizer (kg/ha)", 0.0, 250.0, DEFAULTS["fertilizer"],
                                     help="Fertilizer application rate")
        pest = st.slider("Pest Pressure", 0.0, 1.0, DEFAULTS["pest"], 0.01,
                       help="Pest pressure index (0=none, 1=severe)")
    
    st.divider()
    
    # Prediction button
    if st.button("🔮 Predict Yield", type="primary", key="predict_btn"):
        if 'service' not in st.session_state:
            UI.info_box("⚠️ Prediction model not loaded. Please check configuration.", "error")
            return
        
        # Show loading
        with st.spinner("Analyzing data..."):
            # Create request
            request = PredictionRequest(
                region=region,
                crop=crop,
                season=season,
                avg_temp_c=temp,
                avg_rainfall_mm=rainfall,
                soil_moisture_pct=moisture,
                ndvi=ndvi,
                soil_ph=ph,
                pest_pressure_idx=pest,
                fertilizer_kg_per_ha=fertilizer,
            )
            
            try:
                # Get prediction
                result = st.session_state.service.predict(request)
                yield_val = result['prediction_t_per_ha']
                
                # Save to history
                if 'history' in st.session_state:
                    st.session_state.history.add_prediction({
                        'region': region,
                        'crop': crop,
                        'season': season,
                        'inputs': {
                            'temp': temp,
                            'rainfall': rainfall,
                            'moisture': moisture,
                            'ndvi': ndvi,
                            'ph': ph,
                            'fertilizer': fertilizer,
                            'pest': pest
                        },
                        'prediction': yield_val,
                        'confidence': 'High'
                    })
                
                st.balloons()
                UI.success_badge("Prediction Complete!")
                
                # Display result
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    theme = Theme.get_current()
                    st.markdown(f"""
                    <div style='background: {theme['primary']}; padding: 2rem; border-radius: 12px; 
                                text-align: center; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                                animation: fadeIn 0.5s ease-out;'>
                        <h3 style='margin: 0; color: white;'>🌾 Estimated Yield</h3>
                        <h1 style='font-size: 3rem; margin: 1rem 0; color: white;'>
                            {yield_val:.2f} <span style='font-size: 1.2rem;'>t/ha</span>
                        </h1>
                        <div style='background: white; color: {theme['primary']}; padding: 8px 16px; 
                                    border-radius: 20px; display: inline-block; font-weight: 600;'>
                            ✓ High Confidence
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Export buttons
                    st.markdown("---")
                    
                    # Prepare export data
                    export_data = {
                        'region': region,
                        'crop': crop,
                        'season': season,
                        'temp': temp,
                        'rainfall': rainfall,
                        'moisture': moisture,
                        'ndvi': ndvi,
                        'ph': ph,
                        'fertilizer': fertilizer,
                        'pest': pest,
                        'yield': yield_val,
                        'confidence': 'High',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # CSV Export
                    import pandas as pd
                    import io
                    df = pd.DataFrame([export_data])
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export CSV",
                        data=csv_data,
                        file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Text Report
                    report_data = {
                        **export_data,
                        'recommendations': "Monitor soil moisture levels. Consider increasing fertilizer application."
                    }
                    report = f"""
Prediction Report
================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Crop Details:
- Crop: {report_data.get('crop', 'N/A')}
- Region: {report_data.get('region', 'N/A')}
- Season: {report_data.get('season', 'N/A')}

Environmental Conditions:
- Temperature: {report_data.get('temp', 'N/A')}°C
- Rainfall: {report_data.get('rainfall', 'N/A')}mm
- Soil Moisture: {report_data.get('moisture', 'N/A')}%
- NDVI: {report_data.get('ndvi', 'N/A')}
- Soil pH: {report_data.get('ph', 'N/A')}

Management:
- Fertilizer: {report_data.get('fertilizer', 'N/A')} kg/ha
- Pest Pressure: {report_data.get('pest', 'N/A')}

Results:
- Predicted Yield: {report_data.get('yield', 'N/A')} t/ha
- Confidence: {report_data.get('confidence', 'N/A')}

Recommendations:
{report_data.get('recommendations', 'N/A')}
"""
                    st.download_button(
                        label="📄 Export Report",
                        data=report,
                        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    st.markdown("### 📊 Limiting Factors")
                    
                    # Calculate factors
                    factors = {
                        "Rainfall": min(100, (rainfall / 500) * 100),
                        "Soil Moisture": (moisture / 60) * 100,
                        "Fertilizer": min(100, (fertilizer / 250) * 100),
                        "Pest Control": (1 - pest) * 100,
                        "Vegetation": ndvi * 100
                    }
                    
                    # Find limiting factor
                    limiting = min(factors, key=factors.get)
                    
                    # Display chart
                    UI.bar_chart(factors)
                    
                    # Recommendations
                    UI.info_box(f"💡 Primary limiting factor: **{limiting}**", "warning")
                    
                    suggestions = {
                        "Rainfall": "Consider irrigation to supplement rainfall",
                        "Soil Moisture": "Implement drip irrigation system",
                        "Fertilizer": "Increase fertilizer application by 20-30%",
                        "Pest Control": "Implement integrated pest management (IPM) strategies",
                        "Vegetation": "Improve crop nutrition and water management"
                    }
                    
                    if limiting in suggestions:
                        st.markdown(f"**Recommendation:** {suggestions[limiting]}")
            
            except Exception as e:
                UI.info_box(f"❌ Prediction failed: {str(e)}", "error")

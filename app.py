"""
Smart Agriculture System - Professional Edition
With animations, dark mode, history tracking, weather integration, and export functionality
"""
import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

from src.ui_components import Theme, UI
from src.model_loader import load_service
from src.modules.prediction_studio import render_prediction_studio
from src.modules.data_explorer import render_data_explorer
from src.data.history import PredictionHistory
from src.api.weather import get_weather_for_region

# Page config
st.set_page_config(
    page_title="🌾 Smart Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply theme
Theme.apply()

# Initialize services
if 'service' not in st.session_state and Path("artifacts/metadata.json").exists():
    try:
        st.session_state.service = load_service("artifacts/metadata.json")
    except:
        pass

# Initialize history
if 'history' not in st.session_state:
    st.session_state.history = PredictionHistory()

# ============= Pages =============

def home_page():
    """Enhanced dashboard with modern UI and real-time data visualization"""
    # Custom CSS for better UI
    st.markdown("""<style>.metric-card {background: rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); transition: transform 0.3s;}.metric-card:hover {transform: translateY(-5px);}.metric-value {font-size: 24px; font-weight: 700; color: #2E7D32;}.metric-label {font-size: 14px; color: #666; margin-bottom: 5px;}.metric-change {font-size: 12px; color: #4CAF50;}</style>""", unsafe_allow_html=True)
    
    # Header with theme toggle
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("🌾 Farm Dashboard")
        st.caption("Real-time insights and analytics for your agricultural operations")
    with col2:
        if st.button("🌓 Toggle Theme", key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.get('dark_mode', False)
            st.rerun()
    
    # Key metrics in a grid
    metrics = st.columns(4)
    metrics_data = [
        {"label": "Avg Yield", "value": "4.2 t/ha", "change": "+5%", "icon": "📊"},
        {"label": "Soil Health", "value": "85%", "change": "+2%", "icon": "🌱"},
        {"label": "Active Fields", "value": "12", "change": "+1", "icon": "🌍"},
        {"label": "Next Harvest", "value": "14 days", "change": "", "icon": "📅"}
    ]
    
    for i, metric in enumerate(metrics_data):
        with metrics[i]:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">{metric['icon']} {metric['label']}</div><div class="metric-value">{metric['value']}</div><div class="metric-change">{metric['change']}</div></div>""", unsafe_allow_html=True)
    
    # Main content area with charts
    st.markdown("---")
    
    # Top row: Weather and Crop Distribution
    col1, col2 = st.columns([3, 2], gap="medium")
    
    with col1:
        def weather_chart():
            # Get weather data
            if st.button("🔄 Refresh Weather", key="refresh_weather", width='stretch'):
                weather = get_weather_for_region("Central")
                st.session_state.weather_data = weather
            
            if 'weather_data' not in st.session_state:
                weather = get_weather_for_region("Central")
                st.session_state.weather_data = weather
            
            weather = st.session_state.weather_data
            
            # Current weather card
            st.markdown(f"""<div style="background: linear-gradient(135deg, #3b82f6, #1d4ed8); color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px;"><div style="font-size: 14px; opacity: 0.9;">Current Weather</div><div style="display: flex; justify-content: space-between; align-items: center;"><div style="font-size: 36px; font-weight: 700;">{weather['current_temp']}°C</div><div style="text-align: right;"><div>🌬️ {weather['wind_speed']} km/h</div><div>💧 {weather.get('humidity', 'N/A')}%</div></div></div><div style="font-size: 12px; opacity: 0.8; margin-top: 5px;">Source: {weather['source']} • {datetime.now().strftime('%a, %b %d')}</div></div>""", unsafe_allow_html=True)
            
            # Forecast chart
            forecast = weather['forecast']
            fig = go.Figure()
            
            # Add temperature range as a filled area
            fig.add_trace(go.Scatter(
                x=forecast['dates'] + forecast['dates'][::-1],
                y=forecast['temp_max'] + forecast['temp_min'][::-1],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.2)',
                line=dict(width=0),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Add max and min temperature lines
            fig.add_trace(go.Scatter(
                x=forecast['dates'],
                y=forecast['temp_max'],
                name='Max Temp',
                line=dict(color='#ef4444', width=2),
                mode='lines+markers',
                marker=dict(size=6)
            ))
            fig.add_trace(go.Scatter(
                x=forecast['dates'],
                y=forecast['temp_min'],
                name='Min Temp',
                line=dict(color='#3b82f6', width=2),
                mode='lines+markers',
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=250,
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, showline=True, linecolor='#e0e0e0'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', showline=True, linecolor='#e0e0e0'),
                legend=dict(orientation='h', yanchor='bottom', y=1.1, xanchor='center', x=0.5),
                showlegend=True
            )
            
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        
        UI.card(weather_chart, "🌤️ Weather Forecast", "")
    
    with col2:
        def crop_distribution():
            crops = {
                "Maize": 45, 
                "Rice": 30, 
                "Soybean": 15, 
                "Wheat": 10
            }
            
            # Display crop distribution as a donut chart
            fig = px.pie(
                values=list(crops.values()),
                names=list(crops.keys()),
                color_discrete_sequence=px.colors.sequential.Greens,
                hole=0.6
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='%{label}: %{value}%<extra></extra>',
                marker=dict(line=dict(color='#ffffff', width=1))
            )
            
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=300,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                annotations=[dict(
                    text='Crops',
                    x=0.5, y=0.5,
                    font_size=14,
                    showarrow=False
                )]
            )
            
            st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
            
            # Add a mini table with crop details
            st.markdown("### Crop Details")
            for crop, percentage in crops.items():
                st.markdown(f"""<div style="display: flex; justify-content: space-between; margin: 8px 0; padding: 8px; background: rgba(0,0,0,0.02); border-radius: 8px;"><span>{crop}</span><span style="font-weight: 600;">{percentage}%</span></div>""", unsafe_allow_html=True)
        
        UI.card(crop_distribution, "🌱 Crop Distribution", "")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    cols = st.columns(4)
    actions = [
        ("🔮 Predict Yield", 1),
        ("📊 Explore Data", 3),
        ("📈 View History", 5),
        ("⚙️ Settings", 4)
    ]
    for col, (label, tab_idx) in zip(cols, actions):
        with col:
            if st.button(label):
                st.session_state.active_tab = tab_idx
                st.rerun()

def history_page():
    """View prediction history and trends"""
    UI.header("Prediction History", "Track and analyze your prediction history",
             "View past predictions, trends, and performance")
    
    # Get history stats
    stats = st.session_state.history.get_stats()
    
    # Display stats
    UI.metric_grid([
        ("Total Predictions", stats['total_predictions'], "",
         "Total number of predictions made"),
        ("Average Yield", f"{stats['avg_yield']} t/ha", "",
         "Average predicted yield across all predictions"),
        ("Crops Tracked", len(stats['by_crop']), "",
         "Number of different crops predicted")
    ], cols=3)
    
    st.markdown("---")
    
    # Recent predictions
    recent = st.session_state.history.get_recent(limit=20)
    
    if recent:
        st.subheader("📋 Recent Predictions")
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'Date': r['timestamp'][:10],
            'Time': r['timestamp'][11:19],
            'Region': r['region'],
            'Crop': r['crop'],
            'Season': r['season'],
            'Predicted Yield (t/ha)': round(r['prediction'], 2),
            'Confidence': r['confidence']
        } for r in recent])
        
        # Display table
        st.dataframe(
            df,
            height=400,
            hide_index=True
        )
        
        # Export button
        csv_data = pd.DataFrame([{
            'timestamp': r['timestamp'],
            'region': r['region'],
            'crop': r['crop'],
            'season': r['season'],
            'prediction': r['prediction'],
            'confidence': r['confidence']
        } for r in recent]).to_csv(index=False)
        
        st.download_button(
            label="📥 Export History (CSV)",
            data=csv_data,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Trends chart
        if len(recent) > 1:
            st.subheader("📈 Yield Trends")
            trend_df = pd.DataFrame([{
                'Date': datetime.fromisoformat(r['timestamp']),
                'Yield': r['prediction'],
                'Crop': r['crop']
            } for r in recent])
            
            fig = px.line(
                trend_df,
                x='Date',
                y='Yield',
                color='Crop',
                title='Predicted Yield Over Time',
                markers=True
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, width='stretch')
    else:
        UI.info_box("No prediction history yet. Make your first prediction to start tracking!", "info")
    
    # Clear history button
    if st.button("🗑️ Clear All History", type="secondary"):
        if st.checkbox("Are you sure? This cannot be undone."):
            st.session_state.history.clear_all()
            st.success("History cleared!")
            st.rerun()

def settings_page():
    """Enhanced settings with theme toggle"""
    UI.header("Settings", "Configure your preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎨 Appearance")
        
        # Dark mode toggle
        current_mode = st.session_state.get('dark_mode', False)
        dark_mode = st.toggle("Dark Mode", value=current_mode)
        if dark_mode != current_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        st.selectbox("Language", ["English", "Spanish", "French"])
        st.selectbox("Units", ["Metric", "Imperial"])
    
    with col2:
        st.markdown("#### 🔔 Notifications")
        st.checkbox("Email alerts", value=True)
        st.checkbox("SMS notifications")
        st.checkbox("Weather alerts", value=True)
    
    st.markdown("---")
    
    st.markdown("#### 📊 Data Management")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export All Data", width='stretch'):
            UI.success_badge("Data export initiated")
    
    with col2:
        if st.button("🔄 Backup Data", width='stretch'):
            UI.success_badge("Backup created")
    
    with col3:
        if st.button("💾 Save Settings", type="primary"):
            st.success("✓ Settings saved successfully!")

def health_page():
    """Enhanced Crop Health Analysis with modern UI"""
    UI.header("🌱 Crop Health Analysis", "Monitor and optimize crop conditions", 
             "Analyze environmental factors affecting crop health")
    
    # Create tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["📊 Health Overview", "🌦️ Environmental", "📈 Trends"])
    
    with tab1:
        # Input parameters in a card
        with st.expander("⚙️ Adjust Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### 🌡️ Environmental")
                moisture = st.slider("Soil Moisture (%)", 0, 100, 65, key="moisture")
                temp = st.slider("Temperature (°C)", 0, 50, 24, key="temp")
            with col2:
                st.markdown("#### 🌱 Plant Health")
                ndvi = st.slider("NDVI", 0.0, 1.0, 0.75, 0.01, key="ndvi")
                growth = st.selectbox("Growth Stage", ["Seedling", "Vegetative", "Flowering", "Maturity"], 
                                   index=1, key="growth_stage")
            with col3:
                st.markdown("#### 🐛 Threats")
                pest = st.slider("Pest Pressure", 0.0, 1.0, 0.2, 0.05, key="pest")
                disease = st.slider("Disease Risk", 0.0, 1.0, 0.15, 0.05, key="disease")
        
        # Calculate health score
        health_score = int((moisture/100 * 0.3 + ndvi * 0.4 + (1-pest) * 0.2 + (1-disease) * 0.1) * 100)
        health_color = "#10B981" if health_score > 70 else "#F59E0B" if health_score > 40 else "#EF4444"
        
        st.markdown("---")
        
        # Health score card with visual indicators
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; border-radius: 10px; 
                        background: linear-gradient(135deg, {health_color}20, {health_color}10); 
                        border-left: 4px solid {health_color};">
                <div style="font-size: 14px; color: #666; margin-bottom: 10px;">Overall Health Score</div>
                <div style="font-size: 48px; font-weight: 700; color: {health_color}; line-height: 1;">
                    {health_score}<span style="font-size: 20px; opacity: 0.8;">/100</span>
                </div>
                <div style="height: 6px; background: #e5e7eb; border-radius: 3px; margin: 15px 0;">
                    <div style="width: {health_score}%; height: 6px; background: {health_color}; border-radius: 3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Status indicators
            status = "Excellent" if health_score > 80 else "Good" if health_score > 60 else "Needs Attention"
            status_emoji = "✅" if health_score > 80 else "⚠️" if health_score > 60 else "❌"
            st.metric("Status", f"{status_emoji} {status}", 
                     delta_color="normal" if health_score > 60 else "off",
                     help="Based on environmental and crop conditions")
            
        with col2:
            # Key metrics
            st.markdown("#### 📊 Health Indicators")
            metrics = st.columns(4)
            metrics_data = [
                {"label": "🌡️ Temperature", "value": f"{temp}°C", "status": "optimal" if 20 <= temp <= 30 else "warning"},
                {"label": "💧 Moisture", "value": f"{moisture}%", "status": "good" if moisture > 50 else "warning"},
                {"label": "🌱 NDVI", "value": f"{ndvi:.2f}", "status": "good" if ndvi > 0.6 else "warning"},
                {"label": "🛡️ Threats", "value": "Low" if pest < 0.3 and disease < 0.3 else "Medium" if pest < 0.6 and disease < 0.6 else "High", 
                 "status": "good" if pest < 0.3 and disease < 0.3 else "warning"}
            ]
            
            for i, metric in enumerate(metrics_data):
                status_color = "#10B981" if metric["status"] == "good" else "#F59E0B"
                with metrics[i]:
                    st.markdown(f"""
                    <div style="padding: 12px; border-radius: 8px; background: #f9fafb; 
                                border-left: 3px solid {status_color}; margin-bottom: 10px;">
                        <div style="font-size: 12px; color: #6b7280; margin-bottom: 4px;">
                            {metric['label']}
                        </div>
                        <div style="font-weight: 600; font-size: 14px;">
                            {metric['value']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        # Environmental factors
        st.subheader("🌦️ Environmental Conditions")
        env_cols = st.columns(2)
        
        with env_cols[0]:
            st.markdown("#### Temperature")
            temp_data = pd.DataFrame({
                'Time': ['6AM', '9AM', '12PM', '3PM', '6PM', '9PM', '12AM'],
                'Current': [20, 22, 25, 27, 25, 23, 21],
                'Ideal': [22, 24, 26, 28, 26, 24, 22]
            }).set_index('Time')
            st.line_chart(temp_data, height=200)
            
            st.markdown("#### Soil Moisture")
            st.progress(moisture/100, f"{moisture}% Optimal")
            
        with env_cols[1]:
            st.markdown("#### Light Intensity")
            light_data = pd.DataFrame({
                'Time': ['6AM', '9AM', '12PM', '3PM', '6PM'],
                'Intensity': [200, 1200, 1800, 1500, 800]
            }).set_index('Time')
            st.area_chart(light_data, height=200)
            
            st.markdown("#### Nutrient Levels")
            cols = st.columns(3)
            cols[0].metric("N", "45 ppm", "+5%")
            cols[1].metric("P", "30 ppm", "-2%")
            cols[2].metric("K", "42 ppm", "+3%")
    
    with tab3:
        # Historical trends
        st.subheader("📈 Health Trends")
        trend_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=30, freq='D'),
            'Health Score': [70, 72, 75, 73, 76, 78, 77, 75, 77, 79, 
                            78, 80, 82, 80, 81, 82, 83, 82, 84, 83, 
                            85, 84, 83, 82, 81, 82, 83, 84, 85, 86],
            'Temperature': [22, 23, 24, 23, 24, 25, 24, 23, 24, 25, 
                           26, 25, 24, 25, 26, 25, 24, 25, 26, 25, 
                           26, 25, 24, 25, 26, 25, 24, 25, 26, 27]
        }).set_index('Date')
        
        fig = px.line(trend_data, y=['Health Score', 'Temperature'], 
                     title="30-Day Health & Temperature Trend",
                     labels={'value': 'Score / °C', 'variable': 'Metric'},
                     height=400)
        fig.update_layout(legend_title_text='')
        st.plotly_chart(fig, width='stretch')
    
    # Recommendations section
    st.markdown("---")
    st.subheader("💡 Recommendations")
    
    rec_cols = st.columns(3)
    with rec_cols[0]:
        with st.expander("🌱 Growth Tips", expanded=True):
            st.markdown("""
            - Increase watering frequency by 10%
            - Apply organic fertilizer next week
            - Rotate crops in 2 weeks
            - Monitor for signs of nutrient deficiency
            """)
    
    with rec_cols[1]:
        with st.expander("⚠️ Alerts", expanded=True):
            if moisture < 40:
                st.warning("💧 Low soil moisture detected")
            if pest > 0.4:
                st.warning("🐛 High pest pressure")
            if disease > 0.4:
                st.warning("🦠 Disease risk increasing")
            if health_score > 80:
                st.success("✅ Crops are in excellent condition!")
    
    with rec_cols[2]:
        with st.expander("📅 Schedule", expanded=True):
            st.markdown("""
            - **Today**: Watering scheduled
            - **Tomorrow**: Fertilizer application
            - **In 3 days**: Pest control check
            - **Next week**: Soil testing
            """)

# ============= Main App =============

def main():
    # App title with theme toggle
    col1, col2 = st.columns([6, 1])
    with col1:
        st.markdown("<h1 style='text-align: center; color: #10b981; margin-bottom: 1rem;'>🌾 Smart Agriculture System</h1>", 
                    unsafe_allow_html=True)
    
    # Navigation tabs
    tab_index = st.session_state.get('active_tab', 0)
    
    tabs = st.tabs([
        "🏠 Dashboard",
        "🔮 Prediction",
        "🏥 Health Analysis",  
        "📊 Data Explorer",
        "⚙️ Settings",
        "📈 History"
    ])
    
    with tabs[0]:
        home_page()
    
    with tabs[1]:
        render_prediction_studio()
    
    with tabs[2]:
        health_page()

    
    with tabs[3]:
        render_data_explorer()
    
    with tabs[4]:
        settings_page()
    
    with tabs[5]:
        history_page()
    
    # Footer
    st.markdown("---")
    st.caption("Smart Agriculture System v3.0 | Powered by AI | © 2024")

if __name__ == "__main__":
    main()

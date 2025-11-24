import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

class CropHealthAnalyzer:
    """
    A class to handle the Crop Health Analyzer feature, including metrics,
    visualizations, and recommendations.
    """

    @staticmethod
    def irrigation_recommendation(soil_moisture: float, weather_data: Optional[Dict[str, float]]) -> Dict[str, Any]:
        """
        Provide detailed irrigation recommendations based on soil moisture and weather data.
        """
        if soil_moisture is None:
            return {
                "action": "Unknown",
                "level": "None",
                "message": "Soil moisture data not available.",
                "reasoning": ["Missing soil moisture sensor data"]
            }
            
        # Get precipitation (rain) in mm, default to 0 if not available
        precipitation = weather_data.get('precipitation', 0) if weather_data else 0
        
        # Get temperature in Celsius, default to 25°C if not available
        temperature = weather_data.get('temperature', 25) if weather_data else 25
        
        reasoning = []
        action = "Monitor"
        level = "None"
        amount_mm = 0
        duration_mins = 0
        
        # Check soil moisture levels
        if soil_moisture < 20:
            level = "High"
            action = "Irrigate"
            amount_mm = 25
            reasoning.append(f"Critical soil moisture deficit ({soil_moisture:.1f}%)")
        elif soil_moisture < 40:
            level = "Medium"
            action = "Irrigate"
            amount_mm = 15
            reasoning.append(f"Moderate soil moisture ({soil_moisture:.1f}%)")
        elif soil_moisture > 80:
            level = "None"
            action = "Drain"
            reasoning.append(f"Excessive soil moisture ({soil_moisture:.1f}%)")
        else:
            level = "Low"
            action = "Monitor"
            reasoning.append(f"Optimal soil moisture ({soil_moisture:.1f}%)")
        
        # Adjust based on recent/forecasted precipitation
        if precipitation > 5:
            if action == "Irrigate":
                amount_mm = max(0, amount_mm - precipitation)
                if amount_mm == 0:
                    action = "Monitor"
                    level = "None"
                    reasoning.append(f"Rainfall ({precipitation:.1f}mm) sufficient to cover needs")
                else:
                    reasoning.append(f"Reduced irrigation due to rainfall ({precipitation:.1f}mm)")
            elif action == "Monitor":
                reasoning.append(f"Rainfall expected ({precipitation:.1f}mm)")
                
        # Adjust based on temperature
        if temperature > 30:
            if action == "Irrigate":
                amount_mm *= 1.2
                reasoning.append(f"Increased amount due to high heat ({temperature:.1f}°C)")
            elif action == "Monitor" and soil_moisture < 50:
                action = "Irrigate"
                level = "Low"
                amount_mm = 5
                reasoning.append(f"Light irrigation recommended due to heat ({temperature:.1f}°C)")
                
        # Calculate duration (assuming standard drip rate of 4mm/hour)
        if amount_mm > 0:
            duration_mins = int((amount_mm / 4) * 60)
            
        return {
            "action": action,
            "level": level,
            "amount_mm": round(amount_mm, 1),
            "duration_mins": duration_mins,
            "reasoning": reasoning,
            "message": f"{action} recommended" + (f" ({amount_mm}mm)" if amount_mm > 0 else "")
        }

    def render(self, agronomic_inputs: Dict[str, Any], weather_data: Optional[Dict[str, Any]] = None):
        """
        Render the Crop Health Analyzer UI.
        """
        st.markdown('<div class="section-title">🏥 Crop Health Analyzer</div>', unsafe_allow_html=True)
        
        # Growth Stage Selector
        stage = st.selectbox(
            "Crop Growth Stage",
            ["Seedling", "Vegetative", "Reproductive", "Maturity"],
            help="Select the current growth stage for more specific advice"
        )
        
        # Calculate health score
        health_score = self._calculate_health_score(agronomic_inputs)
        
        # Render Metrics
        self._render_metrics(agronomic_inputs, health_score)
        
        # Detailed Diagnosis
        self._render_diagnosis(agronomic_inputs, stage)
        
        # Risk Assessment Matrix
        self._render_risk_matrix(agronomic_inputs, weather_data)
        
        # Render Visualizations
        self._render_visualizations(agronomic_inputs)
        
        # Render Recommendations
        self._render_recommendations(agronomic_inputs, health_score, weather_data)
        
        # Render History
        self._render_history(health_score)

    def _render_diagnosis(self, inputs: Dict[str, Any], stage: str):
        """Render detailed diagnosis based on growth stage."""
        st.subheader("🩺 Detailed Diagnosis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Current Stage: {stage}**")
            if stage == "Seedling":
                st.markdown("Focus on root establishment. Critical factors: Soil Moisture, Temperature.")
            elif stage == "Vegetative":
                st.markdown("Focus on leaf development. Critical factors: Nitrogen (Fertilizer), Pest Control.")
            elif stage == "Reproductive":
                st.markdown("Focus on flowering/fruiting. Critical factors: Water, Phosphorus/Potassium.")
            else:
                st.markdown("Focus on grain filling/ripening. Critical factors: Pest Control, Dry conditions.")
                
        with col2:
            # Mock diagnosis logic
            issues = []
            if inputs.get('soil_moisture_pct', 0) < 15:
                issues.append("Low moisture may stunt growth.")
            if inputs.get('pest_pressure_idx', 0) > 0.4:
                issues.append("Pest pressure threatens yield.")
            
            if issues:
                st.warning("**Detected Issues:**")
                for issue in issues:
                    st.markdown(f"- {issue}")
            else:
                st.success("**No critical issues detected for this stage.**")

    def _render_risk_matrix(self, inputs: Dict[str, Any], weather: Optional[Dict[str, Any]]):
        """Render a risk assessment heatmap."""
        st.subheader("🛡️ Risk Assessment")
        
        # Calculate risks (0-10 scale)
        pest_risk = inputs.get('pest_pressure_idx', 0) * 10
        disease_risk = (inputs.get('avg_rainfall_mm', 0) / 500) * 8 + (inputs.get('avg_temp_c', 25) / 40) * 2
        climate_risk = 5  # Placeholder
        
        risks = {
            "Pest": pest_risk,
            "Disease": disease_risk,
            "Climate": climate_risk
        }
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[[risks["Pest"], risks["Disease"], risks["Climate"]]],
            x=["Pest", "Disease", "Climate"],
            y=["Risk Level"],
            colorscale="RdYlGn_r",
            zmin=0, zmax=10
        ))
        
        fig.update_layout(
            height=150,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

    def _calculate_health_score(self, inputs: Dict[str, Any]) -> int:
        """Calculate a composite health score."""
        return min(100, int(
            30 +  # Base score
            (inputs.get('ndvi', 0.5) * 40) +  # Up to 40 points for vegetation
            (inputs.get('soil_moisture_pct', 0) * 0.5) +  # Up to 30 points for moisture
            (20 - abs(inputs.get('soil_ph', 7) - 6.5) * 10)  # Up to 20 points for pH
        ))

    def _render_metrics(self, inputs: Dict[str, Any], health_score: int):
        """Render key health metrics."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            soil_quality = min(100, int(inputs.get('soil_moisture_pct', 0) * 1.5 + (inputs.get('soil_ph', 7) * 10)))
            st.metric("Soil Quality", f"{soil_quality}/100")
            
        with col2:
            pest_risk = int(inputs.get('pest_pressure_idx', 0.5) * 100)
            st.metric("Pest Risk", f"{pest_risk}%")
            
        with col3:
            water_balance = inputs.get('avg_rainfall_mm', 0) - (inputs.get('soil_moisture_pct', 0) * 2)
            st.metric("Water Balance", 
                     f"{'Good' if water_balance > 0 else 'Low'}",
                     f"{abs(water_balance):.1f} mm",
                     delta_color="inverse")
            
        with col4:
            st.metric("Health Score", f"{health_score}/100")

    def _render_visualizations(self, inputs: Dict[str, Any]):
        """Render radar and bar charts."""
        fig = make_subplots(
            rows=1, 
            cols=2,
            specs=[[{'type': 'polar'}, {'type': 'xy'}]],
            subplot_titles=("Soil Conditions", "Environmental Factors")
        )
        
        # Soil conditions radar chart
        fig.add_trace(
            go.Scatterpolar(
                r=[
                    inputs.get('soil_moisture_pct', 0),
                    inputs.get('soil_ph', 7) * 10,
                    inputs.get('fertilizer_kg_per_ha', 0) / 2.5,
                    50  # Placeholder for organic matter
                ],
                theta=['Moisture', 'pH', 'Fertilizer', 'Organic Matter'],
                fill='toself',
                name='Soil Conditions'
            ),
            row=1, col=1
        )

        # Environmental factors bar chart
        factors = ['Rainfall', 'Temperature', 'NDVI']
        values = [
            inputs.get('avg_rainfall_mm', 0) / 5,
            inputs.get('avg_temp_c', 25),
            inputs.get('ndvi', 0.5) * 100
        ]
        fig.add_trace(go.Bar(x=factors, y=values, name='Current'), row=1, col=2)

        fig.update_layout(height=400, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        fig.update_polars(radialaxis=dict(visible=True, range=[0, 100]))
        
        st.plotly_chart(fig, use_container_width=True)

    def _render_recommendations(self, inputs: Dict[str, Any], health_score: int, weather_data: Optional[Dict[str, Any]]):
        """Render health analysis and recommendations."""
        st.subheader("📋 Health Analysis & Recommendations")
        
        recommendations = []
        actions = []
        
        # pH Analysis
        ph = inputs.get('soil_ph', 7)
        if ph < 6.0:
            recommendations.append(f"Soil is acidic (pH {ph:.1f}). Consider applying lime to raise pH.")
            actions.append("Apply agricultural lime")
        elif ph > 7.5:
            recommendations.append(f"Soil is alkaline (pH {ph:.1f}). Consider applying sulfur to lower pH.")
            actions.append("Apply elemental sulfur")
            
        # Fertilizer Analysis
        fert = inputs.get('fertilizer_kg_per_ha', 0)
        if fert < 50:
            recommendations.append("Fertilizer application is low. Consider increasing NPK application.")
            actions.append("Schedule fertilizer application")
        elif fert > 200:
            recommendations.append("Fertilizer application is high. Monitor for runoff or nutrient burn.")
            
        # Pest Analysis
        pest = inputs.get('pest_pressure_idx', 0)
        if pest > 0.6:
            recommendations.append("High pest pressure detected. Immediate intervention required.")
            actions.append("Inspect field for pests")
            actions.append("Apply appropriate pesticide/biocontrol")
        elif pest > 0.3:
            recommendations.append("Moderate pest pressure. Increase monitoring frequency.")
            actions.append("Increase scouting frequency")
            
        # NDVI Analysis
        ndvi = inputs.get('ndvi', 0.5)
        if ndvi < 0.3:
            recommendations.append("Low vegetation index indicating potential stress or poor crop cover.")
            actions.append("Check for disease or water stress")
            
        # Display general health status
        if health_score < 50:
            st.warning("⚠️ Crop health requires attention")
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.success("✅ Crop health is generally good")
            if recommendations:
                with st.expander("Optimization Tips"):
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
            else:
                st.markdown("Conditions are optimal. Continue current management practices.")
                
        # Action Plan
        if actions:
            st.subheader("📝 Immediate Action Plan")
            for i, action in enumerate(actions):
                st.checkbox(action, key=f"health_action_{i}")
        
        # Irrigation Advice
        st.subheader("💧 Irrigation Advice")
        soil_moisture = inputs.get('soil_moisture_pct', 20)
        
        # Prepare weather info for irrigation recommendation
        weather_info = {
            'precipitation': weather_data.get('precipitation_sum', 0) if weather_data else 0,
            'temperature': weather_data.get('current_temp', 25) if weather_data else 25
        }
        
        rec = self.irrigation_recommendation(soil_moisture, weather_info)
        
        # Display recommendation card
        rec_color = {
            "High": "#dc3545", "Medium": "#ffc107", "Low": "#17a2b8", "None": "#28a745"
        }.get(rec["level"], "#6c757d")
        
        st.markdown(f"""
        <div style="background-color: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); overflow: hidden; margin-bottom: 1.5rem; border: 1px solid #e9ecef;">
            <div style="background-color: {rec_color}; padding: 1rem 1.5rem; display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.5rem; color: white;">💧</span>
                    <h3 style="margin: 0; color: white; font-size: 1.2rem; font-weight: 600;">{rec['action']} Recommended</h3>
                </div>
                <span style="background-color: rgba(255,255,255,0.2); color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-weight: 500; font-size: 0.85rem;">{rec['level']} Priority</span>
            </div>
            <div style="padding: 1.5rem;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1.5rem; margin-bottom: 1.5rem;">
                    <div style="background-color: #f8f9fa; padding: 1.25rem; border-radius: 10px; text-align: center; border: 1px solid #e9ecef;">
                        <div style="color: #0d6efd; font-size: 1.5rem; margin-bottom: 0.5rem;">💧</div>
                        <small style="color: #6c757d; text-transform: uppercase; font-size: 0.75rem; font-weight: 600;">Water Amount</small>
                        <div style="font-weight: 700; color: #212529; font-size: 1.75rem; margin-top: 0.25rem;">{rec['amount_mm']} <span style="font-size: 1rem; color: #6c757d;">mm</span></div>
                    </div>
                    <div style="background-color: #f8f9fa; padding: 1.25rem; border-radius: 10px; text-align: center; border: 1px solid #e9ecef;">
                        <div style="color: #0d6efd; font-size: 1.5rem; margin-bottom: 0.5rem;">⏱️</div>
                        <small style="color: #6c757d; text-transform: uppercase; font-size: 0.75rem; font-weight: 600;">Duration</small>
                        <div style="font-weight: 700; color: #212529; font-size: 1.75rem; margin-top: 0.25rem;">{rec['duration_mins']} <span style="font-size: 1rem; color: #6c757d;">mins</span></div>
                    </div>
                </div>
                <div style="background-color: #fff; border-left: 4px solid #6c757d; padding: 0.5rem 0 0.5rem 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #343a40; font-size: 0.95rem; font-weight: 600;">Analysis & Reasoning</h4>
                    <ul style="margin: 0; padding-left: 1.2rem; color: #495057;">
                        {''.join(f'<li style="margin-bottom: 0.25rem;">{r}</li>' for r in rec['reasoning'])}
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Store recommendations for history
        self.current_recommendations = recommendations
        self.current_irrigation_action = rec['action']

    def _render_history(self, health_score: int):
        """Render history and save functionality."""
        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("💾 Save Analysis", use_container_width=True):
                if 'analysis_history' not in st.session_state:
                    st.session_state.analysis_history = []
                
                analysis_entry = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "health_score": health_score,
                    "irrigation_action": getattr(self, 'current_irrigation_action', 'Unknown'),
                    "recommendations": len(getattr(self, 'current_recommendations', []))
                }
                st.session_state.analysis_history.append(analysis_entry)
                st.success("Analysis saved to history!")
        
        with col2:
            if st.button("🔄 Refresh Data", key="refresh_health_new", use_container_width=True):
                st.rerun()
                
        if 'analysis_history' in st.session_state and st.session_state.analysis_history:
            with st.expander("📜 Analysis History", expanded=False):
                history_df = pd.DataFrame(st.session_state.analysis_history)
                st.dataframe(
                    history_df.style.background_gradient(subset=['health_score'], cmap='RdYlGn', vmin=0, vmax=100),
                    use_container_width=True
                )

"""
UI Components for Smart Agriculture System

This module contains reusable UI components for the Streamlit application.
"""
import streamlit as st
from typing import Optional, Dict, Any, List
import time

class Theme:
    """Theme configuration for the app"""
    PRIMARY_COLOR = "#2E8B57"  # Sea Green
    SECONDARY_COLOR = "#3CB371"  # Medium Sea Green
    ACCENT_COLOR = "#FFA500"  # Orange
    BACKGROUND_COLOR = "#F5F5F5"
    TEXT_COLOR = "#333333"
    CARD_BACKGROUND = "#FFFFFF"
    BORDER_RADIUS = "10px"
    BOX_SHADOW = "0 4px 6px rgba(0, 0, 0, 0.1)"
    
    @staticmethod
    def apply_custom_theme():
        """Apply custom CSS theme to the app with mobile responsiveness"""
        st.markdown(f"""
        <style>
            /* Responsive base styles */
            @media (max-width: 768px) {{
                .main .block-container {{
                    padding: 1rem 0.5rem !important;
                }}
                
                .stButton > button {{
                    width: 100% !important;
                    margin: 0.25rem 0;
                }}
                
                .stSelectbox, .stTextInput, .stNumberInput, .stSlider > div > div {{
                    width: 100% !important;
                }}
                
                .metric-card {{
                    margin-bottom: 0.5rem !important;
                }}
            }}
            
            /* Main app styling */
            .main .block-container {{
                padding: 1.5rem;
                max-width: 1200px;
                margin: 0 auto;
            }}
            
            /* Sidebar styling */
            [data-testid="stSidebar"] {{
                background-color: #f8f9fa;
                border-right: 1px solid #e9ecef;
            }}
            
            /* Card styling */
            .card {{
                background-color: {Theme.CARD_BACKGROUND};
                border-radius: {Theme.BORDER_RADIUS};
                padding: 1.5rem;
                box-shadow: {Theme.BOX_SHADOW};
                margin-bottom: 1rem;
                border-left: 4px solid {Theme.PRIMARY_COLOR};
            }}
            
            /* Button styling */
            .stButton>button {{
                border-radius: {Theme.BORDER_RADIUS};
                border: none;
                background-color: {Theme.PRIMARY_COLOR};
                color: white;
                padding: 0.5rem 1.5rem;
                font-weight: 500;
                transition: all 0.3s ease;
            }}
            
            .stButton>button:hover {{
                background-color: {Theme.SECONDARY_COLOR};
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            
            /* Metric cards */
            .metric-card {{
                background: linear-gradient(145deg, #ffffff, #f0f0f0);
                border-radius: {Theme.BORDER_RADIUS};
                padding: 1.5rem;
                text-align: center;
                box-shadow: {Theme.BOX_SHADOW};
                transition: all 0.3s ease;
            }}
            
            .metric-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            }}
            
            .metric-value {{
                font-size: 2rem;
                font-weight: 700;
                color: {Theme.PRIMARY_COLOR};
                margin: 0.5rem 0;
            }}
            
            .metric-label {{
                font-size: 0.9rem;
                color: #666;
                margin: 0;
            }}
            
            /* Custom tabs */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 8px;
            }}
            
            .stTabs [data-baseweb="tab"] {{
                height: 40px;
                padding: 0 20px;
                border-radius: 20px;
                background-color: #f0f0f0;
                transition: all 0.3s ease;
            }}
            
            .stTabs [aria-selected="true"] {{
                background-color: {Theme.PRIMARY_COLOR};
                color: white !important;
            }}
            
            /* Form elements */
            .stTextInput>div>div>input, 
            .stSelectbox>div>div>div {{
                border-radius: {Theme.BORDER_RADIUS} !important;
                border: 1px solid #ddd !important;
            }}
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: #f1f1f1;
                border-radius: 10px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: #888;
                border-radius: 10px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: #555;
            }}
        </style>
        """, unsafe_allow_html=True)

class UIComponents:
    """UI Components for the Smart Agriculture System"""
    
    @staticmethod
    def page_header(title: str, description: str = None):
        """Create a beautiful page header"""
        st.markdown(f"""
        <div style="margin-bottom: 2rem;">
            <h1 style="color: {Theme.PRIMARY_COLOR}; margin-bottom: 0.5rem;">{title}</h1>
            {f'<p style="color: #666; margin: 0;">{description}</p>' if description else ''}
            <div style="height: 4px; width: 60px; background: {Theme.ACCENT_COLOR}; margin: 0.5rem 0 1.5rem 0;"></div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def metric_card(title: str, value: Any, delta: str = None, icon: str = None):
        """Create a metric card with optional delta and icon"""
        delta_html = f'<span style="color: #28a745; font-size: 0.9rem;">{delta}</span>' if delta else ''
        icon_html = f'<span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>' if icon else ''
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 0.5rem;">
                {icon_html}
                <h3 style="margin: 0; color: #666; font-size: 0.9rem;">{title}</h3>
            </div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def info_card(title: str, content: str, icon: str = "ℹ️"):
        """Create an info card with icon and content"""
        st.markdown(f"""
        <div class="card">
            <div style="display: flex; align-items: flex-start; gap: 1rem;">
                <div style="font-size: 1.5rem; color: {Theme.PRIMARY_COLOR};">{icon}</div>
                <div>
                    <h3 style="margin: 0 0 0.5rem 0; color: {Theme.TEXT_COLOR};">{title}</h3>
                    <p style="margin: 0; color: #666; line-height: 1.6;">{content}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def status_indicator(status: str, message: str = ""):
        """Create a status indicator with colored dot"""
        status_colors = {
            "success": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545",
            "info": "#17a2b8"
        }
        
        color = status_colors.get(status.lower(), "#6c757d")
        
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.5rem; margin: 0.5rem 0;">
            <div style="
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: {color};
            "></div>
            <span style="color: {color}; font-weight: 500; text-transform: capitalize;">{status}</span>
            {f'<span style="color: #666; margin-left: 0.5rem;">{message}</span>' if message else ''}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def loading_animation(text: str = "Loading..."):
        """Show a loading animation"""
        return st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 2rem;">
            <div class="spinner"></div>
            <p style="margin-top: 1rem; color: #666;">{text}</p>
        </div>
        <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            .spinner {{
                border: 4px solid rgba(0, 0, 0, 0.1);
                width: 36px;
                height: 36px;
                border-radius: 50%;
                border-left-color: {Theme.PRIMARY_COLOR};
                animation: spin 1s linear infinite;
            }}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def success_message(message: str):
        """Display a success message"""
        st.success(message, icon="✅")
    
    @staticmethod
    def error_message(message: str):
        """Display an error message"""
        st.error(message, icon="❌")
    
    @staticmethod
    def warning_message(message: str):
        """Display a warning message"""
        st.warning(message, icon="⚠️")
    
    @staticmethod
    def info_message(message: str):
        """Display an info message"""
        st.info(message, icon="ℹ️")
    
    @staticmethod
    def custom_tabs(tab_names: List[str]):
        """
        Create custom styled tabs that work well on mobile and desktop.
        
        Args:
            tab_names: List of tab names to display
            
        Returns:
            The selected tab index
        """
        # Use columns for better mobile responsiveness
        if len(tab_names) > 3:  # For many tabs, use a selectbox on mobile
            selected = st.radio(
                "Select Tab",
                options=tab_names,
                horizontal=True,
                label_visibility="collapsed"
            )
            return tab_names.index(selected)
        else:
            # For few tabs, use styled buttons
            cols = st.columns(len(tab_names))
            selected = 0
            
            for i, name in enumerate(tab_names):
                with cols[i]:
                    if st.button(name, key=f"tab_{i}", width="stretch"):
                        selected = i
            
            # Add some spacing
            st.markdown("---")
            return selected
            
    def responsive_columns(self, num_columns: int, gap: str = "1rem"):
        """
        Create responsive columns that stack on mobile.
        
        Args:
            num_columns: Number of columns to create
            gap: Gap between columns (CSS value)
            
        Returns:
            List of columns
        """
        return st.columns(num_columns, gap=gap)
        
    def mobile_friendly_metric(self, label: str, value: Any, delta: str = None, 
                             help_text: str = None):
        """
        Display a metric that works well on mobile devices.
        
        Args:
            label: Metric label
            value: Metric value
            delta: Optional delta value (e.g., "+10%")
            help_text: Optional help text to show on hover
        """
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown(f"**{label}**")
            if help_text:
                st.caption(help_text)
        with col2:
            if delta:
                st.metric("", value, delta)
            else:
                st.markdown(f"### {value}")
                
    def responsive_expander(self, title: str, expanded: bool = False):
        """
        Create an expander that works well on mobile.
        
        Args:
            title: Expander title
            expanded: Whether the expander should be expanded by default
            
        Returns:
            The expander object
        """
        return st.expander(title, expanded=expanded)

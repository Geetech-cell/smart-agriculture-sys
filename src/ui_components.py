"""
Enhanced UI Components with Animations, Tooltips & Dark Mode
Professional-grade components for Smart Agriculture System
"""
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List, Callable
import json

class Theme:
    """Enhanced theme system with dark mode support"""
    
    # Light theme
    LIGHT = {
        'primary': '#10b981',
        'secondary': '#3b82f6',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'dark': '#1f2937',
        'light': '#f9fafb',
        'text': '#1f2937',
        'bg': '#ffffff',
        'bg_secondary': '#f9fafb'
    }
    
    # Dark theme
    DARK = {
        'primary': '#10b981',
        'secondary': '#60a5fa',
        'success': '#10b981',
        'warning': '#fbbf24',
        'danger': '#f87171',
        'dark': '#f9fafb',
        'light': '#1f2937',
        'text': '#f9fafb',
        'bg': '#1f2937',
        'bg_secondary': '#374151'
    }
    
    @classmethod
    def get_current(cls):
        """Get current theme based on session state"""
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = False
        return cls.DARK if st.session_state.dark_mode else cls.LIGHT
    
    @classmethod
    def apply(cls):
        """Apply theme with animations"""
        theme = cls.get_current()
        st.markdown(f"""
        <style>
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            @keyframes slideIn {{
                from {{ transform: translateX(-20px); opacity: 0; }}
                to {{ transform: translateX(0); opacity: 1; }}
            }}
            
            @keyframes spin {{
                from {{ transform: rotate(0deg); }}
                to {{ transform: rotate(360deg); }}
            }}
            
            .stApp {{
                background: {theme['bg']};
                color: {theme['text']};
                transition: all 0.3s ease;
            }}
            
            /* Animated page content */
            .main .block-container {{
                animation: fadeIn 0.5s ease-out;
            }}
            
            /* Enhanced buttons */
            .stButton>button {{
                border-radius: 8px;
                font-weight: 500;
                border: none;
                transition: all 0.2s ease;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .stButton>button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            }}
            .stButton>button:active {{
                transform: translateY(0);
            }}
            
            /* Animated cards */
            .card {{
                background: {theme['bg']};
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                transition: all 0.3s ease;
                animation: slideIn 0.4s ease-out;
                border: 1px solid {theme['bg_secondary']};
            }}
            .card:hover {{
                transform: translateY(-4px);
                box-shadow: 0 8px 16px rgba(0,0,0,0.12);
            }}
            
            /* Enhanced metrics */
            [data-testid="stMetricValue"] {{
                font-size: 2rem;
                font-weight: 700;
                color: {theme['primary']};
                animation: fadeIn 0.6s ease-out;
            }}
            [data-testid="stMetricLabel"] {{
                color: {theme['text']};
                opacity: 0.8;
            }}
            
            /* Tooltip styles */
            .tooltip {{
                position: relative;
                display: inline-block;
                cursor: help;
            }}
            .tooltip .tooltiptext {{
                visibility: hidden;
                background-color: {theme['dark']};
                color: {theme['light']};
                text-align: center;
                padding: 8px 12px;
                border-radius: 6px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -80px;
                opacity: 0;
                transition: opacity 0.3s;
                width: 160px;
                font-size: 0.85rem;
            }}
            .tooltip:hover .tooltiptext {{
                visibility: visible;
                opacity: 1;
            }}
            
            /* Loading spinner */
            .spinner {{
                border: 3px solid {theme['bg_secondary']};
                border-top: 3px solid {theme['primary']};
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }}
            
            /* Enhanced Tabs - Modern Pill Style */
            .stTabs [data-baseweb="tab-list"] {{
                gap: 12px;
                background: {theme['bg_secondary']};
                padding: 12px;
                border-radius: 16px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }}
            
            .stTabs [data-baseweb="tab"] {{
                padding: 14px 28px;
                border-radius: 12px;
                font-weight: 600;
                font-size: 0.95rem;
                transition: all 0.3s ease;
                background: transparent;
                color: {theme['text']};
                opacity: 0.7;
                border: none;
            }}
            
            .stTabs [data-baseweb="tab"]:hover {{
                background: {theme['bg']};
                opacity: 1;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            
            .stTabs [aria-selected="true"] {{
                background: {theme['primary']};
                color: white !important;
                opacity: 1;
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
                transform: translateY(-2px);
            }}
            
            .stTabs [data-baseweb="tab-panel"] {{
                padding-top: 2rem;
            }}
            
            /* Input fields */
            .stTextInput>div>div>input,
            .stNumberInput>div>div>input,
            .stSelectbox>div>div>div {{
                background: {theme['bg']};
                color: {theme['text']};
                border-color: {theme['bg_secondary']};
                transition: all 0.2s ease;
            }}
            
            /* Success/Error animations */
            .success-badge {{
                background: {theme['success']};
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                display: inline-block;
                animation: fadeIn 0.5s ease-out;
            }}
        </style>
        """, unsafe_allow_html=True)

class UI:
    """Enhanced UI components with animations and tooltips"""
    
    @staticmethod
    def header(title: str, subtitle: str = None, help_text: str = None):
        """Animated page header with optional tooltip"""
        theme = Theme.get_current()
        
        # Simple header without complex HTML
        st.markdown(f"<h1 style='color: {theme['primary']};'>{title}</h1>", unsafe_allow_html=True)
        
        if subtitle:
            st.markdown(f"<p style='color: {theme['text']}; opacity: 0.8; font-size: 1.1rem;'>{subtitle}</p>", unsafe_allow_html=True)
        
        if help_text:
            st.caption(f"ℹ️ {help_text}")
        
        st.divider()
    
    @staticmethod
    def card(content_func: Callable, title: str = None, icon: str = None):
        """Animated card with hover effects"""
        with st.container():
            if title:
                icon_html = f"{icon} " if icon else ""
                st.markdown(f"### {icon_html}{title}")
            content_func()
    
    @staticmethod
    def metric_grid(metrics: List[tuple], cols: int = 4):
        """Animated metrics grid"""
        columns = st.columns(cols)
        for col, metric_data in zip(columns, metrics):
            with col:
                label, value, delta = metric_data[:3]
                help_text = metric_data[3] if len(metric_data) > 3 else None
                
                # Use Streamlit's built-in help parameter instead of custom tooltips
                st.metric(label, value, delta, help=help_text)
    
    @staticmethod
    def loading_spinner(text: str = "Loading..."):
        """Animated loading spinner"""
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <div class="spinner"></div>
            <p style="margin-top: 1rem; opacity: 0.8;">{text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def success_badge(text: str):
        """Animated success badge"""
        st.markdown(f'<div class="success-badge">✓ {text}</div>', unsafe_allow_html=True)
    
    @staticmethod
    def bar_chart(data: Dict[str, float], title: str = None, color: str = None):
        """Enhanced bar chart with animations"""
        theme = Theme.get_current()
        color = color or theme['primary']
        
        fig = go.Figure(go.Bar(
            x=list(data.values()),
            y=list(data.keys()),
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color=theme['text'], width=0.5)
            ),
            text=[f"{v:.1f}" for v in data.values()],
            textposition='auto',
        ))
        
        fig.update_layout(
            title=title,
            height=max(200, len(data) * 50),
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor=theme['bg'],
            paper_bgcolor=theme['bg'],
            showlegend=False,
            font=dict(color=theme['text']),
            xaxis=dict(
                showgrid=True,
                gridcolor=theme['bg_secondary'],
                color=theme['text']
            ),
            yaxis=dict(showgrid=False, color=theme['text']),
        )
        
        st.plotly_chart(fig, width="stretch", config={'displayModeBar': False})
    
    @staticmethod
    def info_box(message: str, type: str = "info"):
        """Animated info box"""
        theme = Theme.get_current()
        colors = {
            'info': theme['secondary'],
            'success': theme['success'],
            'warning': theme['warning'],
            'error': theme['danger']
        }
        icons = {
            'info': 'ℹ️',
            'success': '✓',
            'warning': '⚠️',
            'error': '✗'
        }
        
        st.markdown(f"""
        <div style="
            background: {colors.get(type, theme['secondary'])}15;
            border-left: 4px solid {colors.get(type, theme['secondary'])};
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            animation: slideIn 0.4s ease-out;">
            <strong>{icons.get(type, 'ℹ️')} {message}</strong>
        </div>
        """, unsafe_allow_html=True)

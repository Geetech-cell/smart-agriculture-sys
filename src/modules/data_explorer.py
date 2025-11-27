"""
Data Explorer - Simple & Powerful
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from src.ui_components import UI

def render_data_explorer():
    """Minimalist data exploration tool"""
    UI.header("Data Explorer", "Analyze and visualize your agricultural data")
    
    # File upload
    path = st.text_input("📁 Dataset Path", "data/sample_crop_data.csv")
    
    try:
        # Load data
        df = pd.read_csv(path)
        
        st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Data editor
        st.markdown("### 📊 Data Table")
        edited_df = st.data_editor(
            df, 
            height=400
        )
        
        st.divider()
        
        # Visualization
        st.markdown("### 📈 Visualization")
        
        cols = edited_df.columns.tolist()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_axis = st.selectbox("X Axis", cols, index=0)
        with col2:
            y_axis = st.selectbox("Y Axis", cols, index=min(1, len(cols)-1))
        with col3:
            color = st.selectbox("Color", [None] + cols)
        with col4:
            chart_type = st.selectbox("Chart", ["Scatter", "Line", "Bar", "Box", "Histogram"])
        
        # Create chart
        if chart_type == "Scatter":
            fig = px.scatter(edited_df, x=x_axis, y=y_axis, color=color, 
                           title=f"{y_axis} vs {x_axis}")
        elif chart_type == "Line":
            fig = px.line(edited_df, x=x_axis, y=y_axis, color=color,
                         title=f"{y_axis} over {x_axis}")
        elif chart_type == "Bar":
            fig = px.bar(edited_df, x=x_axis, y=y_axis, color=color,
                        title=f"{y_axis} by {x_axis}")
        elif chart_type == "Box":
            fig = px.box(edited_df, x=x_axis, y=y_axis, color=color,
                        title=f"Distribution of {y_axis}")
        else:  # Histogram
            fig = px.histogram(edited_df, x=x_axis, color=color,
                             title=f"Distribution of {x_axis}")
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, width="stretch")
        
        # Statistics
        with st.expander("📈 Summary Statistics"):
            st.dataframe(edited_df.describe().T)
    
    except FileNotFoundError:
        st.error(f"❌ File not found: {path}")
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

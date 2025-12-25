"""
CHRONOS Dashboard - Model Weights Analysis
Visualize and analyze the trained model weights.
Uses REAL DATA from model checkpoint.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import os

st.set_page_config(page_title="Model Weights", page_icon="🧮", layout="wide")

st.title("🧮 Model Weights Analysis")
st.markdown("Analyze the trained model weights - **Real Data from Checkpoint**")
st.markdown("---")

# Load real data
@st.cache_data
def load_weight_data():
    data = {}
    if os.path.exists('results/real_data/model_weights.csv'):
        data['weights'] = pd.read_csv('results/real_data/model_weights.csv')
    if os.path.exists('results/real_data/feature_importance_from_weights.csv'):
        data['importance'] = pd.read_csv('results/real_data/feature_importance_from_weights.csv')
    return data

data = load_weight_data()

if 'weights' not in data:
    st.error("Weight data not found! Run: `python scripts/generate_advanced_analysis.py`")
    st.stop()

st.success("✅ All weights from actual trained model checkpoint")

weight_df = data['weights']

# Overview
st.subheader("📊 Model Parameter Overview")

total_params = weight_df['n_params'].sum()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Parameters", f"{total_params:,}")
with col2:
    st.metric("Layers", len(weight_df))
with col3:
    st.metric("Largest Layer", f"{weight_df['n_params'].max():,} params")

st.markdown("---")

# Parameter distribution by layer
st.subheader("📈 Parameters by Layer")

fig = px.bar(weight_df, x='layer', y='n_params',
             title="Parameter Count per Layer",
             color='n_params',
             color_continuous_scale='Viridis')
fig.update_layout(template='plotly_dark', height=400, 
                  xaxis_tickangle=-45)
st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Weight statistics
st.subheader("📉 Weight Statistics")

col1, col2 = st.columns(2)

with col1:
    # Weight distribution bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=weight_df['layer'], y=weight_df['mean'], name='Mean',
                        marker_color='#4ECDC4'))
    fig.add_trace(go.Bar(x=weight_df['layer'], y=weight_df['std'], name='Std',
                        marker_color='#FF6B6B'))
    fig.update_layout(
        template='plotly_dark',
        title="Weight Mean and Std by Layer",
        xaxis_tickangle=-45,
        height=400,
        barmode='group'
    )
    st.plotly_chart(fig, width='stretch')

with col2:
    # Table
    display_df = weight_df[['layer', 'shape', 'n_params', 'mean', 'std', 'abs_mean']].copy()
    display_df['mean'] = display_df['mean'].round(4)
    display_df['std'] = display_df['std'].round(4)
    display_df['abs_mean'] = display_df['abs_mean'].round(4)
    st.dataframe(display_df, width='stretch', hide_index=True, height=400)

st.markdown("---")

# Feature importance from input projection
st.subheader("🎯 Feature Importance (from Input Projection)")

if 'importance' in data:
    imp_df = data['importance'].head(50)  # Top 50
    
    fig = px.bar(imp_df, x='feature_idx', y='importance',
                 title="Top 50 Features by Weight Magnitude",
                 labels={'feature_idx': 'Feature Index', 'importance': 'Avg |Weight|'})
    fig.update_layout(template='plotly_dark', height=400)
    fig.update_traces(marker_color='#4ECDC4')
    st.plotly_chart(fig, width='stretch')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Most Important Features")
        st.dataframe(imp_df.head(20), width='stretch', hide_index=True)
    
    with col2:
        st.markdown("""
        ### Interpretation
        
        **What this shows:**
        - Average absolute weight from the input projection layer
        - Higher = model pays more attention to this feature
        
        **Feature Categories (Elliptic):**
        - **0-93**: Direct transaction features
        - **94-165**: 1-hop neighbor aggregates
        - **166-235**: Engineered graph features
        
        **Caveats:**
        - This is a proxy for importance
        - True importance requires gradient analysis
        """)

st.markdown("---")

# Layer-wise analysis
st.subheader("🔍 Layer-wise Weight Ranges")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=weight_df['layer'],
    y=weight_df['max'],
    mode='markers+lines',
    name='Max',
    marker=dict(color='#FF6B6B')
))
fig.add_trace(go.Scatter(
    x=weight_df['layer'],
    y=weight_df['min'],
    mode='markers+lines',
    name='Min',
    marker=dict(color='#4ECDC4')
))
fig.add_trace(go.Scatter(
    x=weight_df['layer'],
    y=weight_df['mean'],
    mode='markers+lines',
    name='Mean',
    marker=dict(color='#FFE66D')
))

fig.update_layout(
    template='plotly_dark',
    title="Weight Range by Layer",
    xaxis_title="Layer",
    yaxis_title="Weight Value",
    xaxis_tickangle=-45,
    height=400
)
st.plotly_chart(fig, width='stretch')

st.info("**Data Source**: Weights extracted from actual trained checkpoint `best_model.pt`")


"""
CHRONOS Dashboard - Dataset Explorer
Explore the Elliptic dataset with REAL DATA.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Dataset Explorer", page_icon="📊", layout="wide")

st.title("📊 Dataset Explorer")
st.markdown("Explore the Elliptic Bitcoin Transaction Dataset - **Real Data**")
st.markdown("---")

# Load real data
@st.cache_data
def load_real_data():
    """Load real statistics from saved CSV files."""
    data = {}
    
    # Dataset stats
    if os.path.exists('results/real_data/dataset_stats.csv'):
        data['stats'] = pd.read_csv('results/real_data/dataset_stats.csv').iloc[0].to_dict()
    else:
        data['stats'] = None
    
    # Timestep stats
    if os.path.exists('results/real_data/timestep_stats.csv'):
        data['timesteps'] = pd.read_csv('results/real_data/timestep_stats.csv')
    else:
        data['timesteps'] = None
    
    # Class distribution
    if os.path.exists('results/real_data/class_distribution.csv'):
        data['classes'] = pd.read_csv('results/real_data/class_distribution.csv')
    else:
        data['classes'] = None
    
    # Degree distribution
    if os.path.exists('results/real_data/degree_distribution.csv'):
        data['degrees'] = pd.read_csv('results/real_data/degree_distribution.csv')
    else:
        data['degrees'] = None
        
    return data

data = load_real_data()

if data['stats'] is None:
    st.error("Real data not found! Please run: `python scripts/generate_real_stats.py`")
    st.stop()

stats = data['stats']

# Dataset Stats
st.subheader("📈 Dataset Statistics (Real)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Nodes", f"{stats['total_nodes']:,}", "Transactions")
with col2:
    st.metric("Total Edges", f"{stats['total_edges']:,}", "Payment flows")
with col3:
    st.metric("Timesteps", int(stats['n_timesteps']), "~2 weeks each")
with col4:
    st.metric("Features", int(stats['n_features']), "Per transaction")

st.markdown("---")

# Class Distribution
st.subheader("⚖️ Class Distribution (Real)")

col1, col2 = st.columns(2)

with col1:
    if data['classes'] is not None:
        class_df = data['classes']
        
        fig = px.pie(class_df, values='count', names='class',
                     color='class',
                     color_discrete_map={
                         'illicit': '#FF6B6B',
                         'licit': '#4ECDC4',
                         'unknown': '#888'
                     },
                     title="Transaction Class Distribution")
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, width='stretch')

with col2:
    st.markdown(f"""
    ### Class Imbalance (Real Data)
    
    | Class | Count | Percentage |
    |-------|-------|------------|
    | **Illicit** | {stats['n_illicit']:,} | {stats['n_illicit']/stats['total_nodes']*100:.1f}% |
    | **Licit** | {stats['n_licit']:,} | {stats['n_licit']/stats['total_nodes']*100:.1f}% |
    | **Unknown** | {stats['n_unknown']:,} | {stats['n_unknown']/stats['total_nodes']*100:.1f}% |
    
    #### Among labeled data:
    - **{stats['n_illicit']/(stats['n_licit']+1):.0f}:1 ratio** (illicit:licit)
    - This is why we use **Focal Loss**
    """)

st.markdown("---")

# Temporal Distribution
st.subheader("⏱️ Temporal Distribution (Real)")

if data['timesteps'] is not None:
    ts_df = data['timesteps']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ts_df['timestep'], y=ts_df['illicit'], name='Illicit', marker_color='#FF6B6B'))
    fig.add_trace(go.Bar(x=ts_df['timestep'], y=ts_df['licit'], name='Licit', marker_color='#4ECDC4'))
    
    fig.update_layout(
        template='plotly_dark',
        title="Labeled Transactions Over Timesteps",
        xaxis_title="Timestep",
        yaxis_title="Count",
        barmode='stack',
        height=400
    )
    
    # Add train/val/test regions
    fig.add_vrect(x0=1, x1=34, fillcolor="#4ECDC4", opacity=0.1, 
                  annotation_text="Train", annotation_position="top left")
    fig.add_vrect(x0=35, x1=42, fillcolor="#FFE66D", opacity=0.1,
                  annotation_text="Val", annotation_position="top left")
    fig.add_vrect(x0=43, x1=49, fillcolor="#FF6B6B", opacity=0.1,
                  annotation_text="Test", annotation_position="top left")
    
    st.plotly_chart(fig, width='stretch')

col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"**Training Set**\n\nTimesteps 1-34\n\n{stats['n_train']:,} transactions")
with col2:
    st.warning(f"**Validation Set**\n\nTimesteps 35-42\n\n{stats['n_val']:,} transactions")
with col3:
    st.error(f"**Test Set**\n\nTimesteps 43-49\n\n{stats['n_test']:,} transactions")

st.markdown("---")

# Degree Distribution
st.subheader("🕸️ Graph Topology (Real)")

col1, col2 = st.columns(2)

with col1:
    if data['degrees'] is not None:
        deg_df = data['degrees']
        
        fig = px.bar(deg_df[deg_df['degree'] <= 20], x='degree', y='count',
                     title="Node Degree Distribution")
        fig.update_layout(template='plotly_dark', height=350)
        fig.update_traces(marker_color='#4ECDC4')
        st.plotly_chart(fig, width='stretch')

with col2:
    st.markdown(f"""
    ### Graph Properties (Real)
    
    | Property | Value |
    |----------|-------|
    | **Total Nodes** | {stats['total_nodes']:,} |
    | **Total Edges** | {stats['total_edges']:,} |
    | **Avg. Degree** | {2 * stats['total_edges'] / stats['total_nodes']:.2f} |
    | **Density** | {stats['total_edges'] / (stats['total_nodes'] * (stats['total_nodes']-1)):.2e} |
    | **Graph Type** | Directed |
    
    #### Key Insights
    - **Power-law distribution**: Few high-degree nodes
    - **Sparse graph**: Most nodes have few connections
    """)

st.markdown("---")

# Data Source
st.info("""
**Data Source**: All statistics on this page are computed from the actual Elliptic dataset.
No simulated or synthetic data is used.
""")


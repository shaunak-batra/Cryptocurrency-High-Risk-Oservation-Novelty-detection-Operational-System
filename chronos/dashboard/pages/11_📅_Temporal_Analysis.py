"""
CHRONOS Dashboard - Temporal Analysis
Analyze class distribution across timesteps.
Uses REAL DATA from Elliptic dataset.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="Temporal Analysis", page_icon="📅", layout="wide")

st.title("📅 Temporal Analysis")
st.markdown("Analyze how transactions evolve over timesteps - **Real Data**")
st.markdown("---")

# Load real data
@st.cache_data
def load_temporal_data():
    data = {}
    if os.path.exists('results/real_data/class_by_timestep.csv'):
        data['class_ts'] = pd.read_csv('results/real_data/class_by_timestep.csv')
    if os.path.exists('results/real_data/timestep_stats.csv'):
        data['ts_stats'] = pd.read_csv('results/real_data/timestep_stats.csv')
    return data

data = load_temporal_data()

if 'class_ts' not in data:
    st.error("Temporal data not found! Run: `python scripts/generate_real_analysis.py`")
    st.stop()

st.success("✅ All data computed from actual Elliptic dataset - no simulation")

ts_df = data['class_ts']

# Overview
st.subheader("📊 Transaction Volume Over Time")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    subplot_titles=["Transactions by Class", "Illicit Ratio"])

# Stacked bar chart
fig.add_trace(go.Bar(x=ts_df['timestep'], y=ts_df['illicit'], name='Illicit', 
                     marker_color='#FF6B6B'), row=1, col=1)
fig.add_trace(go.Bar(x=ts_df['timestep'], y=ts_df['licit'], name='Licit',
                     marker_color='#4ECDC4'), row=1, col=1)

# Illicit ratio line
fig.add_trace(go.Scatter(x=ts_df['timestep'], y=ts_df['illicit_ratio'], 
                        mode='lines+markers', name='Illicit Ratio',
                        line=dict(color='#FFE66D', width=2)), row=2, col=1)

# Add train/val/test regions
for row in [1, 2]:
    fig.add_vrect(x0=1, x1=34, fillcolor="#4ECDC4", opacity=0.05, row=row, col=1)
    fig.add_vrect(x0=35, x1=42, fillcolor="#FFE66D", opacity=0.05, row=row, col=1)
    fig.add_vrect(x0=43, x1=49, fillcolor="#FF6B6B", opacity=0.05, row=row, col=1)

fig.update_layout(
    template='plotly_dark',
    height=600,
    barmode='stack'
)
fig.update_yaxes(title_text="Count", row=1, col=1)
fig.update_yaxes(title_text="Ratio", row=2, col=1)
fig.update_xaxes(title_text="Timestep", row=2, col=1)

st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Split statistics
st.subheader("📈 Statistics by Data Split")

col1, col2, col3 = st.columns(3)

with col1:
    train_df = ts_df[ts_df['timestep'] <= 34]
    st.markdown("### Train (1-34)")
    st.metric("Total Illicit", f"{train_df['illicit'].sum():,}")
    st.metric("Total Licit", f"{train_df['licit'].sum():,}")
    st.metric("Avg Illicit Ratio", f"{train_df['illicit_ratio'].mean():.2%}")

with col2:
    val_df = ts_df[(ts_df['timestep'] >= 35) & (ts_df['timestep'] <= 42)]
    st.markdown("### Validation (35-42)")
    st.metric("Total Illicit", f"{val_df['illicit'].sum():,}")
    st.metric("Total Licit", f"{val_df['licit'].sum():,}")
    st.metric("Avg Illicit Ratio", f"{val_df['illicit_ratio'].mean():.2%}")

with col3:
    test_df = ts_df[ts_df['timestep'] >= 43]
    st.markdown("### Test (43-49)")
    st.metric("Total Illicit", f"{test_df['illicit'].sum():,}")
    st.metric("Total Licit", f"{test_df['licit'].sum():,}")
    st.metric("Avg Illicit Ratio", f"{test_df['illicit_ratio'].mean():.2%}")

st.markdown("---")

# Temporal drift analysis
st.subheader("🔄 Class Imbalance Shift")

fig = go.Figure()

# Calculate rolling average
ts_df['illicit_ratio_ma'] = ts_df['illicit_ratio'].rolling(3, min_periods=1).mean()

fig.add_trace(go.Scatter(
    x=ts_df['timestep'], y=ts_df['illicit_ratio'],
    mode='lines+markers', name='Illicit Ratio',
    line=dict(color='#FF6B6B', width=1),
    marker=dict(size=5)
))

fig.add_trace(go.Scatter(
    x=ts_df['timestep'], y=ts_df['illicit_ratio_ma'],
    mode='lines', name='3-step Moving Avg',
    line=dict(color='#FFE66D', width=3)
))

fig.update_layout(
    template='plotly_dark',
    title="Illicit Ratio Over Time",
    xaxis_title="Timestep",
    yaxis_title="Illicit Ratio",
    height=400
)

# Add regions
fig.add_vrect(x0=1, x1=34, fillcolor="#4ECDC4", opacity=0.05, 
              annotation_text="Train", annotation_position="top left")
fig.add_vrect(x0=35, x1=42, fillcolor="#FFE66D", opacity=0.05,
              annotation_text="Val", annotation_position="top left")
fig.add_vrect(x0=43, x1=49, fillcolor="#FF6B6B", opacity=0.05,
              annotation_text="Test", annotation_position="top left")

st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Interpretation
st.subheader("📝 Key Observations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Temporal Split Rationale
    
    We use **temporal splits** (not random) because:
    - Real-world deployment is temporal
    - Avoids data leakage from future
    - Tests model's ability to generalize
    
    This is a **more challenging evaluation** than random splits.
    """)

with col2:
    st.markdown("""
    ### Class Imbalance Shift
    
    The test set has **different class distribution** than training:
    - Model must handle distribution shift
    - Validates robustness of Focal Loss
    - More realistic evaluation scenario
    """)

st.info("**Data Source**: All statistics computed from actual Elliptic dataset labels and timesteps")


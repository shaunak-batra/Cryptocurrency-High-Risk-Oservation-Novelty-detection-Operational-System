"""
CHRONOS Dashboard - Training Curves
Display training history and learning curves from model training.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="Training Curves", page_icon="📈", layout="wide")

st.title("📈 Training Curves")
st.markdown("Model training history and learning curves")
st.markdown("---")

@st.cache_data
def load_training_history():
    """Load training history from CSV."""
    path = 'results/real_data/training_history.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df, True
    return None, False

history_df, has_history = load_training_history()

if not has_history:
    st.warning("⚠️ No training history found. Run the training notebook in Colab to generate it.")
    st.stop()

st.success("✅ Training history loaded from Colab training run")

# Summary metrics
st.subheader("📊 Training Summary")

col1, col2, col3, col4 = st.columns(4)

best_epoch = history_df['val_f1'].idxmax()
best_val_f1 = history_df['val_f1'].max()
final_train_loss = history_df['train_loss'].iloc[-1]
final_val_loss = history_df['val_loss'].iloc[-1]

with col1:
    st.metric("Best Val F1", f"{best_val_f1:.4f}", delta=f"Epoch {best_epoch + 1}")
with col2:
    st.metric("Final Train Loss", f"{final_train_loss:.4f}")
with col3:
    st.metric("Final Val Loss", f"{final_val_loss:.4f}")
with col4:
    st.metric("Total Epochs", f"{len(history_df)}")

st.markdown("---")

# Loss curves
st.subheader("📉 Loss Curves")

fig = make_subplots(rows=1, cols=2, subplot_titles=('Training & Validation Loss', 'Loss Gap'))

fig.add_trace(
    go.Scatter(x=history_df['epoch'], y=history_df['train_loss'], 
               name='Train Loss', line=dict(color='#4CAF50')),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=history_df['epoch'], y=history_df['val_loss'], 
               name='Val Loss', line=dict(color='#FF5722')),
    row=1, col=1
)

# Loss gap (overfitting indicator)
loss_gap = history_df['val_loss'] - history_df['train_loss']
fig.add_trace(
    go.Scatter(x=history_df['epoch'], y=loss_gap, 
               name='Gap (Val - Train)', fill='tozeroy', line=dict(color='#9C27B0')),
    row=1, col=2
)

fig.update_layout(template='plotly_dark', height=400)
st.plotly_chart(fig, use_container_width=True, key="loss_curves")

st.markdown("---")

# F1 Score curve
st.subheader("🎯 F1 Score Progress")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=history_df['epoch'], y=history_df['train_f1'],
    name='Train F1', line=dict(color='#4CAF50', width=2)
))
fig.add_trace(go.Scatter(
    x=history_df['epoch'], y=history_df['val_f1'],
    name='Val F1', line=dict(color='#2196F3', width=2)
))

# Add best epoch marker
fig.add_vline(x=best_epoch + 1, line_dash="dash", line_color="gold", 
              annotation_text=f"Best: {best_val_f1:.4f}")

fig.update_layout(
    template='plotly_dark',
    height=400,
    xaxis_title='Epoch',
    yaxis_title='F1 Score',
    yaxis_range=[0, 1]
)
st.plotly_chart(fig, use_container_width=True, key="f1_curve")

st.markdown("---")

# Precision & Recall
st.subheader("⚖️ Precision & Recall Tradeoff")

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df['epoch'], y=history_df['val_precision'],
        name='Precision', line=dict(color='#E91E63', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=history_df['epoch'], y=history_df['val_recall'],
        name='Recall', line=dict(color='#00BCD4', width=2)
    ))
    fig.update_layout(template='plotly_dark', height=350, title='Precision & Recall Over Time')
    st.plotly_chart(fig, use_container_width=True, key="prec_recall")

with col2:
    # PR tradeoff scatter
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df['val_recall'],
        y=history_df['val_precision'],
        mode='markers+lines',
        marker=dict(color=history_df['epoch'], colorscale='Viridis', showscale=True, 
                   colorbar=dict(title='Epoch')),
        line=dict(color='gray', width=1)
    ))
    fig.update_layout(
        template='plotly_dark', height=350,
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision'
    )
    st.plotly_chart(fig, use_container_width=True, key="pr_curve")

st.markdown("---")

# Training progress table
st.subheader("📋 Epoch Details")

# Show every 10th epoch + last epoch
display_df = history_df[history_df['epoch'] % 10 == 0].copy()
if len(history_df) - 1 not in display_df['epoch'].values:
    display_df = pd.concat([display_df, history_df.iloc[[-1]]])

display_df = display_df.round(4)
st.dataframe(display_df, use_container_width=True)

st.info(f"Training completed in {len(history_df)} epochs. Best validation F1: {best_val_f1:.4f} at epoch {best_epoch + 1}")

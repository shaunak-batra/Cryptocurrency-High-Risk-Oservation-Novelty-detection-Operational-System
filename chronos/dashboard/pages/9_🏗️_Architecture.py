"""
CHRONOS Dashboard - Architecture Visualization
Animated architecture diagram showing data flow.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

st.set_page_config(page_title="Architecture", page_icon="🏗️", layout="wide")

st.title("🏗️ CHRONOS Architecture")
st.markdown("Interactive visualization of the model architecture")
st.markdown("---")

# Load model info
@st.cache_data
def get_model_info():
    try:
        checkpoint = torch.load('checkpoints/chronos_experiment/best_model.pt', 
                               map_location='cpu', weights_only=False)
        weights = checkpoint['model']
        
        layers = {}
        total_params = 0
        for name, param in weights.items():
            layer_name = name.split('.')[0]
            if layer_name not in layers:
                layers[layer_name] = {'params': 0, 'shapes': []}
            layers[layer_name]['params'] += param.numel()
            layers[layer_name]['shapes'].append(f"{name}: {list(param.shape)}")
            total_params += param.numel()
        
        return layers, total_params, True
    except:
        return {}, 986626, False

layers, total_params, model_loaded = get_model_info()

if model_loaded:
    st.success(f"✅ Loaded model with {total_params:,} parameters")

# Architecture overview
st.subheader("📊 Model Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Input Features", "235")
with col2:
    st.metric("Hidden Dimension", "256")
with col3:
    st.metric("GAT Layers", "3")
with col4:
    st.metric("Total Parameters", f"{total_params:,}")

st.markdown("---")

# Interactive Architecture Diagram
st.subheader("🎨 Interactive Architecture Diagram")

# Create architecture visualization
fig = go.Figure()

# Layer positions
layer_x = [0, 0.2, 0.4, 0.5, 0.5, 0.7, 0.85, 1.0]
layer_y = [0.5, 0.5, 0.7, 0.85, 0.15, 0.5, 0.5, 0.5]
layer_names = [
    "Input\n(235 features)",
    "Input\nProjection",
    "GAT Layer 1",
    "GAT Layer 2",
    "Temporal\nEncoder",
    "GAT Layer 3",
    "Concat\n& MLP",
    "Output\n(2 classes)"
]
layer_colors = ['#4ECDC4', '#4ECDC4', '#FF6B6B', '#FF6B6B', '#FFE66D', '#FF6B6B', '#4ECDC4', '#4ECDC4']
layer_sizes = [80, 60, 50, 50, 50, 50, 60, 80]

# Add layers
for i, (x, y, name, color, size) in enumerate(zip(layer_x, layer_y, layer_names, layer_colors, layer_sizes)):
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        marker=dict(size=size, color=color, line=dict(width=2, color='white')),
        text=[name],
        textposition='middle center',
        textfont=dict(size=9, color='white'),
        hoverinfo='text',
        hovertext=f"{name.replace(chr(10), ' ')}"
    ))

# Connections
connections = [
    (0, 1), (1, 2), (2, 3), (3, 5),  # Main path
    (1, 4), (4, 6),  # Temporal branch
    (5, 6), (6, 7)   # To output
]

for i, j in connections:
    fig.add_trace(go.Scatter(
        x=[layer_x[i], layer_x[j]],
        y=[layer_y[i], layer_y[j]],
        mode='lines',
        line=dict(color='#888', width=3),
        hoverinfo='none'
    ))

# Add annotations
fig.add_annotation(x=0.3, y=0.8, text="Multi-Head<br>Attention", showarrow=False, 
                   font=dict(size=10, color='#888'))
fig.add_annotation(x=0.35, y=0.25, text="GRU +<br>Temporal", showarrow=False,
                   font=dict(size=10, color='#888'))

fig.update_layout(
    template='plotly_dark',
    showlegend=False,
    xaxis=dict(visible=False, range=[-0.1, 1.1]),
    yaxis=dict(visible=False, range=[-0.1, 1.1]),
    height=500,
    title="CHRONOS-Net Architecture"
)

st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Layer-by-layer breakdown
st.subheader("📋 Layer-by-Layer Breakdown")

# Create detailed table
layer_details = [
    {"Layer": "Input", "Input Shape": "(N, 235)", "Output Shape": "(N, 256)", "Parameters": "60,416", "Description": "Linear projection + ReLU"},
    {"Layer": "Temporal Encoder", "Input Shape": "(N, 235)", "Output Shape": "(N, 256)", "Parameters": "~180K", "Description": "GRU + position encoding"},
    {"Layer": "GAT Layer 1", "Input Shape": "(N, 256)", "Output Shape": "(N, 256)", "Parameters": "~133K", "Description": "8 attention heads"},
    {"Layer": "GAT Layer 2", "Input Shape": "(N, 256)", "Output Shape": "(N, 256)", "Parameters": "~133K", "Description": "8 attention heads"},
    {"Layer": "GAT Layer 3", "Input Shape": "(N, 256)", "Output Shape": "(N, 256)", "Parameters": "~133K", "Description": "8 attention heads"},
    {"Layer": "Concat", "Input Shape": "(N, 512)", "Output Shape": "(N, 512)", "Parameters": "0", "Description": "Concatenate GAT + Temporal"},
    {"Layer": "Classifier", "Input Shape": "(N, 512)", "Output Shape": "(N, 2)", "Parameters": "~132K", "Description": "MLP with dropout"},
]

st.dataframe(pd.DataFrame(layer_details), width='stretch', hide_index=True)

st.markdown("---")

# Parameter distribution
st.subheader("📊 Parameter Distribution")

col1, col2 = st.columns(2)

with col1:
    if layers:
        layer_names_chart = list(layers.keys())
        layer_params = [layers[l]['params'] for l in layer_names_chart]
    else:
        layer_names_chart = ['input_proj', 'temporal', 'gat_layers', 'classifier']
        layer_params = [60416, 180000, 400000, 132000]
    
    fig = go.Figure(data=[go.Pie(
        labels=layer_names_chart,
        values=layer_params,
        hole=0.4,
        marker_colors=['#4ECDC4', '#FFE66D', '#FF6B6B', '#888']
    )])
    fig.update_layout(template='plotly_dark', title="Parameters by Component", height=400)
    st.plotly_chart(fig, width='stretch')

with col2:
    st.markdown("""
    ### Key Design Choices
    
    | Choice | Rationale |
    |--------|-----------|
    | **GAT over GCN** | Learn adaptive neighbor weights |
    | **8 Attention Heads** | Multi-view aggregation |
    | **Temporal Branch** | Capture time patterns |
    | **Skip Connections** | Gradient flow, feature preservation |
    | **Focal Loss** | Handle 9:1 class imbalance |
    | **Dropout (0.3)** | Regularization |
    """)

st.markdown("---")

# Data flow animation
st.subheader("🔄 Data Flow Explanation")

step = st.selectbox("Select Processing Step", [
    "1. Input Features",
    "2. Feature Projection",
    "3. Graph Attention (GAT)",
    "4. Temporal Encoding",
    "5. Feature Concatenation",
    "6. Classification"
])

step_details = {
    "1. Input Features": {
        "title": "Input: 235 Transaction Features",
        "description": """
        Each transaction node has **235 features**:
        - **165 original**: Elliptic dataset features (values, times, aggregated stats)
        - **70 engineered**: PageRank, degree centrality, temporal patterns
        
        ```
        x ∈ ℝ^{N × 235}  where N = number of transactions
        ```
        """,
        "color": "#4ECDC4"
    },
    "2. Feature Projection": {
        "title": "Linear Projection to Hidden Space",
        "description": """
        Project features to 256-dimensional hidden space:
        
        ```
        h = ReLU(W·x + b)   where W ∈ ℝ^{256 × 235}
        ```
        
        This creates a common representation for all downstream processing.
        """,
        "color": "#4ECDC4"
    },
    "3. Graph Attention (GAT)": {
        "title": "3 Graph Attention Layers",
        "description": """
        Learn importance weights for each neighbor:
        
        ```
        α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
        h'_i = σ(Σ α_ij · W·h_j)
        ```
        
        - 8 attention heads per layer
        - Captures structural patterns in transaction graph
        """,
        "color": "#FF6B6B"
    },
    "4. Temporal Encoding": {
        "title": "Temporal Pattern Learning",
        "description": """
        Encode time-based patterns:
        
        ```
        t_norm = timestep / 49
        h_temporal = GRU(features, t_norm)
        ```
        
        - Captures burst patterns (money laundering happens in waves)
        - Encodes relative temporal position
        """,
        "color": "#FFE66D"
    },
    "5. Feature Concatenation": {
        "title": "Combine Spatial and Temporal",
        "description": """
        Concatenate GAT output with temporal encoding:
        
        ```
        h_final = [h_GAT || h_temporal]   dim = 512
        ```
        
        This combines:
        - **Graph structure** (from GAT)
        - **Temporal patterns** (from temporal encoder)
        """,
        "color": "#4ECDC4"
    },
    "6. Classification": {
        "title": "Final Classification",
        "description": """
        MLP classifier with Focal Loss:
        
        ```
        logits = MLP(h_final)   → [licit_score, illicit_score]
        loss = FocalLoss(logits, labels, α=0.25, γ=2.0)
        ```
        
        Output: Probability of transaction being illicit
        """,
        "color": "#4ECDC4"
    }
}

details = step_details[step]
st.markdown(f"### {details['title']}")
st.markdown(details['description'])

import pandas as pd


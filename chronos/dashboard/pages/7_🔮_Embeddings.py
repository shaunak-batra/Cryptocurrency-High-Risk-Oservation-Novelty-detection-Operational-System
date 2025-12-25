"""
CHRONOS Dashboard - Model Embeddings
Visualize learned node representations with t-SNE/UMAP.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import os

st.set_page_config(page_title="Embeddings", page_icon="🔮", layout="wide")

st.title("🔮 Model Embeddings Visualization")
st.markdown("Visualize how the model represents transactions in latent space")
st.markdown("---")

# Load data and compute embeddings
@st.cache_data
def load_and_embed(method='tsne', perplexity=30, n_samples=1000):
    """Load features and compute embeddings."""
    data_dir = 'data/raw/elliptic/raw'
    
    if os.path.exists(f'{data_dir}/elliptic_txs_features.csv'):
        features_df = pd.read_csv(f'{data_dir}/elliptic_txs_features.csv', header=None)
        classes_df = pd.read_csv(f'{data_dir}/elliptic_txs_classes.csv')
        
        X = features_df.iloc[:, 2:].values.astype(np.float32)
        tx_ids = features_df[0].values.astype(str)
        timesteps = features_df[1].values.astype(int)
        
        label_map = {'1': 'Licit', '2': 'Illicit', 'unknown': 'Unknown'}
        classes_dict = dict(zip(classes_df['txId'].astype(str), 
                               classes_df['class'].astype(str).map(lambda x: label_map.get(x, 'Unknown'))))
        y = np.array([classes_dict.get(tx, 'Unknown') for tx in tx_ids])
        
        real_data = True
    else:
        np.random.seed(42)
        n_samples = 1000
        X = np.random.randn(n_samples, 165).astype(np.float32)
        y = np.random.choice(['Licit', 'Illicit', 'Unknown'], n_samples, p=[0.1, 0.3, 0.6])
        timesteps = np.random.randint(1, 50, n_samples)
        tx_ids = [f'tx_{i}' for i in range(n_samples)]
        real_data = False
    
    # Sample for efficiency
    if len(X) > n_samples:
        np.random.seed(42)
        # Stratified sampling
        labeled_idx = np.where(y != 'Unknown')[0]
        unknown_idx = np.where(y == 'Unknown')[0]
        
        n_labeled = min(len(labeled_idx), n_samples // 2)
        n_unknown = min(len(unknown_idx), n_samples - n_labeled)
        
        sample_idx = np.concatenate([
            np.random.choice(labeled_idx, n_labeled, replace=False),
            np.random.choice(unknown_idx, n_unknown, replace=False)
        ])
        
        X = X[sample_idx]
        y = y[sample_idx]
        timesteps = timesteps[sample_idx]
        tx_ids = [tx_ids[i] for i in sample_idx]
    
    # Apply model projection if available
    try:
        checkpoint = torch.load('checkpoints/chronos_experiment/best_model.pt', 
                               map_location='cpu', weights_only=False)
        weights = checkpoint['model']
        input_weight = weights['input_proj.weight'].numpy()
        input_bias = weights['input_proj.bias'].numpy()
        
        # Project through first layer
        X_proj = np.maximum(0, np.dot(X, input_weight.T) + input_bias)
        used_model = True
    except:
        X_proj = X
        used_model = False
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=500)
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
    
    embeddings = reducer.fit_transform(X_proj)
    
    return embeddings, y, timesteps, tx_ids, real_data, used_model

# Sidebar controls
st.sidebar.subheader("🎛️ Embedding Controls")
method = st.sidebar.selectbox("Method", ['t-SNE', 'PCA'])
n_samples = st.sidebar.slider("Samples", 500, 2000, 1000, 100)
if method == 't-SNE':
    perplexity = st.sidebar.slider("Perplexity", 5, 50, 30, 5)
else:
    perplexity = 30
color_by = st.sidebar.selectbox("Color By", ['Class', 'Timestep'])

# Compute embeddings
with st.spinner(f"Computing {method} embeddings..."):
    embeddings, y, timesteps, tx_ids, real_data, used_model = load_and_embed(
        method.lower().replace('-', ''), perplexity, n_samples
    )

if real_data:
    st.success("✅ Using real Elliptic data")
else:
    st.warning("⚠️ Using synthetic data")

if used_model:
    st.info("🧠 Embeddings computed after model projection layer")

# Main visualization
st.subheader(f"📊 {method} Visualization")

df = pd.DataFrame({
    'x': embeddings[:, 0],
    'y': embeddings[:, 1],
    'Class': y,
    'Timestep': timesteps,
    'TX ID': tx_ids
})

if color_by == 'Class':
    fig = px.scatter(df, x='x', y='y', color='Class',
                     color_discrete_map={
                         'Illicit': '#FF6B6B',
                         'Licit': '#4ECDC4',
                         'Unknown': '#888'
                     },
                     hover_data=['TX ID', 'Timestep'],
                     title=f"{method} Embedding of Transaction Features")
else:
    fig = px.scatter(df, x='x', y='y', color='Timestep',
                     color_continuous_scale='Viridis',
                     hover_data=['TX ID', 'Class'],
                     title=f"{method} Embedding Colored by Timestep")

fig.update_layout(
    template='plotly_dark',
    height=600,
    xaxis_title=f"{method} Dimension 1",
    yaxis_title=f"{method} Dimension 2"
)
fig.update_traces(marker=dict(size=6, opacity=0.7))

st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Class separation analysis
st.subheader("📈 Class Separation Analysis")

col1, col2 = st.columns(2)

with col1:
    # Class distribution in embedding space
    illicit_mask = y == 'Illicit'
    licit_mask = y == 'Licit'
    
    if sum(illicit_mask) > 0 and sum(licit_mask) > 0:
        illicit_center = embeddings[illicit_mask].mean(axis=0)
        licit_center = embeddings[licit_mask].mean(axis=0)
        
        # Distance between class centers
        class_distance = np.linalg.norm(illicit_center - licit_center)
        
        st.metric("Class Center Distance", f"{class_distance:.2f}")
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=embeddings[illicit_mask, 0], y=embeddings[illicit_mask, 1],
            mode='markers', name='Illicit',
            marker=dict(color='#FF6B6B', size=5, opacity=0.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=embeddings[licit_mask, 0], y=embeddings[licit_mask, 1],
            mode='markers', name='Licit',
            marker=dict(color='#4ECDC4', size=5, opacity=0.5)
        ))
        
        # Centers
        fig.add_trace(go.Scatter(
            x=[illicit_center[0]], y=[illicit_center[1]],
            mode='markers', name='Illicit Center',
            marker=dict(color='#FF6B6B', size=20, symbol='x', line=dict(width=3))
        ))
        
        fig.add_trace(go.Scatter(
            x=[licit_center[0]], y=[licit_center[1]],
            mode='markers', name='Licit Center',
            marker=dict(color='#4ECDC4', size=20, symbol='x', line=dict(width=3))
        ))
        
        fig.update_layout(
            template='plotly_dark',
            title="Class Centers (X marks)",
            height=400
        )
        st.plotly_chart(fig, width='stretch')

with col2:
    st.markdown("""
    ### Interpretation
    
    **What we're looking for:**
    - **Good separation** = Classes form distinct clusters
    - **Large center distance** = Model learned discriminative features
    
    **What this shows:**
    - The model's first layer projects transactions into a space
    - Illicit and licit transactions should cluster separately
    - Unknown transactions may overlap both (why they're unknown)
    
    **If classes overlap heavily:**
    - Model may need more training
    - Features may not be discriminative enough
    - Could indicate need for more engineered features
    """)

st.markdown("---")

# Temporal evolution
st.subheader("⏱️ Temporal Evolution")

# Animate through timesteps
timestep_range = st.slider("Timestep Range", int(timesteps.min()), int(timesteps.max()), 
                           (int(timesteps.min()), int(timesteps.max())))

mask = (timesteps >= timestep_range[0]) & (timesteps <= timestep_range[1])
df_filtered = df[mask]

fig = px.scatter(df_filtered, x='x', y='y', color='Class',
                 color_discrete_map={
                     'Illicit': '#FF6B6B',
                     'Licit': '#4ECDC4',
                     'Unknown': '#888'
                 },
                 title=f"Transactions from Timesteps {timestep_range[0]}-{timestep_range[1]}")
fig.update_layout(template='plotly_dark', height=400)
fig.update_traces(marker=dict(size=8, opacity=0.7))
st.plotly_chart(fig, width='stretch')

st.info(f"Showing {len(df_filtered)} transactions in selected range")


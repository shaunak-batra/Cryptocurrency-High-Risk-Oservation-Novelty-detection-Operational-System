"""
CHRONOS Dashboard - Graph Visualization
Interactive visualization of the transaction graph.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os

st.set_page_config(page_title="Graph Visualization", page_icon="🕸️", layout="wide")

st.title("🕸️ Interactive Graph Visualization")
st.markdown("Explore the Bitcoin transaction network structure")
st.markdown("---")

# Load graph data
@st.cache_data
def load_graph_sample():
    """Load a sample of the transaction graph."""
    data_dir = 'data/raw/elliptic/raw'
    
    if os.path.exists(f'{data_dir}/elliptic_txs_edgelist.csv'):
        edges_df = pd.read_csv(f'{data_dir}/elliptic_txs_edgelist.csv')
        features_df = pd.read_csv(f'{data_dir}/elliptic_txs_features.csv', header=None)
        classes_df = pd.read_csv(f'{data_dir}/elliptic_txs_classes.csv')
        
        # Sample for visualization
        sample_size = 500
        sample_edges = edges_df.head(sample_size * 2)
        
        # Get unique nodes
        nodes = set(sample_edges['txId1'].astype(str)) | set(sample_edges['txId2'].astype(str))
        nodes = list(nodes)[:sample_size]
        
        # Get labels
        label_map = {'1': 'Licit', '2': 'Illicit', 'unknown': 'Unknown'}
        classes_dict = dict(zip(classes_df['txId'].astype(str), 
                               classes_df['class'].astype(str).map(lambda x: label_map.get(x, 'Unknown'))))
        
        return nodes, sample_edges, classes_dict, True
    else:
        # Generate synthetic graph
        np.random.seed(42)
        n_nodes = 200
        nodes = [f'tx_{i}' for i in range(n_nodes)]
        edges = [(nodes[i], nodes[j]) for i in range(n_nodes) for j in range(i+1, min(i+5, n_nodes)) if np.random.random() > 0.5]
        edges_df = pd.DataFrame({'txId1': [e[0] for e in edges], 'txId2': [e[1] for e in edges]})
        classes_dict = {n: np.random.choice(['Licit', 'Illicit', 'Unknown'], p=[0.1, 0.2, 0.7]) for n in nodes}
        return nodes, edges_df, classes_dict, False

nodes, edges_df, classes_dict, real_data = load_graph_sample()

if real_data:
    st.success("✅ Loaded real Elliptic dataset")
else:
    st.warning("⚠️ Using synthetic data (Elliptic data not found)")

# Sidebar controls
st.sidebar.subheader("🎛️ Visualization Controls")
layout_algo = st.sidebar.selectbox("Layout Algorithm", ['spring', 'kamada_kawai', 'circular', 'random'])
node_size_by = st.sidebar.selectbox("Node Size By", ['Uniform', 'Degree', 'PageRank'])
show_labels = st.sidebar.checkbox("Show Node Labels", False)
color_by_class = st.sidebar.checkbox("Color by Class", True)
n_nodes_display = st.sidebar.slider("Nodes to Display", 50, 500, 200, 50)

# Build NetworkX graph
G = nx.DiGraph()
for _, row in edges_df.head(n_nodes_display * 3).iterrows():
    n1, n2 = str(row['txId1']), str(row['txId2'])
    if n1 in nodes[:n_nodes_display] or n2 in nodes[:n_nodes_display]:
        G.add_edge(n1, n2)

# Filter to connected nodes
G = G.subgraph([n for n in G.nodes() if G.degree(n) > 0]).copy()

# Compute layout
if layout_algo == 'spring':
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
elif layout_algo == 'kamada_kawai':
    pos = nx.kamada_kawai_layout(G)
elif layout_algo == 'circular':
    pos = nx.circular_layout(G)
else:
    pos = nx.random_layout(G, seed=42)

# Node properties
if node_size_by == 'Degree':
    sizes = [max(10, G.degree(n) * 5) for n in G.nodes()]
elif node_size_by == 'PageRank':
    pr = nx.pagerank(G)
    sizes = [max(10, pr.get(n, 0) * 500) for n in G.nodes()]
else:
    sizes = [15] * len(G.nodes())

if color_by_class:
    colors = ['#FF6B6B' if classes_dict.get(n, 'Unknown') == 'Illicit' 
              else '#4ECDC4' if classes_dict.get(n, 'Unknown') == 'Licit'
              else '#888' for n in G.nodes()]
else:
    colors = ['#4ECDC4'] * len(G.nodes())

# Create plot
st.subheader("📊 Transaction Network")

# Edge traces
edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

fig = go.Figure()

# Edges
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    mode='lines',
    line=dict(width=0.5, color='#555'),
    hoverinfo='none'
))

# Nodes
node_x = [pos[n][0] for n in G.nodes()]
node_y = [pos[n][1] for n in G.nodes()]
node_text = [f"{n}<br>Class: {classes_dict.get(n, 'Unknown')}<br>Degree: {G.degree(n)}" for n in G.nodes()]

fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text' if show_labels else 'markers',
    marker=dict(
        size=sizes,
        color=colors,
        line=dict(width=1, color='white')
    ),
    text=[n[:8] for n in G.nodes()] if show_labels else None,
    textposition='top center',
    hovertext=node_text,
    hoverinfo='text'
))

fig.update_layout(
    template='plotly_dark',
    showlegend=False,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    height=600,
    title=f"Transaction Network ({len(G.nodes())} nodes, {len(G.edges())} edges)"
)

st.plotly_chart(fig, width='stretch')

# Legend
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🔴 **Illicit** transactions")
with col2:
    st.markdown("🟢 **Licit** transactions")
with col3:
    st.markdown("⚫ **Unknown** transactions")

st.markdown("---")

# Graph Statistics
st.subheader("📈 Graph Statistics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Nodes", len(G.nodes()))
with col2:
    st.metric("Edges", len(G.edges()))
with col3:
    avg_deg = sum(dict(G.degree()).values()) / len(G.nodes()) if len(G.nodes()) > 0 else 0
    st.metric("Avg Degree", f"{avg_deg:.2f}")
with col4:
    density = nx.density(G) if len(G.nodes()) > 1 else 0
    st.metric("Density", f"{density:.4f}")

# Degree distribution
st.subheader("📊 Degree Distribution")

degrees = [d for _, d in G.degree()]
fig = px.histogram(degrees, nbins=30, title="Node Degree Distribution")
fig.update_layout(template='plotly_dark', height=300)
fig.update_traces(marker_color='#4ECDC4')
st.plotly_chart(fig, width='stretch')

# Top nodes
st.subheader("🔝 Top Nodes by Degree")
top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
top_df = pd.DataFrame({
    'Node': [n[0][:20] for n in top_nodes],
    'Degree': [n[1] for n in top_nodes],
    'Class': [classes_dict.get(n[0], 'Unknown') for n in top_nodes]
})
st.dataframe(top_df, width='stretch', hide_index=True)


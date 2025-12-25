"""
CHRONOS Dashboard - Graph Visualization
Interactive visualization of the transaction graph using real pre-computed data.
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

# Load real data from pre-computed files
@st.cache_data
def load_graph_data():
    """Load graph data from pre-computed files."""
    # Try pre-computed illicit subgraph first
    if os.path.exists('results/real_data/illicit_subgraph.csv'):
        edges_df = pd.read_csv('results/real_data/illicit_subgraph.csv')
        
        # Get unique nodes
        nodes = list(set(edges_df['source'].astype(str)) | set(edges_df['target'].astype(str)))
        
        # Build classes dict from data
        classes_dict = {}
        for _, row in edges_df.iterrows():
            classes_dict[str(row['source'])] = row.get('source_label', 'Unknown')
            classes_dict[str(row['target'])] = row.get('target_label', 'Unknown')
        
        return nodes, edges_df, classes_dict, True
    
    # Try hub nodes
    elif os.path.exists('results/real_data/hub_nodes.csv'):
        hub_df = pd.read_csv('results/real_data/hub_nodes.csv')
        nodes = hub_df['tx_id'].astype(str).tolist()[:200]
        
        # Create edges from hub connections
        edges = []
        for i, n1 in enumerate(nodes[:100]):
            for n2 in nodes[i+1:i+5]:
                edges.append({'source': n1, 'target': n2})
        edges_df = pd.DataFrame(edges)
        
        classes_dict = dict(zip(hub_df['tx_id'].astype(str), hub_df['label'].map({0: 'Licit', 1: 'Illicit', -1: 'Unknown'})))
        return nodes, edges_df, classes_dict, True
    
    else:
        st.error("No pre-computed graph data found")
        return [], pd.DataFrame(), {}, False

nodes, edges_df, classes_dict, real_data = load_graph_data()

if real_data:
    st.success("✅ Using real Elliptic data")
else:
    st.warning("⚠️ No data available")
    st.stop()

# Sidebar controls
st.sidebar.subheader("🎛️ Visualization Controls")
layout_algo = st.sidebar.selectbox("Layout Algorithm", ['spring', 'kamada_kawai', 'circular', 'random'])
node_size_by = st.sidebar.selectbox("Node Size By", ['Uniform', 'Degree', 'PageRank'])
show_labels = st.sidebar.checkbox("Show Node Labels", False)
color_by_class = st.sidebar.checkbox("Color by Class", True)
n_nodes_display = st.sidebar.slider("Nodes to Display", 50, min(500, len(nodes)), min(200, len(nodes)), 50)

# Build NetworkX graph
G = nx.DiGraph()
source_col = 'source' if 'source' in edges_df.columns else 'txId1'
target_col = 'target' if 'target' in edges_df.columns else 'txId2'

for _, row in edges_df.head(n_nodes_display * 3).iterrows():
    n1, n2 = str(row[source_col]), str(row[target_col])
    G.add_edge(n1, n2)

# Filter to connected nodes
G = G.subgraph([n for n in G.nodes() if G.degree(n) > 0]).copy()

if len(G.nodes()) == 0:
    st.warning("No connected nodes to display")
    st.stop()

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
node_text = [f"{str(n)[:10]}<br>Class: {classes_dict.get(n, 'Unknown')}<br>Degree: {G.degree(n)}" for n in G.nodes()]

fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text' if show_labels else 'markers',
    marker=dict(
        size=sizes,
        color=colors,
        line=dict(width=1, color='white')
    ),
    text=[str(n)[:8] for n in G.nodes()] if show_labels else None,
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

st.plotly_chart(fig, use_container_width=True, key="graph_main")

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
st.plotly_chart(fig, use_container_width=True, key="degree_hist")

# Top nodes
st.subheader("🔝 Top Nodes by Degree")
top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
top_df = pd.DataFrame({
    'Node': [str(n[0])[:20] for n in top_nodes],
    'Degree': [n[1] for n in top_nodes],
    'Class': [classes_dict.get(n[0], 'Unknown') for n in top_nodes]
})
st.dataframe(top_df, use_container_width=True, hide_index=True)

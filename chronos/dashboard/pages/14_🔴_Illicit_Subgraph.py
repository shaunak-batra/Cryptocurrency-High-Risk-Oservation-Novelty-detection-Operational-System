"""
CHRONOS Dashboard - Illicit Subgraph
Visualize subgraph around illicit transactions.
Uses REAL DATA from actual graph.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import os

st.set_page_config(page_title="Illicit Subgraph", page_icon="🔴", layout="wide")

st.title("🔴 Illicit Transaction Subgraph")
st.markdown("Visualize the neighborhood around illicit transactions - **Real Data**")
st.markdown("---")

# Load real data
@st.cache_data
def load_subgraph_data():
    if os.path.exists('results/real_data/illicit_subgraph.csv'):
        return pd.read_csv('results/real_data/illicit_subgraph.csv')
    return None

edges_df = load_subgraph_data()

if edges_df is None:
    st.error("Subgraph data not found! Run: `python scripts/generate_advanced_analysis.py`")
    st.stop()

st.success("✅ Subgraph extracted from actual Elliptic edges around illicit nodes")

# Build graph
G = nx.Graph()
for _, row in edges_df.iterrows():
    G.add_edge(row['source'], row['target'])
    G.nodes[row['source']]['label'] = row['source_label']
    G.nodes[row['target']]['label'] = row['target_label']

# Statistics
st.subheader("📊 Subgraph Statistics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Nodes", G.number_of_nodes())
with col2:
    st.metric("Edges", G.number_of_edges())
with col3:
    illicit_nodes = sum(1 for n in G.nodes() if G.nodes[n].get('label') == 'illicit')
    st.metric("Illicit Nodes", illicit_nodes)
with col4:
    licit_nodes = sum(1 for n in G.nodes() if G.nodes[n].get('label') == 'licit')
    st.metric("Licit Nodes", licit_nodes)

st.markdown("---")

# Visualization
st.subheader("🕸️ Subgraph Visualization")

# Layout
n_nodes_display = st.slider("Nodes to display", 50, min(300, G.number_of_nodes()), 100)

# Sample if needed
if G.number_of_nodes() > n_nodes_display:
    nodes = list(G.nodes())[:n_nodes_display]
    G_display = G.subgraph(nodes)
else:
    G_display = G

# Compute layout
pos = nx.spring_layout(G_display, seed=42, k=2)

# Node properties
node_colors = []
for n in G_display.nodes():
    label = G_display.nodes[n].get('label', 'unknown')
    if label == 'illicit':
        node_colors.append('#FF6B6B')
    elif label == 'licit':
        node_colors.append('#4ECDC4')
    else:
        node_colors.append('#888')

# Create figure
fig = go.Figure()

# Edges
edge_x, edge_y = [], []
for edge in G_display.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    mode='lines',
    line=dict(width=0.5, color='#555'),
    hoverinfo='none'
))

# Nodes
node_x = [pos[n][0] for n in G_display.nodes()]
node_y = [pos[n][1] for n in G_display.nodes()]
node_text = [f"{str(n)[:15]}...<br>Label: {G_display.nodes[n].get('label', 'unknown')}" 
             for n in G_display.nodes()]

fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    marker=dict(
        size=10,
        color=node_colors,
        line=dict(width=1, color='white')
    ),
    hovertext=node_text,
    hoverinfo='text'
))

fig.update_layout(
    template='plotly_dark',
    showlegend=False,
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    height=600,
    title=f"Illicit Transaction Neighborhood ({G_display.number_of_nodes()} nodes)"
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

# Edge analysis
st.subheader("🔗 Edge Type Analysis")

edge_types = edges_df.groupby(['source_label', 'target_label']).size().reset_index(name='count')

fig = px.bar(edge_types, x=edge_types['source_label'] + ' → ' + edge_types['target_label'],
             y='count', title="Edge Types in Illicit Subgraph",
             color='count', color_continuous_scale='Reds')
fig.update_layout(template='plotly_dark', height=350, xaxis_title="Edge Type")
st.plotly_chart(fig, width='stretch')

st.info("**Data Source**: Edges sampled from actual Elliptic graph around 100 illicit nodes")


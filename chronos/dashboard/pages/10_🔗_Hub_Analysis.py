"""
CHRONOS Dashboard - Hub Analysis
Analyze high-degree nodes in the transaction graph.
Uses REAL DATA from Elliptic dataset.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Hub Analysis", page_icon="🔗", layout="wide")

st.title("🔗 Hub Analysis")
st.markdown("Analyze high-degree nodes in the transaction graph - **Real Data**")
st.markdown("---")

# Load real data
@st.cache_data
def load_hub_data():
    data = {}
    if os.path.exists('results/real_data/hub_nodes.csv'):
        data['hubs'] = pd.read_csv('results/real_data/hub_nodes.csv')
    if os.path.exists('results/real_data/hub_stats_by_label.csv'):
        data['stats'] = pd.read_csv('results/real_data/hub_stats_by_label.csv')
    if os.path.exists('results/real_data/edge_statistics.csv'):
        data['edges'] = pd.read_csv('results/real_data/edge_statistics.csv')
    return data

data = load_hub_data()

if 'hubs' not in data:
    st.error("Hub data not found! Run: `python scripts/generate_real_analysis.py`")
    st.stop()

st.success("✅ All data computed from actual Elliptic graph - no simulation")

# Hub overview
st.subheader("📊 Top 100 Hub Nodes")

hubs_df = data['hubs']

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Hubs Analyzed", len(hubs_df))
with col2:
    st.metric("Max Degree", int(hubs_df['total_degree'].max()))
with col3:
    st.metric("Median Hub Degree", int(hubs_df['total_degree'].median()))

# Hub degree distribution
fig = px.histogram(hubs_df, x='total_degree', color='label',
                   title="Degree Distribution of Top 100 Hubs",
                   color_discrete_map={'illicit': '#FF6B6B', 'licit': '#4ECDC4', 'unknown': '#888'})
fig.update_layout(template='plotly_dark', height=400)
st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Hub stats by label
st.subheader("🏷️ Hub Statistics by Class")

if 'stats' in data:
    stats_df = data['stats']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(stats_df, values='count', names='label',
                     color='label',
                     color_discrete_map={'illicit': '#FF6B6B', 'licit': '#4ECDC4', 'unknown': '#888'},
                     title="Hub Nodes by Class")
        fig.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.dataframe(stats_df, width='stretch', hide_index=True)
        
        st.markdown("""
        ### Key Insight
        
        If illicit transactions are **over-represented** among hubs, it suggests:
        - Money laundering uses high-connectivity patterns
        - Mixing services create hub structures
        - Graph features are valuable for detection
        """)

st.markdown("---")

# Edge statistics
st.subheader("🔀 Edge Statistics by Class")

if 'edges' in data:
    edges_df = data['edges']
    
    # Only labeled edges
    labeled_edges = edges_df[edges_df['edge_type'] != 'involves_unknown']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(labeled_edges, x='edge_type', y='count',
                     title="Edge Counts by Type (Labeled Only)",
                     color='edge_type',
                     color_discrete_sequence=['#FF6B6B', '#FFE66D', '#4ECDC4', '#88D8C0'])
        fig.update_layout(template='plotly_dark', height=400, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.dataframe(edges_df, width='stretch', hide_index=True)
        
        # Calculate percentages
        total_labeled = labeled_edges['count'].sum()
        if total_labeled > 0:
            illicit_edges = edges_df[edges_df['edge_type'].str.contains('illicit')]['count'].sum()
            st.metric("Edges involving illicit", f"{illicit_edges:,}", 
                     f"{illicit_edges/total_labeled*100:.1f}% of labeled")

st.markdown("---")

# Top hubs table
st.subheader("🏆 Top 20 Hub Nodes")

top_hubs = hubs_df.head(20)

# Color by label
def color_label(val):
    if val == 'illicit':
        return 'background-color: #FF6B6B30'
    elif val == 'licit':
        return 'background-color: #4ECDC430'
    return ''

st.dataframe(
    top_hubs.style.applymap(color_label, subset=['label']),
    width='stretch',
    hide_index=True
)

st.markdown("---")

st.info("**Data Source**: All statistics computed from actual Elliptic dataset edges and labels")


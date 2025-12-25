"""
CHRONOS Dashboard - Community Analysis
Analyze detected communities in the transaction graph.
Uses REAL DATA from community detection.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Communities", page_icon="🏘️", layout="wide")

st.title("🏘️ Community Analysis")
st.markdown("Analyze communities detected in the transaction graph - **Real Data**")
st.markdown("---")

# Load real data
@st.cache_data
def load_community_data():
    data = {}
    if os.path.exists('results/real_data/communities.csv'):
        data['communities'] = pd.read_csv('results/real_data/communities.csv')
    if os.path.exists('results/real_data/neighbor_aggregates.csv'):
        data['neighbors'] = pd.read_csv('results/real_data/neighbor_aggregates.csv')
    if os.path.exists('results/real_data/neighbor_agg_by_label.csv'):
        data['neighbor_agg'] = pd.read_csv('results/real_data/neighbor_agg_by_label.csv')
    return data

data = load_community_data()

if 'communities' not in data:
    st.error("Community data not found! Run: `python scripts/generate_advanced_analysis.py`")
    st.stop()

st.success("✅ Communities detected using greedy modularity on actual Elliptic graph")

comm_df = data['communities']

# Overview
st.subheader("📊 Community Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Communities Found", len(comm_df))
with col2:
    st.metric("Avg Size", f"{comm_df['size'].mean():.0f}")
with col3:
    st.metric("Max Size", int(comm_df['size'].max()))
with col4:
    st.metric("Total Nodes", int(comm_df['size'].sum()))

st.markdown("---")

# Community size distribution
st.subheader("📈 Community Size Distribution")

fig = px.histogram(comm_df, x='size', nbins=30, title="Distribution of Community Sizes")
fig.update_layout(template='plotly_dark', height=400)
fig.update_traces(marker_color='#4ECDC4')
st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Illicit ratio by community
st.subheader("🎯 Illicit Concentration in Communities")

col1, col2 = st.columns(2)

with col1:
    # Scatter: size vs illicit ratio
    fig = px.scatter(comm_df, x='size', y='illicit_ratio',
                     size='size', color='illicit_ratio',
                     color_continuous_scale='RdYlGn_r',
                     title="Community Size vs Illicit Ratio",
                     hover_data=['n_illicit', 'n_licit'])
    fig.update_layout(template='plotly_dark', height=400)
    st.plotly_chart(fig, width='stretch')

with col2:
    # Top illicit communities
    st.markdown("### Top Illicit-Heavy Communities")
    top_illicit = comm_df[comm_df['illicit_ratio'] > 0].sort_values('illicit_ratio', ascending=False).head(10)
    if len(top_illicit) > 0:
        st.dataframe(top_illicit[['community_id', 'size', 'n_illicit', 'n_licit', 'illicit_ratio']], 
                    width='stretch', hide_index=True)
    else:
        st.info("No communities with labeled illicit nodes found")

st.markdown("---")

# Neighbor analysis
st.subheader("👥 Neighbor Homophily Analysis")

if 'neighbor_agg' in data:
    neigh_df = data['neighbor_agg']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(neigh_df, width='stretch', hide_index=True)
    
    with col2:
        st.markdown("""
        ### Homophily in Transaction Networks
        
        **What this shows:**
        - How likely neighbors are to share the same label
        - `illicit_ratio_mean`: Average ratio of illicit neighbors
        
        **Key insight:**
        - If illicit nodes have more illicit neighbors, 
          the network has **homophily**
        - This is why GNNs work well for this task!
        """)

if 'neighbors' in data:
    neighbor_full = data['neighbors']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(neighbor_full, x='illicit_neighbor_ratio', color='label',
                          nbins=20, barmode='overlay', opacity=0.7,
                          color_discrete_map={'illicit': '#FF6B6B', 'licit': '#4ECDC4'},
                          title="Illicit Neighbor Ratio by Node Label")
        fig.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Statistics
        illicit_nodes = neighbor_full[neighbor_full['label'] == 'illicit']
        licit_nodes = neighbor_full[neighbor_full['label'] == 'licit']
        
        st.metric("Illicit nodes: Avg illicit neighbor ratio", 
                 f"{illicit_nodes['illicit_neighbor_ratio'].mean():.2%}")
        st.metric("Licit nodes: Avg illicit neighbor ratio",
                 f"{licit_nodes['illicit_neighbor_ratio'].mean():.2%}")
        
        if illicit_nodes['illicit_neighbor_ratio'].mean() > licit_nodes['illicit_neighbor_ratio'].mean():
            st.success("✅ Network shows homophily - illicit nodes cluster together")
        else:
            st.info("Network shows mixed patterns")

st.markdown("---")
st.info("**Data Source**: Community detection run on actual Elliptic graph edges using greedy modularity")


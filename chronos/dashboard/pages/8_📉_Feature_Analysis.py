"""
CHRONOS Dashboard - Feature Analysis
Compare feature distributions between illicit and licit transactions.
Uses REAL DATA from Elliptic dataset.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Feature Analysis", page_icon="📉", layout="wide")

st.title("📉 Feature Analysis")
st.markdown("Compare feature distributions between illicit and licit transactions - **Real Data**")
st.markdown("---")

# Load real data
@st.cache_data
def load_feature_data():
    if os.path.exists('results/real_data/feature_comparison.csv'):
        return pd.read_csv('results/real_data/feature_comparison.csv')
    return None

feature_df = load_feature_data()

if feature_df is None:
    st.error("Feature comparison data not found! Run: `python scripts/generate_real_analysis.py`")
    st.stop()

st.success("✅ All data computed from actual Elliptic dataset - no simulation")

# Top discriminative features
st.subheader("🎯 Most Discriminative Features (Sorted by |Mean Difference|)")

top_features = feature_df.head(20)

fig = go.Figure()
fig.add_trace(go.Bar(
    name='Illicit Mean',
    x=top_features['feature_name'],
    y=top_features['illicit_mean'],
    marker_color='#FF6B6B'
))
fig.add_trace(go.Bar(
    name='Licit Mean',
    x=top_features['feature_name'],
    y=top_features['licit_mean'],
    marker_color='#4ECDC4'
))

fig.update_layout(
    template='plotly_dark',
    title="Top 20 Features by Class Separation",
    xaxis_title="Feature",
    yaxis_title="Mean Value",
    barmode='group',
    height=500
)
st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Feature table
st.subheader("📊 Feature Statistics Table")

display_df = feature_df.head(30).copy()
display_df = display_df[['feature_name', 'illicit_mean', 'illicit_std', 'licit_mean', 'licit_std', 'mean_diff']]
display_df.columns = ['Feature', 'Illicit Mean', 'Illicit Std', 'Licit Mean', 'Licit Std', 'Mean Diff']

# Round for display
for col in ['Illicit Mean', 'Illicit Std', 'Licit Mean', 'Licit Std', 'Mean Diff']:
    display_df[col] = display_df[col].round(4)

st.dataframe(display_df, width='stretch', hide_index=True)

st.markdown("---")

# Feature comparison - visual
st.subheader("🔍 Detailed Feature Comparison")

selected_feature = st.selectbox(
    "Select a feature to examine",
    feature_df['feature_name'].tolist()
)

if selected_feature:
    row = feature_df[feature_df['feature_name'] == selected_feature].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Illicit Mean", f"{row['illicit_mean']:.4f}")
        st.metric("Illicit Std", f"{row['illicit_std']:.4f}")
    
    with col2:
        st.metric("Licit Mean", f"{row['licit_mean']:.4f}")
        st.metric("Licit Std", f"{row['licit_std']:.4f}")
    
    with col3:
        st.metric("Mean Difference", f"{row['mean_diff']:.4f}")
        if row['mean_diff'] > 0.5:
            st.success("High discrimination")
        elif row['mean_diff'] > 0.1:
            st.warning("Moderate discrimination")
        else:
            st.info("Low discrimination")

st.markdown("---")

# Interpretation
st.subheader("📝 Interpretation")

st.markdown("""
### What This Shows

These are the **actual mean and standard deviation** of feature values computed directly from the Elliptic dataset:
- **Illicit transactions**: 4,545 labeled illicit addresses
- **Licit transactions**: 42,019 labeled licit addresses

### Key Insights

Features with **large mean differences** are more useful for classification:
- The model can more easily distinguish classes when features differ significantly
- High-std features may have outliers or high variance

### Elliptic Feature Structure
- **Features 0-93**: Direct transaction features (local)
- **Features 94-165**: Aggregated neighbor features (1-hop)

*Note: These are the 165 original features before engineering.*
""")

st.info("**Data Source**: All statistics computed from actual Elliptic dataset `elliptic_txs_features.csv`")


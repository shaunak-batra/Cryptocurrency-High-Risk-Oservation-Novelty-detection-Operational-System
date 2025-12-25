"""
CHRONOS Dashboard - Live Demo
Real-time transaction classification simulation.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import time
import random

st.set_page_config(page_title="Live Demo", page_icon="⚡", layout="wide")

st.title("⚡ Live Transaction Monitoring Demo")
st.markdown("Watch CHRONOS classify transactions in real-time")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    try:
        checkpoint = torch.load('checkpoints/chronos_experiment/best_model.pt', 
                               map_location='cpu', weights_only=False)
        weights = checkpoint['model']
        return weights['input_proj.weight'].numpy(), weights['input_proj.bias'].numpy()
    except:
        return np.random.randn(256, 235), np.zeros(256)

input_weight, input_bias = load_model()

def predict(features):
    """Simple prediction using input projection."""
    h = np.dot(features, input_weight.T) + input_bias
    h = np.maximum(0, h)
    score = np.abs(h).mean() / 10
    return min(1.0, score)

# Sidebar controls
st.sidebar.subheader("⚙️ Demo Controls")
speed = st.sidebar.slider("Speed (transactions/sec)", 0.5, 5.0, 2.0, 0.5)
illicit_rate = st.sidebar.slider("Illicit rate (%)", 5, 30, 10, 5) / 100
n_transactions = st.sidebar.slider("Transactions to process", 10, 50, 20, 5)

# Session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'running' not in st.session_state:
    st.session_state.running = False
if 'stats' not in st.session_state:
    st.session_state.stats = {'total': 0, 'flagged': 0, 'illicit': 0, 'correct': 0}

# Layout
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📊 Live Statistics")
    stats_placeholder = st.empty()
    
    st.markdown("---")
    st.subheader("🎯 Classification Breakdown")
    chart_placeholder = st.empty()

with col1:
    st.subheader("📝 Transaction Log")
    
    start_col, stop_col, clear_col = st.columns(3)
    with start_col:
        start_btn = st.button("▶️ Start Monitoring", width='stretch')
    with stop_col:
        stop_btn = st.button("⏹️ Stop", width='stretch')
    with clear_col:
        clear_btn = st.button("🗑️ Clear", width='stretch')
    
    log_placeholder = st.empty()
    
    if clear_btn:
        st.session_state.transactions = []
        st.session_state.stats = {'total': 0, 'flagged': 0, 'illicit': 0, 'correct': 0}
        st.rerun()
    
    if stop_btn:
        st.session_state.running = False
    
    if start_btn:
        st.session_state.running = True
        
        for i in range(n_transactions):
            if not st.session_state.running:
                break
            
            # Generate transaction
            tx_id = f"tx_{random.randint(100000, 999999)}"
            features = np.random.randn(235).astype(np.float32)
            
            # Determine ground truth
            is_illicit = random.random() < illicit_rate
            if is_illicit:
                features[0:5] *= 2.5
                features[165:170] *= 3
            
            # Predict
            score = predict(features)
            is_flagged = score > 0.5
            
            # Update stats
            st.session_state.stats['total'] += 1
            if is_flagged:
                st.session_state.stats['flagged'] += 1
            if is_illicit:
                st.session_state.stats['illicit'] += 1
            if is_flagged == is_illicit:
                st.session_state.stats['correct'] += 1
            
            # Add to log
            status = "🔴 SUSPICIOUS" if is_flagged else "🟢 LEGITIMATE"
            truth = "⚠️ Illicit" if is_illicit else "✓ Licit"
            
            st.session_state.transactions.insert(0, {
                'Time': time.strftime("%H:%M:%S"),
                'TX ID': tx_id,
                'Score': f"{score:.3f}",
                'Status': status,
                'Actual': truth
            })
            
            # Keep only last 100
            st.session_state.transactions = st.session_state.transactions[:100]
            
            # Update display
            df = pd.DataFrame(st.session_state.transactions[:15])
            log_placeholder.dataframe(df, width='stretch', hide_index=True)
            
            # Update stats
            stats = st.session_state.stats
            with stats_placeholder.container():
                st.metric("Total Processed", stats['total'])
                st.metric("Flagged", stats['flagged'])
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total'] * 100
                    st.metric("Accuracy", f"{acc:.1f}%")
            
            # Update chart
            with chart_placeholder.container():
                chart_data = pd.DataFrame({
                    'Category': ['Flagged', 'Clear'],
                    'Count': [stats['flagged'], stats['total'] - stats['flagged']]
                })
                fig = px.pie(chart_data, values='Count', names='Category',
                            color='Category',
                            color_discrete_map={'Flagged': '#FF6B6B', 'Clear': '#4ECDC4'},
                            hole=0.4)
                fig.update_layout(template='plotly_dark', height=250, showlegend=True)
                st.plotly_chart(fig, width='stretch')
            
            time.sleep(1.0 / speed)
        
        st.session_state.running = False
        st.success(f"✅ Processed {st.session_state.stats['total']} transactions!")

# Show existing log if not running
if not st.session_state.running and st.session_state.transactions:
    df = pd.DataFrame(st.session_state.transactions[:15])
    log_placeholder.dataframe(df, width='stretch', hide_index=True)
    
    stats = st.session_state.stats
    with stats_placeholder.container():
        st.metric("Total Processed", stats['total'])
        st.metric("Flagged", stats['flagged'])
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            st.metric("Accuracy", f"{acc:.1f}%")
    
    with chart_placeholder.container():
        if stats['total'] > 0:
            chart_data = pd.DataFrame({
                'Category': ['Flagged', 'Clear'],
                'Count': [stats['flagged'], stats['total'] - stats['flagged']]
            })
            fig = px.pie(chart_data, values='Count', names='Category',
                        color='Category',
                        color_discrete_map={'Flagged': '#FF6B6B', 'Clear': '#4ECDC4'},
                        hole=0.4)
            fig.update_layout(template='plotly_dark', height=250, showlegend=True)
            st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Manual Transaction Analysis
st.subheader("🔬 Manual Transaction Analysis")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Enter Transaction Features")
    
    tx_id = st.text_input("Transaction ID", value="tx_manual_001")
    
    # Feature sliders for key features
    in_degree = st.slider("In-Degree (connections in)", 0, 50, 5)
    out_degree = st.slider("Out-Degree (connections out)", 0, 50, 3)
    pagerank = st.slider("PageRank (0-100)", 0, 100, 10) / 100
    timestep = st.slider("Timestep", 1, 49, 25)
    
    if st.button("🔍 Analyze Transaction", width='stretch'):
        # Create feature vector
        features = np.random.randn(235).astype(np.float32) * 0.1
        features[165] = in_degree / 10
        features[166] = out_degree / 10
        features[168] = pagerank
        features[185] = timestep / 49
        
        score = predict(features)
        
        with col2:
            st.markdown("#### Analysis Results")
            
            if score > 0.7:
                st.error(f"🔴 **HIGH RISK** - Score: {score:.3f}")
                st.markdown("This transaction shows suspicious patterns.")
            elif score > 0.5:
                st.warning(f"🟡 **MEDIUM RISK** - Score: {score:.3f}")
                st.markdown("This transaction requires further review.")
            else:
                st.success(f"🟢 **LOW RISK** - Score: {score:.3f}")
                st.markdown("This transaction appears legitimate.")
            
            # Feature contribution
            st.markdown("#### Risk Factors")
            factors = pd.DataFrame({
                'Factor': ['In-Degree', 'Out-Degree', 'PageRank', 'Temporal'],
                'Value': [in_degree, out_degree, f"{pagerank:.2f}", timestep],
                'Risk': ['High' if in_degree > 20 else 'Low',
                        'High' if out_degree > 20 else 'Low',
                        'High' if pagerank > 0.5 else 'Low',
                        'Medium' if 15 <= timestep <= 38 else 'Low']
            })
            st.dataframe(factors, width='stretch', hide_index=True)


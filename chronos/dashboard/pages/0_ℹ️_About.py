"""
CHRONOS Dashboard - About This Project
Background and context for the CHRONOS system.
"""
import streamlit as st
import os

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About CHRONOS")
st.markdown("---")

# Project Overview
st.subheader("What is CHRONOS?")

st.markdown("""
**CHRONOS** stands for **C**ryptocurrency **H**igh-**R**isk **O**bservation & **N**ovelty-detection **O**perational **S**ystem.

It's a graph neural network system designed to detect money laundering in Bitcoin transactions. 
The system analyzes the Elliptic dataset, which contains real Bitcoin transactions labeled as licit or illicit.
""")

st.markdown("---")

# Technical Approach
st.subheader("Technical Approach")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Core Components
    
    1. **Graph Attention Networks (GAT)**
       - 3 layers with 8 attention heads
       - Learns which neighbors matter for classification
    
    2. **Temporal Encoding**
       - Captures time-based patterns
       - Money laundering often shows temporal bursts
    
    3. **Focal Loss**
       - Handles 9:1 class imbalance
       - Focuses training on hard examples
    """)

with col2:
    st.markdown("""
    ### Why This Works
    
    Money laundering leaves patterns in transaction graphs:
    - **Structural**: Unusual connectivity patterns
    - **Temporal**: Activity bursts and timing
    - **Feature-based**: Abnormal transaction values
    
    GNNs can learn these patterns directly from the graph structure,
    without manual feature engineering.
    """)

st.markdown("---")

# Results
st.subheader("Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("F1 Score", "0.9853")
with col2:
    st.metric("Precision", "0.9749")
with col3:
    st.metric("Recall", "0.9959")

st.markdown("""
These results are from actual training on the Elliptic dataset using temporal splits 
(train on timesteps 1-34, validate on 35-42, test on 43-49).
""")

st.markdown("---")

# Dataset
st.subheader("The Elliptic Dataset")

st.markdown("""
The [Elliptic dataset](https://www.kaggle.com/ellipticco/elliptic-data-set) is a public Bitcoin transaction graph:

| Statistic | Value |
|-----------|-------|
| Nodes | 203,769 transactions |
| Edges | 234,355 payment flows |
| Features | 165 per transaction |
| Labeled | 46,564 (23%) |
| Illicit | 4,545 (9.8% of labeled) |
| Timesteps | 49 (~2 weeks each) |

The dataset was released by Elliptic, a blockchain analytics company, for research purposes.
""")

st.markdown("---")

# Limitations
st.subheader("Limitations")

st.warning("""
**Honest Assessment**

This project combines established techniques (GAT, Focal Loss, temporal encoding) 
rather than introducing fundamentally new methods. The novelty is in the combination 
and application to cryptocurrency AML.

Key limitations:
- No adversarial robustness testing
- No counterfactual explanations implemented
- Single dataset evaluation
- No production deployment testing
""")

st.markdown("---")

# Contact
st.subheader("Project Context")

st.info("""
This project was developed as a demonstration of applying graph neural networks 
to financial crime detection. It uses real data from the Elliptic dataset and 
represents genuine training results.

All visualizations in this dashboard use actual computed statistics from the 
Elliptic dataset and trained model checkpoint.
""")


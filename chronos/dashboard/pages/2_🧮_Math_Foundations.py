"""
CHRONOS Dashboard - Mathematical Foundations
Explain the GNN, GAT, Focal Loss, and Temporal Encoding mathematics.
"""
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Math Foundations", page_icon="🧮", layout="wide")

st.title("🧮 Mathematical Foundations")
st.markdown("Understanding the mathematics behind CHRONOS")
st.markdown("---")

# Graph Neural Networks
st.subheader("1️⃣ Graph Neural Networks (GNNs)")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    ### Message Passing Framework
    
    GNNs learn by passing messages between connected nodes:
    """)
    
    st.latex(r"""
    h_v^{(l+1)} = \sigma\left( W^{(l)} \cdot \text{AGGREGATE}\left(\{h_u^{(l)} : u \in \mathcal{N}(v)\}\right) \right)
    """)
    
    st.markdown("""
    Where:
    - $h_v^{(l)}$ = Node $v$'s representation at layer $l$
    - $\mathcal{N}(v)$ = Neighbors of node $v$
    - $W^{(l)}$ = Learnable weight matrix
    - $\sigma$ = Activation function (ReLU, ELU)
    """)

with col2:
    # Visualization of message passing
    fig = go.Figure()
    
    # Central node
    fig.add_trace(go.Scatter(
        x=[0.5], y=[0.5], mode='markers+text',
        marker=dict(size=60, color='#4ECDC4'),
        text=['v'], textposition='middle center',
        textfont=dict(size=20, color='white')
    ))
    
    # Neighbor nodes
    neighbors = [(0.2, 0.8), (0.8, 0.8), (0.2, 0.2), (0.8, 0.2)]
    for i, (x, y) in enumerate(neighbors):
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=40, color='#FF6B6B'),
            text=[f'u{i+1}'], textposition='middle center',
            textfont=dict(size=14, color='white')
        ))
        # Edge
        fig.add_trace(go.Scatter(
            x=[x, 0.5], y=[y, 0.5], mode='lines',
            line=dict(color='#888', width=2),
            hoverinfo='none'
        ))
        # Arrow annotation
        fig.add_annotation(
            x=0.5, y=0.5, ax=x, ay=y,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowcolor='#FFE66D'
        )
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        template='plotly_dark',
        height=300,
        title="Message Passing: Neighbors → Center"
    )
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Graph Attention
st.subheader("2️⃣ Graph Attention Networks (GAT)")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    ### Attention Mechanism
    
    GAT learns **importance weights** for each neighbor:
    """)
    
    st.latex(r"""
    \alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [W h_i \| W h_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [W h_i \| W h_k]\right)\right)}
    """)
    
    st.markdown("""
    Then aggregate with attention weights:
    """)
    
    st.latex(r"""
    h_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j\right)
    """)
    
    st.info("""
    **Multi-Head Attention**: We use **8 attention heads** and concatenate their outputs 
    for richer representations.
    """)

with col2:
    # Attention weights visualization
    st.markdown("#### Attention Weight Distribution")
    
    np.random.seed(42)
    attention_weights = np.random.dirichlet(np.ones(5), size=1)[0]
    neighbors = ['u1', 'u2', 'u3', 'u4', 'u5']
    
    fig = go.Figure(go.Bar(
        x=neighbors,
        y=attention_weights,
        marker_color=['#FF6B6B' if w == max(attention_weights) else '#4ECDC4' for w in attention_weights],
        text=[f'{w:.2f}' for w in attention_weights],
        textposition='outside'
    ))
    fig.update_layout(
        template='plotly_dark',
        height=300,
        title="Example: Attention Weights α<sub>ij</sub>",
        yaxis_title="Weight",
        xaxis_title="Neighbor"
    )
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Focal Loss
st.subheader("3️⃣ Focal Loss for Class Imbalance")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    ### The Problem
    
    With **9:1 class imbalance**, standard cross-entropy fails because:
    - Easy negatives dominate the loss
    - Model becomes biased toward majority class
    
    ### Focal Loss Solution
    """)
    
    st.latex(r"""
    FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
    """)
    
    st.markdown("""
    Where:
    - $p_t$ = Probability of correct class
    - $\\alpha = 0.25$ = Class balancing factor
    - $\\gamma = 2.0$ = Focusing parameter
    - $(1-p_t)^\\gamma$ = **Down-weights easy examples**
    """)

with col2:
    # Focal vs Cross-Entropy visualization
    p = np.linspace(0.01, 0.99, 100)
    ce_loss = -np.log(p)
    focal_loss_g1 = -(1-p)**1 * np.log(p)
    focal_loss_g2 = -(1-p)**2 * np.log(p)
    focal_loss_g5 = -(1-p)**5 * np.log(p)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p, y=ce_loss, name='Cross-Entropy (γ=0)', line=dict(color='#888')))
    fig.add_trace(go.Scatter(x=p, y=focal_loss_g1, name='Focal (γ=1)', line=dict(color='#FFE66D')))
    fig.add_trace(go.Scatter(x=p, y=focal_loss_g2, name='Focal (γ=2) ✓', line=dict(color='#4ECDC4', width=3)))
    fig.add_trace(go.Scatter(x=p, y=focal_loss_g5, name='Focal (γ=5)', line=dict(color='#FF6B6B')))
    
    fig.update_layout(
        template='plotly_dark',
        height=350,
        title="Focal Loss vs Cross-Entropy",
        xaxis_title="Probability of Correct Class (p)",
        yaxis_title="Loss",
        yaxis_range=[0, 5]
    )
    st.plotly_chart(fig, width='stretch')

st.success("""
**Key Insight**: With γ=2, well-classified examples (high p) contribute very little to the loss, 
allowing the model to focus on hard misclassified examples.
""")

st.markdown("---")

# Temporal Encoding
st.subheader("4️⃣ Temporal Encoding")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    ### Time-Aware Features
    
    Cryptocurrency transactions have **temporal patterns**:
    - Money laundering happens in **bursts**
    - **Time gaps** matter between transactions
    - **Sequencing** reveals suspicious behavior
    
    ### Our Approach
    
    We encode temporal information as features:
    """)
    
    st.latex(r"""
    h_{temporal} = \text{MLP}\left([t_{norm}, \Delta t_{neighbors}, \text{temporal\_stats}]\right)
    """)
    
    st.markdown("""
    Where:
    - $t_{norm}$ = Normalized timestep (0-1)
    - $\\Delta t_{neighbors}$ = Time gap to neighbors
    - Temporal stats = Mean, std of neighbor times
    """)

with col2:
    # Temporal pattern visualization
    np.random.seed(42)
    timesteps = list(range(1, 50))
    illicit_burst = [2 if 15 <= t <= 20 or 35 <= t <= 38 else 0.3 for t in timesteps]
    licit_steady = [0.5 + np.random.normal(0, 0.1) for _ in timesteps]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timesteps, y=[i*20 for i in illicit_burst], 
                             name='Illicit Activity', fill='tozeroy',
                             line=dict(color='#FF6B6B')))
    fig.add_trace(go.Scatter(x=timesteps, y=[l*20 for l in licit_steady],
                             name='Licit Activity', fill='tozeroy',
                             line=dict(color='#4ECDC4')))
    
    fig.update_layout(
        template='plotly_dark',
        height=300,
        title="Temporal Patterns: Bursts vs Steady",
        xaxis_title="Timestep",
        yaxis_title="Activity Level"
    )
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Summary
st.subheader("📚 Summary")

st.markdown("""
| Component | Purpose | CHRONOS Implementation |
|-----------|---------|------------------------|
| **GNN** | Learn from graph structure | 3-layer message passing |
| **GAT** | Weighted neighbor aggregation | 8 attention heads |
| **Focal Loss** | Handle class imbalance | α=0.25, γ=2.0 |
| **Temporal** | Capture time patterns | Normalized encoding + MLP |
""")


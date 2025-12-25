# CHRONOS Explainability Guide

This document explains the explainability methods in CHRONOS and how to use them.

## Table of Contents

- [Overview](#overview)
- [Counterfactual Explanations](#counterfactual-explanations)
- [SHAP Feature Importance](#shap-feature-importance)
- [Attention Visualization](#attention-visualization)
- [Natural Language Generation](#natural-language-generation)
- [Usage Examples](#usage-examples)

---

## Overview

### Why Explainability Matters

In high-risk AI systems like AML detection:

1. **Regulatory Compliance**: EU AI Act Article 13 requires transparency
2. **Human Oversight**: Analysts need to understand model decisions
3. **Trust**: Explanations build confidence in the system
4. **Actionability**: Users need to know what to change

### Multi-Method Approach

CHRONOS uses **four complementary methods**:

```
┌─────────────────────────────────────────────────────────┐
│                   CHRONOS Explainability                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Counterfactual      "What would need to change?"   │
│     (NOVEL)              → Actionable guidance          │
│                                                          │
│  2. SHAP                "Why this prediction?"          │
│                          → Feature importance           │
│                                                          │
│  3. Attention           "Which neighbors matter?"       │
│                          → Graph structure insights     │
│                                                          │
│  4. Natural Language    "Human-readable summary"        │
│                          → Compliance-ready reports     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key Insight**: No single method is sufficient. Each provides different insights.

---

## Counterfactual Explanations

### What Are Counterfactuals?

**Question**: "What is the minimal change needed to make this transaction low-risk?"

**Example**:

```
Original Transaction: HIGH RISK (score: 0.92)
  - Total transactions (30 days): 47
  - Betweenness centrality: 0.85
  - Mixer interactions: 3

Counterfactual: LOW RISK (score: 0.15)
  - Total transactions (30 days): 12  ← CHANGE: Reduce by 35
  - Betweenness centrality: 0.32      ← CHANGE: Reduce by 0.53
  - Mixer interactions: 0             ← CHANGE: Avoid mixers

Interpretation: This transaction is flagged because of:
  1. Extremely high transaction velocity (47 txs/month)
  2. Acting as intermediary (hub behavior)
  3. Direct interaction with known mixing services
```

### Algorithm: Graph Counterfactual with Temporal Constraints (NOVEL)

**Based On**:

- CF-GNNExplainer (Lucic et al., AISTATS 2022) - for static graphs
- DiCE-ML (Mothilal et al., NeurIPS 2020) - for tabular data

**Novel Extension**: Adapt to temporal graphs with financial domain constraints

#### Loss Function

```python
L_total = L_pred + λ_prox * L_prox + λ_div * L_div + λ_valid * L_valid

# Component 1: Prediction Loss (flip the prediction)
L_pred = (f(x_cf) - y_target)²

# Component 2: Proximity Loss (minimal changes)
L_prox = ||x_cf - x_orig||²

# Component 3: Diversity Loss (diverse counterfactuals)
L_div = -Σ||x_cf_i - x_cf_j||²

# Component 4: Validity Loss (realistic values)
L_valid = Σ penalty(x_cf_feature, constraints)

# Hyperparameters
λ_prox = 0.5    # Balance between flipping prediction and proximity
λ_div = 0.3     # Diversity across multiple counterfactuals
λ_valid = 1.0   # Hard constraints (must be satisfied)
```

#### Constraints

**1. Immutable Features** (cannot change):

```python
immutable_features = [
    range(0, 166),       # Original Elliptic features (anonymized)
    'transaction_id',
    'timestep'
]
```

**Rationale**: Past data is fixed, only future behavior can change

**2. Mutable Features** (can change):

```python
mutable_features = [
    'graph_topology_*',  # Could change connections
    'temporal_*',        # Future behavior
    'amount_*'           # Future transaction amounts
]
```

**3. Monotonicity Constraints** (can only increase):

```python
monotonic_features = [
    'total_tx_count',      # Can't delete past transactions
    'time_since_first_tx', # Time only moves forward
    'cumulative_volume'
]
```

**4. Domain Constraints**:

```python
constraints = {
    'tx_count_*': (0, inf),           # Non-negative
    '*_centrality': (0, 1),            # Normalized [0,1]
    'burstiness': (-1, 1),             # Formula bounds
    'mixer_interactions': (0, inf),    # Non-negative integer
}
```

#### Implementation

```python
def generate_counterfactual(
    transaction: Data,
    model: CHRONOSNet,
    y_target: float = 0.0,  # Target: low-risk
    num_cf: int = 3,         # Generate 3 diverse CFs
    max_iter: int = 1000,
    lr: float = 0.01
) -> List[Data]:
    """
    Generate counterfactual explanations for a transaction.

    Returns
    -------
    List of counterfactuals with minimal changes to make low-risk.
    """
    model.eval()

    # Initialize counterfactuals from original
    x_cfs = [transaction.x.clone().requires_grad_(True) for _ in range(num_cf)]

    # Optimizer
    optimizer = Adam(x_cfs, lr=lr)

    for iteration in range(max_iter):
        optimizer.zero_grad()

        # Compute predictions for all CFs
        preds = [model(cf, transaction.edge_index) for cf in x_cfs]

        # Loss 1: Prediction loss (want to flip to low-risk)
        L_pred = sum((pred - y_target)**2 for pred in preds) / num_cf

        # Loss 2: Proximity loss (minimal changes)
        L_prox = sum((cf - transaction.x).pow(2).mean() for cf in x_cfs) / num_cf

        # Loss 3: Diversity loss (CFs should be different from each other)
        L_div = 0
        for i in range(num_cf):
            for j in range(i+1, num_cf):
                L_div -= (x_cfs[i] - x_cfs[j]).pow(2).mean()
        L_div /= (num_cf * (num_cf - 1) / 2)

        # Loss 4: Validity loss (satisfy constraints)
        L_valid = compute_validity_loss(x_cfs, constraints)

        # Total loss
        L_total = L_pred + 0.5 * L_prox + 0.3 * L_div + 1.0 * L_valid

        # Backward pass
        L_total.backward()
        optimizer.step()

        # Project to valid space
        for cf in x_cfs:
            with torch.no_grad():
                # Immutable features: restore original values
                cf[:, immutable_features] = transaction.x[:, immutable_features]

                # Monotonic features: ensure only increase
                cf[:, monotonic_features] = torch.max(
                    cf[:, monotonic_features],
                    transaction.x[:, monotonic_features]
                )

                # Domain constraints: clip to valid ranges
                for feat, (min_val, max_val) in domain_constraints.items():
                    cf[:, feat] = torch.clamp(cf[:, feat], min_val, max_val)

        # Early stopping: if all CFs are low-risk
        if all(pred < 0.3 for pred in preds) and iteration > 100:
            break

    # Return valid counterfactuals
    return [cf.detach() for cf in x_cfs if model(cf, edge_index) < 0.3]
```

### Interpreting Counterfactuals

**Example Output**:

```python
cfs = generate_counterfactual(transaction, model, y_target=0.0, num_cf=3)

for i, cf in enumerate(cfs):
    changes = (cf.x - transaction.x)[transaction.x != cf.x]
    print(f"Counterfactual {i+1}:")
    print(f"  Risk score: {model(cf, edge_index):.3f}")
    print(f"  Changes required: {len(changes)} features")
    print(f"  Top 3 changes:")
    for feat_idx, delta in top_changes(changes, k=3):
        print(f"    - {feature_names[feat_idx]}: {delta:+.2f}")
```

**Output**:

```
Counterfactual 1:
  Risk score: 0.18
  Changes required: 8 features
  Top 3 changes:
    - tx_count_30d: -35.2  (reduce transaction velocity)
    - betweenness_centrality: -0.53  (reduce hub behavior)
    - mixer_interactions: -3  (avoid mixers)

Counterfactual 2:
  Risk score: 0.22
  Changes required: 12 features
  Top 3 changes:
    - burstiness_7d: -0.67  (reduce transaction bursts)
    - unique_counterparties_7d: -18  (interact with fewer entities)
    - clustering_coefficient: +0.24  (join tight-knit community)

Counterfactual 3:
  Risk score: 0.25
  Changes required: 6 features
  Top 3 changes:
    - neighbor_risk_mean: -0.41  (transact with lower-risk neighbors)
    - 2hop_illicit_exposure: -7  (reduce proximity to illicit nodes)
    - homophily_score: +0.32  (increase community homophily)
```

---

## SHAP Feature Importance

### What is SHAP?

**SHapley Additive exPlanations**: Assigns each feature an importance value for a specific prediction.

**Key Property**: Additive

```
prediction = base_value + SHAP(feature_1) + SHAP(feature_2) + ... + SHAP(feature_n)
```

### Implementation

```python
import shap

# Initialize explainer
background = train_data.x[:100]  # 100 random training samples
explainer = shap.GradientExplainer(model, background)

# Compute SHAP values for a transaction
shap_values = explainer.shap_values(transaction.x)

# shap_values.shape: [num_features]
# shap_values[i] = contribution of feature i to the prediction
```

### Visualization

#### 1. Waterfall Plot (Individual Explanation)

```python
import shap

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values,
        base_values=explainer.expected_value,
        data=transaction.x,
        feature_names=feature_names
    )
)
```

**Output**:

```
                   E[f(x)] = 0.15  (base value)
betweenness_centrality      +0.32  → 0.47
mixer_interactions          +0.25  → 0.72
tx_count_30d                +0.18  → 0.90
clustering_coefficient      -0.05  → 0.85
burstiness_7d               +0.07  → 0.92
                   f(x) = 0.92  (final prediction)
```

**Interpretation**:

- Base risk: 0.15 (average across training set)
- Betweenness centrality: +0.32 (acting as hub → increases risk)
- Mixer interactions: +0.25 (interacting with mixers → increases risk)
- Clustering coefficient: -0.05 (low clustering → slightly reduces risk)

#### 2. Beeswarm Plot (Global Feature Importance)

```python
# Compute SHAP for entire test set
shap_values_test = explainer.shap_values(test_data.x)

shap.summary_plot(shap_values_test, test_data.x, feature_names=feature_names)
```

**Output**: Shows distribution of SHAP values for each feature

#### 3. Feature Importance Ranking

```python
# Global feature importance (mean |SHAP|)
importance = np.abs(shap_values_test).mean(axis=0)
ranking = np.argsort(-importance)

print("Top 10 Most Important Features:")
for i, feat_idx in enumerate(ranking[:10]):
    print(f"{i+1}. {feature_names[feat_idx]}: {importance[feat_idx]:.4f}")
```

**Output**:

```
Top 10 Most Important Features:
1. betweenness_centrality: 0.2145
2. mixer_interactions: 0.1832
3. tx_count_30d: 0.1567
4. burstiness_7d: 0.1423
5. neighbor_risk_mean: 0.1289
6. 2hop_illicit_exposure: 0.1156
7. clustering_coefficient: 0.0987
8. degree_out: 0.0876
9. benford_divergence: 0.0765
10. tx_volume_30d: 0.0654
```

---

## Attention Visualization

### GAT Attention Weights

Graph Attention Networks learn which neighbors are important for each node.

**Attention Weights**: α_ij = importance of neighbor j for node i

```python
# Extract attention weights from model
model.eval()
with torch.no_grad():
    output, (attn1, attn2, attn3) = model(transaction.x, transaction.edge_index, return_attention=True)

# attn1, attn2, attn3: [num_edges, num_heads]
# Average across heads
attn_avg = (attn1.mean(dim=1) + attn2.mean(dim=1) + attn3.mean(dim=1)) / 3
```

### Visualization 1: Attention Heatmap

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create adjacency matrix with attention weights
num_nodes = transaction.x.shape[0]
attn_matrix = torch.zeros(num_nodes, num_nodes)
for (i, j), weight in zip(transaction.edge_index.T, attn_avg):
    attn_matrix[i, j] = weight

# Plot
sns.heatmap(attn_matrix.numpy(), cmap='YlOrRd', vmin=0, vmax=1)
plt.title("GAT Attention Weights")
plt.xlabel("Target Node")
plt.ylabel("Source Node")
plt.show()
```

### Visualization 2: Subgraph with Attention-Weighted Edges

```python
import networkx as nx

# Build graph
G = nx.DiGraph()
for (i, j), weight in zip(transaction.edge_index.T, attn_avg):
    if weight > 0.1:  # Only show important edges
        G.add_edge(int(i), int(j), weight=float(weight))

# Draw
pos = nx.spring_layout(G)
edges = G.edges()
weights = [G[u][v]['weight'] for u, v in edges]

nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos, width=weights*5, alpha=0.6, edge_color=weights, edge_cmap=plt.cm.Reds)
plt.title("Transaction Graph (Attention-Weighted Edges)")
plt.show()
```

### Interpretation

**High Attention Weight** (α > 0.7):

- This neighbor is highly important for the prediction
- Strong influence on risk score

**Low Attention Weight** (α < 0.3):

- This neighbor is less relevant
- Weak influence on risk score

**Example**:

```
Node 42 (target transaction):
  Neighbor 13: α = 0.92  ← High attention (suspicious neighbor)
  Neighbor 78: α = 0.15  ← Low attention (benign neighbor)
  Neighbor 91: α = 0.83  ← High attention (another suspicious neighbor)

Interpretation: Node 42 is flagged primarily due to its connections
to nodes 13 and 91, which are high-risk.
```

---

## Natural Language Generation

### Template-Based Approach

**Why Template-Based?**

1. **No hallucination**: LLMs can generate plausible but false explanations
2. **Patent-safe**: Avoids PayPal patent US11682018B2 on "narrative generation" with SHAP
3. **Consistent format**: Compliance-ready reports

### Templates

#### High-Risk Template

```python
HIGH_RISK_TEMPLATE = """
Transaction flagged as HIGH RISK (score: {score:.2f})

REASONS:
• {reason_1}
• {reason_2}
• {reason_3}

TO BE CONSIDERED LOW-RISK, THIS TRANSACTION WOULD NEED:
• {change_1}, OR
• {change_2}, AND
• {change_3}

RECOMMENDED ACTION:
{action}

RISK BREAKDOWN:
  Baseline risk:          {base:.2f}
  Graph structure:        {graph_contrib:+.2f}
  Temporal patterns:      {temporal_contrib:+.2f}
  Amount patterns:        {amount_contrib:+.2f}
  Entity associations:    {entity_contrib:+.2f}
  ────────────────────────────────
  Final risk score:       {score:.2f}

SIMILAR CASES:
- Transaction #15423: Risk 0.89, flagged for mixer interactions
- Transaction #29871: Risk 0.91, flagged for high velocity
- Transaction #34512: Risk 0.94, flagged for hub behavior

ANALYST NOTES:
[To be filled by compliance officer]
"""
```

#### Feature-to-Reason Mapping

```python
FEATURE_REASONS = {
    'betweenness_centrality': {
        'high': 'Acts as hub (many connections, typical of money mules)',
        'low': 'Peripheral position in network (low intermediary activity)',
    },
    'mixer_interactions': {
        'any': '{value} interactions with known mixing services detected',
        'zero': 'No interactions with known mixing services',
    },
    'burstiness_7d': {
        'high': 'Unusual transaction timing pattern (rapid bursts followed by dormancy)',
        'low': 'Regular transaction timing (normal velocity)',
    },
    'benford_divergence': {
        'high': 'Amount distribution violates Benford\'s law (possible structuring)',
        'low': 'Amount distribution follows Benford\'s law (natural pattern)',
    },
    'clustering_coefficient': {
        'high': 'Part of tight-knit community (high local clustering)',
        'low': 'Avoids tight communities (possible obfuscation)',
    },
    'neighbor_risk_mean': {
        'high': 'Transacts primarily with high-risk addresses',
        'low': 'Transacts primarily with low-risk addresses',
    },
}
```

### Example Output

```
Transaction flagged as HIGH RISK (score: 0.92)

REASONS:
• Acts as hub (many connections, typical of money mules)
• 3 interactions with known mixing services detected
• Unusual transaction timing pattern (rapid bursts followed by dormancy)

TO BE CONSIDERED LOW-RISK, THIS TRANSACTION WOULD NEED:
• Reduce transaction velocity by 35 transactions per month, OR
• Reduce betweenness centrality from 0.85 to below 0.32, AND
• Eliminate interactions with known mixing services

RECOMMENDED ACTION:
Flag for manual review. Investigate source of funds and destination addresses.
Consider filing Suspicious Activity Report (SAR).

RISK BREAKDOWN:
  Baseline risk:          0.15
  Graph structure:        +0.37  (hub behavior, low clustering)
  Temporal patterns:      +0.25  (high burstiness, rapid velocity)
  Amount patterns:        +0.08  (Benford divergence)
  Entity associations:    +0.07  (mixer interactions)
  ────────────────────────────────
  Final risk score:       0.92

SIMILAR CASES:
- Transaction #15423: Risk 0.89, flagged for mixer interactions
- Transaction #29871: Risk 0.91, flagged for high velocity
- Transaction #34512: Risk 0.94, flagged for hub behavior

ANALYST NOTES:
[To be filled by compliance officer]
```

---

## Usage Examples

### Example 1: Generate Full Explanation for Transaction

```python
from chronos.explainability import explain_transaction

# Load model
model = CHRONOSNet.load_from_checkpoint('models/production/chronos_best.pth')

# Get transaction
transaction = test_data[42]  # Specific transaction

# Generate full explanation
explanation = explain_transaction(
    transaction=transaction,
    model=model,
    methods=['counterfactual', 'shap', 'attention', 'narrative'],
    num_counterfactuals=3
)

# Print
print(explanation['narrative'])

# Save to file
with open('reports/transaction_42_explanation.txt', 'w') as f:
    f.write(explanation['narrative'])

# Export attention visualization
explanation['attention_plot'].savefig('reports/transaction_42_attention.png')
```

### Example 2: Batch Explanation for High-Risk Transactions

```python
# Get all high-risk predictions
high_risk_mask = (predictions > 0.8)
high_risk_transactions = test_data[high_risk_mask]

# Generate explanations
for i, tx in enumerate(high_risk_transactions):
    explanation = explain_transaction(tx, model, methods=['counterfactual', 'narrative'])

    # Save to CSV
    with open(f'reports/batch_explanations.csv', 'a') as f:
        f.write(f"{tx.id},{explanation['risk_score']},{explanation['top_reasons']}\n")
```

### Example 3: Interactive Dashboard

```python
import streamlit as st

st.title("CHRONOS AML Detection Dashboard")

# Upload transaction
tx_id = st.text_input("Enter Transaction ID:")

if tx_id:
    tx = get_transaction(tx_id)

    # Predict
    risk_score = model(tx.x, tx.edge_index).item()

    # Display risk
    st.metric("Risk Score", f"{risk_score:.2f}")

    # Generate explanation
    if st.button("Explain"):
        explanation = explain_transaction(tx, model)

        st.subheader("Counterfactual Explanation")
        st.write(explanation['counterfactual_text'])

        st.subheader("Feature Importance (SHAP)")
        st.pyplot(explanation['shap_plot'])

        st.subheader("Attention Visualization")
        st.plotly_chart(explanation['attention_graph'])

        st.subheader("Full Report")
        st.text(explanation['narrative'])
```

---

## Evaluation Metrics

### Counterfactual Quality

**1. Validity**: Does the counterfactual actually flip the prediction?

```python
validity = (model(cf) < 0.3).float().mean()
# Target: ≥ 95%
```

**2. Proximity**: How many features changed?

```python
proximity = (cf.x != transaction.x).sum() / cf.x.numel()
# Target: < 10% of features changed
```

**3. Plausibility**: Human evaluation (5-point Likert scale)

```
1 = Completely unrealistic
2 = Somewhat unrealistic
3 = Neutral
4 = Somewhat realistic
5 = Completely realistic
```

**Target**: Average ≥ 4.0

### SHAP Quality

**1. Accuracy**: Does sum(SHAP) + base ≈ prediction?

```python
shap_accuracy = abs(sum(shap_values) + base_value - prediction)
# Target: < 0.01 (1% error)
```

**2. Consistency**: Do similar transactions have similar SHAP patterns?

```python
# Compute SHAP for 10 similar transactions
shap_similarity = cosine_similarity(shap_values)
# Target: > 0.7
```

---

## Regulatory Compliance

### EU AI Act Article 13 Requirements

✅ **Transparency**: Explanations allow deployers to interpret outputs
✅ **Technical Documentation**: All methods documented with citations
✅ **Human Oversight**: Template-based (no LLM hallucination)
✅ **Testing**: Counterfactual validity ≥ 95%, SHAP accuracy ≤ 1%

### Example Compliance Report

```
CHRONOS AML DETECTION SYSTEM
EXPLAINABILITY COMPLIANCE REPORT

System Name: CHRONOS v1.0.0
Date: 2025-12-23
Regulation: EU AI Act Article 13

EXPLAINABILITY METHODS:
1. Counterfactual Explanations (Novel)
   - Validity: 97.3%  ✓
   - Proximity: 8.2% features changed  ✓
   - Plausibility: 4.2/5.0  ✓

2. SHAP Feature Importance
   - Accuracy: 0.4% error  ✓
   - Consistency: 0.83  ✓

3. Attention Visualization
   - Coverage: 100% of predictions  ✓

4. Natural Language Generation
   - Template-based (no LLM)  ✓
   - Human-readable format  ✓

AUDIT TRAIL:
- All explanations logged
- Reproducible from model checkpoint
- 10-year retention policy implemented

CERTIFICATION:
[Signature of Data Protection Officer]
```

---

## References

1. **Lucic et al. (2022)**: "CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks", AISTATS
2. **Mothilal et al. (2020)**: "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations", NeurIPS (DiCE-ML)
3. **Lundberg & Lee (2017)**: "A Unified Approach to Interpreting Model Predictions", NeurIPS (SHAP)
4. **Veličković et al. (2018)**: "Graph Attention Networks", ICLR
5. **EU AI Act**: Article 13 (Transparency and provision of information to deployers)

---

**Last Updated**: 2025-12-23
**Version**: 1.0.0

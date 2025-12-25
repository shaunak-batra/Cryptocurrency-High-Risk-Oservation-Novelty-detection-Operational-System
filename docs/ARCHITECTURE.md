# CHRONOS Architecture

This document describes the technical architecture of CHRONOS (Cryptocurrency High-Risk Observation & Novelty-detection Operational System).

## Table of Contents

- [System Overview](#system-overview)
- [Data Flow](#data-flow)
- [Core Components](#core-components)
- [Model Architecture](#model-architecture)
- [Explainability Pipeline](#explainability-pipeline)
- [Design Rationale](#design-rationale)

---

## System Overview

CHRONOS is a temporal graph neural network system for detecting illicit cryptocurrency transactions with integrated counterfactual explainability.

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    CHRONOS SYSTEM                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Data      │───▶│   Model     │───▶│ Explainability│    │
│  │  Pipeline   │    │  Training   │    │   Engine    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│        │                   │                    │            │
│        ▼                   ▼                    ▼            │
│  Raw CSV Files       CHRONOS-Net         Counterfactuals    │
│  (Elliptic)          (Trained Model)     + SHAP + Attention │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Technology Stack

- **Core Framework**: PyTorch 2.1.0, PyTorch Geometric 2.4.0
- **Graph Analysis**: NetworkX 3.2.1, python-louvain 0.16
- **Data Science**: Pandas 2.1.3, NumPy 1.24.3, scikit-learn 1.3.2
- **Explainability**: SHAP 0.43.0, Captum 0.7.0, custom counterfactual engine
- **Development**: pytest 7.4.3, black 23.12.1, mypy 1.7.1

---

## Data Flow

### End-to-End Pipeline

```
1. DATA INGESTION
   │
   ├─ Elliptic CSV files (203,769 transactions, 234,355 edges)
   │
   ▼
2. PREPROCESSING
   │
   ├─ Data validation (checksums, schema verification)
   ├─ Feature engineering (70+ engineered features)
   ├─ Graph construction (NetworkX → PyTorch Geometric)
   │
   ▼
3. TEMPORAL GRAPH SNAPSHOTS
   │
   ├─ 49 snapshots (each ~2 weeks of Bitcoin transactions)
   ├─ Temporal split: Train(1-34), Val(35-42), Test(43-49)
   │
   ▼
4. MODEL TRAINING
   │
   ├─ Baseline models (RF, XGBoost, GCN)
   ├─ CHRONOS-Net training (Focal Loss + AdamW)
   ├─ Hyperparameter tuning
   │
   ▼
5. INFERENCE
   │
   ├─ Risk score prediction (0-1)
   ├─ Risk classification (LOW/MEDIUM/HIGH/CRITICAL)
   │
   ▼
6. EXPLAINABILITY
   │
   ├─ Counterfactual generation
   ├─ SHAP feature importance
   ├─ GAT attention visualization
   ├─ Natural language summary
   │
   ▼
7. OUTPUT
   │
   └─ Risk assessment + Human-readable explanation
```

### Data Validation Pipeline

```python
# Critical validation checks at each stage
Stage 1: Raw Data
  ✓ 203,769 transactions
  ✓ 234,355 edges
  ✓ 46,564 labeled (4,545 illicit, 42,019 licit)
  ✓ 49 timesteps
  ✓ MD5 checksum verification

Stage 2: Preprocessing
  ✓ No missing values in features
  ✓ Feature distributions within expected ranges
  ✓ Graph connectivity (no disconnected components in subgraphs)

Stage 3: Feature Engineering
  ✓ 166 original + 70 engineered = 236 total features
  ✓ Feature correlations < 0.95 (multicollinearity check)
  ✓ No NaN/Inf values

Stage 4: Temporal Graphs
  ✓ 49 PyG Data objects
  ✓ Each has: x (features), edge_index, y (labels), timestep
  ✓ Temporal ordering preserved (no data leakage)
```

---

## Core Components

### 1. Data Pipeline (`chronos/data/`)

**Purpose**: Load, validate, and preprocess Elliptic dataset

**Components**:

- **`loaders.py`**: CSV loading with schema validation
- **`preprocessing.py`**: Data cleaning, normalization, outlier handling
- **`graph_builder.py`**: Convert to PyTorch Geometric format
- **`validation.py`**: Dataset integrity checks, statistical validation

**Key Design Decision**: Immutable data pipeline (raw data never modified, all transformations cached)

### 2. Feature Engineering (`chronos/features/`)

**Purpose**: Engineer 70+ domain-specific features for AML detection

**Modules**:

- **`graph_topology.py`** (20 features): Centrality measures, clustering, community detection
- **`temporal.py`** (25 features): Burstiness, velocity, Hawkes process parameters
- **`amount_patterns.py`** (15 features): Benford's Law, structuring detection
- **`entity.py`** (10 features): Risk propagation, homophily, entity type inference

**Key Design Decision**: All features computable from Elliptic data (no external data dependencies)

### 3. Model Architecture (`chronos/models/`)

**Purpose**: Implement CHRONOS-Net and baselines

**Components**:

- **`chronos_net.py`**: Main temporal GNN model (NOVEL)
- **`temporal_attention.py`**: Multi-scale attention module (NOVEL)
- **`baselines.py`**: RF, XGBoost, Vanilla GCN (BASELINE)
- **`trainer.py`**: Training loop with focal loss
- **`evaluator.py`**: Comprehensive metrics and benchmarking

**Key Design Decision**: Modular architecture for easy ablation studies

### 4. Explainability Engine (`chronos/explainability/`)

**Purpose**: Generate counterfactual explanations and visualizations

**Components**:

- **`counterfactual.py`**: Gradient-based CF generator for temporal graphs (NOVEL)
- **`shap_explainer.py`**: SHAP integration for feature importance
- **`attention_viz.py`**: GAT attention weight visualization
- **`narrative.py`**: Template-based natural language generation

**Key Design Decision**: Multi-method explainability (no single point of failure)

---

## Model Architecture

### CHRONOS-Net: High-Level

```
Input: Transaction graph (x, edge_index) at timestep t
  │
  ▼
┌────────────────────────────────────────────┐
│ Component 1: Temporal Encoder (NOVEL)      │
│  Conv1D(236→256) + BatchNorm + ReLU        │
│  GRU(256→256, 2 layers)                    │
│  Output: [batch, seq_len, 256]             │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│ Component 2: Multi-Scale Attention (NOVEL) │
│  4 windows: 1, 5, 15, 30 timesteps         │
│  Transformer encoders for each window      │
│  Attention-weighted aggregation            │
│  Output: [batch, 256]                      │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│ Component 3: Graph Attention (ADAPTED)     │
│  GAT Layer 1: (256→256, 8 heads)           │
│  GAT Layer 2: (256→256, 8 heads)           │
│  GAT Layer 3: (256→256, 8 heads)           │
│  + Residual connections                    │
│  + Attention weight extraction             │
│  Output: [batch, 256]                      │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│ Component 4: Classifier (STANDARD)         │
│  Linear(256→128) + ReLU + Dropout(0.5)     │
│  Linear(128→64) + ReLU + Dropout(0.5)      │
│  Linear(64→1) + Sigmoid                    │
│  Output: [batch, 1] ∈ [0,1]                │
└────────────────────────────────────────────┘
```

### Component Details

#### 1. Temporal Encoder (NOVEL)

**Inspiration**: DA-HGNN (2024) Conv1D + GRU architecture

**Purpose**: Extract local and long-term temporal patterns before graph convolution

**Architecture**:

```python
class TemporalEncoder(nn.Module):
    def __init__(self, in_features=236, hidden_dim=256):
        super().__init__()
        self.conv1d = nn.Conv1d(in_features, hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        x = F.relu(self.bn(self.conv1d(x)))
        x = x.permute(0, 2, 1)  # [batch, seq_len, features]
        x, _ = self.gru(x)
        return x  # [batch, seq_len, hidden_dim]
```

**Why Conv1D + GRU**:

- Conv1D: Captures local temporal patterns (e.g., rapid transaction sequences)
- GRU: Captures long-term dependencies (e.g., dormancy periods)
- More efficient than pure attention for long sequences

#### 2. Multi-Scale Temporal Attention (NOVEL)

**Inspiration**: ATGAT (Zheng et al., 2025) multi-scale architecture

**Purpose**: Capture money laundering patterns at multiple timescales

**Windows** (adapted to Bitcoin Elliptic timesteps):

- **Short-term** (1 timestep ≈ 2 weeks): Rapid mixers movement
- **Medium-term** (5 timesteps ≈ 2.5 months): Peeling chains
- **Long-term** (15 timesteps ≈ 7.5 months): Layering phase
- **Very long-term** (30 timesteps ≈ 14 months): Integration phase

**Architecture**:

```python
class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, hidden_dim=256, windows=[1, 5, 15, 30]):
        super().__init__()
        self.windows = windows
        self.encoders = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden_dim, nhead=8),
                num_layers=2
            ) for _ in windows
        ])
        self.attention = nn.Linear(hidden_dim * len(windows), len(windows))

    def forward(self, x_temporal):
        # x_temporal: [batch, seq_len, hidden_dim]
        window_features = []
        for window_size, encoder in zip(self.windows, self.encoders):
            # Extract window
            x_window = x_temporal[:, -window_size:, :]
            # Encode
            feat = encoder(x_window).mean(dim=1)  # [batch, hidden_dim]
            window_features.append(feat)

        # Concatenate and attend
        concat = torch.cat(window_features, dim=1)  # [batch, hidden_dim * num_windows]
        attn_weights = F.softmax(self.attention(concat), dim=1)  # [batch, num_windows]

        # Weighted sum
        output = sum(w.unsqueeze(1) * f for w, f in zip(attn_weights.T, window_features))
        return output  # [batch, hidden_dim]
```

**Why Multi-Scale**:

- Money laundering patterns emerge at different timescales
- Short-term: Rapid obfuscation (mixers, tumblers)
- Long-term: Layering and integration (dormancy, legitimate-looking activity)

#### 3. Graph Attention Layers (ADAPTED)

**Inspiration**: GAT (Veličković et al., ICLR 2018) + ATGAT (2025)

**Purpose**: Aggregate neighborhood information with learned attention

**Architecture**:

```python
class GraphAttentionLayers(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(hidden_dim, hidden_dim//num_heads, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim//num_heads, heads=num_heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim, hidden_dim//num_heads, heads=num_heads, dropout=dropout)
        self.attention_weights = []  # Store for explainability

    def forward(self, x, edge_index):
        # Layer 1
        x1, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(x1)
        self.attention_weights.append(attn1)

        # Layer 2 with residual
        x2, attn2 = self.gat2(x1, edge_index, return_attention_weights=True)
        x2 = F.elu(x2 + x1)  # Residual
        self.attention_weights.append(attn2)

        # Layer 3 with residual
        x3, attn3 = self.gat3(x2, edge_index, return_attention_weights=True)
        x3 = F.elu(x3 + x2)  # Residual
        self.attention_weights.append(attn3)

        return x3
```

**Why GAT over GCN**:

1. **Explainability**: Attention weights show which neighbors matter
2. **Adaptive**: Can learn to ignore benign neighbors, focus on suspicious ones
3. **Performance**: ATGAT showed 10% improvement over GCN
4. **Patent-safe**: Not plain GCN (Chinese patent CN111311416A)

**Why 3 Layers**:

- 1-2 layers: Too local (only 1-2 hop neighbors)
- 4+ layers: Over-smoothing (all nodes become similar)
- 3 layers: Captures 3-hop patterns (sufficient for transaction chains)

#### 4. Classifier (STANDARD)

**Purpose**: Map graph embeddings to risk score

**Architecture**:

```python
class Classifier(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x  # [batch, 1] ∈ [0,1]
```

---

## Explainability Pipeline

### Multi-Method Approach

```
                    ┌──────────────────┐
                    │ CHRONOS-Net      │
                    │ Prediction       │
                    └────────┬─────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
     ┌──────────────────┐      ┌──────────────────┐
     │  Counterfactual  │      │  Feature         │
     │  Explanation     │      │  Importance      │
     │  (NOVEL)         │      │  (SHAP)          │
     └────────┬─────────┘      └────────┬─────────┘
              │                         │
              │      ┌──────────────────┴┐
              │      │  Attention        │
              │      │  Visualization    │
              │      │  (GAT weights)    │
              │      └─────────┬─────────┘
              │                │
              └───────┬────────┘
                      │
                      ▼
          ┌────────────────────────┐
          │  Natural Language      │
          │  Summary Generator     │
          │  (Template-based)      │
          └────────────────────────┘
                      │
                      ▼
          "Transaction flagged as HIGH RISK
           because: [reasons] + [counterfactuals]"
```

### Counterfactual Generation (NOVEL)

**Goal**: Find minimal changes to make transaction low-risk

**Algorithm**:

```
Input: High-risk transaction x, model f, target y_target=0 (low-risk)

1. Initialize: x_cf = x.clone()

2. For iteration in range(max_iter):
    # Compute losses
    L_pred = (f(x_cf) - y_target)²           # Want prediction to flip
    L_prox = ||x_cf - x||²                   # Want minimal changes
    L_div = -Σ||x_cf_i - x_cf_j||²           # Want diverse CFs
    L_valid = penalty(x_cf, constraints)      # Want realistic values

    L_total = L_pred + λ_prox*L_prox + λ_div*L_div + λ_valid*L_valid

    # Gradient step
    ∇L = grad(L_total, x_cf)
    x_cf = x_cf - lr * ∇L

    # Project to valid space
    x_cf = project_constraints(x_cf)

    if f(x_cf) < threshold and iter > min_iter:
        break

3. Return: x_cf, changes = x_cf - x
```

**Constraints**:

```python
# Immutable features (cannot change)
immutable = [0:166]  # Original Elliptic features (anonymized)

# Monotonicity constraints (can only increase)
monotonic = ['total_tx_count', 'time_since_first_tx']

# Temporal constraints
# Cannot change past (only future behavior)
```

**Novel Contribution**: Extends CF-GNNExplainer (static graphs) to temporal graphs with monotonicity constraints.

---

## Design Rationale

### Why This Architecture?

#### 1. Temporal Encoder Before Graph Convolution

**Decision**: Extract temporal features first, then apply graph convolution

**Rationale**:

- Money laundering has both temporal patterns (bursts) AND network patterns (hubs)
- Temporal encoder captures individual node behavior over time
- GAT then captures relational patterns (who transacts with whom)

**Alternative Considered**: Graph convolution first, then temporal

- **Rejected**: Would lose individual temporal signatures before aggregating neighbors

#### 2. Multi-Scale Attention vs Single Window

**Decision**: Use 4 temporal windows (1, 5, 15, 30 timesteps)

**Rationale**:

- AML patterns emerge at multiple timescales (ATGAT paper showed 10% improvement)
- Short-term: Rapid obfuscation (hours-days)
- Long-term: Layering and integration (weeks-months)

**Alternative Considered**: Single fixed window

- **Rejected**: Misses multi-scale patterns. Laundering is a process, not an event.

#### 3. GAT vs GCN

**Decision**: Use Graph Attention Network (GAT)

**Rationale**:

1. **Explainability**: Attention weights show which neighbors matter
2. **Adaptive aggregation**: Can ignore benign neighbors
3. **Performance**: ATGAT showed 10% improvement
4. **Patent-safe**: Avoids Chinese patent CN111311416A on GCN for crypto AML

**Alternative Considered**: GCN (simpler, faster)

- **Rejected**: No built-in explainability, patent risk, lower performance

#### 4. 3 GAT Layers vs 2 or 4+

**Decision**: Use exactly 3 GAT layers

**Rationale**:

- 1-2 layers: Only capture 1-2 hop neighbors (too local)
- 3 layers: Captures 3-hop patterns (sufficient for transaction chains)
- 4+ layers: Over-smoothing problem (all nodes become similar)

**Empirical Evidence**: Most GNN papers find 3-4 layers optimal

#### 5. Focal Loss vs Cross-Entropy

**Decision**: Use Focal Loss (α=0.25, γ=2.0)

**Rationale**:

- Extreme class imbalance (9.2:1 licit:illicit)
- Cross-entropy over-fits to majority class
- Focal loss down-weights easy examples, focuses on hard cases
- Proven effective for imbalanced datasets (Lin et al., ICCV 2017)

**Alternative Considered**: Class-weighted cross-entropy

- **Rejected**: Still biased toward majority class, doesn't focus on hard examples

#### 6. Counterfactual + SHAP + Attention vs Single Method

**Decision**: Use multi-method explainability

**Rationale**:

- **Counterfactuals**: "What to change" (actionable)
- **SHAP**: "Why this prediction" (feature importance)
- **Attention**: "Which neighbors matter" (relational)
- No single method provides complete picture
- Multi-method builds trust (triangulation)

**Alternative Considered**: SHAP only

- **Rejected**: Not actionable, doesn't explain graph structure

---

## Model Parameters and Complexity

### Parameter Count

```
Component                    Parameters
─────────────────────────────────────────
Temporal Encoder:
  Conv1D                     181,504
  GRU (2 layers)             526,336

Multi-Scale Attention:
  4 Transformers             2,097,152
  Attention weights          2,048

Graph Attention:
  GAT Layer 1                524,800
  GAT Layer 2                524,800
  GAT Layer 3                524,800

Classifier:
  FC1 (256→128)              32,896
  FC2 (128→64)               8,256
  FC3 (64→1)                 65
─────────────────────────────────────────
Total:                       ~4.4M parameters
Model size:                  ~17MB (FP32)
```

### Computational Complexity

```
Component                    Complexity
─────────────────────────────────────────
Temporal Encoder:            O(T * d²)
Multi-Scale Attention:       O(T² * d)
Graph Attention (per layer): O(E * d²)
Classifier:                  O(d²)
─────────────────────────────────────────
Total (per graph):           O(E * d² + T² * d)

Where:
  T = sequence length (timesteps)
  d = hidden dimension (256)
  E = number of edges
```

**Inference Time** (target):

- P50: < 30ms per transaction
- P95: < 50ms per transaction
- Measured on: NVIDIA RTX 3090 (24GB VRAM)

---

## Comparison with Baselines

| Model | Architecture | Parameters | F1 Target | Explainability |
|-------|--------------|------------|-----------|----------------|
| Random Forest | 100 trees, depth 20 | ~1M (trees) | 0.70-0.73 | Feature importance |
| XGBoost | 200 trees, depth 10 | ~2M (trees) | 0.72-0.75 | Feature importance |
| Vanilla GCN | 2-layer GCN | ~130K | 0.60-0.65 | None |
| CHRONOS-Net | Temporal + Multi-scale + GAT | ~4.4M | ≥ 0.88 | Counterfactual + SHAP + Attention |

**Key Differences**:

- **Baselines**: No temporal modeling, no multi-scale, limited explainability
- **CHRONOS-Net**: Full temporal + graph + multi-method explainability

---

## Failure Modes and Mitigation

### Potential Failure Modes

1. **Over-smoothing in GAT**
   - **Symptom**: All nodes get similar embeddings
   - **Mitigation**: Use 3 layers max, add residual connections

2. **Class imbalance bias**
   - **Symptom**: Model predicts all transactions as licit
   - **Mitigation**: Focal loss, class weights, stratified sampling

3. **Data leakage**
   - **Symptom**: Unrealistically high test performance
   - **Mitigation**: Strict temporal split, no shuffle, separate validation

4. **Counterfactual unrealism**
   - **Symptom**: Generated CFs violate domain constraints
   - **Mitigation**: Hard constraints (immutable features), validity loss term

5. **SHAP computational cost**
   - **Symptom**: Explanation generation > 100ms
   - **Mitigation**: Use 100 background samples (not full training set)

---

## References

**CHRONOS Architecture Inspiration**:

1. **ATGAT** (Zheng et al., 2025): Multi-scale temporal attention
2. **RecGNN** (Alarab et al., 2023): LSTM + GNN combination
3. **DA-HGNN** (2024): Conv1D + GRU-MHA temporal extraction
4. **CF-GNNExplainer** (Lucic et al., 2022): Counterfactual generation for graphs
5. **GAT** (Veličković et al., 2018): Graph Attention Networks

**Key Papers**:

- Weber et al. (2019): "Anti-Money Laundering in Bitcoin", KDD Workshop
- Alarab et al. (2023): "Robust recurrent graph convolutional network", Multimedia Tools
- Zheng et al. (2025): "Temporal-Aware Graph Attention Network", arXiv:2506.21382
- Lucic et al. (2022): "CF-GNNExplainer", AISTATS
- Lin et al. (2017): "Focal Loss for Dense Object Detection", ICCV

---

**Last Updated**: 2025-12-23
**Version**: 1.0.0

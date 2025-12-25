"""
CHRONOS-Net: Main model architecture.

Combines temporal encoding, multi-scale attention, and graph attention
for cryptocurrency AML detection with explainability.

Architecture:
1. Temporal Encoder (Conv1D + GRU)
2. Multi-Scale Temporal Attention (1, 5, 15, 30 timesteps)
3. Graph Attention Network (3 layers, 8 heads)
4. Classifier (binary: licit vs illicit)

Target Performance:
- F1 ≥ 0.88
- Precision ≥ 0.85
- Recall ≥ 0.85
- AUC-ROC ≥ 0.92
- Inference P95 < 50ms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from typing import Dict, Tuple, Optional, List

from .components import (
    TemporalEncoder,
    MultiScaleTemporalAttention,
    FocalLoss,
    TemporalGraphAttention
)


class CHRONOSNet(nn.Module):
    """
    CHRONOS-Net: Temporal GNN for cryptocurrency AML detection.

    Novel contribution: First system combining temporal GNNs with
    counterfactual explanations for cryptocurrency AML.
    """

    def __init__(
        self,
        in_features: int = 236,
        hidden_dim: int = 256,
        num_gat_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.3,
        window_sizes: List[int] = [1, 5, 15, 30]
    ):
        """
        Initialize CHRONOS-Net.

        Parameters
        ----------
        in_features : int
            Number of input features (166 original + 70 engineered)
        hidden_dim : int
            Hidden dimension throughout network
        num_gat_layers : int
            Number of GAT layers
        num_heads : int
            Number of attention heads per GAT layer
        dropout : float
            Dropout rate
        window_sizes : List[int]
            Window sizes for multi-scale temporal attention
        """
        super().__init__()

        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads
        self.window_sizes = window_sizes

        # ====================================================================
        # 1. TEMPORAL ENCODER
        # ====================================================================
        self.temporal_encoder = TemporalEncoder(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_gru_layers=2,
            dropout=dropout
        )

        # ====================================================================
        # 2. MULTI-SCALE TEMPORAL ATTENTION
        # ====================================================================
        self.temporal_attention = MultiScaleTemporalAttention(
            hidden_dim=hidden_dim,
            window_sizes=window_sizes,
            num_heads=4
        )

        # ====================================================================
        # 3. GRAPH ATTENTION NETWORK
        # ====================================================================
        self.gat_layers = nn.ModuleList()

        # First GAT layer
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True
            )
        )

        # Middle GAT layers
        for _ in range(num_gat_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )

        # Last GAT layer (average heads instead of concat)
        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=False
            )
        )

        # ====================================================================
        # 4. CLASSIFIER
        # ====================================================================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for skip connection
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(num_gat_layers)
        ])
        
        # Input projection for when temporal_sequences is None
        self.input_projection = nn.Linear(in_features, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_sequences: Optional[List[torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor [num_nodes, in_features]
            Node features
        edge_index : Tensor [2, num_edges]
            Edge indices
        temporal_sequences : Optional[List[Tensor]]
            List of temporal sequences for multi-scale attention
            Each: [num_nodes, window_size, in_features]
        return_attention : bool
            Whether to return attention weights for explainability

        Returns
        -------
        logits : Tensor [num_nodes, 2]
            Classification logits
        attention_weights : Optional[Dict]
            Attention weights if return_attention=True
            Keys: 'temporal', 'gat_layer_0', 'gat_layer_1', ...
        """
        attention_weights = {} if return_attention else None

        # ================================================================
        # 1. TEMPORAL ENCODING
        # ================================================================
        if temporal_sequences is not None:
            # Encode each temporal sequence
            encoded_sequences = []
            for seq in temporal_sequences:
                encoded = self.temporal_encoder(seq)
                encoded_sequences.append(encoded)

            # Multi-scale temporal attention
            temporal_out, temporal_attn = self.temporal_attention(
                [seq.unsqueeze(1) for seq in encoded_sequences]
            )

            if return_attention:
                attention_weights['temporal'] = temporal_attn

            # Use temporal encoding
            h = temporal_out
        else:
            # Fallback: use raw features
            h = x.mean(dim=1) if x.dim() == 3 else x
            # Project to hidden_dim
            h = self.input_projection(h)

        # Store for skip connection
        h_temporal = h

        # ================================================================
        # 2. GRAPH ATTENTION LAYERS
        # ================================================================
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.bn_layers)):
            h_prev = h

            # GAT layer
            if return_attention:
                h, (edge_idx, attn) = gat(
                    h,
                    edge_index,
                    return_attention_weights=True
                )
                attention_weights[f'gat_layer_{i}'] = (edge_idx, attn)
            else:
                h = gat(h, edge_index)

            # Batch normalization
            h = bn(h)

            # Activation and dropout (except last layer)
            if i < self.num_gat_layers - 1:
                h = F.elu(h)
                h = F.dropout(h, p=0.3, training=self.training)

                # Residual connection (if dimensions match)
                if h.size(-1) == h_prev.size(-1):
                    h = h + h_prev

        # ================================================================
        # 3. SKIP CONNECTION: Combine graph and temporal features
        # ================================================================
        h_combined = torch.cat([h, h_temporal], dim=1)

        # ================================================================
        # 4. CLASSIFICATION
        # ================================================================
        logits = self.classifier(h_combined)

        if return_attention:
            return logits, attention_weights
        else:
            return logits

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_sequences: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Predict labels.

        Parameters
        ----------
        x : Tensor
            Node features
        edge_index : Tensor
            Edge indices
        temporal_sequences : Optional[List[Tensor]]
            Temporal sequences

        Returns
        -------
        y_pred : Tensor [num_nodes]
            Predicted labels (0=licit, 1=illicit)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, temporal_sequences)
            if isinstance(logits, tuple):
                logits = logits[0]
            y_pred = logits.argmax(dim=1)
        return y_pred

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_sequences: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Predict probabilities.

        Parameters
        ----------
        x : Tensor
            Node features
        edge_index : Tensor
            Edge indices
        temporal_sequences : Optional[List[Tensor]]
            Temporal sequences

        Returns
        -------
        y_proba : Tensor [num_nodes, 2]
            Predicted probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, temporal_sequences)
            if isinstance(logits, tuple):
                logits = logits[0]
            y_proba = F.softmax(logits, dim=1)
        return y_proba

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_sequences: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Get node embeddings (before classification).

        Useful for visualization and analysis.

        Parameters
        ----------
        x : Tensor
            Node features
        edge_index : Tensor
            Edge indices
        temporal_sequences : Optional[List[Tensor]]
            Temporal sequences

        Returns
        -------
        embeddings : Tensor [num_nodes, hidden_dim*2]
            Node embeddings
        """
        self.eval()
        with torch.no_grad():
            # Run forward pass up to classifier
            if temporal_sequences is not None:
                encoded_sequences = []
                for seq in temporal_sequences:
                    encoded = self.temporal_encoder(seq)
                    encoded_sequences.append(encoded)

                temporal_out, _ = self.temporal_attention(
                    [seq.unsqueeze(1) for seq in encoded_sequences]
                )
                h = temporal_out
            else:
                h = x.mean(dim=1) if x.dim() == 3 else x
                h = self.input_projection(h)

            h_temporal = h

            # GAT layers
            for i, (gat, bn) in enumerate(zip(self.gat_layers, self.bn_layers)):
                h_prev = h
                h = gat(h, edge_index)
                h = bn(h)
                if i < self.num_gat_layers - 1:
                    h = F.elu(h)
                    if h.size(-1) == h_prev.size(-1):
                        h = h + h_prev

            # Combine with temporal
            embeddings = torch.cat([h, h_temporal], dim=1)

        return embeddings


def create_chronos_net(
    in_features: int = 236,
    hidden_dim: int = 256,
    config: Optional[Dict] = None
) -> CHRONOSNet:
    """
    Factory function to create CHRONOS-Net with optional config.

    Parameters
    ----------
    in_features : int
        Number of input features
    hidden_dim : int
        Hidden dimension
    config : Optional[Dict]
        Configuration dictionary with hyperparameters

    Returns
    -------
    model : CHRONOSNet
        Initialized model

    Examples
    --------
    >>> # Default configuration
    >>> model = create_chronos_net(in_features=236)

    >>> # Custom configuration
    >>> config = {
    ...     'num_gat_layers': 4,
    ...     'num_heads': 16,
    ...     'dropout': 0.5
    ... }
    >>> model = create_chronos_net(in_features=236, config=config)
    """
    if config is None:
        config = {}

    model = CHRONOSNet(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_gat_layers=config.get('num_gat_layers', 3),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.3),
        window_sizes=config.get('window_sizes', [1, 5, 15, 30])
    )

    return model

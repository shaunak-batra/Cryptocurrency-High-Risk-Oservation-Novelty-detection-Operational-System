"""
Core components for CHRONOS-Net architecture.

This module implements the building blocks:
1. TemporalEncoder: Conv1D + Bidirectional GRU
2. MultiScaleTemporalAttention: Attention over multiple time windows
3. FocalLoss: Loss function for class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class TemporalEncoder(nn.Module):
    """
    Temporal feature encoder using Conv1D and Bidirectional GRU.

    Processes time-series node features to capture temporal patterns.
    """

    def __init__(
        self,
        in_features: int = 236,
        hidden_dim: int = 256,
        num_gru_layers: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize temporal encoder.

        Parameters
        ----------
        in_features : int
            Number of input features (166 original + 70 engineered)
        hidden_dim : int
            Hidden dimension for Conv1D and GRU
        num_gru_layers : int
            Number of GRU layers
        dropout : float
            Dropout rate
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1D Convolution for local temporal patterns
        self.conv1d = nn.Conv1d(
            in_channels=in_features,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        self.bn = nn.BatchNorm1d(hidden_dim)

        # Bidirectional GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0
        )

        # Project bidirectional output back to hidden_dim
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor [batch_size, seq_len, in_features]
            Temporal sequences

        Returns
        -------
        out : Tensor [batch_size, hidden_dim]
            Encoded temporal features
        """
        # Conv1D expects [batch, channels, seq_len]
        x = x.permute(0, 2, 1)  # [batch, in_features, seq_len]
        x = self.conv1d(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, hidden_dim]

        # GRU
        x, _ = self.gru(x)  # [batch, seq_len, hidden_dim*2]

        # Use last timestep output
        x = x[:, -1, :]  # [batch, hidden_dim*2]

        # Project to hidden_dim
        x = self.projection(x)  # [batch, hidden_dim]
        x = self.dropout(x)

        return x


class MultiScaleTemporalAttention(nn.Module):
    """
    Multi-scale temporal attention mechanism.

    Applies attention over multiple time windows (1, 5, 15, 30 timesteps)
    to capture both short-term and long-term temporal patterns.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        window_sizes: List[int] = [1, 5, 15, 30],
        num_heads: int = 4
    ):
        """
        Initialize multi-scale temporal attention.

        Parameters
        ----------
        hidden_dim : int
            Hidden dimension
        window_sizes : List[int]
            List of window sizes (in timesteps)
        num_heads : int
            Number of attention heads per window
        """
        super().__init__()

        self.window_sizes = window_sizes
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Attention layers for each window size
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            for _ in window_sizes
        ])

        # Scale weights (learnable)
        self.scale_weights = nn.Parameter(torch.ones(len(window_sizes)))

        # Output projection
        self.out_proj = nn.Linear(hidden_dim * len(window_sizes), hidden_dim)

    def forward(
        self,
        temporal_sequences: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        temporal_sequences : List[Tensor]
            List of temporal sequences, one per window size
            Each: [batch_size, window_size, hidden_dim]

        Returns
        -------
        out : Tensor [batch_size, hidden_dim]
            Aggregated temporal features
        attention_weights : List[Tensor]
            Attention weights for each scale (for visualization)
        """
        scale_outputs = []
        attention_weights = []

        for i, (seq, attn) in enumerate(zip(temporal_sequences, self.attentions)):
            # Self-attention over temporal sequence
            attn_out, attn_w = attn(seq, seq, seq, need_weights=True)
            # [batch, window_size, hidden_dim], [batch, window_size, window_size]

            # Pool over time (mean of all timesteps)
            pooled = attn_out.mean(dim=1)  # [batch, hidden_dim]

            # Weight by scale importance
            weighted = pooled * self.scale_weights[i]
            scale_outputs.append(weighted)
            attention_weights.append(attn_w)

        # Concatenate all scales
        concatenated = torch.cat(scale_outputs, dim=1)
        # [batch, hidden_dim * num_scales]

        # Project to output dimension
        out = self.out_proj(concatenated)  # [batch, hidden_dim]

        return out, attention_weights


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    From: Lin et al. "Focal Loss for Dense Object Detection" (2017)

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Focuses training on hard examples by down-weighting easy examples.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.

        Parameters
        ----------
        alpha : float
            Weighting factor for class 1 (illicit)
            alpha = 0.25 means class 0 (licit) gets weight 0.75
        gamma : float
            Focusing parameter (higher = more focus on hard examples)
            gamma = 0 reduces to standard cross-entropy
        reduction : str
            'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Parameters
        ----------
        inputs : Tensor [batch_size, num_classes]
            Logits (raw model outputs)
        targets : Tensor [batch_size]
            Ground truth labels

        Returns
        -------
        loss : Tensor
            Focal loss
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get predicted probabilities
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # Compute alpha term
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=targets.device),
            torch.tensor(1 - self.alpha, device=targets.device)
        )

        # Focal loss
        loss = alpha_t * focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TemporalGraphAttention(nn.Module):
    """
    Temporal-aware Graph Attention layer.

    Extends standard GAT by incorporating temporal distance between nodes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 8,
        dropout: float = 0.3,
        concat: bool = True
    ):
        """
        Initialize temporal GAT layer.

        Parameters
        ----------
        in_features : int
            Input feature dimension
        out_features : int
            Output feature dimension (per head)
        num_heads : int
            Number of attention heads
        dropout : float
            Dropout rate
        concat : bool
            If True, concatenate heads; if False, average
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout

        # Linear transformations for each head
        self.W = nn.Parameter(torch.zeros(num_heads, in_features, out_features))
        nn.init.xavier_uniform_(self.W)

        # Attention parameters
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a)

        # Temporal attention parameter
        self.temporal_weight = nn.Parameter(torch.tensor(0.1))

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor [num_nodes, in_features]
            Node features
        edge_index : Tensor [2, num_edges]
            Edge indices
        timestep : Tensor [num_nodes]
            Timestep for each node

        Returns
        -------
        out : Tensor [num_nodes, out_features * num_heads] (if concat)
              or [num_nodes, out_features] (if not concat)
            Updated node features
        """
        num_nodes = x.size(0)
        edge_src, edge_dst = edge_index

        # Apply linear transformation for each head
        # [num_nodes, num_heads, out_features]
        h = torch.einsum('nf,hfo->nho', x, self.W)

        # Compute attention coefficients
        # [num_edges, num_heads]
        h_src = h[edge_src]  # [num_edges, num_heads, out_features]
        h_dst = h[edge_dst]  # [num_edges, num_heads, out_features]

        # Concatenate source and target features
        h_cat = torch.cat([h_src, h_dst], dim=-1)
        # [num_edges, num_heads, 2*out_features]

        # Attention scores
        e = torch.einsum('nho,ho->nh', h_cat, self.a)
        # [num_edges, num_heads]

        # Add temporal component
        temporal_dist = torch.abs(timestep[edge_src] - timestep[edge_dst]).float()
        temporal_factor = torch.exp(-self.temporal_weight * temporal_dist)
        e = e * temporal_factor.unsqueeze(1)

        e = self.leakyrelu(e)

        # Apply softmax per node
        # Normalize attention scores for each destination node
        attention = torch.zeros(num_nodes, self.num_heads, device=x.device)
        attention = attention.index_add(0, edge_dst, torch.exp(e))
        attention = attention[edge_dst]  # [num_edges, num_heads]
        attention = torch.exp(e) / (attention + 1e-8)

        # Apply dropout
        attention = self.dropout_layer(attention)

        # Aggregate messages
        # [num_edges, num_heads, out_features]
        messages = attention.unsqueeze(-1) * h_src

        # Sum messages per destination node
        out = torch.zeros(num_nodes, self.num_heads, self.out_features, device=x.device)
        out = out.index_add(0, edge_dst, messages)
        # [num_nodes, num_heads, out_features]

        if self.concat:
            # Concatenate heads
            out = out.reshape(num_nodes, -1)
            # [num_nodes, num_heads * out_features]
        else:
            # Average heads
            out = out.mean(dim=1)
            # [num_nodes, out_features]

        return out

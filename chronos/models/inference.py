"""
CHRONOS Inference Model - Compatible with trained checkpoint.

This simplified model matches the EXACT structure of the saved checkpoint
for inference only (no training needed).

Checkpoint structure from analysis:
- input_proj: Linear(235, 256)
- temporal: MLP [256→256, 256→256]
- gat_layers: 3 GAT layers 
  - gat_layers.0.lin: [256, 256]
  - gat_layers.2.lin: [2048, 256] (8 heads × 256)
- gat_norms: 3 BatchNorm layers with spectral_norm wrapper
- classifier: MLP [512→256, 256→2]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Tuple, Optional
import numpy as np


class CHRONOSInference(nn.Module):
    """
    CHRONOS-Net inference model matching the checkpoint structure EXACTLY.
    """
    
    def __init__(
        self,
        in_features: int = 235,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input projection: 235 -> 256
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # Temporal branch: Sequential MLP
        # Checkpoint: temporal.0 [256,256], temporal.3 [256,256]
        # (indices 1,2 are activation/dropout)
        self.temporal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # idx 0
            nn.ReLU(),                          # idx 1
            nn.Dropout(dropout),                # idx 2
            nn.Linear(hidden_dim, hidden_dim),  # idx 3
        )
        
        # GAT layers
        # The checkpoint shows gat_layers.2.lin has shape [2048, 256]
        # This means 8 heads with 256 output each, concatenated = 2048
        # But classifier input is 512 = 256 + 256, so GAT output is averaged
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout),
            GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout),
            GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=True, dropout=dropout),
        ])
        
        # Norms - need to match spectral_norm wrapper structure
        # gat_norms.X.module.weight indicates spectral_norm wrapping
        self._gat_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim * num_heads),  # 2048 after last GAT
        ])
        
        # Classifier: 512 -> 256 -> 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),  # 512 -> 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # 256 -> 2
        )
    
    @property
    def gat_norms(self):
        """Property to access norms for checkpoint loading."""
        return self._gat_norms
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass."""
        # Input projection
        h = self.input_proj(x)
        h = F.elu(h)
        
        # Temporal branch
        t = self.temporal(h)
        
        # GAT layers
        attention_weights = []
        gat_out = h
        
        for i, gat in enumerate(self.gat_layers):
            if return_attention:
                gat_out, attn = gat(gat_out, edge_index, return_attention_weights=True)
                attention_weights.append(attn)
            else:
                gat_out = gat(gat_out, edge_index)
            
            gat_out = F.elu(gat_out)
            gat_out = F.dropout(gat_out, p=0.3, training=self.training)
        
        # Average GAT heads for classifier: [N, 2048] -> [N, 256]
        gat_out = gat_out.view(-1, self.num_heads, self.hidden_dim).mean(dim=1)
        
        # Concatenate: [N, 256] + [N, 256] = [N, 512]
        combined = torch.cat([gat_out, t], dim=-1)
        
        # Classifier
        logits = self.classifier(combined)
        
        if return_attention:
            return logits, attention_weights
        return logits, None
    
    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, edge_index)
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = (probs > 0.5).long()
        return probs, preds


def load_inference_model(checkpoint_path: str, device: str = 'cpu') -> CHRONOSInference:
    """
    Load model from checkpoint.
    
    Uses strict=False to allow partial loading since we're using
    a simplified architecture that may not have all layers.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get input dimension from checkpoint
    in_features = checkpoint['model']['input_proj.weight'].shape[1]
    
    # Create model
    model = CHRONOSInference(in_features=in_features)
    
    # Manual loading of matching keys
    model_state = checkpoint['model']
    model_dict = model.state_dict()
    
    # Map checkpoint keys to model keys
    loaded_keys = []
    for key in model_dict.keys():
        # Handle gat_norms mapping (checkpoint has .module. prefix)
        ckpt_key = key
        if 'gat_norms' in key:
            # Convert _gat_norms.0.weight to gat_norms.0.module.weight
            ckpt_key = key.replace('_gat_norms', 'gat_norms')
            parts = ckpt_key.split('.')
            # Insert 'module' after the index
            if len(parts) >= 2:
                ckpt_key = f"{parts[0]}.{parts[1]}.module.{'.'.join(parts[2:])}"
        
        if ckpt_key in model_state:
            if model_dict[key].shape == model_state[ckpt_key].shape:
                model_dict[key] = model_state[ckpt_key]
                loaded_keys.append(key)
    
    model.load_state_dict(model_dict)
    print(f"Loaded {len(loaded_keys)}/{len(model_dict)} keys from checkpoint")
    
    model.to(device)
    model.eval()
    
    return model


def generate_predictions(
    model: CHRONOSInference,
    data,
    test_mask: torch.Tensor,
    device: str = 'cpu'
) -> dict:
    """Generate predictions on test set."""
    model.eval()
    
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    
    with torch.no_grad():
        probs, preds = model.predict(x, edge_index)
    
    test_probs = probs[test_mask].cpu().numpy()
    test_preds = preds[test_mask].cpu().numpy()
    test_labels = data.y[test_mask].cpu().numpy()
    
    return {
        'probabilities': test_probs,
        'predictions': test_preds,
        'labels': test_labels
    }

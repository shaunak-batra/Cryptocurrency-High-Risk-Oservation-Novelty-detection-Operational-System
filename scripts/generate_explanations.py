"""
CHRONOS Explanation Generator (Compatible with Colab-trained model)
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ColabCHRONOSNet(nn.Module):
    """Matches the architecture from Colab training."""
    def __init__(self, in_features=235, hidden_dim=256, num_gat_layers=3, num_heads=8, dropout=0.3):
        super().__init__()
        from torch_geometric.nn import GATConv, BatchNorm
        
        self.input_proj = nn.Linear(in_features, hidden_dim)
        self.temporal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for i in range(num_gat_layers):
            concat = i < num_gat_layers - 1
            out_dim = hidden_dim // num_heads if concat else hidden_dim
            self.gat_layers.append(GATConv(hidden_dim, out_dim, heads=num_heads, concat=concat, dropout=dropout))
            self.gat_norms.append(BatchNorm(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2)
        )
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        h = self.input_proj(x)
        h_temp = self.temporal(h)
        h_graph = h
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            h_graph = norm(gat(h_graph, edge_index))
            if i < len(self.gat_layers) - 1:
                h_graph = F.elu(F.dropout(h_graph, self.dropout, self.training))
        return self.classifier(torch.cat([h_graph, h_temp], dim=1))


def load_model(checkpoint_path, device='cpu'):
    """Load the Colab-trained model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ColabCHRONOSNet(in_features=235, hidden_dim=256, num_gat_layers=3, num_heads=8, dropout=0.0)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, checkpoint.get('metrics', {})


def generate_feature_importance(model, x, edge_index, node_idx, device='cpu'):
    """Generate gradient-based feature importance."""
    model.eval()
    x = x.clone().to(device).requires_grad_(True)
    edge_index = edge_index.to(device)
    
    logits = model(x, edge_index)
    target = logits[node_idx, 1]  # Illicit class
    target.backward()
    
    return x.grad[node_idx].cpu().numpy()


def plot_importance(attributions, top_k=20, save_path=None):
    """Plot feature importances."""
    feature_names = [f'feat_{i}' for i in range(len(attributions))]
    abs_attr = np.abs(attributions)
    top_idx = np.argsort(abs_attr)[-top_k:][::-1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red' if attributions[i] > 0 else 'blue' for i in top_idx]
    ax.barh(range(len(top_idx)), [attributions[i] for i in top_idx], color=colors)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.invert_yaxis()
    ax.set_xlabel('Attribution (Red=Illicit, Blue=Licit)')
    ax.set_title('Top Feature Importances for Illicit Prediction')
    ax.axvline(0, color='black', lw=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
    return fig


def main():
    print("=" * 60)
    print("CHRONOS Explanation Generator")
    print("=" * 60)
    
    checkpoint_path = 'checkpoints/chronos_experiment/best_model.pt'
    output_dir = 'results/explanations'
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading trained model...")
    try:
        model, metrics = load_model(checkpoint_path, device)
        print(f"✓ Model loaded successfully")
        print(f"  Metrics: {metrics}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # Load data
    print("\nLoading dataset...")
    try:
        from chronos.data.loader import load_elliptic_dataset
        data = load_elliptic_dataset('data/elliptic')
        print(f"✓ Data: {data.x.shape[0]} nodes, {data.x.shape[1]} features")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("Creating synthetic test data...")
        from torch_geometric.data import Data
        n_nodes = 1000
        data = Data(
            x=torch.randn(n_nodes, 235),
            edge_index=torch.randint(0, n_nodes, (2, 5000)),
            y=torch.randint(0, 2, (n_nodes,)),
            test_mask=torch.ones(n_nodes, dtype=torch.bool)
        )
    
    # Find illicit samples
    test_mask = data.test_mask if hasattr(data, 'test_mask') else torch.ones(data.x.shape[0], dtype=torch.bool)
    illicit_mask = (data.y == 1) & test_mask
    illicit_indices = torch.where(illicit_mask)[0].numpy()
    
    if len(illicit_indices) == 0:
        print("No illicit samples found. Using random samples.")
        illicit_indices = np.arange(min(10, data.x.shape[0]))
    
    print(f"\n✓ Found {len(illicit_indices)} illicit transactions")
    
    # Generate explanations
    print("\n" + "=" * 60)
    print("Generating Feature Attributions...")
    print("=" * 60)
    
    np.random.seed(42)
    samples = np.random.choice(illicit_indices, min(5, len(illicit_indices)), replace=False)
    
    for i, node_idx in enumerate(samples):
        print(f"\nTransaction {i+1} (Node {node_idx}):")
        try:
            attrs = generate_feature_importance(model, data.x, data.edge_index, node_idx, device)
            save_path = os.path.join(output_dir, f'attribution_node_{node_idx}.png')
            plot_importance(attrs, top_k=15, save_path=save_path)
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print(f"✓ Explanations saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

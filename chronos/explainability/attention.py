"""
Attention visualization for CHRONOS.

Extracts and visualizes attention weights from Graph Attention Network layers
and Multi-Scale Temporal Attention to explain which neighbors and temporal
patterns influence predictions.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import matplotlib.pyplot as plt
import seaborn as sns


class AttentionVisualizer:
    """
    Visualizer for GAT and temporal attention weights in CHRONOS.

    Extracts attention weights from model and creates visualizations.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        """
        Initialize attention visualizer.

        Parameters
        ----------
        model : nn.Module
            Trained CHRONOS model with attention
        device : str
            Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def extract_attention_weights(
        self,
        data: Data,
        node_idx: Optional[int] = None,
        return_all: bool = False
    ) -> Dict:
        """
        Extract attention weights from model forward pass.

        Parameters
        ----------
        data : Data
            PyG Data object
        node_idx : Optional[int]
            If specified, extract attention for this node only
        return_all : bool
            If True, return attention for all layers

        Returns
        -------
        attention_weights : Dict
            Dictionary with attention weights:
            - 'temporal': Temporal attention weights
            - 'gat_layer_0', 'gat_layer_1', etc.: GAT layer attention
        """
        with torch.no_grad():
            # Forward pass with return_attention=True
            logits, attention_dict = self.model(
                data.x.to(self.device),
                data.edge_index.to(self.device),
                return_attention=True
            )

        # If specific node requested, filter attention weights
        if node_idx is not None and 'gat_layer_0' in attention_dict:
            filtered_attention = {}

            # Filter temporal attention
            if 'temporal' in attention_dict:
                filtered_attention['temporal'] = attention_dict['temporal']

            # Filter GAT attention for node
            for layer_name, (edge_index, attn_weights) in attention_dict.items():
                if layer_name.startswith('gat_layer'):
                    # Find edges involving this node
                    mask = (edge_index[1] == node_idx)  # Incoming edges
                    if mask.any():
                        filtered_attention[layer_name] = (
                            edge_index[:, mask],
                            attn_weights[mask]
                        )

            return filtered_attention

        return attention_dict

    def visualize_node_attention(
        self,
        data: Data,
        node_idx: int,
        k_hops: int = 2,
        layer: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention weights for a specific node.

        Parameters
        ----------
        data : Data
            PyG Data object
        node_idx : int
            Node to visualize
        k_hops : int
            Number of hops for neighborhood
        layer : int
            Which GAT layer to visualize
        save_path : Optional[str]
            Path to save figure
        """
        # Extract k-hop subgraph
        subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
            node_idx,
            k_hops,
            data.edge_index,
            relabel_nodes=True,
            num_nodes=data.x.size(0)
        )

        sub_data = Data(
            x=data.x[subset],
            edge_index=edge_index_sub
        )

        # Get attention weights
        attention_dict = self.extract_attention_weights(sub_data)

        layer_name = f'gat_layer_{layer}'
        if layer_name not in attention_dict:
            print(f"Attention weights for layer {layer} not found")
            return

        edge_index, attn_weights = attention_dict[layer_name]

        # Get target node in subgraph
        target_node = mapping.item()

        # Find incoming edges to target node
        incoming_mask = (edge_index[1] == target_node)
        incoming_sources = edge_index[0][incoming_mask].cpu().numpy()
        incoming_attention = attn_weights[incoming_mask].mean(dim=1).cpu().numpy()  # Average over heads

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort by attention weight
        sorted_indices = np.argsort(incoming_attention)[::-1]
        sorted_sources = incoming_sources[sorted_indices]
        sorted_attention = incoming_attention[sorted_indices]

        # Plot top neighbors
        top_k = min(20, len(sorted_sources))
        x_pos = np.arange(top_k)

        ax.bar(x_pos, sorted_attention[:top_k], color='steelblue', alpha=0.7)
        ax.set_xlabel('Neighbor Node Index')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'GAT Attention Weights for Node {node_idx} (Layer {layer})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{subset[src].item()}' for src in sorted_sources[:top_k]],
                           rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention visualization to {save_path}")

        plt.show()

    def visualize_temporal_attention(
        self,
        data: Data,
        node_idx: int,
        save_path: Optional[str] = None
    ):
        """
        Visualize temporal attention weights for multi-scale windows.

        Parameters
        ----------
        data : Data
            PyG Data object
        node_idx : int
            Node to visualize
        save_path : Optional[str]
            Path to save figure
        """
        # Get temporal attention
        attention_dict = self.extract_attention_weights(data, node_idx)

        if 'temporal' not in attention_dict:
            print("Temporal attention weights not found")
            return

        temporal_attention = attention_dict['temporal']

        # temporal_attention is a list of attention matrices, one per window size
        window_sizes = [1, 5, 15, 30]
        num_windows = len(temporal_attention)

        fig, axes = plt.subplots(1, num_windows, figsize=(16, 4))
        if num_windows == 1:
            axes = [axes]

        for i, (window_size, attn) in enumerate(zip(window_sizes[:num_windows], temporal_attention)):
            # attn: [batch, window_size, window_size]
            attn_matrix = attn[node_idx].cpu().numpy()

            sns.heatmap(
                attn_matrix,
                ax=axes[i],
                cmap='YlOrRd',
                cbar=True,
                square=True,
                annot=False
            )
            axes[i].set_title(f'Window Size: {window_size}')
            axes[i].set_xlabel('Timestep (to)')
            axes[i].set_ylabel('Timestep (from)')

        plt.suptitle(f'Multi-Scale Temporal Attention for Node {node_idx}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved temporal attention visualization to {save_path}")

        plt.show()

    def get_most_important_neighbors(
        self,
        data: Data,
        node_idx: int,
        layer: int = 0,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Get most important neighbors based on attention weights.

        Parameters
        ----------
        data : Data
            PyG Data object
        node_idx : int
            Node to analyze
        layer : int
            GAT layer
        top_k : int
            Number of top neighbors

        Returns
        -------
        top_neighbors : List[Tuple[int, float]]
            List of (neighbor_idx, attention_weight) tuples
        """
        # Get attention weights
        attention_dict = self.extract_attention_weights(data, node_idx)

        layer_name = f'gat_layer_{layer}'
        if layer_name not in attention_dict:
            return []

        edge_index, attn_weights = attention_dict[layer_name]

        # Average over attention heads
        avg_attention = attn_weights.mean(dim=1).cpu().numpy()

        # Get source nodes
        source_nodes = edge_index[0].cpu().numpy()

        # Sort by attention weight
        sorted_indices = np.argsort(avg_attention)[::-1]
        top_neighbors = [
            (int(source_nodes[i]), float(avg_attention[i]))
            for i in sorted_indices[:top_k]
        ]

        return top_neighbors

    def compare_attention_across_layers(
        self,
        data: Data,
        node_idx: int,
        save_path: Optional[str] = None
    ):
        """
        Compare attention distributions across GAT layers.

        Parameters
        ----------
        data : Data
            PyG Data object
        node_idx : int
            Node to analyze
        save_path : Optional[str]
            Path to save figure
        """
        attention_dict = self.extract_attention_weights(data, node_idx)

        # Count layers
        num_layers = sum(1 for key in attention_dict.keys() if key.startswith('gat_layer'))

        if num_layers == 0:
            print("No GAT attention found")
            return

        fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 4))
        if num_layers == 1:
            axes = [axes]

        for layer in range(num_layers):
            layer_name = f'gat_layer_{layer}'
            if layer_name in attention_dict:
                edge_index, attn_weights = attention_dict[layer_name]

                # Average over heads
                avg_attention = attn_weights.mean(dim=1).cpu().numpy()

                # Plot distribution
                axes[layer].hist(avg_attention, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
                axes[layer].set_xlabel('Attention Weight')
                axes[layer].set_ylabel('Frequency')
                axes[layer].set_title(f'Layer {layer}')
                axes[layer].grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Attention Weight Distributions for Node {node_idx}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention comparison to {save_path}")

        plt.show()

    def explain_attention(
        self,
        data: Data,
        node_idx: int,
        layer: int = 0,
        top_k: int = 5
    ) -> str:
        """
        Generate text explanation of attention weights.

        Parameters
        ----------
        data : Data
            PyG Data object
        node_idx : int
            Node to explain
        layer : int
            GAT layer
        top_k : int
            Number of top neighbors to include

        Returns
        -------
        explanation : str
            Text explanation
        """
        top_neighbors = self.get_most_important_neighbors(data, node_idx, layer, top_k)

        if not top_neighbors:
            return f"No attention weights found for node {node_idx}"

        text = f"Attention Explanation for Node {node_idx} (Layer {layer})\n"
        text += "=" * 60 + "\n\n"
        text += f"Top {len(top_neighbors)} most influential neighbors:\n\n"

        for rank, (neighbor_idx, attention_weight) in enumerate(top_neighbors, 1):
            percentage = attention_weight * 100
            text += f"{rank}. Node {neighbor_idx}: {percentage:.2f}% of total attention\n"

        text += "\n" + "=" * 60 + "\n"

        return text


def visualize_attention_for_node(
    model: nn.Module,
    data: Data,
    node_idx: int,
    device: str = 'cuda',
    save_dir: Optional[str] = None
) -> Dict:
    """
    Comprehensive attention visualization for a node.

    Parameters
    ----------
    model : nn.Module
        Trained CHRONOS model
    data : Data
        PyG Data object
    node_idx : int
        Node to visualize
    device : str
        Device
    save_dir : Optional[str]
        Directory to save figures

    Returns
    -------
    analysis : Dict
        Attention analysis results

    Examples
    --------
    >>> model = CHRONOSNet(...)
    >>> model.load_state_dict(torch.load('best_model.pt'))
    >>> analysis = visualize_attention_for_node(model, data, node_idx=1234)
    """
    visualizer = AttentionVisualizer(model, device=device)

    # Get top neighbors
    top_neighbors = visualizer.get_most_important_neighbors(data, node_idx)

    # Generate text explanation
    explanation = visualizer.explain_attention(data, node_idx)
    print(explanation)

    # Create visualizations
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)

        visualizer.visualize_node_attention(
            data, node_idx,
            save_path=os.path.join(save_dir, f'node_{node_idx}_attention.png')
        )

        visualizer.visualize_temporal_attention(
            data, node_idx,
            save_path=os.path.join(save_dir, f'node_{node_idx}_temporal_attention.png')
        )

        visualizer.compare_attention_across_layers(
            data, node_idx,
            save_path=os.path.join(save_dir, f'node_{node_idx}_attention_comparison.png')
        )
    else:
        visualizer.visualize_node_attention(data, node_idx)
        visualizer.visualize_temporal_attention(data, node_idx)
        visualizer.compare_attention_across_layers(data, node_idx)

    analysis = {
        'node_idx': node_idx,
        'top_neighbors': top_neighbors,
        'explanation_text': explanation
    }

    return analysis

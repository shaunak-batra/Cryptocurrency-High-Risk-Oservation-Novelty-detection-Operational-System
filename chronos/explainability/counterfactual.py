"""
Counterfactual explanation generator for CHRONOS.

Extends CF-GNNExplainer to temporal graphs for cryptocurrency AML detection.
This is CHRONOS's primary novel contribution.

Key idea: Find minimal changes to graph structure or node features that would
flip the model's prediction (illicit -> licit or vice versa) while maintaining
temporal validity constraints.

Reference: Lucic et al. (2022) "CF-GNNExplainer: Counterfactual Explanations for GNNs"
Extended with temporal constraints for cryptocurrency transactions.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_dense_adj


class TemporalCounterfactualExplainer:
    """
    Temporal counterfactual explanation generator for CHRONOS.

    Finds minimal perturbations to graph structure or node features that flip
    the model's prediction while respecting temporal constraints.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_prox: float = 0.5,
        lambda_div: float = 0.1,
        lambda_valid: float = 1.0,
        lambda_temporal: float = 0.5,
        num_iterations: int = 100,
        learning_rate: float = 0.01,
        device: str = 'cuda'
    ):
        """
        Initialize counterfactual explainer.

        Parameters
        ----------
        model : nn.Module
            Trained CHRONOS model
        lambda_prox : float
            Weight for proximity loss (how close to original)
        lambda_div : float
            Weight for diversity loss (encourage diverse explanations)
        lambda_valid : float
            Weight for validity loss (maintain valid graph structure)
        lambda_temporal : float
            Weight for temporal validity (respect time ordering)
        num_iterations : int
            Number of optimization iterations
        learning_rate : float
            Learning rate for optimization
        device : str
            Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Loss weights
        self.lambda_prox = lambda_prox
        self.lambda_div = lambda_div
        self.lambda_valid = lambda_valid
        self.lambda_temporal = lambda_temporal

        # Optimization parameters
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

    def explain(
        self,
        data: Data,
        node_idx: int,
        target_class: Optional[int] = None,
        num_counterfactuals: int = 3,
        k_hops: int = 2
    ) -> Dict:
        """
        Generate counterfactual explanations for a node.

        Parameters
        ----------
        data : Data
            PyG Data object
        node_idx : int
            Index of node to explain
        target_class : Optional[int]
            Target class for counterfactual (0=licit, 1=illicit)
            If None, flips the current prediction
        num_counterfactuals : int
            Number of diverse counterfactuals to generate
        k_hops : int
            Number of hops for computational subgraph

        Returns
        -------
        explanation : Dict
            Dictionary containing:
            - 'node_idx': Original node index
            - 'original_pred': Original prediction
            - 'target_class': Target class
            - 'counterfactuals': List of counterfactual graphs
            - 'changes': List of changes made
            - 'scores': Quality scores for each counterfactual
        """
        # Get original prediction
        with torch.no_grad():
            logits = self.model(
                data.x.to(self.device),
                data.edge_index.to(self.device)
            )
            if isinstance(logits, tuple):
                logits = logits[0]
            original_pred = logits[node_idx].argmax().item()
            original_proba = torch.softmax(logits[node_idx], dim=0)

        # Determine target class
        if target_class is None:
            target_class = 1 - original_pred  # Flip prediction

        print(f"Node {node_idx}: Generating {num_counterfactuals} counterfactuals")
        print(f"  Original prediction: {original_pred} (prob={original_proba[original_pred]:.4f})")
        print(f"  Target class: {target_class}")

        # Extract k-hop subgraph
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx,
            k_hops,
            data.edge_index,
            relabel_nodes=True,
            num_nodes=data.x.size(0)
        )

        # Create subgraph data
        sub_data = Data(
            x=data.x[subset].to(self.device),
            edge_index=edge_index.to(self.device),
            timestep=data.timestep[subset].to(self.device)
        )
        target_node_idx = mapping.item()

        # Generate multiple diverse counterfactuals
        counterfactuals = []
        all_changes = []
        scores = []

        for i in range(num_counterfactuals):
            print(f"  Generating counterfactual {i+1}/{num_counterfactuals}...")

            cf_data, changes, score = self._generate_single_counterfactual(
                sub_data,
                target_node_idx,
                target_class,
                diversity_ref=counterfactuals
            )

            counterfactuals.append(cf_data)
            all_changes.append(changes)
            scores.append(score)

        explanation = {
            'node_idx': node_idx,
            'original_pred': original_pred,
            'original_proba': original_proba.cpu().numpy(),
            'target_class': target_class,
            'counterfactuals': counterfactuals,
            'changes': all_changes,
            'scores': scores
        }

        return explanation

    def _generate_single_counterfactual(
        self,
        data: Data,
        node_idx: int,
        target_class: int,
        diversity_ref: List[Data] = None
    ) -> Tuple[Data, Dict, float]:
        """
        Generate a single counterfactual explanation.

        Uses gradient-based optimization to perturb graph structure.
        """
        # Initialize perturbation parameters
        # Edge perturbation: learnable adjacency matrix
        num_nodes = data.x.size(0)
        adj_orig = to_dense_adj(data.edge_index, max_num_nodes=num_nodes)[0]

        # Learnable edge probabilities (logits)
        edge_logits = torch.zeros_like(adj_orig, requires_grad=True, device=self.device)
        edge_logits.data = torch.logit(adj_orig + 0.01)  # Initialize close to original

        # Feature perturbation (optional)
        # For now, focus on graph structure perturbation

        optimizer = optim.Adam([edge_logits], lr=self.learning_rate)

        best_loss = float('inf')
        best_adj = None

        for iteration in range(self.num_iterations):
            optimizer.zero_grad()

            # Sample adjacency matrix with Gumbel-Softmax
            adj_perturbed = torch.sigmoid(edge_logits)

            # Binarize for hard edges (during inference)
            if iteration % 10 == 0:
                adj_hard = (adj_perturbed > 0.5).float()
            else:
                adj_hard = adj_perturbed

            # Convert to edge_index
            edge_index_perturbed = adj_hard.nonzero().t()

            # Forward pass with perturbed graph
            logits = self.model(data.x, edge_index_perturbed)
            if isinstance(logits, tuple):
                logits = logits[0]

            # Compute loss
            loss = self._compute_loss(
                logits[node_idx],
                target_class,
                adj_orig,
                adj_perturbed,
                data.timestep,
                edge_index_perturbed,
                diversity_ref
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track best solution
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_adj = adj_perturbed.detach()

            # Check if target class achieved
            pred_class = logits[node_idx].argmax().item()
            if pred_class == target_class and iteration > 20:
                print(f"    Converged at iteration {iteration}")
                break

        # Create counterfactual data
        if best_adj is not None:
            adj_binary = (best_adj > 0.5).float()
            edge_index_cf = adj_binary.nonzero().t()
        else:
            edge_index_cf = data.edge_index

        cf_data = Data(
            x=data.x.clone(),
            edge_index=edge_index_cf,
            timestep=data.timestep.clone()
        )

        # Compute changes
        changes = self._compute_changes(adj_orig, best_adj)

        # Compute quality score
        score = self._compute_quality_score(cf_data, node_idx, target_class, changes)

        return cf_data, changes, score

    def _compute_loss(
        self,
        logits: torch.Tensor,
        target_class: int,
        adj_orig: torch.Tensor,
        adj_perturbed: torch.Tensor,
        timesteps: torch.Tensor,
        edge_index_perturbed: torch.Tensor,
        diversity_ref: List[Data] = None
    ) -> torch.Tensor:
        """
        Compute counterfactual loss.

        L_total = L_pred + λ_prox * L_prox + λ_div * L_div + λ_valid * L_valid + λ_temp * L_temp
        """
        # 1. Prediction loss: encourage target class
        target_tensor = torch.tensor([target_class], device=self.device)
        L_pred = nn.functional.cross_entropy(logits.unsqueeze(0), target_tensor)

        # 2. Proximity loss: stay close to original graph
        L_prox = torch.sum((adj_perturbed - adj_orig) ** 2)

        # 3. Diversity loss: encourage different explanations
        L_div = 0.0
        if diversity_ref is not None and len(diversity_ref) > 0:
            for ref_data in diversity_ref:
                ref_adj = to_dense_adj(ref_data.edge_index, max_num_nodes=adj_perturbed.size(0))[0]
                # Negative similarity (we want to be different)
                similarity = -torch.sum((adj_perturbed - ref_adj) ** 2)
                L_div += similarity
            L_div = L_div / len(diversity_ref)

        # 4. Validity loss: maintain valid graph properties
        # Encourage sparse graphs (avoid too many edges)
        num_edges_orig = adj_orig.sum()
        num_edges_perturbed = adj_perturbed.sum()
        L_valid = torch.abs(num_edges_perturbed - num_edges_orig) / num_edges_orig

        # 5. Temporal validity loss: respect time ordering
        # Edges should only go forward in time or within same timestep
        L_temp = 0.0
        if edge_index_perturbed.size(1) > 0:
            src_times = timesteps[edge_index_perturbed[0]]
            dst_times = timesteps[edge_index_perturbed[1]]
            # Penalize backward-in-time edges
            backward_edges = (src_times > dst_times).float()
            L_temp = backward_edges.sum()

        # Total loss
        total_loss = (
            L_pred +
            self.lambda_prox * L_prox +
            self.lambda_div * L_div +
            self.lambda_valid * L_valid +
            self.lambda_temporal * L_temp
        )

        return total_loss

    def _compute_changes(
        self,
        adj_orig: torch.Tensor,
        adj_cf: torch.Tensor
    ) -> Dict:
        """
        Compute and describe changes made.

        Returns
        -------
        changes : Dict
            Dictionary with:
            - 'edges_added': List of added edges
            - 'edges_removed': List of removed edges
            - 'num_changes': Total number of changes
        """
        adj_orig_binary = (adj_orig > 0.5).cpu().numpy()
        adj_cf_binary = (adj_cf > 0.5).cpu().numpy()

        # Find differences
        diff = adj_cf_binary - adj_orig_binary

        # Added edges
        added = np.argwhere(diff > 0)
        edges_added = [(int(src), int(dst)) for src, dst in added]

        # Removed edges
        removed = np.argwhere(diff < 0)
        edges_removed = [(int(src), int(dst)) for src, dst in removed]

        changes = {
            'edges_added': edges_added,
            'edges_removed': edges_removed,
            'num_changes': len(edges_added) + len(edges_removed)
        }

        return changes

    def _compute_quality_score(
        self,
        cf_data: Data,
        node_idx: int,
        target_class: int,
        changes: Dict
    ) -> float:
        """
        Compute quality score for counterfactual.

        Quality = Prediction confidence * (1 / num_changes)
        """
        with torch.no_grad():
            logits = self.model(cf_data.x, cf_data.edge_index)
            if isinstance(logits, tuple):
                logits = logits[0]

            proba = torch.softmax(logits[node_idx], dim=0)
            target_confidence = proba[target_class].item()

        # Penalize by number of changes (prefer minimal changes)
        num_changes = changes['num_changes']
        if num_changes == 0:
            num_changes = 1  # Avoid division by zero

        quality = target_confidence / np.sqrt(num_changes)

        return quality

    def generate_explanation_text(
        self,
        explanation: Dict,
        verbose: bool = True
    ) -> str:
        """
        Generate human-readable explanation text.

        Parameters
        ----------
        explanation : Dict
            Explanation dictionary from explain()
        verbose : bool
            Whether to include detailed changes

        Returns
        -------
        text : str
            Human-readable explanation
        """
        node_idx = explanation['node_idx']
        original_pred = explanation['original_pred']
        target_class = explanation['target_class']
        num_cf = len(explanation['counterfactuals'])

        class_names = {0: 'licit', 1: 'illicit'}

        text = f"Counterfactual Explanation for Transaction {node_idx}\n"
        text += "=" * 60 + "\n\n"
        text += f"Original Prediction: {class_names[original_pred]}\n"
        text += f"Target Prediction: {class_names[target_class]}\n"
        text += f"Generated {num_cf} counterfactual explanations:\n\n"

        for i, (changes, score) in enumerate(zip(explanation['changes'], explanation['scores'])):
            text += f"Counterfactual {i+1} (Quality Score: {score:.4f}):\n"

            if changes['edges_added']:
                text += f"  - Adding {len(changes['edges_added'])} edge(s):\n"
                if verbose:
                    for src, dst in changes['edges_added'][:5]:  # Show first 5
                        text += f"    + Edge {src} -> {dst}\n"

            if changes['edges_removed']:
                text += f"  - Removing {len(changes['edges_removed'])} edge(s):\n"
                if verbose:
                    for src, dst in changes['edges_removed'][:5]:  # Show first 5
                        text += f"    - Edge {src} -> {dst}\n"

            text += f"  Total changes: {changes['num_changes']}\n\n"

        return text


def explain_node(
    model: nn.Module,
    data: Data,
    node_idx: int,
    device: str = 'cuda'
) -> Dict:
    """
    Convenience function to generate counterfactual explanation.

    Parameters
    ----------
    model : nn.Module
        Trained CHRONOS model
    data : Data
        PyG Data object
    node_idx : int
        Node to explain
    device : str
        Device

    Returns
    -------
    explanation : Dict
        Counterfactual explanation

    Examples
    --------
    >>> model = CHRONOSNet(...)
    >>> model.load_state_dict(torch.load('best_model.pt'))
    >>> explanation = explain_node(model, data, node_idx=1234)
    >>> print(explanation['changes'][0])
    """
    explainer = TemporalCounterfactualExplainer(model, device=device)
    explanation = explainer.explain(data, node_idx)
    text = explainer.generate_explanation_text(explanation)
    print(text)
    return explanation

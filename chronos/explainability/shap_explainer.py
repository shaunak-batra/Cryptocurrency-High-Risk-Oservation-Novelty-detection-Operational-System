"""
SHAP (SHapley Additive exPlanations) integration for CHRONOS.

Provides feature importance explanations using KernelSHAP adapted for graph neural networks.
Identifies which features contribute most to illicit/licit predictions.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
import shap


class CHRONOSSHAPExplainer:
    """
    SHAP explainer for CHRONOS models.

    Uses KernelSHAP to compute feature importance scores for node predictions.
    """

    def __init__(
        self,
        model: nn.Module,
        data: Data,
        background_samples: int = 100,
        device: str = 'cuda'
    ):
        """
        Initialize SHAP explainer.

        Parameters
        ----------
        model : nn.Module
            Trained CHRONOS model
        data : Data
            Full graph data (for background distribution)
        background_samples : int
            Number of background samples for KernelSHAP
        device : str
            Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.data = data

        # Create background dataset
        self.background_data = self._create_background_data(background_samples)

        # Feature names (if available)
        self.feature_names = self._get_feature_names()

    def _create_background_data(self, num_samples: int) -> np.ndarray:
        """
        Create background dataset for SHAP.

        Samples from training set distribution.
        """
        train_mask = self.data.train_mask
        train_features = self.data.x[train_mask].cpu().numpy()

        # Sample random background
        if num_samples < len(train_features):
            indices = np.random.choice(len(train_features), num_samples, replace=False)
            background = train_features[indices]
        else:
            background = train_features

        return background

    def _get_feature_names(self) -> List[str]:
        """
        Get feature names.

        Returns list of feature names for visualization.
        """
        num_features = self.data.x.size(1)

        # Original Elliptic features (1-166)
        feature_names = [f'local_feature_{i}' for i in range(1, 167)]

        # Engineered features (167-236) if present
        if num_features > 166:
            # Graph topology features
            graph_features = [
                'in_degree', 'out_degree', 'total_degree',
                'in_degree_centrality', 'out_degree_centrality',
                'pagerank', 'betweenness', 'closeness', 'eigenvector',
                'clustering_coef', 'num_triangles', 'local_clustering',
                'core_number', 'eccentricity', 'avg_neighbor_degree',
                'degree_assortativity', 'rich_club_coef', 'community',
                'dist_to_illicit', 'dist_to_licit'
            ]

            # Temporal features
            temporal_features = [
                'tx_freq_7d', 'tx_freq_30d',
                'inter_event_mean', 'inter_event_std', 'inter_event_skew', 'inter_event_kurt',
                'burstiness', 'circadian_pattern', 'weekly_pattern',
                'time_since_first', 'time_since_last', 'tx_velocity',
                'active_days', 'inactive_days', 'longest_inactive',
                'temporal_entropy', 'autocorr_lag1', 'trend',
                'seasonality', 'weekend_ratio', 'night_ratio',
                'peak_hour', 'activity_concentration', 'activity_change_zscore',
                'temporal_clustering'
            ]

            # Amount features (placeholders)
            amount_features = [
                'total_sent', 'total_received', 'net_flow',
                'avg_amount', 'std_amount', 'max_amount', 'min_amount',
                'num_tx_10k', 'num_tx_100k', 'coef_variation',
                'amount_gini', 'structuring_score', 'round_number_ratio',
                'amount_entropy', 'unusual_amount_zscore'
            ]

            # Entity behavior features
            entity_features = [
                'account_age', 'num_counterparties', 'counterparty_diversity',
                'repeat_interaction_ratio', 'fan_out', 'fan_in',
                'mixing_score', 'peeling_chain', 'hub_score', 'broker_score'
            ]

            feature_names.extend(graph_features[:num_features-166])
            if num_features > 186:
                feature_names.extend(temporal_features[:num_features-186])
            if num_features > 211:
                feature_names.extend(amount_features[:num_features-211])
            if num_features > 226:
                feature_names.extend(entity_features[:num_features-226])

        return feature_names[:num_features]

    def explain_node(
        self,
        node_idx: int,
        num_samples: int = 100
    ) -> Dict:
        """
        Explain predictions for a specific node using SHAP.

        Parameters
        ----------
        node_idx : int
            Node to explain
        num_samples : int
            Number of samples for KernelSHAP

        Returns
        -------
        explanation : Dict
            Dictionary containing:
            - 'shap_values': SHAP values for each feature
            - 'base_value': Base prediction value
            - 'prediction': Model prediction
            - 'feature_names': Feature names
            - 'top_features': Top contributing features
        """
        print(f"Computing SHAP values for node {node_idx}...")

        # Create prediction function
        def predict_fn(features):
            """Prediction function for SHAP."""
            # features: [num_samples, num_features]
            num_samples = features.shape[0]
            predictions = []

            for i in range(num_samples):
                # Replace target node's features
                modified_x = self.data.x.clone()
                modified_x[node_idx] = torch.tensor(
                    features[i],
                    dtype=torch.float,
                    device=self.device
                )

                # Forward pass
                with torch.no_grad():
                    logits = self.model(
                        modified_x.to(self.device),
                        self.data.edge_index.to(self.device)
                    )
                    if isinstance(logits, tuple):
                        logits = logits[0]

                    # Get probability for illicit class
                    proba = torch.softmax(logits[node_idx], dim=0)[1].cpu().numpy()
                    predictions.append(proba)

            return np.array(predictions)

        # Initialize SHAP explainer
        explainer = shap.KernelExplainer(
            predict_fn,
            self.background_data,
            link='identity'
        )

        # Get node features
        node_features = self.data.x[node_idx].cpu().numpy().reshape(1, -1)

        # Compute SHAP values
        shap_values = explainer.shap_values(
            node_features,
            nsamples=num_samples,
            silent=True
        )

        # Get prediction
        prediction = predict_fn(node_features)[0]

        # Get top features
        top_features = self._get_top_features(shap_values[0], k=20)

        explanation = {
            'shap_values': shap_values[0],
            'base_value': explainer.expected_value,
            'prediction': prediction,
            'feature_names': self.feature_names,
            'top_features': top_features,
            'node_idx': node_idx
        }

        return explanation

    def _get_top_features(
        self,
        shap_values: np.ndarray,
        k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Get top k contributing features.

        Parameters
        ----------
        shap_values : np.ndarray
            SHAP values for features
        k : int
            Number of top features

        Returns
        -------
        top_features : List[Tuple[str, float]]
            List of (feature_name, shap_value) tuples
        """
        # Sort by absolute SHAP value
        indices = np.argsort(np.abs(shap_values))[-k:][::-1]

        top_features = [
            (self.feature_names[i], shap_values[i])
            for i in indices
        ]

        return top_features

    def explain_multiple_nodes(
        self,
        node_indices: List[int],
        num_samples: int = 100
    ) -> Dict:
        """
        Explain multiple nodes and aggregate results.

        Parameters
        ----------
        node_indices : List[int]
            List of nodes to explain
        num_samples : int
            Number of samples for each node

        Returns
        -------
        aggregated_explanation : Dict
            Aggregated SHAP values and top features
        """
        all_shap_values = []
        all_predictions = []

        for node_idx in node_indices:
            explanation = self.explain_node(node_idx, num_samples)
            all_shap_values.append(explanation['shap_values'])
            all_predictions.append(explanation['prediction'])

        # Aggregate SHAP values (mean absolute)
        shap_values_array = np.array(all_shap_values)
        mean_shap_values = np.mean(np.abs(shap_values_array), axis=0)

        # Get top features across all nodes
        top_features = self._get_top_features(mean_shap_values, k=20)

        aggregated = {
            'mean_shap_values': mean_shap_values,
            'all_shap_values': shap_values_array,
            'predictions': all_predictions,
            'feature_names': self.feature_names,
            'top_features': top_features,
            'num_nodes': len(node_indices)
        }

        return aggregated

    def print_explanation(self, explanation: Dict):
        """
        Print human-readable explanation.

        Parameters
        ----------
        explanation : Dict
            Explanation from explain_node()
        """
        print("\n" + "=" * 60)
        print("SHAP Feature Importance Explanation")
        print("=" * 60)
        print(f"Node: {explanation['node_idx']}")
        print(f"Prediction (illicit probability): {explanation['prediction']:.4f}")
        print(f"Base value: {explanation['base_value']:.4f}")
        print("\nTop 20 Contributing Features:")
        print("-" * 60)

        for i, (feature_name, shap_value) in enumerate(explanation['top_features'], 1):
            direction = "increases" if shap_value > 0 else "decreases"
            print(f"{i:2d}. {feature_name:30s}: {shap_value:+.4f} ({direction} illicit probability)")

        print("=" * 60 + "\n")


def explain_with_shap(
    model: nn.Module,
    data: Data,
    node_idx: int,
    device: str = 'cuda'
) -> Dict:
    """
    Convenience function to generate SHAP explanation.

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
        SHAP explanation

    Examples
    --------
    >>> model = CHRONOSNet(...)
    >>> model.load_state_dict(torch.load('best_model.pt'))
    >>> explanation = explain_with_shap(model, data, node_idx=1234)
    """
    explainer = CHRONOSSHAPExplainer(model, data, device=device)
    explanation = explainer.explain_node(node_idx)
    explainer.print_explanation(explanation)
    return explanation

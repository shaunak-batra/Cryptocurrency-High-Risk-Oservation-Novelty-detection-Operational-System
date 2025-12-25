"""
Baseline models for CHRONOS benchmarking.

This module implements three baseline models:
1. Random Forest (traditional ML)
2. XGBoost (gradient boosting)
3. Vanilla GCN (graph neural network)

These baselines establish performance floors for CHRONOS-Net to exceed.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


# ============================================================================
# 1. RANDOM FOREST BASELINE
# ============================================================================

class RandomForestBaseline:
    """
    Random Forest baseline for cryptocurrency AML detection.

    Uses only node features (no graph structure).
    Expected F1: ~0.72 (from literature)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 20,
        min_samples_split: int = 10,
        class_weight: str = 'balanced',
        random_state: int = 42
    ):
        """
        Initialize Random Forest.

        Parameters
        ----------
        n_estimators : int
            Number of trees
        max_depth : int
            Maximum tree depth
        min_samples_split : int
            Minimum samples to split
        class_weight : str
            How to handle class imbalance
        random_state : int
            Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )

    def fit(self, data: Data):
        """
        Train Random Forest on node features.

        Parameters
        ----------
        data : Data
            PyG Data object with train_mask
        """
        X_train = data.x[data.train_mask].numpy()
        y_train = data.y[data.train_mask].numpy()

        print(f"Training Random Forest on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        print("✓ Random Forest training complete")

    def predict(self, data: Data, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Predict labels.

        Parameters
        ----------
        data : Data
            PyG Data object
        mask : Optional[Tensor]
            Mask for which nodes to predict

        Returns
        -------
        y_pred : np.ndarray
            Predicted labels
        """
        if mask is None:
            X = data.x.numpy()
        else:
            X = data.x[mask].numpy()

        return self.model.predict(X)

    def predict_proba(self, data: Data, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Predict probabilities.

        Parameters
        ----------
        data : Data
            PyG Data object
        mask : Optional[Tensor]
            Mask for which nodes to predict

        Returns
        -------
        y_proba : np.ndarray
            Predicted probabilities [n_samples, 2]
        """
        if mask is None:
            X = data.x.numpy()
        else:
            X = data.x[mask].numpy()

        return self.model.predict_proba(X)

    def evaluate(self, data: Data, mask: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model performance.

        Parameters
        ----------
        data : Data
            PyG Data object
        mask : Tensor
            Mask for evaluation set

        Returns
        -------
        metrics : Dict[str, float]
            Dictionary of metrics
        """
        y_true = data.y[mask].numpy()
        y_pred = self.predict(data, mask)
        y_proba = self.predict_proba(data, mask)[:, 1]

        metrics = {
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_proba)
        }

        return metrics


# ============================================================================
# 2. XGBOOST BASELINE
# ============================================================================

class XGBoostBaseline:
    """
    XGBoost baseline for cryptocurrency AML detection.

    Uses only node features (no graph structure).
    Expected F1: ~0.74 (from literature)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42
    ):
        """
        Initialize XGBoost.

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Learning rate
        scale_pos_weight : Optional[float]
            Weight for positive class (for imbalance)
        random_state : int
            Random seed
        """
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=1
        )

    def fit(self, data: Data, eval_set: Optional[Tuple[Data, torch.Tensor]] = None):
        """
        Train XGBoost on node features.

        Parameters
        ----------
        data : Data
            PyG Data object with train_mask
        eval_set : Optional[Tuple[Data, Tensor]]
            Optional validation set for early stopping
        """
        X_train = data.x[data.train_mask].numpy()
        y_train = data.y[data.train_mask].numpy()

        eval_list = None
        if eval_set is not None:
            eval_data, eval_mask = eval_set
            X_val = eval_data.x[eval_mask].numpy()
            y_val = eval_data.y[eval_mask].numpy()
            eval_list = [(X_val, y_val)]

        print(f"Training XGBoost on {len(X_train)} samples...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_list,
            verbose=True
        )
        print("✓ XGBoost training complete")

    def predict(self, data: Data, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """Predict labels."""
        if mask is None:
            X = data.x.numpy()
        else:
            X = data.x[mask].numpy()

        return self.model.predict(X)

    def predict_proba(self, data: Data, mask: Optional[torch.Tensor] = None) -> np.ndarray:
        """Predict probabilities."""
        if mask is None:
            X = data.x.numpy()
        else:
            X = data.x[mask].numpy()

        return self.model.predict_proba(X)

    def evaluate(self, data: Data, mask: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance."""
        y_true = data.y[mask].numpy()
        y_pred = self.predict(data, mask)
        y_proba = self.predict_proba(data, mask)[:, 1]

        metrics = {
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_proba)
        }

        return metrics


# ============================================================================
# 3. VANILLA GCN BASELINE
# ============================================================================

class VanillaGCN(nn.Module):
    """
    Vanilla Graph Convolutional Network baseline.

    Uses graph structure but no temporal modeling.
    Expected F1: ~0.62 (from literature)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        """
        Initialize Vanilla GCN.

        Parameters
        ----------
        in_features : int
            Number of input features
        hidden_dim : int
            Hidden dimension
        num_layers : int
            Number of GCN layers
        dropout : float
            Dropout rate
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Classifier
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor [num_nodes, in_features]
            Node features
        edge_index : Tensor [2, num_edges]
            Edge indices

        Returns
        -------
        out : Tensor [num_nodes, 2]
            Logits for binary classification
        """
        # GCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Classifier
        out = self.classifier(x)
        return out

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict labels.

        Parameters
        ----------
        x : Tensor
            Node features
        edge_index : Tensor
            Edge indices

        Returns
        -------
        y_pred : Tensor
            Predicted labels
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            y_pred = logits.argmax(dim=1)
        return y_pred


class VanillaGCNTrainer:
    """Trainer for Vanilla GCN baseline."""

    def __init__(
        self,
        model: VanillaGCN,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        model : VanillaGCN
            GCN model
        learning_rate : float
            Learning rate
        weight_decay : float
            L2 regularization
        """
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, data: Data) -> float:
        """
        Train for one epoch.

        Parameters
        ----------
        data : Data
            PyG Data object

        Returns
        -------
        loss : float
            Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        out = self.model(data.x, data.edge_index)
        loss = self.criterion(out[data.train_mask], data.y[data.train_mask])

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, data: Data, mask: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate model.

        Parameters
        ----------
        data : Data
            PyG Data object
        mask : Tensor
            Evaluation mask

        Returns
        -------
        metrics : Dict[str, float]
            Dictionary of metrics
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)
            y_pred = logits[mask].argmax(dim=1)
            y_proba = F.softmax(logits[mask], dim=1)[:, 1]
            y_true = data.y[mask]

            metrics = {
                'f1': f1_score(y_true.cpu(), y_pred.cpu()),
                'precision': precision_score(y_true.cpu(), y_pred.cpu()),
                'recall': recall_score(y_true.cpu(), y_pred.cpu()),
                'auc_roc': roc_auc_score(y_true.cpu(), y_proba.cpu())
            }

        return metrics

    def fit(
        self,
        data: Data,
        num_epochs: int = 200,
        patience: int = 20,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train model with early stopping.

        Parameters
        ----------
        data : Data
            PyG Data object
        num_epochs : int
            Maximum number of epochs
        patience : int
            Early stopping patience
        verbose : bool
            Whether to print progress

        Returns
        -------
        best_metrics : Dict[str, float]
            Best validation metrics
        """
        best_val_f1 = 0
        patience_counter = 0
        best_metrics = {}

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(data)

            # Evaluate
            if epoch % 10 == 0:
                val_metrics = self.evaluate(data, data.val_mask)

                if verbose:
                    print(f"Epoch {epoch:03d}: "
                          f"Loss={train_loss:.4f}, "
                          f"Val F1={val_metrics['f1']:.4f}")

                # Early stopping
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    best_metrics = val_metrics
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        print("✓ Vanilla GCN training complete")
        return best_metrics

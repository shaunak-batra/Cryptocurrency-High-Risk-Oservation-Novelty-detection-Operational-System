"""
Visualization utilities for CHRONOS.

Provides plotting functions for:
- Confusion matrices
- ROC and PR curves
- Attention weights
- Embeddings (t-SNE, UMAP)
- Temporal patterns
- Performance over time
"""

from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
import torch


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = ['Licit', 'Illicit'],
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.

    Parameters
    ----------
    cm : np.ndarray [2, 2]
        Confusion matrix
    class_names : list
        Class names
    normalize : bool
        Whether to normalize
    title : str
        Plot title
    save_path : Optional[str]
        Path to save figure
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = 'ROC Curve',
    save_path: Optional[str] = None
):
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    title : str
        Plot title
    save_path : Optional[str]
        Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")

    plt.show()


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = 'Precision-Recall Curve',
    save_path: Optional[str] = None
):
    """
    Plot Precision-Recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    title : str
        Plot title
    save_path : Optional[str]
        Path to save figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curve to {save_path}")

    plt.show()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_metrics: list,
    metric_name: str = 'F1',
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves.

    Parameters
    ----------
    train_losses : list
        Training losses per epoch
    val_losses : list
        Validation losses per epoch
    val_metrics : list
        Validation metrics per epoch
    metric_name : str
        Name of metric being plotted
    save_path : Optional[str]
        Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot metrics
    ax2.plot(epochs, val_metrics, 'g-', label=f'Val {metric_name}', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_name)
    ax2.set_title(f'Validation {metric_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.show()


def plot_attention_weights(
    attention_weights: torch.Tensor,
    node_idx: int,
    neighbor_indices: Optional[list] = None,
    title: str = 'Attention Weights',
    save_path: Optional[str] = None
):
    """
    Plot attention weights for a specific node.

    Parameters
    ----------
    attention_weights : Tensor [num_neighbors]
        Attention weights for node's neighbors
    node_idx : int
        Index of node
    neighbor_indices : Optional[list]
        Indices of neighbors
    title : str
        Plot title
    save_path : Optional[str]
        Path to save figure
    """
    attention_weights = attention_weights.detach().cpu().numpy()

    if neighbor_indices is None:
        neighbor_indices = list(range(len(attention_weights)))

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(attention_weights)), attention_weights, color='steelblue')
    plt.xlabel('Neighbor Index')
    plt.ylabel('Attention Weight')
    plt.title(f'{title} (Node {node_idx})')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention weights to {save_path}")

    plt.show()


def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list = ['Licit', 'Illicit'],
    title: str = 't-SNE Visualization of Node Embeddings',
    save_path: Optional[str] = None
):
    """
    Plot t-SNE visualization of node embeddings.

    Parameters
    ----------
    embeddings : np.ndarray [num_nodes, embedding_dim]
        Node embeddings
    labels : np.ndarray [num_nodes]
        Node labels
    class_names : list
        Class names
    title : str
        Plot title
    save_path : Optional[str]
        Path to save figure
    """
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_name,
            alpha=0.6,
            s=20
        )

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")

    plt.show()


def plot_performance_by_timestep(
    metrics_by_timestep: dict,
    metric_name: str = 'f1',
    title: Optional[str] = None,
    save_path: Optional[str] = None
):
    """
    Plot performance metric across timesteps.

    Parameters
    ----------
    metrics_by_timestep : dict
        Metrics for each timestep
    metric_name : str
        Name of metric to plot
    title : Optional[str]
        Plot title
    save_path : Optional[str]
        Path to save figure
    """
    timesteps = sorted(metrics_by_timestep.keys())
    values = [metrics_by_timestep[t].get(metric_name, 0) for t in timesteps]

    if title is None:
        title = f'{metric_name.upper()} by Timestep'

    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, values, 'o-', linewidth=2, markersize=6)
    plt.xlabel('Timestep')
    plt.ylabel(metric_name.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # Add train/val/test regions
    plt.axvspan(1, 34, alpha=0.1, color='blue', label='Train')
    plt.axvspan(35, 43, alpha=0.1, color='orange', label='Val')
    plt.axvspan(44, 49, alpha=0.1, color='green', label='Test')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved timestep plot to {save_path}")

    plt.show()


def plot_feature_importance(
    feature_importances: np.ndarray,
    feature_names: Optional[list] = None,
    top_k: int = 20,
    title: str = 'Feature Importance',
    save_path: Optional[str] = None
):
    """
    Plot feature importance scores.

    Parameters
    ----------
    feature_importances : np.ndarray
        Feature importance scores
    feature_names : Optional[list]
        Feature names
    top_k : int
        Number of top features to show
    title : str
        Plot title
    save_path : Optional[str]
        Path to save figure
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(feature_importances))]

    # Sort by importance
    indices = np.argsort(feature_importances)[-top_k:]
    top_importances = feature_importances[indices]
    top_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, max(6, top_k * 0.3)))
    plt.barh(range(top_k), top_importances, color='steelblue')
    plt.yticks(range(top_k), top_names)
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance plot to {save_path}")

    plt.show()


def plot_model_comparison(
    model_metrics: dict,
    metrics_to_plot: list = ['f1', 'precision', 'recall', 'auc_roc'],
    title: str = 'Model Comparison',
    save_path: Optional[str] = None
):
    """
    Plot comparison of multiple models.

    Parameters
    ----------
    model_metrics : dict
        Metrics for each model
    metrics_to_plot : list
        List of metrics to compare
    title : str
        Plot title
    save_path : Optional[str]
        Path to save figure
    """
    model_names = list(model_metrics.keys())
    num_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))
    if num_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics_to_plot):
        values = [model_metrics[model].get(metric, 0) for model in model_names]
        axes[i].bar(range(len(model_names)), values, color='steelblue')
        axes[i].set_xticks(range(len(model_names)))
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        axes[i].set_ylabel(metric.upper())
        axes[i].set_title(f'{metric.upper()}')
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].set_ylim([0, 1])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison to {save_path}")

    plt.show()

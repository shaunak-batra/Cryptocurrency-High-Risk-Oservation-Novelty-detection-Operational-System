"""
Evaluation metrics for CHRONOS.

Provides comprehensive metrics for model evaluation including:
- Classification metrics (F1, Precision, Recall, AUC-ROC)
- Confusion matrix
- Performance by timestep
- Inference latency
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import time
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : Optional[np.ndarray]
        Predicted probabilities for positive class

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary of metrics

    Examples
    --------
    >>> metrics = compute_metrics(y_true, y_pred, y_proba)
    >>> print(f"F1: {metrics['f1']:.4f}")
    """
    metrics = {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': (y_true == y_pred).mean(),
    }

    # Add AUC metrics if probabilities provided
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            metrics['auc_pr'] = average_precision_score(y_true, y_proba)
        except ValueError:
            # Handle case where only one class present
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Compute confusion matrix and derived counts.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels

    Returns
    -------
    cm : np.ndarray [2, 2]
        Confusion matrix
    counts : Dict[str, int]
        True positives, false positives, true negatives, false negatives
    """
    cm = confusion_matrix(y_true, y_pred)

    # Handle case where cm might not be 2x2
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fill in missing values
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
        fn = cm[1, 0] if cm.shape[0] > 1 and cm.shape[1] > 0 else 0
        tp = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0

    counts = {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }

    return cm, counts


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list = ['Licit', 'Illicit']
):
    """
    Print detailed classification report.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    target_names : list
        Class names for display
    """
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=target_names))


def evaluate_by_timestep(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    timesteps: np.ndarray
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate performance for each timestep.

    Useful for detecting temporal distribution shift.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray
        Predicted probabilities
    timesteps : np.ndarray
        Timestep for each sample

    Returns
    -------
    metrics_by_timestep : Dict[int, Dict[str, float]]
        Metrics for each timestep
    """
    metrics_by_timestep = {}

    for t in np.unique(timesteps):
        mask = timesteps == t
        if mask.sum() > 0:
            metrics_by_timestep[int(t)] = compute_metrics(
                y_true[mask],
                y_pred[mask],
                y_proba[mask]
            )

    return metrics_by_timestep


def measure_inference_latency(
    model,
    data,
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Measure model inference latency.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    data : Data
        PyG Data object
    num_runs : int
        Number of inference runs
    warmup_runs : int
        Number of warmup runs (not counted)

    Returns
    -------
    latency_stats : Dict[str, float]
        Latency statistics (mean, std, p50, p95, p99) in milliseconds
    """
    model.eval()
    latencies = []

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(data.x, data.edge_index)

    # Measure
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(data.x, data.edge_index)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    stats = {
        'mean_ms': latencies.mean(),
        'std_ms': latencies.std(),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99),
        'min_ms': latencies.min(),
        'max_ms': latencies.max()
    }

    return stats


def check_performance_targets(metrics: Dict[str, float]) -> Dict[str, bool]:
    """
    Check if model meets CHRONOS performance targets.

    Targets:
    - F1 ≥ 0.88
    - Precision ≥ 0.85
    - Recall ≥ 0.85
    - AUC-ROC ≥ 0.92

    Parameters
    ----------
    metrics : Dict[str, float]
        Model metrics

    Returns
    -------
    targets_met : Dict[str, bool]
        Whether each target is met
    """
    targets = {
        'f1': 0.88,
        'precision': 0.85,
        'recall': 0.85,
        'auc_roc': 0.92
    }

    targets_met = {}
    for metric, target in targets.items():
        if metric in metrics:
            targets_met[metric] = metrics[metric] >= target
        else:
            targets_met[metric] = False

    return targets_met


def print_performance_summary(
    metrics: Dict[str, float],
    latency_stats: Optional[Dict[str, float]] = None,
    check_targets: bool = True
):
    """
    Print comprehensive performance summary.

    Parameters
    ----------
    metrics : Dict[str, float]
        Model metrics
    latency_stats : Optional[Dict[str, float]]
        Latency statistics
    check_targets : bool
        Whether to check against CHRONOS targets
    """
    print("\n" + "=" * 60)
    print("CHRONOS Performance Summary")
    print("=" * 60)

    # Classification metrics
    print("\nClassification Metrics:")
    print(f"  F1 Score:       {metrics.get('f1', 0):.4f}")
    print(f"  Precision:      {metrics.get('precision', 0):.4f}")
    print(f"  Recall:         {metrics.get('recall', 0):.4f}")
    print(f"  Accuracy:       {metrics.get('accuracy', 0):.4f}")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:        {metrics['auc_roc']:.4f}")
    if 'auc_pr' in metrics:
        print(f"  AUC-PR:         {metrics['auc_pr']:.4f}")

    # Latency metrics
    if latency_stats is not None:
        print("\nInference Latency:")
        print(f"  Mean:           {latency_stats['mean_ms']:.2f} ms")
        print(f"  Std:            {latency_stats['std_ms']:.2f} ms")
        print(f"  P50:            {latency_stats['p50_ms']:.2f} ms")
        print(f"  P95:            {latency_stats['p95_ms']:.2f} ms")
        print(f"  P99:            {latency_stats['p99_ms']:.2f} ms")

    # Check targets
    if check_targets:
        targets_met = check_performance_targets(metrics)
        print("\nPerformance Targets:")
        print(f"  F1 >= 0.88:      {'[Y]' if targets_met['f1'] else '[N]'}")
        print(f"  Precision >= 0.85: {'[Y]' if targets_met['precision'] else '[N]'}")
        print(f"  Recall >= 0.85:   {'[Y]' if targets_met['recall'] else '[N]'}")
        print(f"  AUC-ROC >= 0.92:  {'[Y]' if targets_met['auc_roc'] else '[N]'}")

        if latency_stats is not None:
            latency_target_met = latency_stats['p95_ms'] < 50
            print(f"  P95 < 50ms:     {'[Y]' if latency_target_met else '[N]'}")

        all_met = all(targets_met.values())
        if latency_stats is not None:
            all_met = all_met and latency_target_met

        print("\n" + "=" * 60)
        if all_met:
            print("[OK] All performance targets met!")
        else:
            print("✗ Some performance targets not met")
        print("=" * 60 + "\n")


def compare_models(
    model_metrics: Dict[str, Dict[str, float]],
    model_names: Optional[list] = None
):
    """
    Compare performance of multiple models.

    Parameters
    ----------
    model_metrics : Dict[str, Dict[str, float]]
        Metrics for each model
    model_names : Optional[list]
        Custom model names (default: use keys)
    """
    if model_names is None:
        model_names = list(model_metrics.keys())

    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(f"{'Model':<20} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC-ROC':>10}")
    print("-" * 80)

    for name in model_names:
        metrics = model_metrics[name]
        print(f"{name:<20} "
              f"{metrics.get('f1', 0):>8.4f} "
              f"{metrics.get('precision', 0):>10.4f} "
              f"{metrics.get('recall', 0):>8.4f} "
              f"{metrics.get('auc_roc', 0):>10.4f}")

    print("=" * 80 + "\n")

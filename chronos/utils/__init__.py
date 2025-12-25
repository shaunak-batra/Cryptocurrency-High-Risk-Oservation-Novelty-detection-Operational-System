"""Utility functions for metrics and visualization."""

from .metrics import (
    compute_metrics,
    compute_confusion_matrix,
    print_classification_report,
    evaluate_by_timestep,
    measure_inference_latency,
    check_performance_targets,
    print_performance_summary,
    compare_models
)

from .visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_training_curves,
    plot_attention_weights,
    plot_embeddings_tsne,
    plot_performance_by_timestep,
    plot_feature_importance,
    plot_model_comparison
)

__all__ = [
    # Metrics
    'compute_metrics',
    'compute_confusion_matrix',
    'print_classification_report',
    'evaluate_by_timestep',
    'measure_inference_latency',
    'check_performance_targets',
    'print_performance_summary',
    'compare_models',
    # Visualization
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_pr_curve',
    'plot_training_curves',
    'plot_attention_weights',
    'plot_embeddings_tsne',
    'plot_performance_by_timestep',
    'plot_feature_importance',
    'plot_model_comparison'
]

"""
Evaluation module for CHRONOS.

Provides comprehensive evaluation utilities including:
- Model evaluation
- Statistical significance testing
- Ablation study framework
"""

from .evaluator import ModelEvaluator, evaluate_model
from .statistical_tests import (
    bootstrap_confidence_interval,
    mcnemar_test,
    friedman_test,
    compute_effect_size
)

__all__ = [
    'ModelEvaluator',
    'evaluate_model',
    'bootstrap_confidence_interval',
    'mcnemar_test',
    'friedman_test',
    'compute_effect_size'
]

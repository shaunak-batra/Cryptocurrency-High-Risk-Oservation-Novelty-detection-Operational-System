"""
Tests for metrics computation.

Tests classification metrics, confusion matrix, and target checking.
"""

import pytest
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class TestMetricsComputation:
    """Tests for compute_metrics function."""
    
    def test_metrics_import(self):
        """Test metrics can be imported."""
        from chronos.utils.metrics import compute_metrics
        assert compute_metrics is not None
    
    def test_compute_metrics_basic(self, mock_predictions):
        """Test basic metrics computation."""
        from chronos.utils.metrics import compute_metrics
        
        y_true, y_pred, y_proba = mock_predictions
        
        metrics = compute_metrics(y_true, y_pred, y_proba)
        
        assert 'f1' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'accuracy' in metrics
    
    def test_metrics_values_in_range(self, mock_predictions):
        """Test that all metrics are in [0, 1]."""
        from chronos.utils.metrics import compute_metrics
        
        y_true, y_pred, y_proba = mock_predictions
        
        metrics = compute_metrics(y_true, y_pred, y_proba)
        
        for name, value in metrics.items():
            if name != 'auc_roc':  # AUC might be NaN for edge cases
                if not np.isnan(value):
                    assert 0 <= value <= 1, f"{name} should be in [0, 1]"
    
    def test_perfect_predictions(self):
        """Test metrics for perfect predictions."""
        from chronos.utils.metrics import compute_metrics
        
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])  # Perfect
        y_proba = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        
        metrics = compute_metrics(y_true, y_pred, y_proba)
        
        assert metrics['f1'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['accuracy'] == 1.0
    
    def test_all_wrong_predictions(self):
        """Test metrics for completely wrong predictions."""
        from chronos.utils.metrics import compute_metrics
        
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])  # All wrong
        y_proba = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        
        metrics = compute_metrics(y_true, y_pred, y_proba)
        
        assert metrics['f1'] == 0.0
        assert metrics['accuracy'] == 0.0


class TestConfusionMatrix:
    """Tests for confusion matrix computation."""
    
    def test_confusion_matrix_import(self):
        """Test confusion matrix can be imported."""
        from chronos.utils.metrics import compute_confusion_matrix
        assert compute_confusion_matrix is not None
    
    def test_confusion_matrix_shape(self, mock_predictions):
        """Test confusion matrix has correct shape."""
        from chronos.utils.metrics import compute_confusion_matrix
        
        y_true, y_pred, _ = mock_predictions
        
        cm, counts = compute_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2), "Confusion matrix should be 2x2"
    
    def test_confusion_matrix_counts(self, mock_predictions):
        """Test confusion matrix counts are correct."""
        from chronos.utils.metrics import compute_confusion_matrix
        
        y_true, y_pred, _ = mock_predictions
        
        cm, counts = compute_confusion_matrix(y_true, y_pred)
        
        assert 'true_positives' in counts
        assert 'true_negatives' in counts
        assert 'false_positives' in counts
        assert 'false_negatives' in counts
        
        # Counts should sum to total samples
        total = counts['true_positives'] + counts['true_negatives'] + counts['false_positives'] + counts['false_negatives']
        assert total == len(y_true)


class TestPerformanceTargets:
    """Tests for performance target checking."""
    
    def test_target_checking_import(self):
        """Test target checking can be imported."""
        from chronos.utils.metrics import check_performance_targets
        assert check_performance_targets is not None
    
    def test_targets_met(self):
        """Test that good metrics meet targets."""
        from chronos.utils.metrics import check_performance_targets
        
        good_metrics = {
            'f1': 0.90,
            'precision': 0.88,
            'recall': 0.92,
            'auc_roc': 0.95
        }
        
        targets_met = check_performance_targets(good_metrics)
        
        assert targets_met['f1'], "F1 should meet target"
        assert targets_met['precision'], "Precision should meet target"
        assert targets_met['recall'], "Recall should meet target"
        assert targets_met['auc_roc'], "AUC-ROC should meet target"
    
    def test_targets_not_met(self):
        """Test that bad metrics don't meet targets."""
        from chronos.utils.metrics import check_performance_targets
        
        bad_metrics = {
            'f1': 0.50,
            'precision': 0.60,
            'recall': 0.55,
            'auc_roc': 0.70
        }
        
        targets_met = check_performance_targets(bad_metrics)
        
        assert not targets_met['f1'], "F1 should NOT meet target"
        assert not targets_met['precision'], "Precision should NOT meet target"
        assert not targets_met['recall'], "Recall should NOT meet target"
        assert not targets_met['auc_roc'], "AUC-ROC should NOT meet target"


class TestTimestepEvaluation:
    """Tests for per-timestep evaluation."""
    
    def test_timestep_eval_import(self):
        """Test timestep evaluation can be imported."""
        from chronos.utils.metrics import evaluate_by_timestep
        assert evaluate_by_timestep is not None
    
    def test_timestep_eval_basic(self):
        """Test per-timestep evaluation."""
        from chronos.utils.metrics import evaluate_by_timestep
        
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.4, 0.1, 0.6, 0.9, 0.8])
        timesteps = np.array([43, 43, 43, 43, 44, 44, 44, 44])
        
        metrics_by_ts = evaluate_by_timestep(y_true, y_pred, y_proba, timesteps)
        
        assert 43 in metrics_by_ts
        assert 44 in metrics_by_ts
        assert 'f1' in metrics_by_ts[43]
        assert 'f1' in metrics_by_ts[44]

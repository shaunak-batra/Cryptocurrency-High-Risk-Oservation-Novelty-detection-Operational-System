"""
Model evaluator for CHRONOS.

Provides a unified interface for evaluating models with comprehensive metrics,
per-timestep analysis, and latency benchmarking.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from chronos.utils.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    evaluate_by_timestep,
    measure_inference_latency,
    check_performance_targets
)


class ModelEvaluator:
    """
    Evaluator for CHRONOS models.
    
    Provides comprehensive evaluation including:
    - Standard classification metrics
    - Confusion matrix analysis
    - Per-timestep performance
    - Inference latency benchmarking
    - Target compliance checking
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate
    device : str
        Device to run evaluation on
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate(
        self,
        data: Data,
        mask: torch.Tensor,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on specified data split.
        
        Parameters
        ----------
        data : Data
            PyG Data object
        mask : Tensor
            Mask for which nodes to evaluate
        return_predictions : bool
            Whether to include predictions in output
            
        Returns
        -------
        results : Dict
            Evaluation results including metrics, confusion matrix, etc.
        """
        data = data.to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            logits = self.model(data.x, data.edge_index)
            if isinstance(logits, tuple):
                logits = logits[0]
            inference_time = time.time() - start_time
        
        # Get predictions
        y_true = data.y[mask].cpu().numpy()
        y_pred = logits[mask].argmax(dim=1).cpu().numpy()
        y_proba = F.softmax(logits[mask], dim=1).cpu().numpy()
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_proba[:, 1])
        
        # Confusion matrix
        cm, counts = compute_confusion_matrix(y_true, y_pred)
        
        # Check targets
        targets_met = check_performance_targets(metrics)
        
        results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'counts': counts,
            'targets_met': targets_met,
            'all_targets_met': all(targets_met.values()),
            'inference_time_ms': inference_time * 1000,
            'num_samples': len(y_true)
        }
        
        if return_predictions:
            results['y_true'] = y_true
            results['y_pred'] = y_pred
            results['y_proba'] = y_proba
        
        return results
    
    def evaluate_by_timestep(
        self,
        data: Data,
        mask: torch.Tensor
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate model performance for each timestep.
        
        Useful for detecting temporal distribution shift.
        
        Parameters
        ----------
        data : Data
            PyG Data object with timestep attribute
        mask : Tensor
            Mask for which nodes to evaluate
            
        Returns
        -------
        metrics_by_timestep : Dict[int, Dict[str, float]]
            Metrics for each timestep
        """
        data = data.to(self.device)
        
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)
            if isinstance(logits, tuple):
                logits = logits[0]
        
        y_true = data.y[mask].cpu().numpy()
        y_pred = logits[mask].argmax(dim=1).cpu().numpy()
        y_proba = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
        timesteps = data.timestep[mask].cpu().numpy()
        
        return evaluate_by_timestep(y_true, y_pred, y_proba, timesteps)
    
    def benchmark_latency(
        self,
        data: Data,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark inference latency.
        
        Parameters
        ----------
        data : Data
            PyG Data object
        num_runs : int
            Number of inference runs
        warmup_runs : int
            Number of warmup runs (not counted)
            
        Returns
        -------
        latency_stats : Dict[str, float]
            Latency statistics (mean, std, p50, p95, p99) in ms
        """
        return measure_inference_latency(
            self.model,
            data,
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
    
    def full_evaluation(
        self,
        data: Data,
        include_timestep_analysis: bool = True,
        include_latency_benchmark: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on test set.
        
        Parameters
        ----------
        data : Data
            PyG Data object with test_mask
        include_timestep_analysis : bool
            Whether to include per-timestep metrics
        include_latency_benchmark : bool
            Whether to include latency benchmarking
            
        Returns
        -------
        results : Dict
            Complete evaluation results
        """
        results = {}
        
        # Main evaluation
        main_results = self.evaluate(data, data.test_mask, return_predictions=True)
        results.update(main_results)
        
        # Per-timestep analysis
        if include_timestep_analysis:
            results['by_timestep'] = self.evaluate_by_timestep(data, data.test_mask)
        
        # Latency benchmark
        if include_latency_benchmark:
            results['latency'] = self.benchmark_latency(data)
        
        return results


def evaluate_model(
    model: torch.nn.Module,
    data: Data,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate
    data : Data
        PyG Data object with test_mask
    device : str
        Device to run evaluation on
        
    Returns
    -------
    results : Dict
        Evaluation results
        
    Examples
    --------
    >>> results = evaluate_model(model, data, device='cuda')
    >>> print(f"F1: {results['metrics']['f1']:.4f}")
    """
    evaluator = ModelEvaluator(model, device)
    return evaluator.full_evaluation(data)

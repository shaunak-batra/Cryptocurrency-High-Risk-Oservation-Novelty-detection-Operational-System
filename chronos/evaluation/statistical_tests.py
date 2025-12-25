"""
Statistical significance tests for CHRONOS benchmarking.

Implements:
- Bootstrap confidence intervals
- McNemar's test (paired comparison)
- Friedman test (multiple models comparison)
- Effect size calculations
"""

from typing import Dict, List, Tuple, Callable
import numpy as np
from scipy import stats


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    metric_fn: Callable = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Parameters
    ----------
    y_true : ndarray
        Ground truth labels
    y_pred : ndarray
        Predicted labels
    y_proba : ndarray, optional
        Predicted probabilities (for AUC-based metrics)
    metric_fn : Callable
        Function that computes the metric: metric_fn(y_true, y_pred, y_proba)
        If None, uses F1 score
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int
        Random seed
        
    Returns
    -------
    result : Dict[str, float]
        Dictionary with 'mean', 'lower', 'upper', 'std'
        
    Examples
    --------
    >>> from sklearn.metrics import f1_score
    >>> ci = bootstrap_confidence_interval(y_true, y_pred, metric_fn=lambda t, p, _: f1_score(t, p))
    >>> print(f"F1: {ci['mean']:.3f} ({ci['lower']:.3f}, {ci['upper']:.3f})")
    """
    np.random.seed(random_state)
    
    if metric_fn is None:
        from sklearn.metrics import f1_score
        metric_fn = lambda t, p, _: f1_score(t, p)
    
    n_samples = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        try:
            if y_proba is not None:
                score = metric_fn(y_true[idx], y_pred[idx], y_proba[idx])
            else:
                score = metric_fn(y_true[idx], y_pred[idx], None)
            scores.append(score)
        except Exception:
            continue
    
    scores = np.array(scores)
    alpha = 1 - confidence_level
    
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'lower': np.percentile(scores, alpha / 2 * 100),
        'upper': np.percentile(scores, (1 - alpha / 2) * 100)
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    correction: bool = True
) -> Dict[str, float]:
    """
    McNemar's test for comparing two classifiers.
    
    Tests if two classifiers have statistically different error rates.
    
    Parameters
    ----------
    y_true : ndarray
        Ground truth labels
    y_pred_1 : ndarray
        Predictions from classifier 1
    y_pred_2 : ndarray
        Predictions from classifier 2
    correction : bool
        Whether to apply continuity correction
        
    Returns
    -------
    result : Dict[str, float]
        Dictionary with 'chi2', 'p_value', 'significant' (p < 0.05)
        
    Examples
    --------
    >>> result = mcnemar_test(y_true, chronos_pred, baseline_pred)
    >>> if result['significant']:
    ...     print("CHRONOS significantly outperforms baseline")
    """
    # Build contingency table
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)
    
    # n01: classifier 1 correct, classifier 2 wrong
    n01 = np.sum(correct_1 & ~correct_2)
    # n10: classifier 1 wrong, classifier 2 correct
    n10 = np.sum(~correct_1 & correct_2)
    
    # McNemar's test statistic
    if n01 + n10 == 0:
        return {
            'chi2': 0.0,
            'p_value': 1.0,
            'significant': False,
            'n01': int(n01),
            'n10': int(n10)
        }
    
    if correction:
        # With continuity correction
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    else:
        # Without correction
        chi2 = (n01 - n10) ** 2 / (n01 + n10)
    
    # P-value from chi-squared distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return {
        'chi2': float(chi2),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'n01': int(n01),
        'n10': int(n10)
    }


def friedman_test(
    scores: Dict[str, List[float]]
) -> Dict[str, float]:
    """
    Friedman test for comparing multiple classifiers across multiple datasets/folds.
    
    Non-parametric alternative to repeated-measures ANOVA.
    
    Parameters
    ----------
    scores : Dict[str, List[float]]
        Dictionary mapping model names to list of scores (one per fold/timestep)
        All lists must have the same length
        
    Returns
    -------
    result : Dict[str, float]
        Dictionary with 'chi2', 'p_value', 'significant', 'ranks'
        
    Examples
    --------
    >>> scores = {
    ...     'Random Forest': [0.71, 0.72, 0.70, 0.73],
    ...     'XGBoost': [0.74, 0.73, 0.72, 0.75],
    ...     'CHRONOS': [0.88, 0.89, 0.87, 0.90]
    ... }
    >>> result = friedman_test(scores)
    >>> print(f"Friedman χ² = {result['chi2']:.2f}, p = {result['p_value']:.4f}")
    """
    model_names = list(scores.keys())
    n_models = len(model_names)
    n_folds = len(scores[model_names[0]])
    
    # Check all lists have same length
    for name, score_list in scores.items():
        if len(score_list) != n_folds:
            raise ValueError(f"All score lists must have same length. {name} has {len(score_list)}")
    
    # Create score matrix [n_folds, n_models]
    score_matrix = np.array([scores[name] for name in model_names]).T
    
    # Compute ranks for each fold (row)
    ranks = np.zeros_like(score_matrix)
    for i in range(n_folds):
        # Higher score = better = lower rank
        order = np.argsort(-score_matrix[i])
        ranks[i, order] = np.arange(1, n_models + 1)
    
    # Average ranks for each model
    avg_ranks = ranks.mean(axis=0)
    
    # Friedman statistic
    chi2 = (12 * n_folds / (n_models * (n_models + 1))) * \
           (np.sum(avg_ranks ** 2) - (n_models * (n_models + 1) ** 2) / 4)
    
    # P-value
    p_value = 1 - stats.chi2.cdf(chi2, df=n_models - 1)
    
    return {
        'chi2': float(chi2),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'ranks': {name: float(rank) for name, rank in zip(model_names, avg_ranks)}
    }


def compute_effect_size(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray
) -> Dict[str, float]:
    """
    Compute effect size (Cohen's d) for difference between two classifiers.
    
    Parameters
    ----------
    y_true : ndarray
        Ground truth labels
    y_pred_1 : ndarray
        Predictions from classifier 1
    y_pred_2 : ndarray
        Predictions from classifier 2
        
    Returns
    -------
    result : Dict[str, float]
        Dictionary with 'cohens_d', 'interpretation'
        
    Notes
    -----
    Cohen's d interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    # Accuracy for each sample
    correct_1 = (y_pred_1 == y_true).astype(float)
    correct_2 = (y_pred_2 == y_true).astype(float)
    
    # Paired difference
    diff = correct_1 - correct_2
    
    # Cohen's d = mean difference / pooled std
    mean_diff = np.mean(diff)
    pooled_std = np.sqrt((np.var(correct_1) + np.var(correct_2)) / 2)
    
    if pooled_std == 0:
        cohens_d = 0.0
    else:
        cohens_d = mean_diff / pooled_std
    
    # Interpretation
    d_abs = abs(cohens_d)
    if d_abs < 0.2:
        interpretation = 'negligible'
    elif d_abs < 0.5:
        interpretation = 'small'
    elif d_abs < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    
    return {
        'cohens_d': float(cohens_d),
        'interpretation': interpretation,
        'mean_difference': float(mean_diff)
    }


def paired_t_test(
    scores_1: List[float],
    scores_2: List[float]
) -> Dict[str, float]:
    """
    Paired t-test for comparing two classifiers across folds.
    
    Parameters
    ----------
    scores_1 : List[float]
        Scores from classifier 1 (one per fold)
    scores_2 : List[float]
        Scores from classifier 2 (one per fold)
        
    Returns
    -------
    result : Dict[str, float]
        Dictionary with 't_statistic', 'p_value', 'significant'
    """
    scores_1 = np.array(scores_1)
    scores_2 = np.array(scores_2)
    
    t_stat, p_value = stats.ttest_rel(scores_1, scores_2)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'mean_diff': float(np.mean(scores_1 - scores_2))
    }

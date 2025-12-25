#!/usr/bin/env python
"""
Benchmark script for CHRONOS.

Usage:
    python scripts/run_benchmarks.py --data-dir data/raw/elliptic
    python scripts/run_benchmarks.py --help

This script:
1. Trains and evaluates all baseline models (RF, XGBoost, GCN)
2. Trains and evaluates CHRONOS-Net
3. Generates comparison table
4. Runs statistical significance tests
5. Outputs comprehensive benchmark report
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos.data.loader import load_elliptic_dataset, verify_dataset
from chronos.data.features import add_engineered_features
from chronos.models.baselines import (
    RandomForestBaseline,
    XGBoostBaseline,
    VanillaGCN
)
from chronos.models.chronos_net import create_chronos_net
from chronos.training.trainer import CHRONOSTrainer
from chronos.training.config import CHRONOSConfig
from chronos.utils.metrics import compute_metrics, compare_models


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    McNemar's test for comparing two classifiers.
    
    Returns chi-squared statistic and p-value.
    """
    # Build contingency table
    # n01: model 1 correct, model 2 wrong
    # n10: model 1 wrong, model 2 correct
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    n01 = np.sum(correct1 & ~correct2)
    n10 = np.sum(~correct1 & correct2)
    
    # McNemar's test
    if n01 + n10 == 0:
        return 0, 1.0
    
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return chi2, p_value


def bootstrap_ci(y_true, y_pred, y_proba, metric_fn, n_bootstrap=1000, ci=0.95):
    """
    Compute bootstrap confidence interval for a metric.
    """
    n_samples = len(y_true)
    scores = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx], y_proba[idx])
        scores.append(score)
    
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    mean = np.mean(scores)
    
    return mean, lower, upper


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run CHRONOS benchmarks against baselines'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/elliptic',
        help='Path to Elliptic dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/benchmarks',
        help='Directory to save benchmark results'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training, use existing checkpoints'
    )
    parser.add_argument(
        '--chronos-epochs',
        type=int,
        default=50,
        help='Number of epochs for CHRONOS training (default: 50)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device: cuda, cpu, or auto'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def main():
    """Main benchmark function."""
    args = parse_args()
    set_seed(args.seed)
    
    print(f"\n{'='*70}")
    print("CHRONOS Benchmark Suite")
    print(f"{'='*70}")
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ================================================================
    # 1. LOAD DATA
    # ================================================================
    print(f"\n{'='*70}")
    print("Loading Dataset")
    print(f"{'='*70}")
    
    data = load_elliptic_dataset(args.data_dir, include_unknown=True)
    verify_dataset(data)
    
    # Compute engineered features
    print("Computing engineered features...")
    data = add_engineered_features(data)
    data.x_full = torch.cat([data.x, data.x_engineered], dim=1)
    
    print(f"Original features: {data.x.size(1)}")
    print(f"Engineered features: {data.x_engineered.size(1)}")
    print(f"Total features: {data.x_full.size(1)}")
    
    # Store results
    all_results = {}
    all_predictions = {}
    
    # Get ground truth
    y_true_test = data.y[data.test_mask].numpy()
    
    # ================================================================
    # 2. RANDOM FOREST BASELINE
    # ================================================================
    print(f"\n{'='*70}")
    print("Baseline 1: Random Forest (BASELINE)")
    print(f"{'='*70}")
    
    rf = RandomForestBaseline(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        class_weight='balanced',
        random_state=args.seed
    )
    
    # Train on original features only (to match literature)
    rf.fit(data)
    rf_metrics = rf.evaluate(data, data.test_mask)
    rf_pred = rf.predict(data, data.test_mask)
    rf_proba = rf.predict_proba(data, data.test_mask)[:, 1]
    
    all_results['Random Forest'] = rf_metrics
    all_predictions['Random Forest'] = rf_pred
    
    print(f"F1: {rf_metrics['f1']:.4f} (Expected: 0.70-0.73)")
    
    # ================================================================
    # 3. XGBOOST BASELINE
    # ================================================================
    print(f"\n{'='*70}")
    print("Baseline 2: XGBoost (BASELINE)")
    print(f"{'='*70}")
    
    # Calculate class imbalance
    y_train = data.y[data.train_mask]
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    xgb = XGBoostBaseline(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed
    )
    
    xgb.fit(data)
    xgb_metrics = xgb.evaluate(data, data.test_mask)
    xgb_pred = xgb.predict(data, data.test_mask)
    xgb_proba = xgb.predict_proba(data, data.test_mask)[:, 1]
    
    all_results['XGBoost'] = xgb_metrics
    all_predictions['XGBoost'] = xgb_pred
    
    print(f"F1: {xgb_metrics['f1']:.4f} (Expected: 0.72-0.75)")
    
    # ================================================================
    # 4. VANILLA GCN BASELINE
    # ================================================================
    print(f"\n{'='*70}")
    print("Baseline 3: Vanilla GCN (BASELINE)")
    print(f"{'='*70}")
    
    gcn = VanillaGCN(
        in_features=data.x.size(1),  # Original features only
        hidden_dim=128,
        dropout=0.5
    )
    gcn = gcn.to(device)
    
    # Simple training loop for GCN
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
    
    data_device = data.to(device)
    
    print("Training GCN...")
    for epoch in range(100):
        gcn.train()
        optimizer.zero_grad()
        logits = gcn(data_device.x, data_device.edge_index)
        loss = torch.nn.functional.cross_entropy(
            logits[data_device.train_mask],
            data_device.y[data_device.train_mask]
        )
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    # Evaluate GCN
    gcn.eval()
    with torch.no_grad():
        gcn_logits = gcn(data_device.x, data_device.edge_index)
        gcn_pred = gcn_logits[data_device.test_mask].argmax(dim=1).cpu().numpy()
        gcn_proba = torch.softmax(gcn_logits[data_device.test_mask], dim=1)[:, 1].cpu().numpy()
    
    gcn_metrics = compute_metrics(y_true_test, gcn_pred, gcn_proba)
    all_results['Vanilla GCN'] = gcn_metrics
    all_predictions['Vanilla GCN'] = gcn_pred
    
    print(f"F1: {gcn_metrics['f1']:.4f} (Expected: 0.60-0.65)")
    
    # ================================================================
    # 5. CHRONOS-Net (NOVEL)
    # ================================================================
    print(f"\n{'='*70}")
    print("CHRONOS-Net (NOVEL)")
    print(f"{'='*70}")
    
    if not args.skip_training:
        # Create model
        model = create_chronos_net(
            in_features=data.x_full.size(1),
            hidden_dim=256,
            config={
                'num_gat_layers': 3,
                'num_heads': 8,
                'dropout': 0.3
            }
        )
        
        # Create config
        config = CHRONOSConfig(
            num_epochs=args.chronos_epochs,
            learning_rate=0.001,
            weight_decay=1e-4,
            early_stopping_patience=10,
            checkpoint_dir='checkpoints',
            log_dir='logs'
        )
        
        # Train
        trainer = CHRONOSTrainer(model, config, device=device)
        
        # Use full features
        data.x_original = data.x
        data.x = data.x_full
        data = data.to(device)
        
        trainer.fit(data)
        
        # Test
        chronos_metrics = trainer.test(data)
    else:
        # Load from checkpoint
        checkpoint_path = 'checkpoints/chronos_best.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = create_chronos_net(in_features=data.x_full.size(1))
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            data.x = data.x_full
            data = data.to(device)
            
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                if isinstance(logits, tuple):
                    logits = logits[0]
            
            chronos_pred = logits[data.test_mask].argmax(dim=1).cpu().numpy()
            chronos_proba = torch.softmax(logits[data.test_mask], dim=1)[:, 1].cpu().numpy()
            chronos_metrics = compute_metrics(y_true_test, chronos_pred, chronos_proba)
        else:
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return
    
    # Get CHRONOS predictions
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        if isinstance(logits, tuple):
            logits = logits[0]
    chronos_pred = logits[data.test_mask].argmax(dim=1).cpu().numpy()
    chronos_proba = torch.softmax(logits[data.test_mask], dim=1)[:, 1].cpu().numpy()
    
    all_results['CHRONOS-Net'] = chronos_metrics
    all_predictions['CHRONOS-Net'] = chronos_pred
    
    print(f"F1: {chronos_metrics['f1']:.4f} (Target: ≥0.88)")
    
    # ================================================================
    # 6. COMPARISON TABLE
    # ================================================================
    print(f"\n{'='*70}")
    print("Model Comparison")
    print(f"{'='*70}")
    compare_models(all_results)
    
    # ================================================================
    # 7. STATISTICAL TESTS
    # ================================================================
    print(f"\n{'='*70}")
    print("Statistical Significance Tests (McNemar's)")
    print(f"{'='*70}")
    
    stat_tests = {}
    for model_name in ['Random Forest', 'XGBoost', 'Vanilla GCN']:
        chi2, p_val = mcnemar_test(
            y_true_test,
            all_predictions['CHRONOS-Net'],
            all_predictions[model_name]
        )
        stat_tests[f'CHRONOS vs {model_name}'] = {
            'chi2': chi2,
            'p_value': p_val,
            'significant': p_val < 0.05
        }
        sig = "✅" if p_val < 0.05 else "❌"
        print(f"CHRONOS vs {model_name}: χ²={chi2:.2f}, p={p_val:.4f} {sig}")
    
    # ================================================================
    # 8. SAVE RESULTS
    # ================================================================
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}")
    
    results = {
        'model_metrics': {k: v for k, v in all_results.items()},
        'statistical_tests': stat_tests,
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed
    }
    
    output_file = os.path.join(args.output_dir, 'benchmark_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {output_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"CHRONOS-Net achieves F1={chronos_metrics['f1']:.4f}")
    print(f"  +{(chronos_metrics['f1'] - rf_metrics['f1'])*100:.1f}% vs Random Forest")
    print(f"  +{(chronos_metrics['f1'] - xgb_metrics['f1'])*100:.1f}% vs XGBoost")
    print(f"  +{(chronos_metrics['f1'] - gcn_metrics['f1'])*100:.1f}% vs Vanilla GCN")
    
    return results


if __name__ == '__main__':
    results = main()

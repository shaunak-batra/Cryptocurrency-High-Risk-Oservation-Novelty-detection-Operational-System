"""
Train baseline models for CHRONOS.

Trains Random Forest, XGBoost, and Vanilla GCN baselines.
Used to validate data pipeline and establish performance floor.
"""

import os
import sys
import argparse
import torch
import yaml
import joblib
from pathlib import Path
from datetime import datetime
import numpy as np

# Add chronos to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos.data.loader import load_elliptic_dataset, verify_dataset
from chronos.data.preprocessing import normalize_features
from chronos.models.baselines import (
    RandomForestBaseline,
    XGBoostBaseline,
    VanillaGCN,
    train_sklearn_baseline,
    train_gcn_baseline
)
from chronos.utils.metrics import compute_metrics
from chronos.utils.visualization import plot_confusion_matrix, plot_roc_curve


def train_random_forest(data, device='cpu'):
    """Train Random Forest baseline."""
    print("\n" + "=" * 70)
    print("RANDOM FOREST BASELINE")
    print("=" * 70)

    model = RandomForestBaseline(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    print("[INFO] Training Random Forest...")
    print(f"      n_estimators: 100")
    print(f"      max_depth: 20")
    print(f"      class_weight: balanced")

    # Train
    train_metrics, val_metrics, test_metrics = train_sklearn_baseline(
        model, data
    )

    # Print results
    print("\n[RESULTS] Random Forest")
    print(f"  Train F1: {train_metrics['f1']:.4f}")
    print(f"  Val F1:   {val_metrics['f1']:.4f}")
    print(f"  Test F1:  {test_metrics['f1']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall:    {test_metrics['recall']:.4f}")
    print(f"  Test AUC-ROC:   {test_metrics['auc']:.4f}")

    # Expected from Weber et al. 2019: F1 = 0.70-0.73
    if 0.68 <= test_metrics['f1'] <= 0.75:
        print(f"\n[OK] F1 score within expected range (0.70-0.73)")
    else:
        print(f"\n[WARNING] F1 score outside expected range (0.70-0.73)")
        print(f"          This may indicate issues with data pipeline")

    return model, test_metrics


def train_xgboost(data, device='cpu'):
    """Train XGBoost baseline."""
    print("\n" + "=" * 70)
    print("XGBOOST BASELINE")
    print("=" * 70)

    # Calculate scale_pos_weight (class imbalance ratio)
    train_labels = data.y[data.train_mask]
    num_licit = (train_labels == 0).sum().item()
    num_illicit = (train_labels == 1).sum().item()
    scale_pos_weight = num_licit / num_illicit

    model = XGBoostBaseline(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    print("[INFO] Training XGBoost...")
    print(f"      n_estimators: 200")
    print(f"      max_depth: 10")
    print(f"      scale_pos_weight: {scale_pos_weight:.2f}")

    # Train
    train_metrics, val_metrics, test_metrics = train_sklearn_baseline(
        model, data
    )

    # Print results
    print("\n[RESULTS] XGBoost")
    print(f"  Train F1: {train_metrics['f1']:.4f}")
    print(f"  Val F1:   {val_metrics['f1']:.4f}")
    print(f"  Test F1:  {test_metrics['f1']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall:    {test_metrics['recall']:.4f}")
    print(f"  Test AUC-ROC:   {test_metrics['auc']:.4f}")

    # Expected: F1 = 0.72-0.75
    if 0.70 <= test_metrics['f1'] <= 0.77:
        print(f"\n[OK] F1 score within expected range (0.72-0.75)")
    else:
        print(f"\n[WARNING] F1 score outside expected range (0.72-0.75)")

    return model, test_metrics


def train_vanilla_gcn(data, device='cpu'):
    """Train Vanilla GCN baseline."""
    print("\n" + "=" * 70)
    print("VANILLA GCN BASELINE")
    print("=" * 70)

    model = VanillaGCN(
        in_features=data.x.size(1),
        hidden_dim=128,
        dropout=0.5
    )

    print("[INFO] Training Vanilla GCN...")
    print(f"      in_features: {data.x.size(1)}")
    print(f"      hidden_dim: 128")
    print(f"      dropout: 0.5")

    # Train
    train_metrics, val_metrics, test_metrics = train_gcn_baseline(
        model, data, device=device
    )

    # Print results
    print("\n[RESULTS] Vanilla GCN")
    print(f"  Train F1: {train_metrics['f1']:.4f}")
    print(f"  Val F1:   {val_metrics['f1']:.4f}")
    print(f"  Test F1:  {test_metrics['f1']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall:    {test_metrics['recall']:.4f}")
    print(f"  Test AUC-ROC:   {test_metrics['auc']:.4f}")

    # Expected from Weber et al. 2019: F1 = 0.60-0.65
    if 0.58 <= test_metrics['f1'] <= 0.67:
        print(f"\n[OK] F1 score within expected range (0.60-0.65)")
    else:
        print(f"\n[WARNING] F1 score outside expected range (0.60-0.65)")

    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/raw/elliptic',
        help='Root directory for dataset'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['rf', 'xgb', 'gcn', 'all'],
        default=['all'],
        help='Which models to train (default: all)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for GCN training (cuda/cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 70)
    print("CHRONOS BASELINE MODEL TRAINING")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print()

    # Determine which models to train
    if 'all' in args.models:
        models_to_train = ['rf', 'xgb', 'gcn']
    else:
        models_to_train = args.models

    print(f"Models to train: {', '.join(models_to_train)}")
    print()

    # ========================================================================
    # Load Dataset
    # ========================================================================
    print("[INFO] Loading Elliptic dataset...")

    data = load_elliptic_dataset(
        root=args.data_root,
        timestep=None,
        include_unknown=True
    )

    print(f"      Nodes: {data.x.size(0):,}")
    print(f"      Edges: {data.edge_index.size(1):,}")
    print(f"      Features: {data.x.size(1)}")
    print(f"      Train: {data.train_mask.sum().item():,}")
    print(f"      Val: {data.val_mask.sum().item():,}")
    print(f"      Test: {data.test_mask.sum().item():,}")
    print()

    # Normalize features
    print("[INFO] Normalizing features...")
    data = normalize_features(data, method='standard', train_mask=data.train_mask)
    print()

    # Move to device (for GCN)
    data = data.to(args.device)

    # ========================================================================
    # Train Models
    # ========================================================================
    results = {}

    if 'rf' in models_to_train:
        model_rf, metrics_rf = train_random_forest(data, device=args.device)
        results['RandomForest'] = metrics_rf

        # Save model
        Path('models/baselines').mkdir(parents=True, exist_ok=True)
        joblib.dump(model_rf, 'models/baselines/random_forest.pkl')
        print(f"\n[INFO] Saved: models/baselines/random_forest.pkl")

        # Plot confusion matrix
        plot_confusion_matrix(
            metrics_rf['y_true'],
            metrics_rf['y_pred'],
            save_path='results/figures/confusion_matrix_rf.png'
        )
        print(f"[INFO] Saved: results/figures/confusion_matrix_rf.png")

    if 'xgb' in models_to_train:
        model_xgb, metrics_xgb = train_xgboost(data, device=args.device)
        results['XGBoost'] = metrics_xgb

        # Save model
        Path('models/baselines').mkdir(parents=True, exist_ok=True)
        joblib.dump(model_xgb, 'models/baselines/xgboost.pkl')
        print(f"\n[INFO] Saved: models/baselines/xgboost.pkl")

        # Plot confusion matrix
        plot_confusion_matrix(
            metrics_xgb['y_true'],
            metrics_xgb['y_pred'],
            save_path='results/figures/confusion_matrix_xgb.png'
        )
        print(f"[INFO] Saved: results/figures/confusion_matrix_xgb.png")

    if 'gcn' in models_to_train:
        model_gcn, metrics_gcn = train_vanilla_gcn(data, device=args.device)
        results['VanillaGCN'] = metrics_gcn

        # Save model
        Path('models/baselines').mkdir(parents=True, exist_ok=True)
        torch.save(model_gcn.state_dict(), 'models/baselines/vanilla_gcn.pt')
        print(f"\n[INFO] Saved: models/baselines/vanilla_gcn.pt")

        # Plot confusion matrix
        plot_confusion_matrix(
            metrics_gcn['y_true'],
            metrics_gcn['y_pred'],
            save_path='results/figures/confusion_matrix_gcn.png'
        )
        print(f"[INFO] Saved: results/figures/confusion_matrix_gcn.png")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'F1':>8} {'Precision':>12} {'Recall':>10} {'AUC-ROC':>10}")
    print("-" * 70)

    for model_name, metrics in results.items():
        print(f"{model_name:<20} {metrics['f1']:>8.4f} {metrics['precision']:>12.4f} "
              f"{metrics['recall']:>10.4f} {metrics['auc']:>10.4f}")

    print("=" * 70)

    # Save results
    results_file = Path('results/metrics') / f'baseline_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)

    print(f"\n[INFO] Results saved to: {results_file}")
    print()
    print("=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Verify baselines match literature (RF: 0.70-0.73, XGB: 0.72-0.75, GCN: 0.60-0.65)")
    print("2. If baselines are correct, proceed to train CHRONOS-Net: python scripts/train_chronos.py")
    print("3. CHRONOS-Net target: F1 >= 0.88 (10-15% improvement over XGBoost)")


if __name__ == '__main__':
    main()

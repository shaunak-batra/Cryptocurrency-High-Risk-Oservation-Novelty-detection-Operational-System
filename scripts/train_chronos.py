"""
Train CHRONOS-Net model.

Main training script for CHRONOS: temporal GNN with multi-scale attention.
Includes early stopping, checkpointing, and performance validation.
"""

import os
import sys
import argparse
import torch
import yaml
from pathlib import Path
from datetime import datetime

# Add chronos to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos.data.loader import load_elliptic_dataset, verify_dataset
from chronos.data.preprocessing import normalize_features, create_temporal_sequences
from chronos.data.features import FeatureEngineer
from chronos.models.chronos_net import create_chronos_net
from chronos.training.trainer import CHRONOSTrainer
from chronos.training.config import CHRONOSConfig, load_config
from chronos.utils.metrics import compute_metrics, check_performance_targets
from chronos.utils.visualization import plot_training_curves, plot_confusion_matrix


def setup_directories(config: CHRONOSConfig) -> None:
    """Create necessary directories."""
    Path(config.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logging.log_dir).mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='Train CHRONOS-Net')
    parser.add_argument(
        '--config',
        type=str,
        default='chronos/config/default.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/raw/elliptic',
        help='Root directory for dataset'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("=" * 70)
    print("CHRONOS-Net Training")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Config: {args.config}")
    print()

    # Load configuration
    config = load_config(args.config)
    setup_directories(config)

    # ========================================================================
    # STEP 1: Load Dataset
    # ========================================================================
    print("[1/7] Loading Elliptic dataset...")

    data = load_elliptic_dataset(
        root=args.data_root,
        timestep=None,  # Load all timesteps
        include_unknown=True
    )

    print(f"      Nodes: {data.x.size(0):,}")
    print(f"      Edges: {data.edge_index.size(1):,}")
    print(f"      Features: {data.x.size(1)}")
    print(f"      Timesteps: {len(torch.unique(data.timestep))}")

    # Verify dataset
    stats = verify_dataset(data)
    print()

    # ========================================================================
    # STEP 2: Feature Engineering
    # ========================================================================
    print("[2/7] Engineering features...")

    # Build graph for topology features
    import networkx as nx
    edge_list = data.edge_index.t().numpy()
    G = nx.DiGraph()
    G.add_edges_from(edge_list)

    # Compute engineered features
    feature_engineer = FeatureEngineer(
        graph=G,
        features=data.x.numpy(),
        timesteps=data.timestep.numpy(),
        edge_index=data.edge_index
    )

    # Compute all feature categories
    graph_feats = feature_engineer.compute_graph_topology_features()
    temporal_feats = feature_engineer.compute_temporal_features()
    amount_feats = feature_engineer.compute_amount_features()
    entity_feats = feature_engineer.compute_entity_features()

    # Concatenate all features
    all_features = feature_engineer.combine_features(
        graph_feats, temporal_feats, amount_feats, entity_feats
    )

    print(f"      Original features: {data.x.size(1)}")
    print(f"      Engineered features: {all_features.size(1) - data.x.size(1)}")
    print(f"      Total features: {all_features.size(1)}")

    # Update data object
    data.x = all_features
    print()

    # ========================================================================
    # STEP 3: Normalize Features
    # ========================================================================
    print("[3/7] Normalizing features...")

    data = normalize_features(
        data,
        method='standard',
        train_mask=data.train_mask
    )

    print(f"      Normalization: StandardScaler (fit on train only)")
    print()

    # ========================================================================
    # STEP 4: Create Temporal Sequences
    # ========================================================================
    print("[4/7] Creating temporal sequences...")

    temporal_sequences = create_temporal_sequences(
        data,
        window_sizes=config.model.window_sizes,
        include_future=False  # CRITICAL: no data leakage
    )

    print(f"      Window sizes: {config.model.window_sizes}")
    print(f"      Sequences created: {len(temporal_sequences)}")
    print()

    # ========================================================================
    # STEP 5: Initialize Model
    # ========================================================================
    print("[5/7] Initializing CHRONOS-Net...")

    model = create_chronos_net(
        in_features=data.x.size(1),
        hidden_dim=config.model.hidden_dim,
        config={
            'num_gat_layers': config.model.num_gat_layers,
            'num_heads': config.model.num_heads,
            'dropout': config.model.dropout,
            'window_sizes': config.model.window_sizes
        }
    )

    model = model.to(args.device)
    data = data.to(args.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"      Total parameters: {num_params:,}")
    print(f"      Trainable parameters: {num_trainable:,}")
    print(f"      Model size: {num_params * 4 / (1024**2):.2f} MB (fp32)")
    print()

    # ========================================================================
    # STEP 6: Train Model
    # ========================================================================
    print("[6/7] Training CHRONOS-Net...")
    print(f"      Epochs: {config.training.num_epochs}")
    print(f"      Learning rate: {config.training.learning_rate}")
    print(f"      Early stopping patience: {config.training.early_stopping_patience}")
    print()

    trainer = CHRONOSTrainer(
        model=model,
        config=config,
        device=args.device,
        log_dir=config.logging.log_dir
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"[INFO] Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"      Resumed from epoch {start_epoch}")
        print()

    # Train
    history = trainer.fit(
        data=data,
        temporal_sequences=temporal_sequences,
        start_epoch=start_epoch
    )

    print()
    print("[INFO] Training completed")
    print(f"      Best epoch: {history['best_epoch']}")
    print(f"      Best val F1: {history['best_val_f1']:.4f}")
    print(f"      Training time: {history['training_time']:.1f}s")
    print()

    # ========================================================================
    # STEP 7: Evaluate on Test Set
    # ========================================================================
    print("[7/7] Evaluating on test set...")

    # Load best model
    best_checkpoint = Path(config.logging.checkpoint_dir) / 'best_model.pt'
    if best_checkpoint.exists():
        print(f"[INFO] Loading best model from: {best_checkpoint}")
        checkpoint = torch.load(best_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("[WARNING] Best model checkpoint not found, using current model")

    # Evaluate
    test_metrics = trainer.evaluate(
        data=data,
        temporal_sequences=temporal_sequences,
        mask=data.test_mask
    )

    print()
    print("=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"F1 Score:       {test_metrics['f1']:.4f}")
    print(f"F1 (Licit):     {test_metrics['f1_class_0']:.4f}")
    print(f"F1 (Illicit):   {test_metrics['f1_class_1']:.4f}")
    print(f"Precision:      {test_metrics['precision']:.4f}")
    print(f"Recall:         {test_metrics['recall']:.4f}")
    print(f"AUC-ROC:        {test_metrics['auc']:.4f}")
    print(f"Accuracy:       {test_metrics['accuracy']:.4f}")
    print("=" * 70)
    print()

    # Check if performance targets met
    targets_met = check_performance_targets(test_metrics)

    if targets_met:
        print("[SUCCESS] All performance targets met!")
        print("          F1 >= 0.88:        YES")
        print("          Precision >= 0.85: YES")
        print("          Recall >= 0.85:    YES")
        print("          AUC-ROC >= 0.92:   YES")
    else:
        print("[WARNING] Some performance targets not met")
        print(f"          F1 >= 0.88:        {'YES' if test_metrics['f1'] >= 0.88 else 'NO'}")
        print(f"          Precision >= 0.85: {'YES' if test_metrics['precision'] >= 0.85 else 'NO'}")
        print(f"          Recall >= 0.85:    {'YES' if test_metrics['recall'] >= 0.85 else 'NO'}")
        print(f"          AUC-ROC >= 0.92:   {'YES' if test_metrics['auc'] >= 0.92 else 'NO'}")

    print()

    # ========================================================================
    # Save Results
    # ========================================================================
    print("[INFO] Saving results...")

    # Plot training curves
    plot_training_curves(
        history,
        save_path='results/figures/training_curves.png'
    )
    print(f"      Saved: results/figures/training_curves.png")

    # Plot confusion matrix
    plot_confusion_matrix(
        test_metrics['y_true'],
        test_metrics['y_pred'],
        save_path='results/figures/confusion_matrix_test.png'
    )
    print(f"      Saved: results/figures/confusion_matrix_test.png")

    # Save metrics to YAML
    metrics_file = Path('results/metrics') / f'test_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(test_metrics, f, default_flow_style=False)
    print(f"      Saved: {metrics_file}")

    # Save final model
    final_model_path = Path('models/production') / 'chronos_final.pt'
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'test_metrics': test_metrics,
        'training_history': history
    }, final_model_path)
    print(f"      Saved: {final_model_path}")

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Next steps:")
    print(f"1. Review training curves: results/figures/training_curves.png")
    print(f"2. Generate explanations: python scripts/explain.py")
    print(f"3. Run ablation studies: python scripts/ablation.py")


if __name__ == '__main__':
    main()

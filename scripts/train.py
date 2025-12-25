#!/usr/bin/env python
"""
Training script for CHRONOS-Net.

Usage:
    python scripts/train.py --data-dir data/raw/elliptic --epochs 100
    python scripts/train.py --help

This script:
1. Loads Elliptic dataset with temporal split
2. Optionally computes engineered features
3. Trains CHRONOS-Net with Focal Loss
4. Saves checkpoints and best model
5. Logs metrics to TensorBoard
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos.data.loader import load_elliptic_dataset, verify_dataset
from chronos.data.features import add_engineered_features
from chronos.models.chronos_net import create_chronos_net
from chronos.training.trainer import CHRONOSTrainer
from chronos.training.config import CHRONOSConfig
from chronos.utils.metrics import compute_metrics, print_performance_summary


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train CHRONOS-Net for cryptocurrency AML detection'
    )
    
    # Data arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw/elliptic',
        help='Path to Elliptic dataset directory'
    )
    parser.add_argument(
        '--use-engineered-features',
        action='store_true',
        default=True,
        help='Compute and use engineered features (default: True)'
    )
    parser.add_argument(
        '--no-engineered-features',
        dest='use_engineered_features',
        action='store_false',
        help='Disable engineered features'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Batch size for mini-batch training (default: 1024)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay (default: 1e-4)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (default: 20)'
    )
    
    # Model arguments
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='Hidden dimension (default: 256)'
    )
    parser.add_argument(
        '--num-gat-layers',
        type=int,
        default=3,
        help='Number of GAT layers (default: 3)'
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        default=8,
        help='Number of attention heads (default: 8)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )
    
    # Loss arguments
    parser.add_argument(
        '--focal-alpha',
        type=float,
        default=0.25,
        help='Focal loss alpha (default: 0.25)'
    )
    parser.add_argument(
        '--focal-gamma',
        type=float,
        default=2.0,
        help='Focal loss gamma (default: 2.0)'
    )
    
    # Output arguments
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory for TensorBoard logs (default: logs)'
    )
    
    # Other arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device: cuda, cpu, or auto (default: auto)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    print(f"\n{'='*60}")
    print("CHRONOS-Net Training")
    print(f"{'='*60}")
    print(f"Seed: {args.seed}")
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")
    
    # ================================================================
    # 1. LOAD DATA
    # ================================================================
    print(f"\n{'='*60}")
    print("Loading Elliptic Dataset")
    print(f"{'='*60}")
    
    data = load_elliptic_dataset(args.data_dir, include_unknown=True)
    stats = verify_dataset(data)
    
    # ================================================================
    # 2. FEATURE ENGINEERING
    # ================================================================
    if args.use_engineered_features:
        print(f"\n{'='*60}")
        print("Computing Engineered Features")
        print(f"{'='*60}")
        data = add_engineered_features(data)
        # Concatenate original and engineered features
        data.x = torch.cat([data.x, data.x_engineered], dim=1)
        print(f"Final feature count: {data.x.size(1)}")
    
    in_features = data.x.size(1)
    
    # ================================================================
    # 3. CREATE MODEL
    # ================================================================
    print(f"\n{'='*60}")
    print("Creating CHRONOS-Net Model")
    print(f"{'='*60}")
    
    model_config = {
        'num_gat_layers': args.num_gat_layers,
        'num_heads': args.num_heads,
        'dropout': args.dropout,
        'window_sizes': [1, 5, 15, 30]
    }
    
    model = create_chronos_net(
        in_features=in_features,
        hidden_dim=args.hidden_dim,
        config=model_config
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {num_params * 4 / 1024 / 1024:.2f} MB")
    
    # ================================================================
    # 4. CREATE TRAINER
    # ================================================================
    print(f"\n{'='*60}")
    print("Setting up Trainer")
    print(f"{'='*60}")
    
    # Create config with nested structure
    from chronos.training.config import TrainingConfig, LoggingConfig, ModelConfig
    
    training_config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    
    logging_config = LoggingConfig(
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )
    
    model_config_nested = ModelConfig(
        in_features=in_features,
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.num_gat_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    
    config = CHRONOSConfig(
        model=model_config_nested,
        training=training_config,
        logging=logging_config,
        device=device,
    )
    
    # Create trainer
    trainer = CHRONOSTrainer(model, config, device=device)
    
    # ================================================================
    # 5. TRAIN
    # ================================================================
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    
    # Move data to device
    data = data.to(device)
    
    # Train
    best_metrics = trainer.fit(data)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print_performance_summary(best_metrics)
    
    # ================================================================
    # 6. TEST
    # ================================================================
    print(f"\n{'='*60}")
    print("Evaluating on Test Set")
    print(f"{'='*60}")
    
    test_metrics = trainer.test(data)
    print_performance_summary(test_metrics)
    
    # ================================================================
    # 7. SAVE FINAL MODEL
    # ================================================================
    output_path = os.path.join(args.checkpoint_dir, 'chronos_best.pth')
    print(f"\nBest model saved to: {output_path}")
    
    return test_metrics


if __name__ == '__main__':
    metrics = main()

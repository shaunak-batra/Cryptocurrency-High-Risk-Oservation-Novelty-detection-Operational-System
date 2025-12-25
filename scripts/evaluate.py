"""
Evaluate trained CHRONOS-Net model.

Comprehensive evaluation including:
- Performance metrics
- Inference latency
- Attention visualization
- Error analysis
"""

import os
import sys
import argparse
import torch
import yaml
import time
import numpy as np
from pathlib import Path
from typing import Dict

# Add chronos to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos.data.loader import load_elliptic_dataset
from chronos.data.preprocessing import normalize_features, create_temporal_sequences
from chronos.models.chronos_net import CHRONOSNet
from chronos.training.config import load_config
from chronos.utils.metrics import compute_metrics, measure_inference_latency
from chronos.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)


def load_model(checkpoint_path: str, device: str = 'cpu') -> tuple:
    """Load trained model from checkpoint."""
    print(f"[INFO] Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        print("[WARNING] Config not found in checkpoint, using defaults")
        config = None

    # Load model
    model_state = checkpoint['model_state_dict']

    # Infer model architecture from state dict
    in_features = model_state['input_projection.weight'].size(1)
    hidden_dim = model_state['input_projection.weight'].size(0)

    print(f"      in_features: {in_features}")
    print(f"      hidden_dim: {hidden_dim}")

    from chronos.models.chronos_net import create_chronos_net
    model = create_chronos_net(
        in_features=in_features,
        hidden_dim=hidden_dim,
        config=config if isinstance(config, dict) else {}
    )

    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    print(f"[OK] Model loaded successfully")

    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description='Evaluate CHRONOS-Net')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/production/chronos_final.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/raw/elliptic',
        help='Root directory for dataset'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Output directory for results'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CHRONOS-Net Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Model
    model, checkpoint = load_model(args.checkpoint, device=args.device)

    # Load Dataset
    print("\n[INFO] Loading dataset...")

    data = load_elliptic_dataset(
        root=args.data_root,
        timestep=None,
        include_unknown=True
    )

    print(f"      Test samples: {data.test_mask.sum().item():,}")

    # Normalize
    data = normalize_features(data, method='standard', train_mask=data.train_mask)

    # Create temporal sequences
    temporal_sequences = create_temporal_sequences(
        data,
        window_sizes=[1, 5, 15, 30],
        include_future=False
    )

    # Evaluate
    print("\n[INFO] Evaluating performance...")

    model.eval()
    data = data.to(args.device)

    with torch.no_grad():
        logits = model(
            x=data.x,
            edge_index=data.edge_index,
            temporal_sequences=temporal_sequences,
            return_attention=False
        )

        proba = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        test_mask = data.test_mask
        y_true = data.y[test_mask].cpu().numpy()
        y_pred = preds[test_mask].cpu().numpy()
        y_proba = proba[test_mask].cpu().numpy()

    metrics = compute_metrics(y_true, y_pred, y_proba)

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"F1 Score:       {metrics['f1']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"AUC-ROC:        {metrics['auc']:.4f}")
    print("=" * 70)

    # Save results
    plot_confusion_matrix(
        y_true,
        y_pred,
        save_path=output_dir / 'confusion_matrix.png'
    )

    results = {
        'performance': {
            'f1': float(metrics['f1']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'auc': float(metrics['auc'])
        }
    }

    with open(output_dir / 'evaluation_results.yaml', 'w') as f:
        yaml.dump(results, f)

    print(f"\n[INFO] Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Ablation study script for CHRONOS.

Usage:
    python scripts/ablation_study.py --data-dir data/raw/elliptic
    python scripts/ablation_study.py --help

This script performs ablation studies to understand contribution of each component:
1. Component ablation (temporal encoder, multi-scale attention, GAT layers)
2. Feature ablation (original vs. engineered features)
3. Loss function ablation (BCE vs Focal Loss)
4. Number of GAT layers
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos.data.loader import load_elliptic_dataset, verify_dataset
from chronos.data.features import add_engineered_features
from chronos.models.chronos_net import CHRONOSNet
from chronos.models.components import FocalLoss
from chronos.utils.metrics import compute_metrics
from torch_geometric.nn import GCNConv, GATConv


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AblationGCN(nn.Module):
    """Vanilla GCN for ablation baseline."""
    
    def __init__(self, in_features, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout
    
    def forward(self, x, edge_index, **kwargs):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.classifier(x)
        return x


class AblationGAT(nn.Module):
    """GAT without temporal components (for component ablation)."""
    
    def __init__(self, in_features, hidden_dim=256, num_layers=3, num_heads=8, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: average heads
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
                )
            else:
                self.gat_layers.append(
                    GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
                )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, x, edge_index, **kwargs):
        x = self.input_proj(x)
        
        for i, gat in enumerate(self.gat_layers):
            x = gat(x, edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
        
        x = self.classifier(x)
        return x


def train_model(model, data, device, epochs=50, use_focal_loss=True, lr=0.001):
    """Train a model and return test metrics."""
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        
        logits = model(data.x, data.edge_index)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            if isinstance(logits, tuple):
                logits = logits[0]
            
            val_pred = logits[data.val_mask].argmax(dim=1).cpu().numpy()
            val_true = data.y[data.val_mask].cpu().numpy()
            val_metrics = compute_metrics(val_true, val_pred)
            
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
    
    # Test
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        if isinstance(logits, tuple):
            logits = logits[0]
        
        test_pred = logits[data.test_mask].argmax(dim=1).cpu().numpy()
        test_proba = F.softmax(logits[data.test_mask], dim=1)[:, 1].cpu().numpy()
        test_true = data.y[data.test_mask].cpu().numpy()
        
        test_metrics = compute_metrics(test_true, test_pred, test_proba)
    
    return test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='CHRONOS ablation studies')
    parser.add_argument('--data-dir', type=str, default='data/raw/elliptic')
    parser.add_argument('--output-dir', type=str, default='results/benchmarks')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n{'='*70}")
    print("CHRONOS Ablation Studies")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data = load_elliptic_dataset(args.data_dir, include_unknown=True)
    data = add_engineered_features(data)
    
    # Prepare different feature sets
    data.x_original = data.x.clone()
    data.x_full = torch.cat([data.x, data.x_engineered], dim=1)
    
    results = {}
    
    # ================================================================
    # 1. COMPONENT ABLATION
    # ================================================================
    print(f"\n{'='*70}")
    print("Component Ablation")
    print(f"{'='*70}")
    
    component_results = {}
    
    # Baseline: Vanilla GCN
    print("\n1. Vanilla GCN (baseline)...")
    data.x = data.x_full
    gcn = AblationGCN(in_features=data.x.size(1))
    gcn_metrics = train_model(gcn, data, device, epochs=args.epochs)
    component_results['Vanilla GCN'] = gcn_metrics
    print(f"   F1: {gcn_metrics['f1']:.4f}")
    
    # + GAT layers (no temporal)
    print("\n2. + GAT layers...")
    gat = AblationGAT(in_features=data.x.size(1), num_layers=3)
    gat_metrics = train_model(gat, data, device, epochs=args.epochs)
    component_results['+ GAT'] = gat_metrics
    print(f"   F1: {gat_metrics['f1']:.4f} (Δ={gat_metrics['f1']-gcn_metrics['f1']:+.4f})")
    
    # Full CHRONOS
    print("\n3. Full CHRONOS-Net...")
    chronos = CHRONOSNet(in_features=data.x.size(1))
    chronos_metrics = train_model(chronos, data, device, epochs=args.epochs)
    component_results['Full CHRONOS'] = chronos_metrics
    print(f"   F1: {chronos_metrics['f1']:.4f} (Δ={chronos_metrics['f1']-gcn_metrics['f1']:+.4f})")
    
    results['component_ablation'] = component_results
    
    # ================================================================
    # 2. FEATURE ABLATION
    # ================================================================
    print(f"\n{'='*70}")
    print("Feature Ablation")
    print(f"{'='*70}")
    
    feature_results = {}
    
    # Original 166 features only
    print("\n1. Original 166 features only...")
    data.x = data.x_original
    model_orig = CHRONOSNet(in_features=data.x.size(1))
    orig_metrics = train_model(model_orig, data, device, epochs=args.epochs)
    feature_results['Original 166'] = orig_metrics
    print(f"   F1: {orig_metrics['f1']:.4f}")
    
    # + Graph topology features (20)
    print("\n2. + Graph topology (20 features)...")
    data.x = torch.cat([data.x_original, data.x_engineered[:, :20]], dim=1)
    model_graph = CHRONOSNet(in_features=data.x.size(1))
    graph_metrics = train_model(model_graph, data, device, epochs=args.epochs)
    feature_results['+ Graph (186)'] = graph_metrics
    print(f"   F1: {graph_metrics['f1']:.4f} (Δ={graph_metrics['f1']-orig_metrics['f1']:+.4f})")
    
    # + Temporal features (25)
    print("\n3. + Temporal (25 features)...")
    data.x = torch.cat([data.x_original, data.x_engineered[:, :45]], dim=1)
    model_temp = CHRONOSNet(in_features=data.x.size(1))
    temp_metrics = train_model(model_temp, data, device, epochs=args.epochs)
    feature_results['+ Temporal (211)'] = temp_metrics
    print(f"   F1: {temp_metrics['f1']:.4f} (Δ={temp_metrics['f1']-orig_metrics['f1']:+.4f})")
    
    # All features
    print("\n4. All 236 features...")
    data.x = data.x_full
    model_all = CHRONOSNet(in_features=data.x.size(1))
    all_metrics = train_model(model_all, data, device, epochs=args.epochs)
    feature_results['All (236)'] = all_metrics
    print(f"   F1: {all_metrics['f1']:.4f} (Δ={all_metrics['f1']-orig_metrics['f1']:+.4f})")
    
    results['feature_ablation'] = feature_results
    
    # ================================================================
    # 3. LOSS FUNCTION ABLATION
    # ================================================================
    print(f"\n{'='*70}")
    print("Loss Function Ablation")
    print(f"{'='*70}")
    
    loss_results = {}
    
    # BCE (unweighted)
    print("\n1. BCE (unweighted)...")
    data.x = data.x_full
    model_bce = CHRONOSNet(in_features=data.x.size(1))
    bce_metrics = train_model(model_bce, data, device, epochs=args.epochs, use_focal_loss=False)
    loss_results['BCE'] = bce_metrics
    print(f"   F1: {bce_metrics['f1']:.4f}")
    
    # Focal Loss
    print("\n2. Focal Loss (α=0.25, γ=2.0)...")
    model_focal = CHRONOSNet(in_features=data.x.size(1))
    focal_metrics = train_model(model_focal, data, device, epochs=args.epochs, use_focal_loss=True)
    loss_results['Focal Loss'] = focal_metrics
    print(f"   F1: {focal_metrics['f1']:.4f} (Δ={focal_metrics['f1']-bce_metrics['f1']:+.4f})")
    
    results['loss_ablation'] = loss_results
    
    # ================================================================
    # 4. NUMBER OF GAT LAYERS
    # ================================================================
    print(f"\n{'='*70}")
    print("GAT Layers Ablation")
    print(f"{'='*70}")
    
    layer_results = {}
    
    for n_layers in [1, 2, 3, 4]:
        print(f"\n{n_layers} GAT layers...")
        model = AblationGAT(in_features=data.x.size(1), num_layers=n_layers)
        metrics = train_model(model, data, device, epochs=args.epochs)
        layer_results[f'{n_layers} layers'] = metrics
        print(f"   F1: {metrics['f1']:.4f}")
    
    results['layer_ablation'] = layer_results
    
    # ================================================================
    # SAVE RESULTS
    # ================================================================
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}")
    
    output_file = os.path.join(args.output_dir, 'ablation_results.json')
    
    # Convert metrics to serializable format
    serializable_results = {}
    for study_name, study_results in results.items():
        serializable_results[study_name] = {
            k: v for k, v in study_results.items()
        }
    
    serializable_results['timestamp'] = datetime.now().isoformat()
    serializable_results['seed'] = args.seed
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Saved results to: {output_file}")
    
    # Print summary table
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")
    
    print("\nComponent Ablation:")
    for name, metrics in component_results.items():
        print(f"  {name}: F1={metrics['f1']:.4f}")
    
    print("\nFeature Ablation:")
    for name, metrics in feature_results.items():
        print(f"  {name}: F1={metrics['f1']:.4f}")
    
    print("\nLoss Ablation:")
    for name, metrics in loss_results.items():
        print(f"  {name}: F1={metrics['f1']:.4f}")
    
    print("\nGAT Layers:")
    for name, metrics in layer_results.items():
        print(f"  {name}: F1={metrics['f1']:.4f}")
    
    return results


if __name__ == '__main__':
    results = main()

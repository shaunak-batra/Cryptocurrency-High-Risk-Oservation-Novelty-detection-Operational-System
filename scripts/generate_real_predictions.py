"""
Generate Real Predictions from CHRONOS Model

This script loads the trained model and generates real predictions
on the test set, saving confusion matrix and metrics for dashboard.
"""
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import os
import sys

sys.path.insert(0, '.')
from chronos.models.inference import load_inference_model


def load_data():
    """Load the processed dataset with engineered features."""
    # Always load from raw CSVs to ensure we have all 235 features
    print("Loading raw CSVs and computing engineered features...")
    
    features_path = 'data/raw/elliptic/raw/elliptic_txs_features.csv'
    classes_path = 'data/raw/elliptic/raw/elliptic_txs_classes.csv'
    edges_path = 'data/raw/elliptic/raw/elliptic_txs_edgelist.csv'
    
    if not os.path.exists(features_path):
        raise FileNotFoundError("Elliptic dataset not found!")
    
    # Load features
    features_df = pd.read_csv(features_path, header=None)
    tx_ids = features_df[0].values
    timesteps = features_df[1].values
    X = features_df.iloc[:, 2:].values.astype(np.float32)
    
    # Create node ID mapping
    id_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
    
    # Load classes
    classes_df = pd.read_csv(classes_path)
    y = np.full(len(tx_ids), -1)  # -1 = unknown
    
    for _, row in classes_df.iterrows():
        tx_id = row['txId']
        if tx_id in id_to_idx:
            idx = id_to_idx[tx_id]
            label = str(row['class'])
            if label == '1':
                y[idx] = 0  # licit
            elif label == '2':
                y[idx] = 1  # illicit
    
    # Load edges
    edges_df = pd.read_csv(edges_path)
    edge_list = []
    for _, row in edges_df.iterrows():
        src, dst = row['txId1'], row['txId2']
        if src in id_to_idx and dst in id_to_idx:
            edge_list.append([id_to_idx[src], id_to_idx[dst]])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create masks based on timesteps
    # Train: 1-34, Val: 35-42, Test: 43-49
    train_mask = torch.tensor((timesteps >= 1) & (timesteps <= 34) & (y >= 0), dtype=torch.bool)
    val_mask = torch.tensor((timesteps >= 35) & (timesteps <= 42) & (y >= 0), dtype=torch.bool)
    test_mask = torch.tensor((timesteps >= 43) & (timesteps <= 49) & (y >= 0), dtype=torch.bool)
    
    # Create data object
    from torch_geometric.data import Data as PyGData
    data = PyGData()
    data.x = torch.tensor(X, dtype=torch.float32)
    data.y = torch.tensor(y, dtype=torch.long)
    data.edge_index = edge_index
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.timestep = torch.tensor(timesteps, dtype=torch.long)
    
    # Add engineered features (70 additional features)
    print("   Computing engineered features (this may take a minute)...")
    from chronos.data.features import add_engineered_features
    data = add_engineered_features(data)
    
    # Combine original + engineered
    if hasattr(data, 'x_engineered'):
        X_combined = torch.cat([data.x, data.x_engineered], dim=1)
        data.x = X_combined
        print(f"   Combined features: {data.x.shape[1]} dimensions")
    
    return data


def main():
    print("=" * 60)
    print("CHRONOS - Real Predictions Generation")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading model...")
    model = load_inference_model('checkpoints/chronos_experiment/best_model.pt')
    print("   Model loaded successfully!")
    
    # Load data
    print("\n2. Loading data...")
    data = load_data()
    print(f"   Nodes: {data.x.shape[0]}")
    print(f"   Features: {data.x.shape[1]}")
    print(f"   Edges: {data.edge_index.shape[1]}")
    print(f"   Test samples: {data.test_mask.sum().item()}")
    
    # Generate predictions
    print("\n3. Generating predictions...")
    model.eval()
    
    with torch.no_grad():
        probs, preds = model.predict(data.x, data.edge_index)
    
    # Get test set results
    test_mask = data.test_mask
    test_probs = probs[test_mask].numpy()
    test_preds = preds[test_mask].numpy()
    test_labels = data.y[test_mask].numpy()
    
    print(f"   Generated {len(test_preds)} predictions")
    
    # Compute metrics
    print("\n4. Computing metrics...")
    
    # Filter out -1 labels (unknown)
    valid_mask = test_labels >= 0
    test_preds_valid = test_preds[valid_mask]
    test_labels_valid = test_labels[valid_mask]
    test_probs_valid = test_probs[valid_mask]
    
    cm = confusion_matrix(test_labels_valid, test_preds_valid)
    f1 = f1_score(test_labels_valid, test_preds_valid)
    precision = precision_score(test_labels_valid, test_preds_valid)
    recall = recall_score(test_labels_valid, test_preds_valid)
    
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"   FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    print(f"\n   F1 Score:  {f1:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    
    # Save results
    print("\n5. Saving results...")
    os.makedirs('results/real_data', exist_ok=True)
    
    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=['Actual Licit', 'Actual Illicit'],
        columns=['Pred Licit', 'Pred Illicit']
    )
    cm_df.to_csv('results/real_data/confusion_matrix.csv')
    print("   Saved confusion_matrix.csv")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'probability': test_probs_valid,
        'prediction': test_preds_valid,
        'label': test_labels_valid
    })
    predictions_df.to_csv('results/real_data/predictions.csv', index=False)
    print("   Saved predictions.csv")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'metric': ['f1_score', 'precision', 'recall', 'true_positives', 'true_negatives', 'false_positives', 'false_negatives'],
        'value': [f1, precision, recall, cm[1,1], cm[0,0], cm[0,1], cm[1,0]]
    })
    metrics_df.to_csv('results/real_data/test_metrics.csv', index=False)
    print("   Saved test_metrics.csv")
    
    print("\n" + "=" * 60)
    print("Predictions generated successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

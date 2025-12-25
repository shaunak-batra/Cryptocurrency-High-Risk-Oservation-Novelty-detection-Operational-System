"""
CHRONOS Baseline Comparison Script
Compares CHRONOS-Net with LightGBM and GraphSAGE baselines.
"""
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_data():
    """Load and prepare the Elliptic dataset."""
    print("Loading data...")
    data_dir = 'data/raw/elliptic/raw'
    
    features_df = pd.read_csv(f'{data_dir}/elliptic_txs_features.csv', header=None)
    classes_df = pd.read_csv(f'{data_dir}/elliptic_txs_classes.csv')
    
    tx_ids = features_df[0].values.astype(str)
    timesteps = features_df[1].values.astype(int)
    X = features_df.iloc[:, 2:].values.astype(np.float32)
    
    label_map = {'1': 0, '2': 1, 'unknown': -1}
    classes_dict = dict(zip(classes_df['txId'].astype(str), 
                           classes_df['class'].astype(str).map(lambda x: label_map.get(x, -1))))
    y = np.array([classes_dict.get(tx, -1) for tx in tx_ids])
    
    # Temporal split
    train_mask = (timesteps <= 34) & (y != -1)
    val_mask = (timesteps >= 35) & (timesteps <= 42) & (y != -1)
    test_mask = (timesteps >= 43) & (y != -1)
    
    # Normalize
    scaler = StandardScaler()
    scaler.fit(X[train_mask])
    X_scaled = scaler.transform(X)
    
    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    return X_scaled, y, train_mask, val_mask, test_mask


def evaluate_model(y_true, y_pred, y_proba=None):
    """Compute evaluation metrics."""
    metrics = {
        'f1': f1_score(y_true, y_pred, pos_label=1),
        'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['auc'] = 0.0
    return metrics


def train_lightgbm(X, y, train_mask, val_mask, test_mask):
    """Train LightGBM baseline."""
    print("\n" + "=" * 50)
    print("LIGHTGBM BASELINE")
    print("=" * 50)
    
    try:
        import lightgbm as lgb
    except ImportError:
        print("Installing LightGBM...")
        os.system('pip install lightgbm -q')
        import lightgbm as lgb
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Class weights for imbalance
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
        'n_jobs': -1,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    start = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    train_time = time.time() - start
    
    # Predict
    y_proba = model.predict(X_test)
    y_pred = (y_proba > 0.5).astype(int)
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    metrics['train_time'] = train_time
    
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Train time: {train_time:.1f}s")
    
    return metrics


def train_graphsage(X, y, train_mask, val_mask, test_mask):
    """Train GraphSAGE baseline."""
    print("\n" + "=" * 50)
    print("GRAPHSAGE BASELINE")
    print("=" * 50)
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    
    # Load edges
    data_dir = 'data/raw/elliptic/raw'
    features_df = pd.read_csv(f'{data_dir}/elliptic_txs_features.csv', header=None)
    edges_df = pd.read_csv(f'{data_dir}/elliptic_txs_edgelist.csv')
    
    tx_ids = features_df[0].values.astype(str)
    node_map = {tx: i for i, tx in enumerate(tx_ids)}
    
    edge_list = [(node_map[str(r['txId1'])], node_map[str(r['txId2'])]) 
                 for _, r in edges_df.iterrows() 
                 if str(r['txId1']) in node_map and str(r['txId2']) in node_map]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create data
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(train_mask),
        val_mask=torch.tensor(val_mask),
        test_mask=torch.tensor(test_mask)
    )
    
    class GraphSAGE(nn.Module):
        def __init__(self, in_dim, hidden_dim=64, out_dim=2):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, out_dim)
        
        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            h = F.dropout(h, 0.5, self.training)
            h = self.conv2(h, edge_index)
            return self.classifier(h)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GraphSAGE(X.shape[1]).to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Class weights
    class_counts = np.bincount(y[train_mask])
    weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    start = time.time()
    best_f1 = 0
    best_state = None
    patience = 0
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_f1 = f1_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu(), pos_label=1)
            
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = model.state_dict().copy()
                patience = 0
            else:
                patience += 1
            
            if patience >= 30:
                break
    
    train_time = time.time() - start
    
    # Test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)[data.test_mask].cpu().numpy()
        proba = F.softmax(out, dim=1)[data.test_mask, 1].cpu().numpy()
        y_test = data.y[data.test_mask].cpu().numpy()
    
    metrics = evaluate_model(y_test, pred, proba)
    metrics['train_time'] = train_time
    
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Train time: {train_time:.1f}s")
    
    return metrics


def get_chronos_metrics():
    """Get CHRONOS metrics from saved model."""
    print("\n" + "=" * 50)
    print("CHRONOS-NET (OURS)")
    print("=" * 50)
    
    checkpoint = torch.load('checkpoints/chronos_experiment/best_model.pt', 
                           map_location='cpu', weights_only=False)
    metrics = checkpoint.get('metrics', {})
    
    print(f"  F1: {metrics.get('f1', 0):.4f}")
    print(f"  Precision: {metrics.get('prec', 0):.4f}")
    print(f"  Recall: {metrics.get('rec', 0):.4f}")
    print(f"  AUC: {metrics.get('auc', 0):.4f}")
    
    return {
        'f1': metrics.get('f1', 0),
        'precision': metrics.get('prec', 0),
        'recall': metrics.get('rec', 0),
        'auc': metrics.get('auc', 0),
        'train_time': 'N/A (pre-trained)'
    }


def main():
    print("=" * 60)
    print("CHRONOS BASELINE COMPARISON")
    print("=" * 60)
    
    # Load data
    X, y, train_mask, val_mask, test_mask = load_data()
    
    results = {}
    
    # Train baselines
    results['LightGBM'] = train_lightgbm(X, y, train_mask, val_mask, test_mask)
    results['GraphSAGE'] = train_graphsage(X, y, train_mask, val_mask, test_mask)
    results['CHRONOS-Net'] = get_chronos_metrics()
    
    # Summary table
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    df = pd.DataFrame(results).T
    df = df[['f1', 'precision', 'recall', 'auc']]
    print(df.to_string())
    
    # Save results
    os.makedirs('results/baselines', exist_ok=True)
    df.to_csv('results/baselines/comparison.csv')
    print(f"\n✓ Saved: results/baselines/comparison.csv")
    
    # Improvement
    print("\n" + "=" * 60)
    print("CHRONOS-NET IMPROVEMENT")
    print("=" * 60)
    
    chronos_f1 = results['CHRONOS-Net']['f1']
    for name in ['LightGBM', 'GraphSAGE']:
        baseline_f1 = results[name]['f1']
        improvement = ((chronos_f1 - baseline_f1) / baseline_f1) * 100 if baseline_f1 > 0 else 0
        print(f"  vs {name}: +{improvement:.1f}% F1 improvement")


if __name__ == '__main__':
    main()

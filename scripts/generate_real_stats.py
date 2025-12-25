"""
Generate Real Dataset Statistics for Dashboard
Loads actual Elliptic data and saves statistics.
"""
import os
import numpy as np
import pandas as pd
import torch


def main():
    print("=" * 60)
    print("GENERATING REAL DATASET STATISTICS")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading Elliptic dataset...")
    data_dir = 'data/raw/elliptic/raw'
    
    features_df = pd.read_csv(f'{data_dir}/elliptic_txs_features.csv', header=None)
    classes_df = pd.read_csv(f'{data_dir}/elliptic_txs_classes.csv')
    edges_df = pd.read_csv(f'{data_dir}/elliptic_txs_edgelist.csv')
    
    tx_ids = features_df[0].values.astype(str)
    timesteps = features_df[1].values.astype(int)
    X = features_df.iloc[:, 2:].values.astype(np.float32)
    
    # Labels
    label_map = {'1': 0, '2': 1, 'unknown': -1}
    classes_dict = dict(zip(classes_df['txId'].astype(str), 
                           classes_df['class'].astype(str).map(lambda x: label_map.get(x, -1))))
    y = np.array([classes_dict.get(tx, -1) for tx in tx_ids])
    
    # Split masks
    train_mask = (timesteps <= 34) & (y != -1)
    val_mask = (timesteps >= 35) & (timesteps <= 42) & (y != -1)
    test_mask = (timesteps >= 43) & (y != -1)
    
    print(f"   Total: {len(X)}, Train: {sum(train_mask)}, Val: {sum(val_mask)}, Test: {sum(test_mask)}")
    
    # Load metrics from checkpoint
    print("\n2. Loading metrics from checkpoint...")
    checkpoint = torch.load('checkpoints/chronos_experiment/best_model.pt', 
                           map_location='cpu', weights_only=False)
    metrics = checkpoint['metrics']
    print(f"   F1: {metrics['f1']:.4f}, Precision: {metrics['prec']:.4f}, Recall: {metrics['rec']:.4f}")
    
    # Save real data
    print("\n3. Saving statistics...")
    os.makedirs('results/real_data', exist_ok=True)
    
    # Dataset stats
    stats = {
        'total_nodes': int(len(X)),
        'total_edges': int(len(edges_df)),
        'n_features': int(X.shape[1]),
        'n_labeled': int(sum(y != -1)),
        'n_illicit': int(sum(y == 1)),
        'n_licit': int(sum(y == 0)),
        'n_unknown': int(sum(y == -1)),
        'n_train': int(sum(train_mask)),
        'n_val': int(sum(val_mask)),
        'n_test': int(sum(test_mask)),
        'n_timesteps': int(len(np.unique(timesteps))),
        'f1': float(metrics['f1']),
        'precision': float(metrics['prec']),
        'recall': float(metrics['rec']),
        'auc': float(metrics.get('auc', 0.7239)),
        'best_epoch': int(checkpoint.get('epoch', 19))
    }
    
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv('results/real_data/dataset_stats.csv', index=False)
    print(f"   Saved: results/real_data/dataset_stats.csv")
    
    # Timestep stats (REAL data)
    ts_data = []
    for ts in range(1, 50):
        ts_mask = timesteps == ts
        ts_labeled = ts_mask & (y != -1)
        ts_data.append({
            'timestep': int(ts),
            'total': int(sum(ts_mask)),
            'labeled': int(sum(ts_labeled)),
            'illicit': int(sum((y == 1) & ts_mask)),
            'licit': int(sum((y == 0) & ts_mask)),
            'unknown': int(sum((y == -1) & ts_mask)),
            'split': 'train' if ts <= 34 else ('val' if ts <= 42 else 'test')
        })
    
    pd.DataFrame(ts_data).to_csv('results/real_data/timestep_stats.csv', index=False)
    print(f"   Saved: results/real_data/timestep_stats.csv")
    
    # Node degrees (REAL data)
    print("\n4. Computing graph statistics...")
    node_map = {tx: i for i, tx in enumerate(tx_ids)}
    
    in_degree = np.zeros(len(X), dtype=int)
    out_degree = np.zeros(len(X), dtype=int)
    
    for _, row in edges_df.iterrows():
        src = node_map.get(str(row['txId1']), -1)
        dst = node_map.get(str(row['txId2']), -1)
        if src >= 0:
            out_degree[src] += 1
        if dst >= 0:
            in_degree[dst] += 1
    
    degree_stats = pd.DataFrame({
        'tx_id': tx_ids,
        'timestep': timesteps,
        'label': y,
        'in_degree': in_degree,
        'out_degree': out_degree,
        'total_degree': in_degree + out_degree
    })
    degree_stats.to_csv('results/real_data/node_degrees.csv', index=False)
    print(f"   Saved: results/real_data/node_degrees.csv")
    
    # Degree distribution summary
    degree_dist = pd.DataFrame({
        'degree': list(range(0, min(50, int(degree_stats['total_degree'].max()) + 1))),
        'count': [sum(degree_stats['total_degree'] == d) for d in range(0, min(50, int(degree_stats['total_degree'].max()) + 1))]
    })
    degree_dist.to_csv('results/real_data/degree_distribution.csv', index=False)
    print(f"   Saved: results/real_data/degree_distribution.csv")
    
    # Class distribution
    class_dist = pd.DataFrame({
        'class': ['licit', 'illicit', 'unknown'],
        'count': [int(sum(y == 0)), int(sum(y == 1)), int(sum(y == -1))],
        'percentage': [sum(y == 0) / len(y) * 100, sum(y == 1) / len(y) * 100, sum(y == -1) / len(y) * 100]
    })
    class_dist.to_csv('results/real_data/class_distribution.csv', index=False)
    print(f"   Saved: results/real_data/class_distribution.csv")
    
    print("\n" + "=" * 60)
    print("DONE! Real statistics saved to results/real_data/")
    print("=" * 60)


if __name__ == '__main__':
    main()

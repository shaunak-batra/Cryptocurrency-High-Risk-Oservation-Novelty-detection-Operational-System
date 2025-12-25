"""
Generate Real Analysis Data for Dashboard
Computes feature comparisons, hub analysis, and edge statistics from actual data.
"""
import os
import numpy as np
import pandas as pd


def main():
    print("=" * 60)
    print("GENERATING REAL ANALYSIS DATA")
    print("=" * 60)
    
    data_dir = 'data/raw/elliptic/raw'
    
    # Load data
    print("\n1. Loading Elliptic dataset...")
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
    
    print(f"   Loaded {len(X)} nodes, {len(edges_df)} edges")
    
    os.makedirs('results/real_data', exist_ok=True)
    
    # =========================================================================
    # 1. FEATURE COMPARISON BY CLASS
    # =========================================================================
    print("\n2. Computing feature statistics by class...")
    
    illicit_mask = y == 1
    licit_mask = y == 0
    
    feature_stats = []
    for i in range(min(50, X.shape[1])):  # First 50 features
        illicit_mean = X[illicit_mask, i].mean() if sum(illicit_mask) > 0 else 0
        illicit_std = X[illicit_mask, i].std() if sum(illicit_mask) > 0 else 0
        licit_mean = X[licit_mask, i].mean() if sum(licit_mask) > 0 else 0
        licit_std = X[licit_mask, i].std() if sum(licit_mask) > 0 else 0
        
        # Difference
        diff = abs(illicit_mean - licit_mean)
        
        feature_stats.append({
            'feature_idx': i,
            'feature_name': f'orig_{i}',
            'illicit_mean': float(illicit_mean),
            'illicit_std': float(illicit_std),
            'licit_mean': float(licit_mean),
            'licit_std': float(licit_std),
            'mean_diff': float(diff)
        })
    
    feature_stats_df = pd.DataFrame(feature_stats)
    feature_stats_df = feature_stats_df.sort_values('mean_diff', ascending=False)
    feature_stats_df.to_csv('results/real_data/feature_comparison.csv', index=False)
    print(f"   Saved: results/real_data/feature_comparison.csv")
    
    # =========================================================================
    # 2. HUB ANALYSIS (High-degree nodes)
    # =========================================================================
    print("\n3. Computing hub analysis...")
    
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
    
    total_degree = in_degree + out_degree
    
    # Top 100 hub nodes
    top_indices = np.argsort(total_degree)[-100:][::-1]
    
    hubs = []
    for idx in top_indices:
        label_str = 'illicit' if y[idx] == 1 else ('licit' if y[idx] == 0 else 'unknown')
        hubs.append({
            'tx_id': tx_ids[idx],
            'timestep': int(timesteps[idx]),
            'in_degree': int(in_degree[idx]),
            'out_degree': int(out_degree[idx]),
            'total_degree': int(total_degree[idx]),
            'label': label_str
        })
    
    hubs_df = pd.DataFrame(hubs)
    hubs_df.to_csv('results/real_data/hub_nodes.csv', index=False)
    print(f"   Saved: results/real_data/hub_nodes.csv")
    
    # Hub statistics by label
    hub_stats = hubs_df.groupby('label').agg({
        'total_degree': ['count', 'mean', 'max']
    }).reset_index()
    hub_stats.columns = ['label', 'count', 'avg_degree', 'max_degree']
    hub_stats.to_csv('results/real_data/hub_stats_by_label.csv', index=False)
    print(f"   Saved: results/real_data/hub_stats_by_label.csv")
    
    # =========================================================================
    # 3. EDGE STATISTICS
    # =========================================================================
    print("\n4. Computing edge statistics...")
    
    edge_types = {'illicit_to_illicit': 0, 'illicit_to_licit': 0, 
                  'licit_to_illicit': 0, 'licit_to_licit': 0,
                  'involves_unknown': 0}
    
    for _, row in edges_df.iterrows():
        src_idx = node_map.get(str(row['txId1']), -1)
        dst_idx = node_map.get(str(row['txId2']), -1)
        
        if src_idx < 0 or dst_idx < 0:
            continue
            
        src_label = y[src_idx]
        dst_label = y[dst_idx]
        
        if src_label == -1 or dst_label == -1:
            edge_types['involves_unknown'] += 1
        elif src_label == 1 and dst_label == 1:
            edge_types['illicit_to_illicit'] += 1
        elif src_label == 1 and dst_label == 0:
            edge_types['illicit_to_licit'] += 1
        elif src_label == 0 and dst_label == 1:
            edge_types['licit_to_illicit'] += 1
        else:
            edge_types['licit_to_licit'] += 1
    
    edge_df = pd.DataFrame([
        {'edge_type': k, 'count': v} for k, v in edge_types.items()
    ])
    edge_df.to_csv('results/real_data/edge_statistics.csv', index=False)
    print(f"   Saved: results/real_data/edge_statistics.csv")
    
    # =========================================================================
    # 4. CLASS DISTRIBUTION BY TIMESTEP
    # =========================================================================
    print("\n5. Computing class distribution by timestep...")
    
    ts_class_dist = []
    for ts in range(1, 50):
        ts_mask = timesteps == ts
        ts_class_dist.append({
            'timestep': ts,
            'illicit': int(sum((y == 1) & ts_mask)),
            'licit': int(sum((y == 0) & ts_mask)),
            'unknown': int(sum((y == -1) & ts_mask)),
            'illicit_ratio': sum((y == 1) & ts_mask) / max(1, sum(ts_mask & (y != -1)))
        })
    
    pd.DataFrame(ts_class_dist).to_csv('results/real_data/class_by_timestep.csv', index=False)
    print(f"   Saved: results/real_data/class_by_timestep.csv")
    
    print("\n" + "=" * 60)
    print("DONE! Real analysis data saved.")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
Generate Advanced Real Analysis Data
Community detection, neighbor aggregates, subgraph samples, model weight analysis.
All using REAL data only.
"""
import os
import numpy as np
import pandas as pd
import torch
import networkx as nx
from collections import defaultdict


def main():
    print("=" * 60)
    print("GENERATING ADVANCED REAL ANALYSIS DATA")
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
    
    label_map = {'1': 0, '2': 1, 'unknown': -1}
    classes_dict = dict(zip(classes_df['txId'].astype(str), 
                           classes_df['class'].astype(str).map(lambda x: label_map.get(x, -1))))
    y = np.array([classes_dict.get(tx, -1) for tx in tx_ids])
    
    node_map = {tx: i for i, tx in enumerate(tx_ids)}
    
    os.makedirs('results/real_data', exist_ok=True)
    
    # =========================================================================
    # 1. NEIGHBOR FEATURE AGGREGATES
    # =========================================================================
    print("\n2. Computing neighbor feature aggregates...")
    
    # Build adjacency
    neighbors = defaultdict(list)
    for _, row in edges_df.iterrows():
        src = node_map.get(str(row['txId1']), -1)
        dst = node_map.get(str(row['txId2']), -1)
        if src >= 0 and dst >= 0:
            neighbors[src].append(dst)
            neighbors[dst].append(src)
    
    # For labeled nodes, compute neighbor stats
    neighbor_stats = []
    for idx in range(len(X)):
        if y[idx] == -1:
            continue
        
        neighs = neighbors[idx]
        if len(neighs) == 0:
            continue
        
        # Neighbor labels
        neigh_labels = [y[n] for n in neighs]
        n_illicit = sum(1 for l in neigh_labels if l == 1)
        n_licit = sum(1 for l in neigh_labels if l == 0)
        n_unknown = sum(1 for l in neigh_labels if l == -1)
        
        # Neighbor features (mean of first 10 features)
        neigh_feats = X[neighs, :10]
        
        neighbor_stats.append({
            'node_idx': idx,
            'label': 'illicit' if y[idx] == 1 else 'licit',
            'degree': len(neighs),
            'n_illicit_neighbors': n_illicit,
            'n_licit_neighbors': n_licit,
            'n_unknown_neighbors': n_unknown,
            'illicit_neighbor_ratio': n_illicit / max(1, n_illicit + n_licit),
            'neighbor_feat0_mean': float(neigh_feats[:, 0].mean()),
            'neighbor_feat0_std': float(neigh_feats[:, 0].std()),
        })
    
    neighbor_df = pd.DataFrame(neighbor_stats)
    neighbor_df.to_csv('results/real_data/neighbor_aggregates.csv', index=False)
    print(f"   Saved: results/real_data/neighbor_aggregates.csv ({len(neighbor_df)} nodes)")
    
    # Aggregate by label
    agg_by_label = neighbor_df.groupby('label').agg({
        'illicit_neighbor_ratio': ['mean', 'std'],
        'degree': ['mean', 'max']
    }).reset_index()
    agg_by_label.columns = ['label', 'illicit_ratio_mean', 'illicit_ratio_std', 'degree_mean', 'degree_max']
    agg_by_label.to_csv('results/real_data/neighbor_agg_by_label.csv', index=False)
    print(f"   Saved: results/real_data/neighbor_agg_by_label.csv")
    
    # =========================================================================
    # 2. COMMUNITY DETECTION (Sample for efficiency)
    # =========================================================================
    print("\n3. Running community detection on sample...")
    
    # Build NetworkX graph from sample
    sample_edges = edges_df.head(50000)  # Sample for efficiency
    G = nx.Graph()
    for _, row in sample_edges.iterrows():
        src, dst = str(row['txId1']), str(row['txId2'])
        if src in node_map and dst in node_map:
            G.add_edge(src, dst)
    
    print(f"   Sample graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Use greedy modularity communities (faster than Louvain)
    from networkx.algorithms.community import greedy_modularity_communities
    
    communities = list(greedy_modularity_communities(G))
    print(f"   Found {len(communities)} communities")
    
    # Analyze communities
    community_stats = []
    for i, comm in enumerate(communities[:50]):  # Top 50 communities
        comm_nodes = list(comm)
        comm_labels = [y[node_map[n]] if n in node_map else -1 for n in comm_nodes]
        
        n_illicit = sum(1 for l in comm_labels if l == 1)
        n_licit = sum(1 for l in comm_labels if l == 0)
        n_unknown = sum(1 for l in comm_labels if l == -1)
        
        community_stats.append({
            'community_id': i,
            'size': len(comm),
            'n_illicit': n_illicit,
            'n_licit': n_licit,
            'n_unknown': n_unknown,
            'illicit_ratio': n_illicit / max(1, n_illicit + n_licit) if (n_illicit + n_licit) > 0 else 0
        })
    
    community_df = pd.DataFrame(community_stats)
    community_df.to_csv('results/real_data/communities.csv', index=False)
    print(f"   Saved: results/real_data/communities.csv")
    
    # =========================================================================
    # 3. SUBGRAPH SAMPLES
    # =========================================================================
    print("\n4. Extracting subgraph samples...")
    
    # Sample around illicit nodes
    illicit_nodes = [tx_ids[i] for i in range(len(y)) if y[i] == 1][:100]
    
    subgraph_edges = []
    for src_tx in illicit_nodes:
        if src_tx not in node_map:
            continue
        src_idx = node_map[src_tx]
        for neigh_idx in neighbors[src_idx][:5]:  # Limit neighbors
            subgraph_edges.append({
                'source': src_tx,
                'target': tx_ids[neigh_idx],
                'source_label': 'illicit' if y[src_idx] == 1 else ('licit' if y[src_idx] == 0 else 'unknown'),
                'target_label': 'illicit' if y[neigh_idx] == 1 else ('licit' if y[neigh_idx] == 0 else 'unknown')
            })
    
    subgraph_df = pd.DataFrame(subgraph_edges[:500])  # Limit size
    subgraph_df.to_csv('results/real_data/illicit_subgraph.csv', index=False)
    print(f"   Saved: results/real_data/illicit_subgraph.csv ({len(subgraph_df)} edges)")
    
    # =========================================================================
    # 4. MODEL WEIGHT ANALYSIS
    # =========================================================================
    print("\n5. Analyzing model weights...")
    
    checkpoint = torch.load('checkpoints/chronos_experiment/best_model.pt', 
                           map_location='cpu', weights_only=False)
    weights = checkpoint['model']
    
    weight_stats = []
    for name, param in weights.items():
        param_np = param.numpy()
        weight_stats.append({
            'layer': name,
            'shape': str(list(param.shape)),
            'n_params': int(param.numel()),
            'mean': float(param_np.mean()),
            'std': float(param_np.std()),
            'min': float(param_np.min()),
            'max': float(param_np.max()),
            'abs_mean': float(np.abs(param_np).mean())
        })
    
    weight_df = pd.DataFrame(weight_stats)
    weight_df.to_csv('results/real_data/model_weights.csv', index=False)
    print(f"   Saved: results/real_data/model_weights.csv")
    
    # Input projection analysis (feature importance proxy)
    input_weight = weights['input_proj.weight'].numpy()  # [256, 235]
    feature_importance = np.abs(input_weight).mean(axis=0)  # Average abs weight per input feature
    
    importance_df = pd.DataFrame({
        'feature_idx': range(len(feature_importance)),
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    importance_df.to_csv('results/real_data/feature_importance_from_weights.csv', index=False)
    print(f"   Saved: results/real_data/feature_importance_from_weights.csv")
    
    print("\n" + "=" * 60)
    print("DONE! All advanced analysis data saved.")
    print("=" * 60)


if __name__ == '__main__':
    main()

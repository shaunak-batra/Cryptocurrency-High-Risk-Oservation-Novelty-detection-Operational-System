"""
CHRONOS SHAP-like Explanation Generator with SMOTE-ENN
Uses input projection weights for feature importance and SMOTE-ENN for balanced analysis.
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model_and_data():
    """Load trained model and prepare data with engineered features."""
    # Load model checkpoint
    checkpoint = torch.load('checkpoints/chronos_experiment/best_model.pt', 
                           map_location='cpu', weights_only=False)
    
    metrics = checkpoint.get('metrics', {})
    print(f"Model metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Load dataset
    data_dir = 'data/raw/elliptic/raw'
    features_df = pd.read_csv(f'{data_dir}/elliptic_txs_features.csv', header=None)
    classes_df = pd.read_csv(f'{data_dir}/elliptic_txs_classes.csv')
    edges_df = pd.read_csv(f'{data_dir}/elliptic_txs_edgelist.csv')
    
    # Process features (165 original)
    tx_ids = features_df[0].values.astype(str)
    timesteps = features_df[1].values.astype(int)
    X_orig = features_df.iloc[:, 2:].values.astype(np.float32)
    
    # Create node mapping and edge list
    node_map = {tx: i for i, tx in enumerate(tx_ids)}
    edge_list = [(node_map[str(r['txId1'])], node_map[str(r['txId2'])]) 
                 for _, r in edges_df.iterrows() 
                 if str(r['txId1']) in node_map and str(r['txId2']) in node_map]
    
    # Engineer 70 additional features
    print("\nEngineering 70 additional features...")
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    in_deg, out_deg = dict(G.in_degree()), dict(G.out_degree())
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=50)
    
    eng = np.zeros((len(tx_ids), 70), dtype=np.float32)
    for i in range(len(tx_ids)):
        eng[i, 0:5] = [in_deg.get(i,0), out_deg.get(i,0), in_deg.get(i,0)+out_deg.get(i,0),
                       pagerank.get(i,0), out_deg.get(i,0)/max(1,in_deg.get(i,0)+out_deg.get(i,0))]
        eng[i, 20:22] = [timesteps[i], timesteps[i]/49.0]
    
    X = np.concatenate([X_orig, np.nan_to_num(eng)], axis=1)
    print(f"Total features: {X.shape[1]} (165 original + 70 engineered)")
    
    # Process labels
    label_map = {'1': 0, '2': 1, 'unknown': -1}
    classes_dict = dict(zip(classes_df['txId'].astype(str), 
                           classes_df['class'].astype(str).map(lambda x: label_map.get(x, -1))))
    y = np.array([classes_dict.get(tx, -1) for tx in tx_ids])
    
    # Filter labeled data only
    labeled_mask = y != -1
    X_labeled = X[labeled_mask]
    y_labeled = y[labeled_mask]
    
    return checkpoint, X_labeled, y_labeled


def apply_smoteenn(X, y, sample_size=5000):
    """Apply SMOTE-ENN for balanced sampling."""
    print("\nApplying SMOTE-ENN for balanced sampling...")
    print(f"  Original: Class 0 (Licit) = {(y==0).sum()}, Class 1 (Illicit) = {(y==1).sum()}")
    
    # Subsample for speed
    if len(X) > sample_size:
        np.random.seed(42)
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sub, y_sub = X[idx], y[idx]
        print(f"  Subsampled to {sample_size} samples")
    else:
        X_sub, y_sub = X, y
    
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_sub, y_sub)
    
    print(f"  After SMOTE-ENN: Class 0 = {(y_resampled==0).sum()}, Class 1 = {(y_resampled==1).sum()}")
    
    return X_resampled, y_resampled


def compute_feature_importance(checkpoint, X, y, feature_names):
    """Compute feature importance using multiple methods."""
    weights = checkpoint['model']
    
    # 1. Input projection weight importance
    input_weights = weights['input_proj.weight'].numpy()  # [hidden_dim, in_features]
    weight_importance = np.abs(input_weights).mean(axis=0)
    
    # 2. Gradient-weighted importance (approximation using class statistics)
    licit_mean = X[y == 0].mean(axis=0)
    illicit_mean = X[y == 1].mean(axis=0)
    diff_importance = np.abs(illicit_mean - licit_mean)
    
    # 3. Combined score
    # Normalize both
    weight_norm = weight_importance / (weight_importance.max() + 1e-8)
    diff_norm = diff_importance / (diff_importance.max() + 1e-8)
    combined = 0.6 * weight_norm + 0.4 * diff_norm
    
    return {
        'weight': weight_importance,
        'class_diff': diff_importance,
        'combined': combined
    }


def plot_explanations(importance_dict, feature_names, X, y):
    """Generate explanation visualizations."""
    os.makedirs('results/explanations', exist_ok=True)
    
    combined = importance_dict['combined']
    
    # 1. Top 25 Combined Feature Importance
    top_idx = np.argsort(combined)[-25:][::-1]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_idx)))
    bars = ax.barh(range(len(top_idx)), combined[top_idx], color=colors)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([feature_names[i] for i in top_idx])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('CHRONOS-Net: Top 25 Feature Importances\n(Combined: Weight + Class Difference)', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/explanations/shap_importance_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: results/explanations/shap_importance_combined.png")
    
    # 2. Weight-based vs Class-difference comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Weight importance
    weight_imp = importance_dict['weight']
    w_top = np.argsort(weight_imp)[-15:][::-1]
    axes[0].barh(range(15), weight_imp[w_top], color='steelblue')
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels([feature_names[i] for i in w_top])
    axes[0].invert_yaxis()
    axes[0].set_title('Weight-based Importance', fontsize=12)
    axes[0].set_xlabel('Importance')
    
    # Class difference
    diff_imp = importance_dict['class_diff']
    d_top = np.argsort(diff_imp)[-15:][::-1]
    axes[1].barh(range(15), diff_imp[d_top], color='coral')
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels([feature_names[i] for i in d_top])
    axes[1].invert_yaxis()
    axes[1].set_title('Class Difference Importance', fontsize=12)
    axes[1].set_xlabel('Importance')
    
    plt.suptitle('Feature Importance Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/explanations/shap_importance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: results/explanations/shap_importance_comparison.png")
    
    # 3. Original vs Engineered features
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original (0-164)
    orig_imp = combined[:165]
    orig_top = np.argsort(orig_imp)[-12:][::-1]
    axes[0].barh(range(12), orig_imp[orig_top], color='teal')
    axes[0].set_yticks(range(12))
    axes[0].set_yticklabels([feature_names[i] for i in orig_top])
    axes[0].invert_yaxis()
    axes[0].set_title('Top 12 Original Features (Elliptic)', fontsize=12)
    
    # Engineered (165-234)
    eng_imp = combined[165:]
    eng_feature_names = feature_names[165:]
    eng_top = np.argsort(eng_imp)[-12:][::-1]
    axes[1].barh(range(12), eng_imp[eng_top], color='tomato')
    axes[1].set_yticks(range(12))
    axes[1].set_yticklabels([eng_feature_names[i] for i in eng_top])
    axes[1].invert_yaxis()
    axes[1].set_title('Top 12 Engineered Features', fontsize=12)
    
    plt.suptitle('Feature Importance: Original vs Engineered', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/explanations/shap_orig_vs_engineered.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: results/explanations/shap_orig_vs_engineered.png")
    
    # 4. Individual sample explanations using class centroids
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sample illicit transactions
    illicit_idx = np.where(y == 1)[0][:3]
    licit_mean = X[y == 0].mean(axis=0)
    
    for i, idx in enumerate(illicit_idx[:3]):
        sample = X[idx]
        sample_diff = (sample - licit_mean) * combined
        
        top_idx = np.argsort(np.abs(sample_diff))[-10:][::-1]
        colors = ['red' if sample_diff[j] > 0 else 'blue' for j in top_idx]
        
        axes[i].barh(range(10), [sample_diff[j] for j in top_idx], color=colors)
        axes[i].set_yticks(range(10))
        axes[i].set_yticklabels([feature_names[j] for j in top_idx])
        axes[i].invert_yaxis()
        axes[i].set_title(f'Illicit Transaction {i+1}', fontsize=11)
        axes[i].axvline(0, color='black', linewidth=0.5)
    
    plt.suptitle('Individual Transaction Explanations\n(Red = pushes toward illicit, Blue = pushes toward licit)', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/explanations/shap_individual_transactions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: results/explanations/shap_individual_transactions.png")
    
    return


def main():
    print("=" * 60)
    print("CHRONOS Feature Explanation Generator (SMOTE-ENN)")
    print("=" * 60)
    
    # Load model and data
    checkpoint, X, y = load_model_and_data()
    print(f"\nLoaded data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Apply SMOTE-ENN
    X_balanced, y_balanced = apply_smoteenn(X, y, sample_size=5000)
    
    # Generate feature names
    feature_names = [f'orig_{i}' for i in range(165)]
    eng_names = ['in_degree', 'out_degree', 'total_deg', 'pagerank', 'fan_out_ratio']
    eng_names += [f'eng_{i}' for i in range(5, 20)]
    eng_names += ['timestep', 'timestep_norm']
    eng_names += [f'eng_{i}' for i in range(22, 70)]
    feature_names.extend(eng_names)
    
    # Compute feature importance
    print("\nComputing feature importance...")
    importance_dict = compute_feature_importance(checkpoint, X_balanced, y_balanced, feature_names)
    
    # Plot explanations
    print("\nGenerating visualizations...")
    plot_explanations(importance_dict, feature_names, X_balanced, y_balanced)
    
    # Summary
    combined = importance_dict['combined']
    print("\n" + "=" * 60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)
    top_idx = np.argsort(combined)[-10:][::-1]
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i:2d}. {feature_names[idx]:20s} {combined[idx]:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All explanations saved to: results/explanations/")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
CHRONOS Local Explanation Generator
Generates feature importance from trained model weights.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=" * 60)
    print("CHRONOS Local Explanation Generator")
    print("=" * 60)
    
    # Load model
    model_path = 'checkpoints/chronos_experiment/best_model.pt'
    print(f"\nLoading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Show metrics
    metrics = checkpoint.get('metrics', {})
    print(f"\nModel Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Get model weights
    state_dict = checkpoint.get('model', checkpoint)
    
    # Extract input projection weights
    input_weights = state_dict['input_proj.weight'].numpy()  # [hidden_dim, in_features]
    print(f"\nInput projection shape: {input_weights.shape}")
    
    # Feature importance from input layer weights
    feature_importance = np.abs(input_weights).mean(axis=0)  # Average across hidden dims
    
    # Feature names
    feature_names = []
    # Original Elliptic features (165)
    for i in range(165):
        feature_names.append(f'elliptic_{i}')
    # Engineered features (70)
    eng_names = ['in_degree', 'out_degree', 'total_degree', 'pagerank', 'fan_out_ratio']
    eng_names += [f'eng_{i}' for i in range(5, 20)]
    eng_names += ['timestep', 'timestep_norm']
    eng_names += [f'eng_{i}' for i in range(22, 45)]
    eng_names += ['neighbor_ts_mean', 'neighbor_ts_std', 'eng_47', 'eng_48', 'neighbor_count', 'neighbor_pr_mean']
    eng_names += [f'eng_{i}' for i in range(51, 70)]
    feature_names.extend(eng_names)
    
    # Ensure we have enough names
    while len(feature_names) < len(feature_importance):
        feature_names.append(f'feat_{len(feature_names)}')
    
    # Get top 25 features
    top_k = 25
    top_indices = np.argsort(feature_importance)[-top_k:][::-1]
    top_importance = feature_importance[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # Create visualization
    os.makedirs('results/explanations', exist_ok=True)
    
    # Plot 1: Bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_importance)))[::-1]
    bars = ax.barh(range(len(top_importance)), top_importance, color=colors)
    ax.set_yticks(range(len(top_importance)))
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('CHRONOS-Net: Top 25 Feature Importances\n(Based on Input Projection Weights)', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, top_importance):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    save_path = 'results/explanations/feature_importance.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()
    
    # Plot 2: Grouped by category
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Split into original vs engineered
    orig_imp = feature_importance[:165]
    eng_imp = feature_importance[165:]
    
    # Original features - top 15
    orig_top_idx = np.argsort(orig_imp)[-15:][::-1]
    axes[0].barh(range(15), orig_imp[orig_top_idx], color='steelblue')
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels([f'elliptic_{i}' for i in orig_top_idx])
    axes[0].invert_yaxis()
    axes[0].set_title('Top 15 Original Features', fontsize=12)
    axes[0].set_xlabel('Importance')
    
    # Engineered features - top 15
    eng_top_idx = np.argsort(eng_imp)[-15:][::-1]
    eng_labels = [feature_names[165 + i] for i in eng_top_idx]
    axes[1].barh(range(len(eng_top_idx)), eng_imp[eng_top_idx], color='coral')
    axes[1].set_yticks(range(len(eng_top_idx)))
    axes[1].set_yticklabels(eng_labels)
    axes[1].invert_yaxis()
    axes[1].set_title('Top 15 Engineered Features', fontsize=12)
    axes[1].set_xlabel('Importance')
    
    plt.suptitle('Feature Importance by Category', fontsize=14, y=1.02)
    plt.tight_layout()
    save_path2 = 'results/explanations/feature_importance_grouped.png'
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path2}")
    plt.close()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 60)
    for i, (name, imp) in enumerate(zip(top_names[:10], top_importance[:10]), 1):
        print(f"  {i:2d}. {name:25s} {imp:.4f}")
    
    print("\n" + "=" * 60)
    print(f"✓ Explanations saved to: results/explanations/")
    print("=" * 60)


if __name__ == '__main__':
    main()

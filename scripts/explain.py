"""
Generate explanations for CHRONOS predictions.

Provides comprehensive multi-method explanations:
1. Counterfactual explanations (novel contribution)
2. SHAP feature importance
3. GAT attention visualization
4. Natural language summaries
"""

import os
import sys
import argparse
import torch
import yaml
from pathlib import Path

# Add chronos to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos.data.loader import load_elliptic_dataset
from chronos.data.preprocessing import normalize_features, create_temporal_sequences
from chronos.models.chronos_net import CHRONOSNet
from chronos.explainability import (
    TemporalCounterfactualExplainer,
    CHRONOSSHAPExplainer,
    AttentionVisualizer,
    generate_explanation
)


def load_model(checkpoint_path: str, device: str = 'cpu') -> CHRONOSNet:
    """Load trained model from checkpoint."""
    print(f"[INFO] Loading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer model architecture
    model_state = checkpoint['model_state_dict']
    in_features = model_state['input_projection.weight'].size(1)
    hidden_dim = model_state['input_projection.weight'].size(0)

    from chronos.models.chronos_net import create_chronos_net
    model = create_chronos_net(
        in_features=in_features,
        hidden_dim=hidden_dim,
        config=checkpoint.get('config', {})
    )

    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    print(f"[OK] Model loaded")
    return model


def main():
    parser = argparse.ArgumentParser(description='Generate CHRONOS explanations')
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
        '--node-idx',
        type=int,
        required=True,
        help='Node index to explain'
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['counterfactual', 'shap', 'attention', 'all'],
        default=['all'],
        help='Explanation methods to use (default: all)'
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
        default='results/explanations',
        help='Output directory'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CHRONOS Explanation Generation")
    print("=" * 70)
    print(f"Node: {args.node_idx}")
    print(f"Device: {args.device}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which methods to use
    if 'all' in args.methods:
        methods = ['counterfactual', 'shap', 'attention']
    else:
        methods = args.methods

    # ========================================================================
    # Load Model & Data
    # ========================================================================
    model = load_model(args.checkpoint, device=args.device)

    print("[INFO] Loading dataset...")
    data = load_elliptic_dataset(
        root=args.data_root,
        timestep=None,
        include_unknown=True
    )

    data = normalize_features(data, method='standard', train_mask=data.train_mask)
    data = data.to(args.device)

    temporal_sequences = create_temporal_sequences(
        data,
        window_sizes=[1, 5, 15, 30],
        include_future=False
    )

    # Verify node is in test set
    if not data.test_mask[args.node_idx]:
        print(f"[WARNING] Node {args.node_idx} is not in test set")

    # ========================================================================
    # Get Prediction
    # ========================================================================
    print(f"\n[INFO] Getting prediction for node {args.node_idx}...")

    with torch.no_grad():
        logits, attention_weights = model(
            x=data.x,
            edge_index=data.edge_index,
            temporal_sequences=temporal_sequences,
            return_attention=True
        )

        proba = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1)[args.node_idx].item()
        pred_proba = proba[args.node_idx, pred_class].item()

    true_class = data.y[args.node_idx].item()

    print(f"      True label:       {'Illicit' if true_class == 1 else 'Licit'}")
    print(f"      Predicted:        {'Illicit' if pred_class == 1 else 'Licit'}")
    print(f"      Confidence:       {pred_proba:.2%}")
    print(f"      Correct:          {'Yes' if pred_class == true_class else 'No'}")

    # ========================================================================
    # Generate Explanations
    # ========================================================================
    cf_explanation = None
    shap_explanation = None
    attention_analysis = None

    # 1. Counterfactual Explanation
    if 'counterfactual' in methods:
        print("\n[1/3] Generating counterfactual explanation...")
        try:
            cf_explainer = TemporalCounterfactualExplainer(
                model=model,
                data=data,
                device=args.device
            )

            cf_explanation = cf_explainer.explain(
                node_idx=args.node_idx,
                target_class=1 - pred_class,  # Flip prediction
                num_counterfactuals=3
            )

            print(f"      Generated {len(cf_explanation['changes'])} counterfactuals")
            print(f"      Validity: {cf_explanation['validity']:.1%}")

        except Exception as e:
            print(f"[ERROR] Counterfactual generation failed: {e}")

    # 2. SHAP Explanation
    if 'shap' in methods:
        print("\n[2/3] Computing SHAP values...")
        try:
            shap_explainer = CHRONOSSHAPExplainer(
                model=model,
                data=data,
                background_samples=100,
                device=args.device
            )

            shap_explanation = shap_explainer.explain_node(
                node_idx=args.node_idx
            )

            print(f"      Top 5 features:")
            for feature_name, shap_value in shap_explanation['top_features'][:5]:
                print(f"        {feature_name}: {shap_value:+.4f}")

        except Exception as e:
            print(f"[ERROR] SHAP computation failed: {e}")

    # 3. Attention Analysis
    if 'attention' in methods:
        print("\n[3/3] Analyzing attention weights...")
        try:
            attention_viz = AttentionVisualizer(
                model=model,
                data=data,
                device=args.device
            )

            attention_analysis = attention_viz.analyze_node_attention(
                node_idx=args.node_idx,
                attention_weights=attention_weights
            )

            print(f"      Top 3 neighbors by attention:")
            for neighbor_idx, weight in attention_analysis['top_neighbors'][:3]:
                print(f"        Node {neighbor_idx}: {weight:.4f}")

        except Exception as e:
            print(f"[ERROR] Attention analysis failed: {e}")

    # ========================================================================
    # Generate Natural Language Explanation
    # ========================================================================
    print("\n[INFO] Generating natural language explanation...")

    explanation_text = generate_explanation(
        node_idx=args.node_idx,
        prediction_proba=pred_proba,
        prediction_class=pred_class,
        shap_explanation=shap_explanation,
        cf_explanation=cf_explanation,
        attention_analysis=attention_analysis
    )

    # Print explanation
    print("\n" + "=" * 70)
    print(explanation_text)
    print("=" * 70)

    # ========================================================================
    # Save Results
    # ========================================================================
    print(f"\n[INFO] Saving results...")

    # Save text explanation
    text_file = output_dir / f'explanation_node_{args.node_idx}.txt'
    with open(text_file, 'w') as f:
        f.write(explanation_text)
    print(f"      Saved: {text_file}")

    # Save structured explanation
    structured_explanation = {
        'node_idx': args.node_idx,
        'prediction': {
            'class': int(pred_class),
            'probability': float(pred_proba),
            'label': 'illicit' if pred_class == 1 else 'licit'
        },
        'ground_truth': {
            'class': int(true_class),
            'label': 'illicit' if true_class == 1 else 'licit'
        },
        'counterfactual': cf_explanation,
        'shap': shap_explanation,
        'attention': attention_analysis
    }

    yaml_file = output_dir / f'explanation_node_{args.node_idx}.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(structured_explanation, f, default_flow_style=False)
    print(f"      Saved: {yaml_file}")

    print("\n" + "=" * 70)
    print("EXPLANATION COMPLETE")
    print("=" * 70)
    print(f"\nTo explain another transaction, run:")
    print(f"  python scripts/explain.py --node-idx <NODE_ID>")


if __name__ == '__main__':
    main()

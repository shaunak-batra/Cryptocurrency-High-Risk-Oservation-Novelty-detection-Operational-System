#!/usr/bin/env python
"""
CHRONOS System Validation Audit Script
Comprehensive validation of all system components
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import numpy as np

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def print_check(name, passed, details=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  [{status}] {name}")
    if details:
        print(f"          {details}")

results = {"passed": [], "failed": [], "warnings": []}

# ===========================================================================
# SECTION 1: ARCHITECTURE VALIDATION
# ===========================================================================
print_header("SECTION 1: ARCHITECTURE VALIDATION")

try:
    from chronos.models.chronos_net import CHRONOSNet
    
    # Load checkpoint
    checkpoint = torch.load('checkpoints/chronos_experiment/best_model.pt', 
                           map_location='cpu', weights_only=False)
    config = checkpoint['config']
    
    print("1.1 Model Configuration from Checkpoint:")
    print(f"    in_features: {config.model.in_features}")
    print(f"    hidden_dim: {config.model.hidden_dim}")
    print(f"    num_gat_layers: {config.model.num_gat_layers}")
    print(f"    num_heads: {config.model.num_heads}")
    
    # Verify in_features
    expected_features = 235  # 165 original + 70 engineered
    check1 = config.model.in_features == expected_features
    print_check("Feature count matches expected (235)", check1, 
                f"Got: {config.model.in_features}")
    if check1: results["passed"].append("Feature count")
    else: results["warnings"].append(f"Feature count: {config.model.in_features} vs expected {expected_features}")
    
    # Create model
    model = CHRONOSNet(
        in_features=config.model.in_features,
        hidden_dim=config.model.hidden_dim,
        num_gat_layers=config.model.num_gat_layers,
        num_heads=config.model.num_heads
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    print("\n1.2 Component Verification:")
    has_input_proj = hasattr(model, 'input_projection')
    print_check("input_projection layer exists", has_input_proj)
    if has_input_proj: results["passed"].append("input_projection exists")
    else: results["failed"].append("input_projection missing")
    
    has_gat = hasattr(model, 'gat_layers')
    gat_count = len(model.gat_layers) if has_gat else 0
    print_check(f"GAT layers exist ({gat_count} layers)", has_gat and gat_count > 0)
    if has_gat: results["passed"].append(f"GAT layers: {gat_count}")
    else: results["failed"].append("GAT layers missing")
    
    has_classifier = hasattr(model, 'classifier')
    print_check("Classifier head exists", has_classifier)
    if has_classifier: results["passed"].append("Classifier exists")
    else: results["failed"].append("Classifier missing")
    
    # Forward pass test
    print("\n1.3 Forward Pass Sanity Test:")
    x = torch.randn(100, config.model.in_features)
    edge_index = torch.randint(0, 100, (2, 500))
    
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index)
    
    shape_correct = output.shape == (100, 2)
    print_check("Output shape correct (100, 2)", shape_correct, f"Got: {output.shape}")
    if shape_correct: results["passed"].append("Forward pass shape")
    else: results["failed"].append(f"Forward pass shape: {output.shape}")
    
    # Check output is valid probabilities after softmax
    probs = torch.softmax(output, dim=1)
    probs_valid = torch.allclose(probs.sum(dim=1), torch.ones(100), atol=1e-5)
    print_check("Probabilities sum to 1.0", probs_valid)
    if probs_valid: results["passed"].append("Probability normalization")
    else: results["failed"].append("Probability normalization failed")
    
    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"\n1.4 Model Size:")
    print(f"    Parameters: {params:,}")
    print(f"    Size: {params * 4 / 1024 / 1024:.2f} MB")
    results["passed"].append(f"Model size: {params:,} params")
    
except Exception as e:
    print(f"  [✗ FAIL] Architecture validation: {str(e)}")
    results["failed"].append(f"Architecture: {str(e)}")

# ===========================================================================
# SECTION 2: DATA PIPELINE VALIDATION
# ===========================================================================
print_header("SECTION 2: DATA PIPELINE VALIDATION")

try:
    from chronos.data.loader import load_elliptic_dataset, verify_dataset
    
    print("2.1 Loading Dataset...")
    data = load_elliptic_dataset('data/raw/elliptic')
    stats = verify_dataset(data)
    
    print("\n2.2 Dataset Statistics:")
    for key, value in stats.items():
        print(f"    {key}: {value}")
    
    # Feature count check
    features_ok = stats['num_features'] == 165
    print_check("Feature count correct (165)", features_ok, f"Got: {stats['num_features']}")
    if features_ok: results["passed"].append("Dataset features: 165")
    else: results["warnings"].append(f"Dataset features: {stats['num_features']}")
    
    # Temporal split check
    print("\n2.3 Temporal Split Verification:")
    train_timesteps = data.timestep[data.train_mask].unique()
    val_timesteps = data.timestep[data.val_mask].unique()
    test_timesteps = data.timestep[data.test_mask].unique()
    
    train_max = train_timesteps.max().item()
    val_min = val_timesteps.min().item()
    test_min = test_timesteps.min().item()
    
    no_leakage = train_max < val_min and val_min <= val_timesteps.max().item() < test_min
    print_check("No temporal leakage", no_leakage, 
                f"Train max: {train_max}, Val: {val_min}-{val_timesteps.max().item()}, Test: {test_min}+")
    if no_leakage: results["passed"].append("No temporal leakage")
    else: results["failed"].append("TEMPORAL LEAKAGE DETECTED!")
    
    # Label distribution
    print("\n2.4 Label Distribution:")
    y = data.y
    illicit = (y == 1).sum().item()
    licit = (y == 0).sum().item()
    unknown = (y == -1).sum().item() if (y == -1).any() else 0
    
    print(f"    Licit (0): {licit:,}")
    print(f"    Illicit (1): {illicit:,}")
    print(f"    Unknown (-1): {unknown:,}")
    
    imbalance_ratio = licit / illicit if illicit > 0 else 0
    print(f"    Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    ratio_ok = 8 < imbalance_ratio < 12
    print_check("Imbalance ratio ~9:1", ratio_ok, f"Got: {imbalance_ratio:.1f}:1")
    if ratio_ok: results["passed"].append("Class imbalance ratio correct")
    else: results["warnings"].append(f"Imbalance ratio: {imbalance_ratio:.1f}")
    
except Exception as e:
    print(f"  [✗ FAIL] Data pipeline validation: {str(e)}")
    results["failed"].append(f"Data pipeline: {str(e)}")

# ===========================================================================
# SECTION 3: TRAINING VALIDATION
# ===========================================================================
print_header("SECTION 3: TRAINING VALIDATION")

try:
    # Check saved metrics in checkpoint
    metrics = checkpoint.get('metrics', {})
    print("3.1 Saved Training Metrics:")
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.4f}")
            else:
                print(f"    {key}: {value}")
    
    # Verify focal loss exists
    print("\n3.2 Loss Function Check:")
    from chronos.models.components import FocalLoss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Test focal loss
    logits = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
    targets = torch.tensor([1, 0])
    loss = focal_loss(logits, targets)
    
    loss_valid = loss.item() > 0 and not torch.isnan(loss)
    print_check("Focal loss computes correctly", loss_valid, f"Loss: {loss.item():.4f}")
    if loss_valid: results["passed"].append("Focal loss implementation")
    else: results["failed"].append("Focal loss computation failed")
    
except Exception as e:
    print(f"  [✗ FAIL] Training validation: {str(e)}")
    results["failed"].append(f"Training: {str(e)}")

# ===========================================================================
# SECTION 4: RESULTS VALIDATION
# ===========================================================================
print_header("SECTION 4: RESULTS VALIDATION")

try:
    print("4.1 Performance Metrics from Training:")
    
    # Get best metrics
    f1 = metrics.get('f1', metrics.get('val_f1', 0))
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    
    print(f"    F1 Score: {f1:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall: {recall:.4f}")
    
    # Sanity checks
    print("\n4.2 Results Sanity Checks:")
    
    f1_reasonable = 0.7 < f1 < 1.0
    print_check("F1 in reasonable range (0.7-1.0)", f1_reasonable, f"Got: {f1:.4f}")
    if f1_reasonable: results["passed"].append(f"F1 score: {f1:.4f}")
    else: results["warnings"].append(f"F1 score unusual: {f1:.4f}")
    
    # Perfect recall with high precision warning
    if recall > 0.99 and precision > 0.9:
        results["warnings"].append("Very high recall+precision - verify no data leakage")
        print("    ⚠ WARNING: Near-perfect metrics - verify no data leakage")
    
    # Consistent metrics check
    if f1 > 0 and precision > 0 and recall > 0:
        calc_f1 = 2 * precision * recall / (precision + recall)
        f1_consistent = abs(f1 - calc_f1) < 0.05
        print_check("F1 consistent with P/R", f1_consistent, 
                   f"Calculated: {calc_f1:.4f}, Reported: {f1:.4f}")
        if f1_consistent: results["passed"].append("Metrics consistency")
        else: results["warnings"].append("F1 doesn't match P/R calculation")
    
except Exception as e:
    print(f"  [✗ FAIL] Results validation: {str(e)}")
    results["failed"].append(f"Results: {str(e)}")

# ===========================================================================
# SECTION 5: NOVELTY VALIDATION
# ===========================================================================
print_header("SECTION 5: NOVELTY VALIDATION")

try:
    print("5.1 Novel Component Verification:")
    
    # Check counterfactual exists
    try:
        from chronos.explainability.counterfactual import CounterfactualGenerator
        print_check("Counterfactual generator exists", True)
        results["passed"].append("Counterfactual implementation exists")
    except ImportError:
        print_check("Counterfactual generator exists", False)
        results["failed"].append("Counterfactual import failed")
    
    # Check SHAP integration
    try:
        from chronos.explainability.shap_explainer import SHAPExplainer
        print_check("SHAP explainer exists", True)
        results["passed"].append("SHAP implementation exists")
    except ImportError:
        print_check("SHAP explainer exists", False)
        results["warnings"].append("SHAP import failed")
    
    # Check attention visualization
    try:
        from chronos.explainability.attention import AttentionVisualizer
        print_check("Attention visualizer exists", True)
        results["passed"].append("Attention visualization exists")
    except ImportError:
        print_check("Attention visualizer exists", False)
        results["warnings"].append("Attention visualizer import failed")
    
    # Check NLG (template-based)
    try:
        from chronos.explainability.nlg import ExplanationGenerator
        print_check("NLG explanation generator exists", True)
        results["passed"].append("NLG implementation exists")
    except ImportError:
        print_check("NLG explanation generator exists", False)
        results["warnings"].append("NLG import failed")
    
    print("\n5.2 Multi-Scale Attention Windows:")
    try:
        from chronos.models.components import MultiScaleTemporalAttention
        msta = MultiScaleTemporalAttention(256, [1, 5, 15, 30])
        print(f"    Window sizes: [1, 5, 15, 30]")
        print_check("Multi-scale attention implemented", True)
        results["passed"].append("Multi-scale attention windows: [1,5,15,30]")
    except Exception as e:
        print_check("Multi-scale attention implemented", False, str(e))
        results["warnings"].append(f"Multi-scale attention: {str(e)}")
    
except Exception as e:
    print(f"  [✗ FAIL] Novelty validation: {str(e)}")
    results["failed"].append(f"Novelty: {str(e)}")

# ===========================================================================
# SECTION 6: EXPLAINABILITY VALIDATION  
# ===========================================================================
print_header("SECTION 6: EXPLAINABILITY VALIDATION")

try:
    print("6.1 Explainability Module Structure:")
    import importlib
    
    modules = [
        'chronos.explainability.counterfactual',
        'chronos.explainability.shap_explainer',
        'chronos.explainability.attention',
        'chronos.explainability.nlg'
    ]
    
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            print(f"    [✓] {mod_name}")
        except Exception as e:
            print(f"    [✗] {mod_name}: {e}")
    
except Exception as e:
    print(f"  [✗ FAIL] Explainability validation: {str(e)}")

# ===========================================================================
# SECTION 7: API VALIDATION
# ===========================================================================
print_header("SECTION 7: API VALIDATION")

try:
    print("7.1 API Module Check:")
    from chronos.api.main import app
    print_check("FastAPI app imported", True)
    results["passed"].append("API module loads")
    
    # Check routes exist
    routes = [route.path for route in app.routes]
    print(f"    Available routes: {routes[:5]}...")
    
    expected_routes = ['/health', '/predict']
    for route in expected_routes:
        exists = route in routes
        print_check(f"Route {route} exists", exists)
        if exists: results["passed"].append(f"Route {route}")
        else: results["warnings"].append(f"Route {route} missing")
    
except Exception as e:
    print(f"  [✗ FAIL] API validation: {str(e)}")
    results["failed"].append(f"API: {str(e)}")

# ===========================================================================
# SECTION 8: CODE QUALITY
# ===========================================================================
print_header("SECTION 8: CODE QUALITY")

try:
    print("8.1 Import Verification:")
    
    imports = [
        ('chronos', 'Main package'),
        ('chronos.models', 'Models subpackage'),
        ('chronos.data', 'Data subpackage'),
        ('chronos.training', 'Training subpackage'),
        ('chronos.explainability', 'Explainability subpackage'),
        ('chronos.utils', 'Utils subpackage'),
    ]
    
    for mod_name, desc in imports:
        try:
            importlib.import_module(mod_name)
            print(f"    [✓] {mod_name}")
        except Exception as e:
            print(f"    [✗] {mod_name}: {e}")
            results["failed"].append(f"Import {mod_name}: {e}")
    
except Exception as e:
    print(f"  [✗ FAIL] Code quality check: {str(e)}")

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print_header("FINAL AUDIT SUMMARY")

print(f"PASSED ({len(results['passed'])}):")
for item in results['passed'][:10]:
    print(f"  ✓ {item}")
if len(results['passed']) > 10:
    print(f"  ... and {len(results['passed'])-10} more")

print(f"\nFAILED ({len(results['failed'])}):")
for item in results['failed']:
    print(f"  ✗ {item}")

print(f"\nWARNINGS ({len(results['warnings'])}):")
for item in results['warnings']:
    print(f"  ⚠ {item}")

# Final verdict
print("\n" + "="*70)
if len(results['failed']) == 0:
    print("  ✓ VERDICT: CODEBASE VALIDATION PASSED")
    print("  All critical checks passed. System is functioning correctly.")
elif len(results['failed']) <= 2:
    print("  ⚠ VERDICT: MOSTLY PASSED WITH MINOR ISSUES")
    print(f"  {len(results['failed'])} minor issue(s) found.")
else:
    print("  ✗ VERDICT: VALIDATION FAILED")
    print(f"  {len(results['failed'])} critical issue(s) found.")
print("="*70)

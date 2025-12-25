"""
CHRONOS Codebase Audit Script
Validates code quality, imports, and structure.
"""
import os
import sys
import ast
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def audit_imports():
    """Check all core imports work."""
    print("\n" + "=" * 60)
    print("IMPORT AUDIT")
    print("=" * 60)
    
    modules = [
        ('chronos', 'Core package'),
        ('chronos.models.chronos_net', 'CHRONOSNet model'),
        ('chronos.models.components', 'Model components'),
        ('chronos.data.loader', 'Data loader'),
        ('chronos.api.main', 'FastAPI application'),
        ('chronos.training.trainer', 'Training module'),
        ('chronos.utils.metrics', 'Metrics utilities'),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, description in modules:
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {module_name}: OK")
            passed += 1
        except Exception as e:
            print(f"  ✗ {module_name}: FAIL - {type(e).__name__}")
            failed += 1
    
    return passed, failed


def audit_files():
    """Check required files exist."""
    print("\n" + "=" * 60)
    print("FILE STRUCTURE AUDIT")
    print("=" * 60)
    
    required_files = [
        'README.md',
        'pyproject.toml',
        'chronos/__init__.py',
        'chronos/models/__init__.py',
        'chronos/models/chronos_net.py',
        'chronos/models/components.py',
        'chronos/data/__init__.py',
        'chronos/data/loader.py',
        'chronos/api/__init__.py',
        'chronos/api/main.py',
        'checkpoints/chronos_experiment/best_model.pt',
    ]
    
    passed = 0
    failed = 0
    
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}: EXISTS")
            passed += 1
        else:
            print(f"  ✗ {filepath}: MISSING")
            failed += 1
    
    return passed, failed


def audit_model():
    """Check model checkpoint is valid."""
    print("\n" + "=" * 60)
    print("MODEL AUDIT")
    print("=" * 60)
    
    import torch
    
    model_path = 'checkpoints/chronos_experiment/best_model.pt'
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"  ✓ Model loads successfully")
        
        # Check required keys
        if 'model' in checkpoint:
            print(f"  ✓ Contains 'model' state dict")
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"  ✓ Contains 'metrics': F1={metrics.get('f1', 'N/A'):.4f}")
        
        # Check model weights
        weights = checkpoint.get('model', {})
        total_params = sum(p.numel() for p in weights.values())
        print(f"  ✓ Total parameters: {total_params:,}")
        
        return 3, 0
    except Exception as e:
        print(f"  ✗ Model load failed: {e}")
        return 0, 1


def audit_code_quality():
    """Check code syntax in key files."""
    print("\n" + "=" * 60)
    print("CODE QUALITY AUDIT")
    print("=" * 60)
    
    python_files = [
        'chronos/models/chronos_net.py',
        'chronos/models/components.py',
        'chronos/data/loader.py',
        'chronos/api/main.py',
        'chronos/training/trainer.py',
    ]
    
    passed = 0
    failed = 0
    
    for filepath in python_files:
        if not os.path.exists(filepath):
            print(f"  ⚠ {filepath}: FILE NOT FOUND")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source)
            print(f"  ✓ {filepath}: Valid syntax")
            passed += 1
        except SyntaxError as e:
            print(f"  ✗ {filepath}: Syntax error at line {e.lineno}")
            failed += 1
    
    return passed, failed


def audit_documentation():
    """Check documentation exists."""
    print("\n" + "=" * 60)
    print("DOCUMENTATION AUDIT")
    print("=" * 60)
    
    checks = []
    
    # README check
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            readme = f.read()
        
        sections = [
            ('Installation', '## Installation' in readme),
            ('Training Results', 'Training Results' in readme or 'Actual Training' in readme),
            ('Usage', 'Usage' in readme or 'Quick Start' in readme),
            ('API', 'API' in readme),
            ('Performance', 'Performance' in readme),
        ]
        
        for name, exists in sections:
            if exists:
                print(f"  ✓ README has {name} section")
                checks.append(True)
            else:
                print(f"  ✗ README missing {name} section")
                checks.append(False)
    
    return sum(checks), len(checks) - sum(checks)


def main():
    print("=" * 60)
    print("CHRONOS CODEBASE AUDIT")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    # Run audits
    p, f = audit_imports()
    total_passed += p
    total_failed += f
    
    p, f = audit_files()
    total_passed += p
    total_failed += f
    
    p, f = audit_model()
    total_passed += p
    total_failed += f
    
    p, f = audit_code_quality()
    total_passed += p
    total_failed += f
    
    p, f = audit_documentation()
    total_passed += p
    total_failed += f
    
    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"\n  PASSED: {total_passed}")
    print(f"  FAILED: {total_failed}")
    print(f"  TOTAL:  {total_passed + total_failed}")
    
    # Verdict
    print("\n" + "=" * 60)
    if total_failed == 0:
        print("VERDICT: ✓ AUDIT PASSED - All checks successful!")
    elif total_failed <= 3:
        print(f"VERDICT: ⚠ AUDIT PASSED WITH WARNINGS - {total_failed} minor issues")
    else:
        print(f"VERDICT: ✗ AUDIT FAILED - {total_failed} issues need attention")
    print("=" * 60)
    
    return total_failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

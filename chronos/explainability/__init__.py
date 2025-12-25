"""
Explainability module for CHRONOS.

Provides multiple complementary explanation methods:
1. Counterfactual explanations (novel contribution)
2. SHAP feature importance
3. GAT attention visualization
4. Natural language generation

This is CHRONOS's core novel contribution: first system combining
temporal GNNs with counterfactual explanations for cryptocurrency AML.
"""

from .counterfactual import (
    TemporalCounterfactualExplainer,
    explain_node
)

from .shap_explainer import (
    CHRONOSSHAPExplainer,
    explain_with_shap
)

from .attention import (
    AttentionVisualizer,
    visualize_attention_for_node
)

from .nlg import (
    ExplanationGenerator,
    generate_explanation
)

__all__ = [
    # Counterfactual
    'TemporalCounterfactualExplainer',
    'explain_node',
    # SHAP
    'CHRONOSSHAPExplainer',
    'explain_with_shap',
    # Attention
    'AttentionVisualizer',
    'visualize_attention_for_node',
    # Natural Language Generation
    'ExplanationGenerator',
    'generate_explanation'
]

"""
Natural Language Generation for CHRONOS explanations.

Generates human-readable explanations combining counterfactual, SHAP,
and attention information. Patent-safe: avoids "narrative generation"
terminology (Mastercard US20220020026A1).

Uses template-based generation for EU AI Act Article 13 compliance.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class ExplanationGenerator:
    """
    Template-based explanation generator for CHRONOS.

    Combines multiple explanation types into coherent, human-readable text.
    """

    def __init__(self):
        """Initialize explanation generator."""
        self.risk_thresholds = {
            'CRITICAL': 0.90,
            'HIGH': 0.75,
            'MEDIUM': 0.50,
            'LOW': 0.0
        }

    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level from probability."""
        for level, threshold in self.risk_thresholds.items():
            if probability >= threshold:
                return level
        return 'LOW'

    def generate_prediction_summary(
        self,
        node_idx: int,
        prediction_proba: float,
        prediction_class: int
    ) -> str:
        """
        Generate summary of prediction.

        Parameters
        ----------
        node_idx : int
            Transaction node
        prediction_proba : float
            Prediction probability
        prediction_class : int
            Predicted class (0=licit, 1=illicit)

        Returns
        -------
        summary : str
            Prediction summary text
        """
        class_name = 'ILLICIT' if prediction_class == 1 else 'LICIT'
        risk_level = self._get_risk_level(prediction_proba)
        confidence = prediction_proba * 100

        text = f"Transaction #{node_idx} Analysis\n"
        text += "=" * 70 + "\n\n"
        text += f"Classification: {class_name}\n"
        text += f"Confidence: {confidence:.1f}%\n"
        text += f"Risk Level: {risk_level}\n\n"

        # Risk level explanation
        if risk_level == 'CRITICAL':
            text += "⚠️  CRITICAL RISK: This transaction exhibits strong indicators of illicit activity.\n"
            text += "Immediate investigation and potential regulatory reporting recommended.\n\n"
        elif risk_level == 'HIGH':
            text += "⚠️  HIGH RISK: This transaction shows significant suspicious patterns.\n"
            text += "Enhanced due diligence and monitoring recommended.\n\n"
        elif risk_level == 'MEDIUM':
            text += "⚠️  MEDIUM RISK: This transaction displays some concerning characteristics.\n"
            text += "Standard monitoring and review recommended.\n\n"
        else:
            text += "✓ LOW RISK: This transaction appears to follow normal patterns.\n"
            text += "Routine processing may continue.\n\n"

        return text

    def generate_feature_explanation(
        self,
        shap_explanation: Dict,
        top_n: int = 5
    ) -> str:
        """
        Generate explanation from SHAP feature importance.

        Parameters
        ----------
        shap_explanation : Dict
            SHAP explanation dictionary
        top_n : int
            Number of top features to include

        Returns
        -------
        explanation : str
            Feature importance explanation
        """
        text = "Key Contributing Factors:\n"
        text += "-" * 70 + "\n\n"

        top_features = shap_explanation['top_features'][:top_n]

        for rank, (feature_name, shap_value) in enumerate(top_features, 1):
            impact = "increases" if shap_value > 0 else "decreases"
            magnitude = abs(shap_value)

            # Describe magnitude
            if magnitude > 0.1:
                strength = "strongly"
            elif magnitude > 0.05:
                strength = "moderately"
            else:
                strength = "slightly"

            # Generate human-readable feature description
            feature_desc = self._describe_feature(feature_name)

            text += f"{rank}. {feature_desc} {strength} {impact} risk "
            text += f"(SHAP: {shap_value:+.4f})\n"

        text += "\n"
        return text

    def _describe_feature(self, feature_name: str) -> str:
        """Convert feature name to human-readable description."""
        # Mapping of technical names to descriptions
        descriptions = {
            # Graph topology
            'pagerank': 'Network centrality',
            'betweenness': 'Bridge position in network',
            'clustering_coef': 'Clustering with neighbors',
            'in_degree': 'Number of incoming transactions',
            'out_degree': 'Number of outgoing transactions',
            'total_degree': 'Total transaction count',

            # Temporal patterns
            'tx_freq_7d': 'Recent transaction frequency',
            'tx_freq_30d': 'Monthly transaction volume',
            'burstiness': 'Transaction timing irregularity',
            'tx_velocity': 'Transaction velocity change',
            'time_since_first': 'Account age',

            # Amount patterns
            'total_sent': 'Total amount sent',
            'total_received': 'Total amount received',
            'net_flow': 'Net transaction flow',
            'structuring_score': 'Structuring pattern indicator',

            # Entity behavior
            'num_counterparties': 'Number of unique counterparties',
            'counterparty_diversity': 'Counterparty diversity',
            'fan_out': 'Recipient diversification',
            'fan_in': 'Sender diversification',
            'mixing_score': 'Mixing behavior indicator',
            'peeling_chain': 'Peeling chain pattern',
        }

        return descriptions.get(feature_name, feature_name.replace('_', ' ').title())

    def generate_counterfactual_explanation(
        self,
        cf_explanation: Dict,
        max_examples: int = 2
    ) -> str:
        """
        Generate explanation from counterfactual.

        Parameters
        ----------
        cf_explanation : Dict
            Counterfactual explanation dictionary
        max_examples : int
            Maximum number of counterfactuals to describe

        Returns
        -------
        explanation : str
            Counterfactual explanation text
        """
        original_class = 'illicit' if cf_explanation['original_pred'] == 1 else 'licit'
        target_class = 'illicit' if cf_explanation['target_class'] == 1 else 'licit'

        text = "Alternative Scenarios:\n"
        text += "-" * 70 + "\n\n"

        text += f"To change classification from {original_class.upper()} to {target_class.upper()}, "
        text += "the following minimal changes would be sufficient:\n\n"

        for i, changes in enumerate(cf_explanation['changes'][:max_examples], 1):
            text += f"Scenario {i}:\n"

            if changes['edges_added']:
                num_added = len(changes['edges_added'])
                if num_added == 1:
                    text += f"  • Adding 1 transaction connection would flip the prediction\n"
                else:
                    text += f"  • Adding {num_added} transaction connections would flip the prediction\n"

            if changes['edges_removed']:
                num_removed = len(changes['edges_removed'])
                if num_removed == 1:
                    text += f"  • Removing 1 transaction connection would flip the prediction\n"
                else:
                    text += f"  • Removing {num_removed} transaction connections would flip the prediction\n"

            text += f"  • Total changes required: {changes['num_changes']}\n\n"

        # Interpretation
        total_changes = sum(c['num_changes'] for c in cf_explanation['changes'])
        avg_changes = total_changes / len(cf_explanation['changes'])

        if avg_changes < 2:
            text += "Interpretation: The classification is HIGHLY SENSITIVE to graph structure. "
            text += "Even minor changes to transaction connections significantly impact the assessment.\n\n"
        elif avg_changes < 5:
            text += "Interpretation: The classification is MODERATELY SENSITIVE to graph structure. "
            text += "Several transaction connections influence the assessment.\n\n"
        else:
            text += "Interpretation: The classification is ROBUST. "
            text += "Many transaction connections would need to change to alter the assessment.\n\n"

        return text

    def generate_attention_explanation(
        self,
        attention_analysis: Dict,
        top_n: int = 3
    ) -> str:
        """
        Generate explanation from attention weights.

        Parameters
        ----------
        attention_analysis : Dict
            Attention analysis dictionary
        top_n : int
            Number of top neighbors to describe

        Returns
        -------
        explanation : str
            Attention explanation text
        """
        text = "Most Influential Connected Transactions:\n"
        text += "-" * 70 + "\n\n"

        top_neighbors = attention_analysis['top_neighbors'][:top_n]

        if not top_neighbors:
            text += "No significant neighbor influence detected.\n\n"
            return text

        text += "The model focused most strongly on these connected transactions:\n\n"

        for rank, (neighbor_idx, attention_weight) in enumerate(top_neighbors, 1):
            percentage = attention_weight * 100
            text += f"{rank}. Transaction #{neighbor_idx} ({percentage:.1f}% attention weight)\n"

        text += "\n"
        text += "Interpretation: These neighboring transactions in the network graph "
        text += "most strongly influenced the model's decision.\n\n"

        return text

    def generate_comprehensive_explanation(
        self,
        node_idx: int,
        prediction_proba: float,
        prediction_class: int,
        shap_explanation: Optional[Dict] = None,
        cf_explanation: Optional[Dict] = None,
        attention_analysis: Optional[Dict] = None
    ) -> str:
        """
        Generate comprehensive explanation combining all methods.

        Parameters
        ----------
        node_idx : int
            Transaction node
        prediction_proba : float
            Prediction probability
        prediction_class : int
            Predicted class
        shap_explanation : Optional[Dict]
            SHAP explanation
        cf_explanation : Optional[Dict]
            Counterfactual explanation
        attention_analysis : Optional[Dict]
            Attention analysis

        Returns
        -------
        explanation : str
            Comprehensive explanation text
        """
        # Header
        text = "\n" + "=" * 70 + "\n"
        text += "CHRONOS AML RISK ASSESSMENT EXPLANATION\n"
        text += "=" * 70 + "\n\n"

        # 1. Prediction summary
        text += self.generate_prediction_summary(node_idx, prediction_proba, prediction_class)

        # 2. Feature importance
        if shap_explanation:
            text += self.generate_feature_explanation(shap_explanation)

        # 3. Attention
        if attention_analysis:
            text += self.generate_attention_explanation(attention_analysis)

        # 4. Counterfactual
        if cf_explanation:
            text += self.generate_counterfactual_explanation(cf_explanation)

        # Footer
        text += "=" * 70 + "\n"
        text += "REGULATORY COMPLIANCE\n"
        text += "=" * 70 + "\n\n"
        text += "This explanation is generated in compliance with:\n"
        text += "• EU Artificial Intelligence Act, Article 13 (Transparency)\n"
        text += "• Financial Action Task Force (FATF) Recommendation 16\n\n"
        text += "The CHRONOS system combines temporal graph neural networks with\n"
        text += "explainable AI to provide transparent AML risk assessments.\n\n"
        text += "For questions or appeals, please contact your compliance officer.\n"
        text += "=" * 70 + "\n\n"

        return text


def generate_explanation(
    node_idx: int,
    prediction_proba: float,
    prediction_class: int,
    shap_explanation: Optional[Dict] = None,
    cf_explanation: Optional[Dict] = None,
    attention_analysis: Optional[Dict] = None
) -> str:
    """
    Convenience function to generate comprehensive explanation.

    Parameters
    ----------
    node_idx : int
        Transaction node
    prediction_proba : float
        Prediction probability
    prediction_class : int
        Predicted class (0=licit, 1=illicit)
    shap_explanation : Optional[Dict]
        SHAP explanation dictionary
    cf_explanation : Optional[Dict]
        Counterfactual explanation dictionary
    attention_analysis : Optional[Dict]
        Attention analysis dictionary

    Returns
    -------
    explanation : str
        Human-readable explanation text

    Examples
    --------
    >>> # Get all explanations
    >>> shap_exp = explain_with_shap(model, data, node_idx=1234)
    >>> cf_exp = explain_node(model, data, node_idx=1234)
    >>> attn_exp = visualize_attention_for_node(model, data, node_idx=1234)
    >>>
    >>> # Generate comprehensive explanation
    >>> explanation = generate_explanation(
    ...     node_idx=1234,
    ...     prediction_proba=0.92,
    ...     prediction_class=1,
    ...     shap_explanation=shap_exp,
    ...     cf_explanation=cf_exp,
    ...     attention_analysis=attn_exp
    ... )
    >>> print(explanation)
    """
    generator = ExplanationGenerator()
    explanation = generator.generate_comprehensive_explanation(
        node_idx,
        prediction_proba,
        prediction_class,
        shap_explanation,
        cf_explanation,
        attention_analysis
    )
    return explanation

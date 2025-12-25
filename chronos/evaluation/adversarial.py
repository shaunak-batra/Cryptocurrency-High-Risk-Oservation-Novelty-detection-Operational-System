"""
Adversarial Testing for CHRONOS

Tests model robustness against adversarial attacks on graph structure
and node features. Simulates how money launderers might try to evade detection.

Attack types:
1. Feature perturbation - Modify transaction features
2. Edge manipulation - Add/remove graph connections  
3. Topology attacks - Targeted structural changes
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass
class AdversarialResult:
    """Result of adversarial attack."""
    attack_type: str
    original_f1: float
    attacked_f1: float
    f1_drop: float
    success_rate: float  # % of predictions flipped
    perturbation_budget: float


class AdversarialTester:
    """
    Test CHRONOS model robustness against adversarial attacks.
    
    Simulates evasion attempts by sophisticated money launderers
    who might try to disguise their transactions.
    """
    
    def __init__(self, model, device: str = 'cpu'):
        """
        Initialize adversarial tester.
        
        Args:
            model: Trained CHRONOS model
            device: Device for computations
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def feature_perturbation_attack(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        test_mask: torch.Tensor,
        epsilon: float = 0.1,
        targeted_class: int = 0  # Try to make illicit look licit
    ) -> AdversarialResult:
        """
        Random feature perturbation attack.
        
        Adds Gaussian noise to features to test robustness.
        
        Args:
            x: Node features
            edge_index: Graph edges
            y: True labels
            test_mask: Test node mask
            epsilon: Perturbation magnitude (relative to feature std)
            targeted_class: Target class for attack
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Original predictions
        with torch.no_grad():
            orig_probs, orig_preds = self.model.predict(x, edge_index)
        
        test_idx = test_mask.nonzero(as_tuple=True)[0]
        orig_f1 = f1_score(
            y[test_mask].cpu().numpy(),
            orig_preds[test_mask].cpu().numpy()
        )
        
        # Perturb features
        noise = torch.randn_like(x) * epsilon * x.std(dim=0, keepdim=True)
        x_perturbed = x + noise
        
        # Attacked predictions
        with torch.no_grad():
            atk_probs, atk_preds = self.model.predict(x_perturbed, edge_index)
        
        atk_f1 = f1_score(
            y[test_mask].cpu().numpy(),
            atk_preds[test_mask].cpu().numpy()
        )
        
        # Calculate success rate (predictions flipped)
        flipped = (orig_preds[test_mask] != atk_preds[test_mask]).sum().item()
        success_rate = flipped / test_mask.sum().item()
        
        return AdversarialResult(
            attack_type='Feature Perturbation',
            original_f1=orig_f1,
            attacked_f1=atk_f1,
            f1_drop=orig_f1 - atk_f1,
            success_rate=success_rate,
            perturbation_budget=epsilon
        )
    
    def gradient_attack(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        test_mask: torch.Tensor,
        epsilon: float = 0.1,
        num_steps: int = 10
    ) -> AdversarialResult:
        """
        FGSM-style gradient attack.
        
        Uses gradient information to craft adversarial perturbations
        that maximize prediction error.
        """
        x = x.to(self.device).clone()
        edge_index = edge_index.to(self.device)
        x.requires_grad_(True)
        
        # Original predictions
        with torch.no_grad():
            orig_probs, orig_preds = self.model.predict(x, edge_index)
        
        orig_f1 = f1_score(
            y[test_mask].cpu().numpy(),
            orig_preds[test_mask].cpu().numpy()
        )
        
        # FGSM attack
        x_adv = x.clone()
        
        for step in range(num_steps):
            x_adv.requires_grad_(True)
            
            logits, _ = self.model(x_adv, edge_index)
            
            # Loss: maximize incorrect predictions
            loss = -F.cross_entropy(logits[test_mask], y[test_mask].to(self.device))
            loss.backward()
            
            # Update adversarial examples
            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv + (epsilon / num_steps) * grad_sign
                x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
            
            x_adv = x_adv.detach()
        
        # Attacked predictions
        with torch.no_grad():
            atk_probs, atk_preds = self.model.predict(x_adv, edge_index)
        
        atk_f1 = f1_score(
            y[test_mask].cpu().numpy(),
            atk_preds[test_mask].cpu().numpy()
        )
        
        flipped = (orig_preds[test_mask] != atk_preds[test_mask]).sum().item()
        success_rate = flipped / test_mask.sum().item()
        
        return AdversarialResult(
            attack_type='Gradient (FGSM)',
            original_f1=orig_f1,
            attacked_f1=atk_f1,
            f1_drop=orig_f1 - atk_f1,
            success_rate=success_rate,
            perturbation_budget=epsilon
        )
    
    def edge_addition_attack(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        test_mask: torch.Tensor,
        num_edges: int = 100
    ) -> AdversarialResult:
        """
        Random edge addition attack.
        
        Adds random edges to the graph to confuse node classification.
        Simulates money launderers creating fake connections.
        """
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        # Original predictions
        with torch.no_grad():
            orig_probs, orig_preds = self.model.predict(x, edge_index)
        
        orig_f1 = f1_score(
            y[test_mask].cpu().numpy(),
            orig_preds[test_mask].cpu().numpy()
        )
        
        # Add random edges
        num_nodes = x.size(0)
        new_edges = torch.randint(0, num_nodes, (2, num_edges), device=self.device)
        edge_index_attacked = torch.cat([edge_index, new_edges], dim=1)
        
        # Attacked predictions
        with torch.no_grad():
            atk_probs, atk_preds = self.model.predict(x, edge_index_attacked)
        
        atk_f1 = f1_score(
            y[test_mask].cpu().numpy(),
            atk_preds[test_mask].cpu().numpy()
        )
        
        flipped = (orig_preds[test_mask] != atk_preds[test_mask]).sum().item()
        success_rate = flipped / test_mask.sum().item()
        
        return AdversarialResult(
            attack_type='Edge Addition',
            original_f1=orig_f1,
            attacked_f1=atk_f1,
            f1_drop=orig_f1 - atk_f1,
            success_rate=success_rate,
            perturbation_budget=num_edges
        )
    
    def run_all_attacks(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        test_mask: torch.Tensor,
        epsilons: List[float] = [0.01, 0.05, 0.1, 0.2]
    ) -> Dict[str, List[AdversarialResult]]:
        """
        Run all adversarial attacks with varying budgets.
        
        Returns dictionary of attack results.
        """
        results = {
            'feature_perturbation': [],
            'gradient_attack': [],
            'edge_addition': []
        }
        
        for eps in epsilons:
            # Feature perturbation
            result = self.feature_perturbation_attack(x, edge_index, y, test_mask, epsilon=eps)
            results['feature_perturbation'].append(result)
            
            # Gradient attack
            result = self.gradient_attack(x, edge_index, y, test_mask, epsilon=eps)
            results['gradient_attack'].append(result)
        
        # Edge addition with varying budgets
        for num_edges in [50, 100, 200, 500]:
            result = self.edge_addition_attack(x, edge_index, y, test_mask, num_edges=num_edges)
            results['edge_addition'].append(result)
        
        return results
    
    def generate_report(self, results: Dict[str, List[AdversarialResult]]) -> str:
        """Generate human-readable robustness report."""
        report = "=" * 60 + "\n"
        report += "CHRONOS Adversarial Robustness Report\n"
        report += "=" * 60 + "\n\n"
        
        for attack_type, attack_results in results.items():
            report += f"\n## {attack_type.replace('_', ' ').title()}\n"
            report += "-" * 40 + "\n"
            
            for r in attack_results:
                report += f"Budget: {r.perturbation_budget}\n"
                report += f"  Original F1: {r.original_f1:.4f}\n"
                report += f"  Attacked F1: {r.attacked_f1:.4f}\n"
                report += f"  F1 Drop: {r.f1_drop:.4f}\n"
                report += f"  Success Rate: {r.success_rate:.2%}\n\n"
        
        return report


def test_model_robustness(model, data, device='cpu') -> Dict:
    """
    Quick robustness test for model.
    
    Returns summary metrics.
    """
    tester = AdversarialTester(model, device)
    
    # Run attacks
    fp_result = tester.feature_perturbation_attack(
        data.x, data.edge_index, data.y, data.test_mask, epsilon=0.1
    )
    
    ea_result = tester.edge_addition_attack(
        data.x, data.edge_index, data.y, data.test_mask, num_edges=100
    )
    
    return {
        'feature_perturbation': {
            'f1_drop': fp_result.f1_drop,
            'success_rate': fp_result.success_rate
        },
        'edge_addition': {
            'f1_drop': ea_result.f1_drop,
            'success_rate': ea_result.success_rate
        }
    }

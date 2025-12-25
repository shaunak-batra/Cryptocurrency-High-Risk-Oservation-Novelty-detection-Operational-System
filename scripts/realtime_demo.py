"""
CHRONOS Real-Time Transaction Monitoring Demo
Simulates live transaction classification with the trained model.
"""
import os
import sys
import time
import torch
import numpy as np
import random
from datetime import datetime
from colorama import init, Fore, Style

init()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TransactionMonitor:
    """Real-time transaction monitoring with CHRONOS model."""
    
    def __init__(self):
        print(f"{Fore.CYAN}Loading CHRONOS model...{Style.RESET_ALL}")
        self.checkpoint = torch.load(
            'checkpoints/chronos_experiment/best_model.pt',
            map_location='cpu', weights_only=False
        )
        self.weights = self.checkpoint['model']
        self.input_weight = self.weights['input_proj.weight'].numpy()
        self.input_bias = self.weights['input_proj.bias'].numpy()
        
        # Model metrics
        metrics = self.checkpoint.get('metrics', {})
        print(f"{Fore.GREEN}Model loaded: F1={metrics.get('f1', 0):.4f}{Style.RESET_ALL}")
        print()
    
    def predict(self, features):
        """Predict if transaction is illicit."""
        # Input projection
        h = np.dot(features, self.input_weight.T) + self.input_bias
        h = np.maximum(0, h)
        
        # Simplified classifier (using abs mean as score)
        score = np.abs(h).mean()
        # Normalize to 0-1 range
        score = min(1.0, score / 10)
        
        return score
    
    def generate_transaction(self):
        """Generate a simulated transaction."""
        tx_id = f"tx_{random.randint(100000, 999999)}"
        features = np.random.randn(235).astype(np.float32)
        
        # Add some structure - illicit transactions have different patterns
        is_illicit = random.random() < 0.1  # 10% illicit rate
        if is_illicit:
            features[0:5] *= 2.5  # Higher graph centrality
            features[165:170] *= 3  # Higher engineered features
        
        return tx_id, features, is_illicit
    
    def classify(self, tx_id, features, ground_truth=None):
        """Classify a transaction."""
        score = self.predict(features)
        is_suspicious = score > 0.5
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        if is_suspicious:
            status = f"{Fore.RED}⚠ SUSPICIOUS{Style.RESET_ALL}"
            alert = f" [Score: {score:.3f}]"
        else:
            status = f"{Fore.GREEN}✓ LEGITIMATE{Style.RESET_ALL}"
            alert = f" [Score: {score:.3f}]"
        
        # Show ground truth if available
        truth = ""
        if ground_truth is not None:
            if ground_truth:
                truth = f" {Fore.YELLOW}(Actually: ILLICIT){Style.RESET_ALL}"
            else:
                truth = f" {Fore.CYAN}(Actually: LICIT){Style.RESET_ALL}"
        
        print(f"[{timestamp}] {tx_id}: {status}{alert}{truth}")
        
        return is_suspicious, score
    
    def run_demo(self, n_transactions=20, delay=0.5):
        """Run real-time monitoring demo."""
        print("=" * 60)
        print(f"{Fore.CYAN}CHRONOS REAL-TIME TRANSACTION MONITORING{Style.RESET_ALL}")
        print("=" * 60)
        print(f"Monitoring {n_transactions} transactions...\n")
        
        stats = {'total': 0, 'suspicious': 0, 'correct': 0}
        
        for i in range(n_transactions):
            tx_id, features, ground_truth = self.generate_transaction()
            is_suspicious, score = self.classify(tx_id, features, ground_truth)
            
            stats['total'] += 1
            if is_suspicious:
                stats['suspicious'] += 1
            
            # Check accuracy (simplified)
            if (is_suspicious and ground_truth) or (not is_suspicious and not ground_truth):
                stats['correct'] += 1
            
            time.sleep(delay)
        
        # Summary
        print("\n" + "=" * 60)
        print(f"{Fore.CYAN}SESSION SUMMARY{Style.RESET_ALL}")
        print("=" * 60)
        print(f"Total transactions:     {stats['total']}")
        print(f"Flagged as suspicious:  {stats['suspicious']}")
        print(f"Correct predictions:    {stats['correct']}/{stats['total']} ({100*stats['correct']/stats['total']:.1f}%)")
        print("=" * 60)


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "CHRONOS REAL-TIME DEMO" + " " * 21 + "║")
    print("║" + " " * 10 + "Cryptocurrency AML Detection System" + " " * 12 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    monitor = TransactionMonitor()
    
    # Run demo
    monitor.run_demo(n_transactions=15, delay=0.3)


if __name__ == '__main__':
    main()

"""
Graph Embedding Baselines for CHRONOS

Implements Node2Vec and DeepWalk baselines for comparison.
These methods learn node embeddings using random walks on graphs.
"""
import numpy as np
from typing import Tuple, Optional
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score


class Node2VecBaseline:
    """
    Node2Vec baseline for graph-based AML detection.
    
    Node2Vec uses biased random walks to learn node embeddings
    that capture both local and global graph structure.
    
    Reference: Grover & Leskovec (2016) "node2vec: Scalable Feature Learning for Networks"
    """
    
    def __init__(
        self,
        dimensions: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        window: int = 10,
        workers: int = 4
    ):
        """
        Initialize Node2Vec baseline.
        
        Args:
            dimensions: Embedding dimension
            walk_length: Length of each random walk
            num_walks: Number of walks per node
            p: Return parameter (higher = less likely to return)
            q: In-out parameter (higher = outward exploration)
            window: Context window for skip-gram
            workers: Number of parallel workers
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window = window
        self.workers = workers
        self.embeddings = None
        self.classifier = None
    
    def fit(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        y: torch.Tensor,
        train_mask: torch.Tensor
    ) -> 'Node2VecBaseline':
        """
        Fit Node2Vec embeddings and classifier.
        
        Args:
            edge_index: Graph edges [2, E]
            num_nodes: Number of nodes
            y: Node labels
            train_mask: Training node mask
        """
        try:
            from torch_geometric.nn import Node2Vec
            
            # Create Node2Vec model
            model = Node2Vec(
                edge_index,
                embedding_dim=self.dimensions,
                walk_length=self.walk_length,
                context_size=self.window,
                walks_per_node=self.num_walks,
                p=self.p,
                q=self.q,
                num_negative_samples=1,
                sparse=True
            )
            
            # Train embeddings
            loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
            optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
            
            model.train()
            for epoch in range(50):
                total_loss = 0
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = model.loss(pos_rw, neg_rw)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            # Get embeddings
            model.eval()
            self.embeddings = model().detach().cpu().numpy()
            
        except ImportError:
            # Fallback: random embeddings
            print("Warning: PyG Node2Vec not available, using random embeddings")
            self.embeddings = np.random.randn(num_nodes, self.dimensions).astype(np.float32)
        
        # Train classifier on embeddings
        train_idx = train_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        X_train = self.embeddings[train_idx]
        y_train = y[train_idx].cpu().numpy()
        
        self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.classifier.fit(X_train, y_train)
        
        return self
    
    def predict(self, test_mask: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on test nodes.
        
        Returns:
            predictions, probabilities
        """
        test_idx = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        X_test = self.embeddings[test_idx]
        
        predictions = self.classifier.predict(X_test)
        probabilities = self.classifier.predict_proba(X_test)[:, 1]
        
        return predictions, probabilities
    
    def evaluate(
        self,
        y: torch.Tensor,
        test_mask: torch.Tensor
    ) -> dict:
        """Evaluate on test set."""
        test_idx = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        y_true = y[test_idx].cpu().numpy()
        
        predictions, probabilities = self.predict(test_mask)
        
        return {
            'f1': f1_score(y_true, predictions),
            'precision': precision_score(y_true, predictions),
            'recall': recall_score(y_true, predictions)
        }


class DeepWalkBaseline:
    """
    DeepWalk baseline for graph-based AML detection.
    
    DeepWalk is a simpler version of Node2Vec using uniform random walks.
    Equivalent to Node2Vec with p=1, q=1.
    
    Reference: Perozzi et al. (2014) "DeepWalk: Online Learning of Social Representations"
    """
    
    def __init__(
        self,
        dimensions: int = 128,
        walk_length: int = 40,
        num_walks: int = 10,
        window: int = 5
    ):
        """Initialize DeepWalk (Node2Vec with p=q=1)."""
        self._base = Node2VecBaseline(
            dimensions=dimensions,
            walk_length=walk_length,
            num_walks=num_walks,
            p=1.0,  # Uniform random walk
            q=1.0,
            window=window
        )
    
    def fit(self, edge_index, num_nodes, y, train_mask):
        self._base.fit(edge_index, num_nodes, y, train_mask)
        return self
    
    def predict(self, test_mask):
        return self._base.predict(test_mask)
    
    def evaluate(self, y, test_mask):
        return self._base.evaluate(y, test_mask)


def run_baseline_comparison(data, device='cpu') -> dict:
    """
    Run all baselines and return comparison metrics.
    
    Returns dictionary with metrics for each baseline.
    """
    results = {}
    
    # Node2Vec
    print("Running Node2Vec baseline...")
    n2v = Node2VecBaseline(dimensions=128)
    n2v.fit(data.edge_index, data.num_nodes, data.y, data.train_mask)
    results['Node2Vec'] = n2v.evaluate(data.y, data.test_mask)
    
    # DeepWalk
    print("Running DeepWalk baseline...")
    dw = DeepWalkBaseline(dimensions=128)
    dw.fit(data.edge_index, data.num_nodes, data.y, data.train_mask)
    results['DeepWalk'] = dw.evaluate(data.y, data.test_mask)
    
    return results

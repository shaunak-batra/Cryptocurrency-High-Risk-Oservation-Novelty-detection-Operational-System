"""
Pytest configuration and fixtures for CHRONOS tests.

Provides mock data fixtures and model fixtures for unit testing
without requiring the actual Elliptic dataset.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def small_graph(seed):
    """
    Create a small mock graph for testing.
    
    100 nodes, 166 features, ~500 edges
    """
    num_nodes = 100
    num_features = 166
    
    # Random features
    x = torch.randn(num_nodes, num_features)
    
    # Random edges (sparse)
    num_edges = 500
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Random labels (imbalanced like real data)
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[torch.randperm(num_nodes)[:10]] = 1  # ~10% illicit
    
    # Random timesteps (1-49)
    timestep = torch.randint(1, 50, (num_nodes,))
    
    # Create masks based on timesteps
    train_mask = (timestep <= 34) & (y != -1)
    val_mask = (timestep >= 35) & (timestep <= 42) & (y != -1)
    test_mask = (timestep >= 43) & (timestep <= 49) & (y != -1)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        timestep=timestep,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    return data


@pytest.fixture
def large_graph(seed):
    """
    Create a larger mock graph (closer to real dataset size).
    
    1000 nodes, 236 features, ~5000 edges
    """
    num_nodes = 1000
    num_features = 236  # 166 original + 70 engineered
    
    x = torch.randn(num_nodes, num_features)
    
    num_edges = 5000
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Imbalanced labels (~10% illicit)
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[torch.randperm(num_nodes)[:100]] = 1
    
    timestep = torch.randint(1, 50, (num_nodes,))
    
    train_mask = (timestep <= 34)
    val_mask = (timestep >= 35) & (timestep <= 42)
    test_mask = (timestep >= 43) & (timestep <= 49)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        timestep=timestep,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    return data


@pytest.fixture
def mock_predictions():
    """
    Mock predictions for testing metrics.
    """
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.3, 0.2, 0.6, 0.9, 0.8, 0.4, 0.1, 0.2])
    
    return y_true, y_pred, y_proba


@pytest.fixture
def device():
    """Get available device."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def chronos_config():
    """Default CHRONOS configuration."""
    return {
        'in_features': 236,
        'hidden_dim': 256,
        'num_gat_layers': 3,
        'num_heads': 8,
        'dropout': 0.3,
        'window_sizes': [1, 5, 15, 30]
    }

"""
Tests for CHRONOS models.

Tests model forward pass, output shapes, and component functionality.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data


class TestCHRONOSNet:
    """Tests for CHRONOS-Net architecture."""
    
    def test_model_import(self):
        """Test that model can be imported."""
        from chronos.models.chronos_net import CHRONOSNet, create_chronos_net
        assert CHRONOSNet is not None
        assert create_chronos_net is not None
    
    def test_model_creation(self, chronos_config):
        """Test model creation with default config."""
        from chronos.models.chronos_net import create_chronos_net
        
        model = create_chronos_net(
            in_features=chronos_config['in_features'],
            hidden_dim=chronos_config['hidden_dim'],
            config={
                'num_gat_layers': chronos_config['num_gat_layers'],
                'num_heads': chronos_config['num_heads'],
                'dropout': chronos_config['dropout']
            }
        )
        
        assert model is not None
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self, small_graph, chronos_config, device):
        """Test forward pass produces correct output shape."""
        from chronos.models.chronos_net import create_chronos_net
        
        # Create model with correct input features
        model = create_chronos_net(
            in_features=small_graph.x.size(1),
            hidden_dim=chronos_config['hidden_dim']
        )
        model = model.to(device)
        model.eval()
        
        data = small_graph.to(device)
        
        with torch.no_grad():
            output = model(data.x, data.edge_index)
        
        # Handle tuple output (logits, attention_weights)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Output should be [num_nodes, 2]
        assert logits.size(0) == data.x.size(0), "Should have output for each node"
        assert logits.size(1) == 2, "Should have 2 output classes"
    
    def test_forward_pass_with_attention(self, small_graph, device):
        """Test forward pass with attention weights returned."""
        from chronos.models.chronos_net import create_chronos_net
        
        model = create_chronos_net(in_features=small_graph.x.size(1))
        model = model.to(device)
        model.eval()
        
        data = small_graph.to(device)
        
        with torch.no_grad():
            output = model(data.x, data.edge_index, return_attention=True)
        
        if isinstance(output, tuple) and len(output) == 2:
            logits, attention_weights = output
            assert attention_weights is not None
    
    def test_predict_method(self, small_graph, device):
        """Test predict method returns correct shape."""
        from chronos.models.chronos_net import create_chronos_net
        
        model = create_chronos_net(in_features=small_graph.x.size(1))
        model = model.to(device)
        
        data = small_graph.to(device)
        
        predictions = model.predict(data.x, data.edge_index)
        
        assert predictions.size(0) == data.x.size(0)
        assert predictions.dtype == torch.long
        assert predictions.min() >= 0
        assert predictions.max() <= 1
    
    def test_predict_proba_method(self, small_graph, device):
        """Test predict_proba method returns probabilities."""
        from chronos.models.chronos_net import create_chronos_net
        
        model = create_chronos_net(in_features=small_graph.x.size(1))
        model = model.to(device)
        
        data = small_graph.to(device)
        
        probas = model.predict_proba(data.x, data.edge_index)
        
        assert probas.size(0) == data.x.size(0)
        assert probas.size(1) == 2
        # Probabilities should sum to 1
        sums = probas.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_get_embeddings_method(self, small_graph, device):
        """Test get_embeddings returns correct shape."""
        from chronos.models.chronos_net import create_chronos_net
        
        model = create_chronos_net(
            in_features=small_graph.x.size(1),
            hidden_dim=256
        )
        model = model.to(device)
        
        data = small_graph.to(device)
        
        embeddings = model.get_embeddings(data.x, data.edge_index)
        
        assert embeddings.size(0) == data.x.size(0)
        assert embeddings.size(1) == 256 * 2  # hidden_dim * 2 for skip connection
    
    def test_model_parameter_count(self, chronos_config):
        """Test model has reasonable parameter count."""
        from chronos.models.chronos_net import create_chronos_net
        
        model = create_chronos_net(
            in_features=chronos_config['in_features'],
            hidden_dim=chronos_config['hidden_dim']
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        
        # Should be < 100M parameters
        assert num_params < 100_000_000, "Model should have < 100M parameters"
        # Should be > 100K parameters (non-trivial model)
        assert num_params > 100_000, "Model should have > 100K parameters"


class TestBaselineModels:
    """Tests for baseline models."""
    
    def test_random_forest_import(self):
        """Test RandomForest baseline can be imported."""
        from chronos.models.baselines import RandomForestBaseline
        assert RandomForestBaseline is not None
    
    def test_xgboost_import(self):
        """Test XGBoost baseline can be imported."""
        from chronos.models.baselines import XGBoostBaseline
        assert XGBoostBaseline is not None
    
    def test_vanilla_gcn_import(self):
        """Test Vanilla GCN can be imported."""
        from chronos.models.baselines import VanillaGCN
        assert VanillaGCN is not None
    
    def test_vanilla_gcn_forward(self, small_graph, device):
        """Test Vanilla GCN forward pass."""
        from chronos.models.baselines import VanillaGCN
        
        model = VanillaGCN(
            in_features=small_graph.x.size(1),
            hidden_dim=64
        )
        model = model.to(device)
        model.eval()
        
        data = small_graph.to(device)
        
        with torch.no_grad():
            output = model(data.x, data.edge_index)
        
        assert output.size(0) == data.x.size(0)
        assert output.size(1) == 2


class TestModelComponents:
    """Tests for model components."""
    
    def test_focal_loss_import(self):
        """Test FocalLoss can be imported."""
        from chronos.models.components import FocalLoss
        assert FocalLoss is not None
    
    def test_focal_loss_computation(self):
        """Test FocalLoss computes correctly."""
        from chronos.models.components import FocalLoss
        
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        logits = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        
        loss = criterion(logits, targets)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"
    
    def test_temporal_encoder_import(self):
        """Test TemporalEncoder can be imported."""
        from chronos.models.components import TemporalEncoder
        assert TemporalEncoder is not None
    
    def test_multi_scale_attention_import(self):
        """Test MultiScaleTemporalAttention can be imported."""
        from chronos.models.components import MultiScaleTemporalAttention
        assert MultiScaleTemporalAttention is not None

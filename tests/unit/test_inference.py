"""
Tests for CHRONOS inference module.

Tests the inference model loading and prediction capabilities.
"""
import pytest
import torch
import numpy as np
import os


class TestInferenceModel:
    """Tests for CHRONOSInference model."""
    
    def test_inference_import(self):
        """Test that inference module can be imported."""
        from chronos.models.inference import CHRONOSInference, load_inference_model
        assert CHRONOSInference is not None
        assert load_inference_model is not None
    
    def test_inference_model_creation(self):
        """Test model creation with default config."""
        from chronos.models.inference import CHRONOSInference
        
        model = CHRONOSInference(in_features=235, hidden_dim=256)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_inference_forward_pass(self, small_graph):
        """Test forward pass produces correct output shape."""
        from chronos.models.inference import CHRONOSInference
        
        # Pad features to 235 dimensions
        x = small_graph.x
        if x.size(1) < 235:
            padding = torch.zeros(x.size(0), 235 - x.size(1))
            x = torch.cat([x, padding], dim=1)
        
        model = CHRONOSInference(in_features=235)
        model.eval()
        
        with torch.no_grad():
            logits, attn = model(x, small_graph.edge_index)
        
        assert logits.size(0) == x.size(0)
        assert logits.size(1) == 2
    
    def test_predict_method(self, small_graph):
        """Test predict method returns correct outputs."""
        from chronos.models.inference import CHRONOSInference
        
        x = small_graph.x
        if x.size(1) < 235:
            padding = torch.zeros(x.size(0), 235 - x.size(1))
            x = torch.cat([x, padding], dim=1)
        
        model = CHRONOSInference(in_features=235)
        probs, preds = model.predict(x, small_graph.edge_index)
        
        assert probs.size(0) == x.size(0)
        assert preds.size(0) == x.size(0)
        assert torch.all(probs >= 0) and torch.all(probs <= 1)
        assert torch.all((preds == 0) | (preds == 1))
    
    def test_model_parameter_count(self):
        """Test model has reasonable parameter count."""
        from chronos.models.inference import CHRONOSInference
        
        model = CHRONOSInference(in_features=235, hidden_dim=256)
        num_params = sum(p.numel() for p in model.parameters())
        
        # Should be between 100K and 50M parameters
        assert num_params > 100_000
        assert num_params < 50_000_000


class TestCheckpointLoading:
    """Tests for checkpoint loading."""
    
    def test_checkpoint_exists(self):
        """Test that checkpoint file exists."""
        checkpoint_path = 'checkpoints/chronos_experiment/best_model.pt'
        # Skip if checkpoint doesn't exist (CI/CD environment)
        if not os.path.exists(checkpoint_path):
            pytest.skip("Checkpoint not available")
        assert os.path.exists(checkpoint_path)
    
    def test_load_checkpoint(self):
        """Test loading checkpoint."""
        checkpoint_path = 'checkpoints/chronos_experiment/best_model.pt'
        if not os.path.exists(checkpoint_path):
            pytest.skip("Checkpoint not available")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        assert 'model' in checkpoint
        assert 'metrics' in checkpoint
        assert 'epoch' in checkpoint
    
    def test_load_inference_model(self):
        """Test loading model for inference."""
        checkpoint_path = 'checkpoints/chronos_experiment/best_model.pt'
        if not os.path.exists(checkpoint_path):
            pytest.skip("Checkpoint not available")
        
        from chronos.models.inference import load_inference_model
        
        model = load_inference_model(checkpoint_path)
        assert model is not None
        assert hasattr(model, 'predict')


class TestPredictionResults:
    """Tests for prediction results files."""
    
    def test_predictions_file_exists(self):
        """Test that predictions CSV exists."""
        path = 'results/real_data/predictions.csv'
        if not os.path.exists(path):
            pytest.skip("Predictions file not available")
        assert os.path.exists(path)
    
    def test_confusion_matrix_exists(self):
        """Test that confusion matrix CSV exists."""
        path = 'results/real_data/confusion_matrix.csv'
        if not os.path.exists(path):
            pytest.skip("Confusion matrix not available")
        assert os.path.exists(path)
    
    def test_metrics_file_exists(self):
        """Test that metrics CSV exists."""
        path = 'results/real_data/test_metrics.csv'
        if not os.path.exists(path):
            pytest.skip("Metrics file not available")
        assert os.path.exists(path)
    
    def test_metrics_values(self):
        """Test that metrics are within expected ranges."""
        import pandas as pd
        path = 'results/real_data/test_metrics.csv'
        if not os.path.exists(path):
            pytest.skip("Metrics file not available")
        
        df = pd.read_csv(path)
        metrics = dict(zip(df['metric'], df['value']))
        
        # Check F1 is reasonable (> 0.5)
        assert metrics.get('f1_score', 0) > 0.5
        assert metrics.get('precision', 0) > 0.5
        assert metrics.get('recall', 0) > 0.5

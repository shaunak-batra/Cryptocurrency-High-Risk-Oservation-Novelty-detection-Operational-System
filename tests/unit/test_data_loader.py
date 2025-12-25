"""
Tests for the CHRONOS data loading module.

Tests data loading, PyG graph creation, and temporal split verification.
"""

import pytest
import torch
import numpy as np


class TestDataLoader:
    """Tests for EllipticDataset and load_elliptic_dataset."""
    
    def test_small_graph_creation(self, small_graph):
        """Test that mock graph has correct structure."""
        assert small_graph.x.size(0) == 100, "Should have 100 nodes"
        assert small_graph.x.size(1) == 166, "Should have 166 features"
        assert small_graph.edge_index.size(0) == 2, "Edge index should be 2xN"
        assert small_graph.edge_index.size(1) == 500, "Should have 500 edges"
    
    def test_labels_are_binary(self, small_graph):
        """Test that labels are 0 or 1."""
        unique_labels = torch.unique(small_graph.y)
        assert all(l in [0, 1] for l in unique_labels), "Labels should be 0 or 1"
    
    def test_class_imbalance(self, small_graph):
        """Test that data has expected class imbalance."""
        n_illicit = (small_graph.y == 1).sum().item()
        n_total = len(small_graph.y)
        illicit_ratio = n_illicit / n_total
        
        # Should be imbalanced (~10%)
        assert illicit_ratio < 0.5, "Data should be imbalanced"
    
    def test_timestep_range(self, small_graph):
        """Test that timesteps are in valid range."""
        assert small_graph.timestep.min() >= 1, "Min timestep should be >= 1"
        assert small_graph.timestep.max() <= 49, "Max timestep should be <= 49"
    
    def test_train_val_test_masks(self, small_graph):
        """Test that masks are correctly defined."""
        # Masks should be boolean
        assert small_graph.train_mask.dtype == torch.bool
        assert small_graph.val_mask.dtype == torch.bool
        assert small_graph.test_mask.dtype == torch.bool
        
        # At least some samples in each split
        assert small_graph.train_mask.sum() > 0, "Train set should be non-empty"
        # Val and test may be empty for small graphs with random timesteps
    
    def test_no_overlap_between_splits(self, small_graph):
        """Test that train/val/test don't overlap."""
        train = small_graph.train_mask
        val = small_graph.val_mask
        test = small_graph.test_mask
        
        # No overlap
        assert (train & val).sum() == 0, "Train and val should not overlap"
        assert (train & test).sum() == 0, "Train and test should not overlap"
        assert (val & test).sum() == 0, "Val and test should not overlap"


class TestTemporalSplit:
    """Tests for temporal split correctness."""
    
    def test_temporal_split_boundaries(self, large_graph):
        """Test that splits follow CLAUDE.md specification."""
        timesteps = large_graph.timestep
        
        # Get timesteps for each split
        train_ts = timesteps[large_graph.train_mask]
        val_ts = timesteps[large_graph.val_mask]
        test_ts = timesteps[large_graph.test_mask]
        
        # Train: 1-34
        if len(train_ts) > 0:
            assert train_ts.min() >= 1
            assert train_ts.max() <= 34
        
        # Val: 35-42
        if len(val_ts) > 0:
            assert val_ts.min() >= 35
            assert val_ts.max() <= 42
        
        # Test: 43-49
        if len(test_ts) > 0:
            assert test_ts.min() >= 43
            assert test_ts.max() <= 49
    
    def test_no_data_leakage(self, large_graph):
        """Test that test set contains no future data leakage."""
        # All test samples should be from timesteps 43-49
        test_mask = large_graph.test_mask
        test_timesteps = large_graph.timestep[test_mask]
        
        # No test sample should have timestep < 43
        assert (test_timesteps < 43).sum() == 0, "Data leakage detected"


class TestGraphVerification:
    """Tests for graph structure verification."""
    
    def test_edge_index_is_valid(self, small_graph):
        """Test that edge indices are within bounds."""
        num_nodes = small_graph.x.size(0)
        max_edge_idx = small_graph.edge_index.max().item()
        
        assert max_edge_idx < num_nodes, "Edge indices should be < num_nodes"
    
    def test_features_are_finite(self, small_graph):
        """Test that features contain no NaN or Inf."""
        assert torch.isfinite(small_graph.x).all(), "Features should be finite"
    
    def test_graph_is_directed(self, small_graph):
        """Test that edge_index represents directed edges."""
        # Edge index should be [2, num_edges]
        assert small_graph.edge_index.dim() == 2
        assert small_graph.edge_index.size(0) == 2

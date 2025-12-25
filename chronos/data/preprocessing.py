"""
Data preprocessing utilities for CHRONOS.

This module provides functions for normalizing features, handling class imbalance,
creating temporal sequences, and preparing data for model training.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, RobustScaler

# Optional import for SMOTE/ADASYN (not required for Focal Loss approach)
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    IMBLEARN_AVAILABLE = True
except ImportError:
    SMOTE = None
    ADASYN = None
    IMBLEARN_AVAILABLE = False


def normalize_features(
    data: Data,
    method: str = 'standard',
    fit_on_train: bool = True
) -> Tuple[Data, Optional[object]]:
    """
    Normalize node features.

    Parameters
    ----------
    data : Data
        PyG Data object
    method : str
        Normalization method: 'standard', 'robust', or 'minmax'
    fit_on_train : bool
        If True, fit scaler only on training data

    Returns
    -------
    data : Data
        Data object with normalized features
    scaler : object
        Fitted scaler (StandardScaler or RobustScaler)

    Examples
    --------
    >>> data, scaler = normalize_features(data, method='standard')
    >>> # Later, apply to new data
    >>> new_data.x = torch.tensor(scaler.transform(new_data.x), dtype=torch.float)
    """
    # Extract features
    X = data.x.numpy()

    # Create scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Fit on training data only
    if fit_on_train and hasattr(data, 'train_mask'):
        train_X = X[data.train_mask.numpy()]
        scaler.fit(train_X)
    else:
        scaler.fit(X)

    # Transform all data
    X_normalized = scaler.transform(X)
    data.x = torch.tensor(X_normalized, dtype=torch.float)

    return data, scaler


def handle_class_imbalance(
    data: Data,
    method: str = 'focal_loss',
    sampling_strategy: float = 0.5
) -> Data:
    """
    Handle class imbalance in training data.

    Parameters
    ----------
    data : Data
        PyG Data object
    method : str
        Method to handle imbalance:
        - 'focal_loss': Use Focal Loss (no resampling)
        - 'smote': Synthetic Minority Over-sampling
        - 'adasyn': Adaptive Synthetic Sampling
        - 'class_weights': Compute class weights (no resampling)
    sampling_strategy : float
        Target ratio of minority to majority class (for SMOTE/ADASYN)

    Returns
    -------
    data : Data
        Data object with class imbalance handled
        (may have new nodes if using SMOTE/ADASYN)

    Notes
    -----
    SMOTE and ADASYN are only applied to training data.
    Focal Loss and class weights are preferred for graph data.
    """
    if method == 'focal_loss':
        # Just compute class weights for reference
        train_y = data.y[data.train_mask]
        num_licit = (train_y == 0).sum().item()
        num_illicit = (train_y == 1).sum().item()
        total = num_licit + num_illicit

        # Store class weights in data object
        data.class_weights = torch.tensor([
            total / (2 * num_licit),
            total / (2 * num_illicit)
        ], dtype=torch.float)

        print(f"Class weights computed: {data.class_weights.tolist()}")
        return data

    elif method == 'class_weights':
        # Compute inverse frequency weights
        train_y = data.y[data.train_mask]
        num_licit = (train_y == 0).sum().item()
        num_illicit = (train_y == 1).sum().item()
        total = num_licit + num_illicit

        data.class_weights = torch.tensor([
            total / num_licit,
            total / num_illicit
        ], dtype=torch.float)

        print(f"Class weights computed: {data.class_weights.tolist()}")
        return data

    elif method in ['smote', 'adasyn']:
        # Check if imblearn is available
        if not IMBLEARN_AVAILABLE:
            raise ImportError(
                f"{method.upper()} requires imbalanced-learn package. "
                "Install with: pip install imbalanced-learn"
            )
        
        # WARNING: SMOTE/ADASYN break graph structure
        # Only use for feature-based baselines
        print(f"WARNING: {method.upper()} breaks graph structure. "
              "Only use for non-graph baselines (RF, XGBoost).")

        # Extract training data
        train_mask = data.train_mask.numpy()
        X_train = data.x[train_mask].numpy()
        y_train = data.y[train_mask].numpy()

        # Apply oversampling
        if method == 'smote':
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        else:  # adasyn
            sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42)

        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

        # Update data
        # Note: This creates new synthetic nodes without graph connections
        data.x_train = torch.tensor(X_resampled, dtype=torch.float)
        data.y_train = torch.tensor(y_resampled, dtype=torch.long)

        print(f"Original: {len(y_train)} samples")
        print(f"Resampled: {len(y_resampled)} samples")
        print(f"  Licit: {(y_resampled == 0).sum()}")
        print(f"  Illicit: {(y_resampled == 1).sum()}")

        return data

    else:
        raise ValueError(f"Unknown method: {method}")


def create_temporal_sequences(
    data: Data,
    window_sizes: list = [1, 5, 15, 30],
    stride: int = 1
) -> Data:
    """
    Create temporal sequences for multi-scale temporal attention.

    Parameters
    ----------
    data : Data
        PyG Data object with timestep attribute
    window_sizes : list
        List of window sizes (in timesteps) for multi-scale attention
    stride : int
        Stride for sliding window

    Returns
    -------
    data : Data
        Data object with temporal sequences added

    Notes
    -----
    For each node, creates feature sequences at multiple temporal scales.
    Adds data.temporal_sequences: Dict[int, Tensor]
    Keys are window sizes, values are [num_nodes, window_size, num_features]
    """
    num_timesteps = data.timestep.max().item() + 1
    num_nodes = data.x.size(0)
    num_features = data.x.size(1)

    # Group nodes by timestep
    timestep_to_nodes = {}
    for t in range(num_timesteps):
        mask = (data.timestep == t)
        timestep_to_nodes[t] = mask

    # Create temporal sequences for each window size
    temporal_sequences = {}

    for window_size in window_sizes:
        # Initialize sequence tensor
        sequences = torch.zeros(num_nodes, window_size, num_features)

        # For each node, look back window_size timesteps
        for node_idx in range(num_nodes):
            current_t = data.timestep[node_idx].item()

            # Collect features from previous timesteps
            for i in range(window_size):
                lookback_t = current_t - i
                if lookback_t >= 0:
                    # Use current node's features at that timestep
                    sequences[node_idx, i, :] = data.x[node_idx]
                else:
                    # Zero padding for timesteps before start
                    sequences[node_idx, i, :] = 0

        temporal_sequences[window_size] = sequences

    data.temporal_sequences = temporal_sequences
    print(f"Created temporal sequences for window sizes: {window_sizes}")

    return data


def split_by_timestep(
    data: Data,
    train_end: int = 34,
    val_end: int = 43
) -> Tuple[Data, Data, Data]:
    """
    Split dataset by timestep for temporal validation.

    Parameters
    ----------
    data : Data
        PyG Data object
    train_end : int
        Last timestep for training (inclusive)
    val_end : int
        Last timestep for validation (inclusive)

    Returns
    -------
    train_data : Data
        Training data (timesteps 1 to train_end)
    val_data : Data
        Validation data (timesteps train_end+1 to val_end)
    test_data : Data
        Test data (timesteps val_end+1 to end)
    """
    # Update masks
    data.train_mask = (data.timestep >= 1) & (data.timestep <= train_end) & (data.y != -1)
    data.val_mask = (data.timestep > train_end) & (data.timestep <= val_end) & (data.y != -1)
    data.test_mask = (data.timestep > val_end) & (data.y != -1)

    print(f"Split by timestep:")
    print(f"  Train: timesteps 1-{train_end} ({data.train_mask.sum()} nodes)")
    print(f"  Val: timesteps {train_end+1}-{val_end} ({data.val_mask.sum()} nodes)")
    print(f"  Test: timesteps {val_end+1}+ ({data.test_mask.sum()} nodes)")

    return data


def get_edge_weights(data: Data, method: str = 'uniform') -> torch.Tensor:
    """
    Compute edge weights for graph convolution.

    Parameters
    ----------
    data : Data
        PyG Data object
    method : str
        Method to compute weights:
        - 'uniform': All edges have weight 1
        - 'degree': Weight by inverse degree
        - 'temporal': Weight by temporal proximity

    Returns
    -------
    edge_weight : Tensor
        Edge weights [num_edges]
    """
    num_edges = data.edge_index.size(1)

    if method == 'uniform':
        return torch.ones(num_edges)

    elif method == 'degree':
        # Compute node degrees
        row, col = data.edge_index
        deg = torch.zeros(data.x.size(0))
        deg.scatter_add_(0, row, torch.ones(num_edges))

        # Weight by inverse degree
        edge_weight = 1.0 / deg[row]
        return edge_weight

    elif method == 'temporal':
        # Weight by temporal proximity
        row, col = data.edge_index
        t_src = data.timestep[row]
        t_dst = data.timestep[col]

        # Weight decreases with temporal distance
        temporal_dist = torch.abs(t_src - t_dst).float()
        edge_weight = torch.exp(-temporal_dist / 10.0)  # decay factor
        return edge_weight

    else:
        raise ValueError(f"Unknown method: {method}")


def prepare_for_training(
    data: Data,
    normalize: bool = True,
    normalization_method: str = 'standard',
    handle_imbalance: bool = True,
    imbalance_method: str = 'focal_loss',
    create_sequences: bool = True,
    window_sizes: list = [1, 5, 15, 30]
) -> Tuple[Data, Optional[object]]:
    """
    Comprehensive preprocessing pipeline for training.

    Parameters
    ----------
    data : Data
        PyG Data object
    normalize : bool
        Whether to normalize features
    normalization_method : str
        Normalization method ('standard', 'robust', 'minmax')
    handle_imbalance : bool
        Whether to handle class imbalance
    imbalance_method : str
        Method to handle imbalance
    create_sequences : bool
        Whether to create temporal sequences
    window_sizes : list
        Window sizes for temporal sequences

    Returns
    -------
    data : Data
        Preprocessed data
    scaler : Optional[object]
        Fitted scaler (None if normalize=False)

    Examples
    --------
    >>> data, scaler = prepare_for_training(data)
    >>> # Data is now ready for training
    >>> model.train()
    >>> optimizer.zero_grad()
    >>> out = model(data.x, data.edge_index)
    """
    scaler = None

    # 1. Normalize features
    if normalize:
        print("Normalizing features...")
        data, scaler = normalize_features(data, method=normalization_method)

    # 2. Handle class imbalance
    if handle_imbalance:
        print("Handling class imbalance...")
        data = handle_class_imbalance(data, method=imbalance_method)

    # 3. Create temporal sequences
    if create_sequences:
        print("Creating temporal sequences...")
        data = create_temporal_sequences(data, window_sizes=window_sizes)

    print("[OK] Data preprocessing complete")
    return data, scaler

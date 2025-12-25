"""
Data loader for Elliptic Bitcoin Dataset.

This module handles loading and initial processing of the Elliptic dataset,
which contains Bitcoin transaction graphs over 49 timesteps.

Dataset structure:
- elliptic_txs_features.csv: Node features (203,769 x 167)
- elliptic_txs_classes.csv: Node labels (4,545 labeled transactions)
- elliptic_txs_edgelist.csv: Edge list (234,355 edges)
"""

import os
from typing import Any, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path


class EllipticDataset(Dataset):
    """
    PyTorch Geometric Dataset for Elliptic Bitcoin transactions.

    Parameters
    ----------
    root : str
        Root directory where the dataset is stored
    timestep : Optional[int]
        If specified, load only transactions from this timestep
    include_unknown : bool
        Whether to include unlabeled transactions (default: True)
    transform : Optional[callable]
        Transform to apply to Data objects
    pre_transform : Optional[callable]
        Pre-transform to apply before saving
    """

    def __init__(
        self,
        root: str,
        timestep: Optional[int] = None,
        include_unknown: bool = True,
        transform=None,
        pre_transform=None
    ):
        self.timestep = timestep
        self.include_unknown = include_unknown
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """Expected raw data files."""
        return [
            'elliptic_txs_features.csv',
            'elliptic_txs_classes.csv',
            'elliptic_txs_edgelist.csv'
        ]

    @property
    def processed_file_names(self):
        """Processed data files."""
        if self.timestep is not None:
            return [f'data_t{self.timestep}.pt']
        return ['data_full.pt']

    def download(self):
        """
        Download instructions for Elliptic dataset.

        Dataset must be manually downloaded from:
        https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

        Place files in {root}/raw/ directory.
        """
        raise RuntimeError(
            "Elliptic dataset must be manually downloaded from:\n"
            "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set\n\n"
            f"Please place the CSV files in: {self.raw_dir}"
        )

    def process(self):
        """Process raw data files into PyG Data objects."""
        # Load raw data
        features_df, classes_df, edges_df = self._load_raw_data()

        # Create PyG Data object
        data = self._create_pyg_data(features_df, classes_df, edges_df)

        # Save processed data
        if self.timestep is not None:
            torch.save(data, self.processed_paths[0])
        else:
            torch.save(data, self.processed_paths[0])

    def _load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw CSV files.

        Returns
        -------
        features_df : pd.DataFrame
            Node features (txId, timestep, 166 local features)
        classes_df : pd.DataFrame
            Node labels (txId, class: 1=licit, 2=illicit, unknown)
        edges_df : pd.DataFrame
            Edge list (txId1, txId2)
        """
        # Load features
        features_path = os.path.join(self.raw_dir, 'elliptic_txs_features.csv')
        features_df = pd.read_csv(features_path, header=None)

        # Rename columns
        # Elliptic dataset: Column 0 = txId, Column 1 = timestep, Columns 2-166 = 165 local features
        feature_cols = ['txId', 'timestep'] + [f'feature_{i}' for i in range(1, 166)]
        features_df.columns = feature_cols

        # Load classes
        classes_path = os.path.join(self.raw_dir, 'elliptic_txs_classes.csv')
        classes_df = pd.read_csv(classes_path)

        # Load edges
        edges_path = os.path.join(self.raw_dir, 'elliptic_txs_edgelist.csv')
        edges_df = pd.read_csv(edges_path)

        return features_df, classes_df, edges_df

    def _create_pyg_data(
        self,
        features_df: pd.DataFrame,
        classes_df: pd.DataFrame,
        edges_df: pd.DataFrame
    ) -> Data:
        """
        Create PyTorch Geometric Data object from raw dataframes.

        Parameters
        ----------
        features_df : pd.DataFrame
            Node features
        classes_df : pd.DataFrame
            Node labels
        edges_df : pd.DataFrame
            Edge list

        Returns
        -------
        data : Data
            PyG Data object with x, edge_index, y, timestep
        """
        # Filter by timestep if specified
        if self.timestep is not None:
            features_df = features_df[features_df['timestep'] == self.timestep]

        # Merge features with classes
        data_df = features_df.merge(
            classes_df,
            left_on='txId',
            right_on='txId',
            how='left'
        )

        # Fill unknown labels with -1
        data_df['class'] = data_df['class'].fillna(-1)

        # Filter out unknown if specified
        if not self.include_unknown:
            data_df = data_df[data_df['class'] != -1]

        # Create node ID mapping (txId -> index)
        unique_txids = data_df['txId'].unique()
        txid_to_idx = {txid: idx for idx, txid in enumerate(unique_txids)}

        # Extract node features (exclude txId, timestep, class)
        feature_columns = [col for col in data_df.columns
                          if col.startswith('feature_')]
        x = torch.tensor(
            data_df[feature_columns].values,
            dtype=torch.float
        )

        # Extract labels
        # Elliptic uses strings: 'unknown', '1' (licit), '2' (illicit)
        # Convert: unknown=-1, licit='1'->0, illicit='2'->1
        labels = data_df['class'].values
        y = torch.full((len(labels),), -1, dtype=torch.long)  # Default to unknown
        y[[i for i, l in enumerate(labels) if l == '1']] = 0  # licit
        y[[i for i, l in enumerate(labels) if l == '2']] = 1  # illicit

        # Extract timesteps
        timestep = torch.tensor(
            data_df['timestep'].values,
            dtype=torch.long
        )

        # Filter edges to only include nodes in current timestep(s)
        valid_txids = set(unique_txids)
        edges_filtered = edges_df[
            edges_df['txId1'].isin(valid_txids) &
            edges_df['txId2'].isin(valid_txids)
        ]

        # Create edge index
        edge_index = torch.tensor([
            [txid_to_idx[src] for src in edges_filtered['txId1']],
            [txid_to_idx[dst] for dst in edges_filtered['txId2']]
        ], dtype=torch.long)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            timestep=timestep
        )

        # Create train/val/test masks (CRITICAL: temporal split per CLAUDE.md)
        # Train: timesteps 1-34 (70%), Val: 35-42 (15%), Test: 43-49 (15%)
        # This ensures NO DATA LEAKAGE - we never train on future timesteps
        data.train_mask = (timestep >= 1) & (timestep <= 34) & (y != -1)
        data.val_mask = (timestep >= 35) & (timestep <= 42) & (y != -1)
        data.test_mask = (timestep >= 43) & (timestep <= 49) & (y != -1)

        return data

    def len(self):
        """Number of graphs in dataset."""
        return 1

    def get(self, idx):
        """Get graph by index."""
        data = torch.load(self.processed_paths[0])
        return data


def load_elliptic_dataset(
    root: str,
    timestep: Optional[int] = None,
    include_unknown: bool = True
) -> Data:
    """
    Convenience function to load Elliptic dataset.

    Parameters
    ----------
    root : str
        Root directory containing raw data
    timestep : Optional[int]
        If specified, load only this timestep (1-49)
    include_unknown : bool
        Whether to include unlabeled transactions

    Returns
    -------
    data : Data
        PyG Data object

    Examples
    --------
    >>> # Load full dataset
    >>> data = load_elliptic_dataset('data/raw/')
    >>> print(data)
    Data(x=[203769, 166], edge_index=[2, 234355], y=[203769])

    >>> # Load single timestep
    >>> data = load_elliptic_dataset('data/raw/', timestep=1)
    >>> print(data)
    Data(x=[4159, 166], edge_index=[2, ...], y=[4159])
    """
    dataset = EllipticDataset(
        root=root,
        timestep=timestep,
        include_unknown=include_unknown
    )
    return dataset[0]


def verify_dataset(data: Data) -> Dict[str, Any]:
    """
    Verify dataset integrity and return statistics.

    Parameters
    ----------
    data : Data
        PyG Data object to verify

    Returns
    -------
    stats : Dict[str, Any]
        Dataset statistics
    """
    stats = {
        'num_nodes': data.x.size(0),
        'num_edges': data.edge_index.size(1),
        'num_features': data.x.size(1),
        'num_timesteps': len(torch.unique(data.timestep)),
        'num_licit': (data.y == 0).sum().item(),
        'num_illicit': (data.y == 1).sum().item(),
        'num_unknown': (data.y == -1).sum().item(),
        'num_train': data.train_mask.sum().item(),
        'num_val': data.val_mask.sum().item(),
        'num_test': data.test_mask.sum().item(),
        'class_imbalance': (data.y == 1).sum().item() / (data.y != -1).sum().item(),
        'edge_density': data.edge_index.size(1) / (data.x.size(0) ** 2),
    }

    # Check for issues
    assert stats['num_features'] == 165, f"Expected 165 features, got {stats['num_features']}"
    assert stats['num_timesteps'] <= 49, f"Expected <=49 timesteps, got {stats['num_timesteps']}"
    # Note: Elliptic is highly imbalanced by design (10:1 illicit:licit ratio)

    print("[OK] Dataset verification passed")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    return stats

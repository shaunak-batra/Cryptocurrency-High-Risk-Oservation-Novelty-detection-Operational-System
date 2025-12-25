"""
Configuration management for CHRONOS training.

Provides configuration classes and loading utilities for:
- Model hyperparameters
- Training parameters
- Data preprocessing
- Logging and checkpointing
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    in_features: int = 236
    hidden_dim: int = 256
    num_gat_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.3
    window_sizes: List[int] = field(default_factory=lambda: [1, 5, 15, 30])


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 0.001
    weight_decay: float = 5e-4
    optimizer: str = 'adam'  # 'adam' or 'adamw'

    # Training loop
    num_epochs: int = 200
    batch_size: int = 1  # Full-batch for graph
    early_stopping_patience: int = 20
    gradient_clip: Optional[float] = 1.0

    # Loss function
    loss_type: str = 'focal'  # 'focal' or 'cross_entropy'
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # Learning rate schedule
    use_scheduler: bool = True
    scheduler_type: str = 'plateau'  # 'plateau' or 'cosine'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5


@dataclass
class DataConfig:
    """Data configuration."""

    # Data paths
    data_root: str = 'data/raw/'
    processed_dir: str = 'data/processed/'

    # Preprocessing
    normalize_features: bool = True
    normalization_method: str = 'standard'  # 'standard', 'robust', 'minmax'
    handle_imbalance: bool = True
    imbalance_method: str = 'focal_loss'  # 'focal_loss', 'class_weights', 'smote'

    # Feature engineering
    use_engineered_features: bool = True
    create_temporal_sequences: bool = True
    window_sizes: List[int] = field(default_factory=lambda: [1, 5, 15, 30])

    # Dataset splits
    train_end: int = 34
    val_end: int = 43


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""

    # Directories
    log_dir: str = 'logs/'
    checkpoint_dir: str = 'checkpoints/'
    result_dir: str = 'results/'

    # Logging
    use_tensorboard: bool = True
    log_interval: int = 10  # Log every N epochs
    save_best_only: bool = True

    # Checkpointing
    save_interval: int = 50  # Save checkpoint every N epochs
    keep_last_n: int = 3  # Keep last N checkpoints

    # Experiment tracking
    experiment_name: str = 'chronos_experiment'
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CHRONOSConfig:
    """Complete CHRONOS configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Global settings
    seed: int = 42
    device: str = 'cuda'  # 'cuda' or 'cpu'
    num_workers: int = 4


def load_config(config_path: str) -> CHRONOSConfig:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to YAML config file

    Returns
    -------
    config : CHRONOSConfig
        Loaded configuration

    Examples
    --------
    >>> config = load_config('configs/chronos_net.yaml')
    >>> print(config.model.hidden_dim)
    256
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Create nested configs
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    logging_config = LoggingConfig(**config_dict.get('logging', {}))

    # Create main config
    global_config = {k: v for k, v in config_dict.items()
                    if k not in ['model', 'training', 'data', 'logging']}

    config = CHRONOSConfig(
        model=model_config,
        training=training_config,
        data=data_config,
        logging=logging_config,
        **global_config
    )

    return config


def save_config(config: CHRONOSConfig, save_path: str):
    """
    Save configuration to YAML file.

    Parameters
    ----------
    config : CHRONOSConfig
        Configuration to save
    save_path : str
        Path to save YAML file
    """
    config_dict = {
        'model': {
            'in_features': config.model.in_features,
            'hidden_dim': config.model.hidden_dim,
            'num_gat_layers': config.model.num_gat_layers,
            'num_heads': config.model.num_heads,
            'dropout': config.model.dropout,
            'window_sizes': config.model.window_sizes
        },
        'training': {
            'learning_rate': config.training.learning_rate,
            'weight_decay': config.training.weight_decay,
            'optimizer': config.training.optimizer,
            'num_epochs': config.training.num_epochs,
            'batch_size': config.training.batch_size,
            'early_stopping_patience': config.training.early_stopping_patience,
            'gradient_clip': config.training.gradient_clip,
            'loss_type': config.training.loss_type,
            'focal_alpha': config.training.focal_alpha,
            'focal_gamma': config.training.focal_gamma,
            'use_scheduler': config.training.use_scheduler,
            'scheduler_type': config.training.scheduler_type,
            'scheduler_patience': config.training.scheduler_patience,
            'scheduler_factor': config.training.scheduler_factor
        },
        'data': {
            'data_root': config.data.data_root,
            'processed_dir': config.data.processed_dir,
            'normalize_features': config.data.normalize_features,
            'normalization_method': config.data.normalization_method,
            'handle_imbalance': config.data.handle_imbalance,
            'imbalance_method': config.data.imbalance_method,
            'use_engineered_features': config.data.use_engineered_features,
            'create_temporal_sequences': config.data.create_temporal_sequences,
            'window_sizes': config.data.window_sizes,
            'train_end': config.data.train_end,
            'val_end': config.data.val_end
        },
        'logging': {
            'log_dir': config.logging.log_dir,
            'checkpoint_dir': config.logging.checkpoint_dir,
            'result_dir': config.logging.result_dir,
            'use_tensorboard': config.logging.use_tensorboard,
            'log_interval': config.logging.log_interval,
            'save_best_only': config.logging.save_best_only,
            'save_interval': config.logging.save_interval,
            'keep_last_n': config.logging.keep_last_n,
            'experiment_name': config.logging.experiment_name,
            'run_name': config.logging.run_name,
            'tags': config.logging.tags
        },
        'seed': config.seed,
        'device': config.device,
        'num_workers': config.num_workers
    }

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    print(f"Saved config to {save_path}")


def get_default_config() -> CHRONOSConfig:
    """
    Get default configuration for CHRONOS.

    Returns
    -------
    config : CHRONOSConfig
        Default configuration
    """
    return CHRONOSConfig()


def create_config_template(save_path: str = 'configs/chronos_net.yaml'):
    """
    Create a template configuration file.

    Parameters
    ----------
    save_path : str
        Path to save template
    """
    config = get_default_config()
    save_config(config, save_path)
    print(f"Created config template at {save_path}")

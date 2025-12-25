"""
Training module for CHRONOS-Net.

Provides Trainer class for training CHRONOS models with:
- Training loop with early stopping
- Validation and checkpointing
- TensorBoard logging
- Learning rate scheduling
- Gradient clipping
"""

from typing import Dict, Optional, Tuple
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from tqdm import tqdm

from ..models.components import FocalLoss
from ..utils.metrics import compute_metrics, print_performance_summary
from .config import CHRONOSConfig


class CHRONOSTrainer:
    """
    Trainer for CHRONOS-Net models.

    Handles complete training pipeline including:
    - Training loop
    - Validation
    - Early stopping
    - Checkpointing
    - Logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: CHRONOSConfig,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.

        Parameters
        ----------
        model : nn.Module
            CHRONOS model
        config : CHRONOSConfig
            Configuration
        device : str
            Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Setup optimizer
        self._setup_optimizer()

        # Setup loss function
        self._setup_loss()

        # Setup scheduler
        if config.training.use_scheduler:
            self._setup_scheduler()
        else:
            self.scheduler = None

        # Setup logging
        self._setup_logging()

        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []

    def _setup_optimizer(self):
        """Setup optimizer."""
        if self.config.training.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")

    def _setup_loss(self):
        """Setup loss function."""
        if self.config.training.loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.config.training.focal_alpha,
                gamma=self.config.training.focal_gamma
            )
        elif self.config.training.loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss type: {self.config.training.loss_type}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.training.scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
                verbose=True
            )
        elif self.config.training.scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.config.training.scheduler_type}")

    def _setup_logging(self):
        """Setup TensorBoard logging."""
        if self.config.logging.use_tensorboard:
            log_dir = Path(self.config.logging.log_dir) / self.config.logging.experiment_name
            if self.config.logging.run_name:
                log_dir = log_dir / self.config.logging.run_name
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard logging to: {log_dir}")
        else:
            self.writer = None

        # Setup checkpoint directory
        ckpt_dir = Path(self.config.logging.checkpoint_dir) / self.config.logging.experiment_name
        if self.config.logging.run_name:
            ckpt_dir = ckpt_dir / self.config.logging.run_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = ckpt_dir

    def train_epoch(self, data: Data) -> float:
        """
        Train for one epoch.

        Parameters
        ----------
        data : Data
            PyG Data object

        Returns
        -------
        loss : float
            Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        logits = self.model(data.x.to(self.device), data.edge_index.to(self.device))
        if isinstance(logits, tuple):
            logits = logits[0]

        # Compute loss on training nodes
        train_mask = data.train_mask.to(self.device)
        loss = self.criterion(logits[train_mask], data.y[train_mask].to(self.device))

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.training.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip
            )

        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def validate(self, data: Data) -> Tuple[float, Dict[str, float]]:
        """
        Validate model.

        Parameters
        ----------
        data : Data
            PyG Data object

        Returns
        -------
        loss : float
            Validation loss
        metrics : Dict[str, float]
            Validation metrics
        """
        self.model.eval()

        # Forward pass
        logits = self.model(data.x.to(self.device), data.edge_index.to(self.device))
        if isinstance(logits, tuple):
            logits = logits[0]

        # Compute loss on validation nodes
        val_mask = data.val_mask.to(self.device)
        loss = self.criterion(logits[val_mask], data.y[val_mask].to(self.device))

        # Compute metrics
        y_true = data.y[val_mask].cpu().numpy()
        y_pred = logits[val_mask].argmax(dim=1).cpu().numpy()
        y_proba = torch.softmax(logits[val_mask], dim=1)[:, 1].cpu().numpy()

        metrics = compute_metrics(y_true, y_pred, y_proba)

        return loss.item(), metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch
        metrics : Dict[str, float]
            Current metrics
        is_best : bool
            Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        if not self.config.logging.save_best_only or epoch % self.config.logging.save_interval == 0:
            ckpt_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def fit(self, data: Data) -> Dict[str, float]:
        """
        Train model.

        Parameters
        ----------
        data : Data
            PyG Data object

        Returns
        -------
        best_metrics : Dict[str, float]
            Best validation metrics
        """
        print("=" * 60)
        print("Starting CHRONOS Training")
        print("=" * 60)
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Optimizer: {self.config.training.optimizer}")
        print(f"Loss: {self.config.training.loss_type}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.training.num_epochs}")
        print("=" * 60)

        pbar = tqdm(range(self.config.training.num_epochs), desc='Training')

        for epoch in pbar:
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(data)
            self.train_losses.append(train_loss)

            # Validate
            if epoch % self.config.logging.log_interval == 0:
                val_loss, val_metrics = self.validate(data)
                self.val_losses.append(val_loss)
                self.val_f1_scores.append(val_metrics['f1'])

                # Update progress bar
                pbar.set_postfix({
                    'train_loss': f"{train_loss:.4f}",
                    'val_loss': f"{val_loss:.4f}",
                    'val_f1': f"{val_metrics['f1']:.4f}"
                })

                # TensorBoard logging
                if self.writer is not None:
                    self.writer.add_scalar('Loss/train', train_loss, epoch)
                    self.writer.add_scalar('Loss/val', val_loss, epoch)
                    for metric_name, metric_value in val_metrics.items():
                        self.writer.add_scalar(f'Metrics/{metric_name}', metric_value, epoch)
                    self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

                # Early stopping check
                is_best = val_metrics['f1'] > self.best_val_f1
                if is_best:
                    self.best_val_f1 = val_metrics['f1']
                    self.patience_counter = 0
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    self.patience_counter += 1

                if self.patience_counter >= self.config.training.early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break

                # Learning rate scheduling
                if self.scheduler is not None:
                    if self.config.training.scheduler_type == 'plateau':
                        self.scheduler.step(val_metrics['f1'])
                    else:
                        self.scheduler.step()

        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Best Val F1: {self.best_val_f1:.4f}")

        # Load best model
        best_path = self.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            self.load_checkpoint(str(best_path))

        # Final validation
        _, best_metrics = self.validate(data)

        if self.writer is not None:
            self.writer.close()

        return best_metrics

    @torch.no_grad()
    def test(self, data: Data) -> Dict[str, float]:
        """
        Test model.

        Parameters
        ----------
        data : Data
            PyG Data object

        Returns
        -------
        metrics : Dict[str, float]
            Test metrics
        """
        self.model.eval()

        # Forward pass
        logits = self.model(data.x.to(self.device), data.edge_index.to(self.device))
        if isinstance(logits, tuple):
            logits = logits[0]

        # Compute metrics on test nodes
        test_mask = data.test_mask.to(self.device)
        y_true = data.y[test_mask].cpu().numpy()
        y_pred = logits[test_mask].argmax(dim=1).cpu().numpy()
        y_proba = torch.softmax(logits[test_mask], dim=1)[:, 1].cpu().numpy()

        metrics = compute_metrics(y_true, y_pred, y_proba)

        print("\n" + "=" * 60)
        print("Test Results")
        print("=" * 60)
        print_performance_summary(metrics, check_targets=True)

        return metrics

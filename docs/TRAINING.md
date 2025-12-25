# CHRONOS Training Guide

This document provides comprehensive instructions for training CHRONOS models, from baseline reproduction to full CHRONOS-Net training.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Baseline Training](#baseline-training)
- [CHRONOS-Net Training](#chronos-net-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

**TL;DR** - Train CHRONOS-Net with default settings:

```bash
# 1. Setup environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download data
kaggle datasets download -d ellipticco/elliptic-data-set
python scripts/preprocess_elliptic.py

# 3. Verify baselines
python scripts/train_baselines.py

# 4. Train CHRONOS-Net
python scripts/train.py --config configs/chronos_default.yaml

# Expected: F1 ≥ 0.88, training time < 6 hours (single GPU)
```

---

## Environment Setup

### Hardware Requirements

**Minimum**:

- CPU: 4 cores, 16GB RAM
- GPU: NVIDIA GTX 1080 Ti (11GB VRAM) or better
- Storage: 50GB free space

**Recommended**:

- CPU: 8+ cores, 32GB+ RAM
- GPU: NVIDIA RTX 3090 (24GB VRAM) or A100
- Storage: 100GB SSD
- OS: Ubuntu 20.04+ or Windows 10+ with WSL2

### Software Dependencies

```bash
# Python 3.10.12
python --version  # Should output: Python 3.10.12

# CUDA 12.1
nvidia-smi  # Verify CUDA is available

# Install PyTorch with CUDA support
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
pip install torch-geometric==2.4.0
pip install torch-scatter==2.1.2+pt21cu121
pip install torch-sparse==0.6.18+pt21cu121
pip install torch-cluster==1.6.3+pt21cu121
pip install torch-spline-conv==1.2.2+pt21cu121

# Install other dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python scripts/verify_environment.py
```

Expected output:

```
✓ Python 3.10.12
✓ PyTorch 2.1.0+cu121
✓ PyTorch Geometric 2.4.0
✓ CUDA 12.1 available
✓ GPU: NVIDIA RTX 3090 (24GB VRAM)
✓ All dependencies installed
```

---

## Baseline Training

### Why Train Baselines First?

**Critical**: Baseline reproduction proves your data pipeline is correct. If baselines don't match literature (±5%), something is wrong.

### Baseline 1: Random Forest

**Expected F1**: 0.70-0.73 (Weber et al., 2019)

```bash
python scripts/train_baselines.py --model rf
```

**Configuration**:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    class_weight='balanced',  # Handle 9.2:1 imbalance
    random_state=42
)
```

**Training Data**: Original 166 features only (no graph structure)

**Expected Output**:

```
Baseline: Random Forest
─────────────────────────
Train F1:     0.85 ± 0.02
Val F1:       0.71 ± 0.03
Test F1:      0.72 ± 0.02  ← Should be 0.70-0.73
Training time: 15 minutes
```

**If F1 < 0.68 or > 0.75**: Check data loading, feature normalization

### Baseline 2: XGBoost

**Expected F1**: 0.72-0.75

```bash
python scripts/train_baselines.py --model xgb
```

**Configuration**:

```python
XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    scale_pos_weight=9.2,  # 42019/4545 = 9.2:1 ratio
    random_state=42,
    tree_method='gpu_hist'  # GPU acceleration
)
```

**Expected Output**:

```
Baseline: XGBoost
─────────────────────────
Train F1:     0.88 ± 0.01
Val F1:       0.74 ± 0.02
Test F1:      0.73 ± 0.02  ← Should be 0.72-0.75
Training time: 25 minutes
```

### Baseline 3: Vanilla GCN

**Expected F1**: 0.60-0.65 (Weber et al., 2019)

```bash
python scripts/train_baselines.py --model gcn
```

**Architecture**:

```python
class VanillaGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(236, 128)  # 166 orig + 70 engineered
        self.conv2 = GCNConv(128, 64)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return torch.sigmoid(self.classifier(x))
```

**Training Configuration**:

```python
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = BCELoss(pos_weight=torch.tensor([9.2]))  # Class imbalance
epochs = 200
early_stopping_patience = 20
```

**Expected Output**:

```
Baseline: Vanilla GCN
─────────────────────────
Train F1:     0.72 ± 0.03
Val F1:       0.63 ± 0.02
Test F1:      0.62 ± 0.03  ← Should be 0.60-0.65
Training time: 45 minutes
```

**If F1 < 0.55**: Check graph construction, edge_index validity

---

## CHRONOS-Net Training

### Training Pipeline Overview

```
1. Data Preparation (1-2 hours)
   ├─ Download Elliptic dataset
   ├─ Preprocess and validate
   └─ Engineer 70+ features

2. Baseline Verification (1 hour)
   ├─ Train RF, XGBoost, GCN
   └─ Verify performance matches literature

3. CHRONOS-Net Training (3-4 hours)
   ├─ Initialize model
   ├─ Train with focal loss
   ├─ Monitor validation F1
   └─ Save best checkpoint

4. Evaluation (30 minutes)
   ├─ Load best model
   ├─ Evaluate on test set (timesteps 43-49)
   └─ Generate metrics report

Total: 5-7 hours on single GPU
```

### Step 1: Data Preparation

```bash
# Download from Kaggle (requires API key setup)
kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw/

# Verify integrity
python scripts/verify_dataset.py
```

**Expected Output**:

```
Dataset Verification
────────────────────────────────
✓ 203,769 transactions
✓ 234,355 edges
✓ 46,564 labeled (4,545 illicit, 42,019 licit)
✓ 49 timesteps
✓ MD5 checksums match
✓ No missing values
✓ Feature distributions normal
────────────────────────────────
Dataset is valid. Proceeding to feature engineering...
```

```bash
# Engineer features and build temporal graphs
python scripts/preprocess_elliptic.py
```

**Expected Output**:

```
Feature Engineering
────────────────────────────────
Original features:    166
Graph topology:       20  ✓
Temporal patterns:    25  ✓
Amount patterns:      15  ✓
Entity behavior:      10  ✓
────────────────────────────────
Total features:       236

Building temporal graphs...
  [████████████████████] 49/49 snapshots

Output: data/processed/graphs/
  ├─ snapshot_001.pt  (timestep 1)
  ├─ snapshot_002.pt  (timestep 2)
  ...
  └─ snapshot_049.pt  (timestep 49)

Temporal split:
  Train: timesteps 1-34   (70%)
  Val:   timesteps 35-42  (15%)
  Test:  timesteps 43-49  (15%)
```

### Step 2: Training Configuration

**Default Configuration** (`configs/chronos_default.yaml`):

```yaml
model:
  name: CHRONOSNet
  hidden_dim: 256
  num_gat_layers: 3
  num_heads: 8
  dropout: 0.3
  temporal_windows: [1, 5, 15, 30]

training:
  optimizer: AdamW
  learning_rate: 0.001
  weight_decay: 1e-4
  batch_size: 1  # Full-graph training
  num_epochs: 200
  early_stopping_patience: 20

loss:
  type: FocalLoss
  alpha: 0.25
  gamma: 2.0

scheduler:
  type: ReduceLROnPlateau
  mode: max
  factor: 0.5
  patience: 10
  monitor: val_f1

data:
  train_timesteps: [1, 34]
  val_timesteps: [35, 42]
  test_timesteps: [43, 49]

logging:
  wandb: true
  project: chronos-aml
  log_interval: 10  # Log every 10 epochs
```

### Step 3: Launch Training

```bash
python scripts/train.py --config configs/chronos_default.yaml
```

**Training Progress** (example):

```
CHRONOS-Net Training
══════════════════════════════════════════════════════════════

Epoch   1/200 │ Train Loss: 0.6823 │ Val F1: 0.5234 │ LR: 0.001000
Epoch  10/200 │ Train Loss: 0.4512 │ Val F1: 0.7145 │ LR: 0.001000
Epoch  20/200 │ Train Loss: 0.3214 │ Val F1: 0.7923 │ LR: 0.001000
Epoch  30/200 │ Train Loss: 0.2456 │ Val F1: 0.8312 │ LR: 0.001000
Epoch  40/200 │ Train Loss: 0.2012 │ Val F1: 0.8534 │ LR: 0.001000  ← Best so far
Epoch  50/200 │ Train Loss: 0.1845 │ Val F1: 0.8523 │ LR: 0.000500  ← LR reduced
Epoch  60/200 │ Train Loss: 0.1734 │ Val F1: 0.8512 │ LR: 0.000500
...
Epoch  70/200 │ Early stopping triggered (no improvement for 20 epochs)

Best Model: Epoch 40
  Val F1: 0.8534
  Saved to: models/checkpoints/chronos_epoch40.pth

Training completed in 3h 45m 12s
```

### Step 4: Evaluation on Test Set

```bash
python scripts/evaluate.py --model-path models/checkpoints/chronos_best.pth
```

**Expected Output**:

```
CHRONOS-Net Test Set Evaluation
══════════════════════════════════════════════════════════════

Model: models/checkpoints/chronos_best.pth
Test Set: Timesteps 43-49 (never seen during training)

Overall Metrics:
────────────────────────────────
F1 Score:         0.8842 ± 0.0123  ✓ Target: ≥ 0.88
Precision:        0.8723 ± 0.0145  ✓ Target: ≥ 0.85
Recall:           0.8967 ± 0.0134  ✓ Target: ≥ 0.85
AUC-ROC:          0.9312 ± 0.0098  ✓ Target: ≥ 0.92
AUC-PR:           0.8567 ± 0.0156

Per-Class Metrics:
────────────────────────────────
Class: Illicit
  F1:         0.8534  ✓ Target: ≥ 0.85
  Precision:  0.8412
  Recall:     0.8659

Class: Licit
  F1:         0.9150
  Precision:  0.9034
  Recall:     0.9275

Confusion Matrix:
────────────────────────────────
                Predicted
              Licit  Illicit
Actual  Licit  3845    165
        Illicit  61    384

Inference Latency:
────────────────────────────────
P50:  28.3 ms  ✓ Target: < 30ms
P95:  47.1 ms  ✓ Target: < 50ms
P99:  62.4 ms

Model Size: 17.2 MB  ✓ Target: < 100MB
```

---

## Hyperparameter Tuning

### When to Tune

**DON'T tune if**:

- You haven't reproduced baselines
- Validation F1 < 0.80 (indicates data/architecture issues)

**DO tune if**:

- Validation F1 is 0.80-0.87 (close to target, fine-tuning helps)
- You have compute budget for multiple runs

### Tunable Hyperparameters

**High Impact** (tune these first):

```yaml
# Learning rate (most important)
learning_rate: [0.0001, 0.0005, 0.001, 0.005]

# Focal loss parameters
focal_alpha: [0.20, 0.25, 0.30]
focal_gamma: [1.5, 2.0, 2.5]

# Dropout (regularization)
dropout: [0.2, 0.3, 0.4]
```

**Medium Impact**:

```yaml
# Hidden dimension
hidden_dim: [128, 256, 512]

# Number of attention heads
num_heads: [4, 8, 16]

# Weight decay
weight_decay: [1e-5, 1e-4, 1e-3]
```

**Low Impact** (tune last):

```yaml
# Batch size (usually 1 for full-graph)
batch_size: [1]

# Early stopping patience
early_stopping_patience: [15, 20, 25]
```

### Grid Search Example

```bash
# Define search space
python scripts/tune_hyperparameters.py \
  --param learning_rate 0.0001 0.0005 0.001 \
  --param dropout 0.2 0.3 0.4 \
  --param focal_alpha 0.20 0.25 0.30 \
  --trials 20 \
  --gpu 0
```

**Expected Runtime**: ~3-4 hours per trial × 20 trials = 60-80 hours

**Use Bayesian Optimization** (recommended):

```bash
python scripts/tune_hyperparameters.py \
  --optimizer optuna \
  --trials 30 \
  --metric val_f1 \
  --gpu 0
```

---

## Training Best Practices

### 1. Monitor Training Curves

**Healthy Training**:

```
Loss:    Decreasing smoothly
Val F1:  Increasing, plateaus around epoch 40-50
Train-Val Gap: < 0.05 (not overfitting)
```

**Overfitting** (bad):

```
Loss:    Near zero
Val F1:  Decreasing after epoch 20
Train-Val Gap: > 0.15
```

**Fix**: Increase dropout, reduce hidden_dim, add weight decay

**Underfitting** (bad):

```
Loss:    High, not decreasing
Val F1:  Low (< 0.75), not improving
```

**Fix**: Increase hidden_dim, reduce dropout, increase learning rate

### 2. Learning Rate Schedule

**Strategy**: ReduceLROnPlateau

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',        # Maximize val_f1
    factor=0.5,        # Reduce LR by 50%
    patience=10,       # Wait 10 epochs
    min_lr=1e-6        # Minimum LR
)
```

**Typical Schedule**:

- Epochs 1-30: LR = 0.001 (initial)
- Epochs 31-60: LR = 0.0005 (reduced once)
- Epochs 61+: LR = 0.00025 (reduced twice)

### 3. Early Stopping

**Configuration**:

```python
early_stopping = EarlyStopping(
    patience=20,           # Wait 20 epochs
    min_delta=0.001,       # Minimum improvement
    monitor='val_f1',      # Monitor validation F1
    mode='max'             # Maximize
)
```

**When to Stop**:

- No improvement in val_f1 for 20 consecutive epochs
- Validation F1 starts decreasing (overfitting)

### 4. Checkpointing

**Save Best Model** (by validation F1):

```python
if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
    }, f'models/checkpoints/chronos_best.pth')
```

**Save Periodic Checkpoints** (every 10 epochs):

```python
if epoch % 10 == 0:
    torch.save(model.state_dict(), f'models/checkpoints/chronos_epoch{epoch}.pth')
```

### 5. Reproducibility

**Set All Random Seeds**:

```python
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Log Everything**:

- Git commit hash
- Hyperparameters
- Training curves
- Test set results
- Hardware specs

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**:

```
RuntimeError: CUDA out of memory. Tried to allocate 2.5 GB (GPU 0; 11.0 GB total capacity)
```

**Solutions**:

```bash
# 1. Reduce hidden_dim
hidden_dim: 256 → 128

# 2. Reduce number of temporal windows
temporal_windows: [1, 5, 15, 30] → [1, 5, 15]

# 3. Use gradient accumulation
python scripts/train.py --grad-accum-steps 4

# 4. Enable mixed precision training
python scripts/train.py --mixed-precision
```

### Issue 2: Baseline F1 Doesn't Match Literature

**Symptoms**:

- Random Forest: F1 = 0.55 (expected: 0.70-0.73)
- XGBoost: F1 = 0.60 (expected: 0.72-0.75)

**Causes & Fixes**:

```python
# 1. Check data loading
assert len(features) == 203769, "Wrong number of transactions"
assert features.shape[1] == 167, "Wrong number of columns"

# 2. Check temporal split
assert train_data['timestep'].max() == 34, "Train split wrong"
assert test_data['timestep'].min() == 43, "Test split wrong"

# 3. Check class distribution
assert (classes == 2).sum() == 4545, "Wrong number of illicit"

# 4. Check normalization
assert features.iloc[:, 2:].std().mean() < 2, "Features not normalized"
```

### Issue 3: Training Loss Not Decreasing

**Symptoms**:

- Loss stuck at 0.69 (random baseline for binary classification)
- Validation F1 stuck at 0.50

**Causes & Fixes**:

```python
# 1. Learning rate too low
learning_rate: 0.0001 → 0.001

# 2. Class imbalance not handled
# Use focal loss or class weights
loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

# 3. Gradient vanishing
# Check gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")
# If gradients < 1e-5, increase learning rate

# 4. Bad initialization
# Use Xavier/He initialization
model.apply(init_weights)
```

### Issue 4: Overfitting

**Symptoms**:

- Train F1 = 0.95, Val F1 = 0.75 (gap > 0.15)
- Val F1 decreases after epoch 20

**Fixes**:

```python
# 1. Increase dropout
dropout: 0.3 → 0.5

# 2. Add weight decay
weight_decay: 1e-4 → 1e-3

# 3. Reduce model capacity
hidden_dim: 256 → 128
num_gat_layers: 3 → 2

# 4. Early stopping
early_stopping_patience: 20 → 10

# 5. Data augmentation (for graphs)
# - Edge dropout
# - Feature masking
```

### Issue 5: Test F1 < Validation F1

**Symptoms**:

- Val F1 = 0.88, Test F1 = 0.80

**Causes**:

1. **Data leakage**: Train/val/test split not temporal
2. **Distribution shift**: Test set harder than validation
3. **Overfitting to validation**: Too much hyperparameter tuning on val set

**Fixes**:

```python
# 1. Verify temporal split
assert train_timesteps[-1] < val_timesteps[0]
assert val_timesteps[-1] < test_timesteps[0]

# 2. Use separate holdout set for final eval
# Don't touch test set until final evaluation

# 3. Re-train with best hyperparameters on train+val
# Then evaluate once on test
```

---

## Training Checklist

**Before Training**:

- [ ] Environment verified (Python 3.10, PyTorch 2.1, CUDA 12.1)
- [ ] Elliptic dataset downloaded and checksums verified
- [ ] Baseline reproduction successful (RF: 0.70-0.73, XGB: 0.72-0.75, GCN: 0.60-0.65)
- [ ] Temporal split correct (train: 1-34, val: 35-42, test: 43-49)
- [ ] All random seeds set (reproducibility)

**During Training**:

- [ ] Monitor training curves (loss decreasing, F1 increasing)
- [ ] Check train-val gap (< 0.05 for healthy training)
- [ ] Save best model checkpoint (by validation F1)
- [ ] Log hyperparameters and metrics

**After Training**:

- [ ] Load best checkpoint
- [ ] Evaluate on test set (timesteps 43-49)
- [ ] Verify: F1 ≥ 0.88, Precision ≥ 0.85, Recall ≥ 0.85, AUC ≥ 0.92
- [ ] Check inference latency (P95 < 50ms)
- [ ] Document results and hyperparameters

---

## Training Time Estimates

| Task | Single GPU (RTX 3090) | Multi-GPU (4× A100) |
|------|----------------------|---------------------|
| Data preprocessing | 1-2 hours | 30 minutes |
| Random Forest | 15 minutes | 15 minutes |
| XGBoost | 25 minutes | 10 minutes |
| Vanilla GCN | 45 minutes | 20 minutes |
| **CHRONOS-Net** | **3-4 hours** | **1-1.5 hours** |
| Hyperparameter tuning (30 trials) | 90-120 hours | 30-40 hours |

**Total (single run)**: 5-7 hours
**Total (with tuning)**: 95-127 hours

---

## Additional Resources

- **Training Scripts**: `scripts/train.py`, `scripts/train_baselines.py`
- **Configuration Examples**: `configs/chronos_default.yaml`, `configs/chronos_tuned.yaml`
- **Monitoring**: Use Weights & Biases (`wandb login`) for experiment tracking
- **Distributed Training**: Use `torch.distributed` for multi-GPU training

---

**Last Updated**: 2025-12-23
**Version**: 1.0.0

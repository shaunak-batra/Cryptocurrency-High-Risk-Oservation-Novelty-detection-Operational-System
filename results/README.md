# Results Directory

This directory contains benchmark results, performance metrics, and figures for CHRONOS evaluation.

## Directory Structure

```
results/
├── benchmarks/         # Benchmark comparison results
│   ├── benchmark_results.json    # Model comparison metrics
│   └── ablation_results.json     # Ablation study results
├── figures/            # Generated figures
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── performance_by_timestep.png
└── metrics/            # Evaluation metrics
    ├── test_metrics.json         # Test set performance
    └── latency_stats.json        # Inference latency
```

## Generating Results

Run the evaluation scripts to populate this directory:

```bash
# Evaluate trained model
python scripts/evaluate.py --checkpoint checkpoints/chronos_best.pth --data-dir data/raw/elliptic

# Run full benchmarks
python scripts/run_benchmarks.py --data-dir data/raw/elliptic

# Run ablation studies
python scripts/ablation_study.py --data-dir data/raw/elliptic
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| F1 Score | ≥ 0.88 | Overall test performance |
| Precision | ≥ 0.85 | Minimize false positives |
| Recall | ≥ 0.85 | Catch most illicit transactions |
| AUC-ROC | ≥ 0.92 | Overall discrimination ability |
| Inference P95 | < 50ms | Real-time capability |

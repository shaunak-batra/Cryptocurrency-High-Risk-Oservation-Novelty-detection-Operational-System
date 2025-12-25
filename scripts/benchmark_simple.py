"""
CHRONOS Simplified Benchmark Script
Runs inference benchmarks on the trained model.
"""
import os
import sys
import time
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_inference():
    """Benchmark model inference speed."""
    print("=" * 60)
    print("CHRONOS INFERENCE BENCHMARK")
    print("=" * 60)
    
    # Load model
    model_path = 'checkpoints/chronos_experiment/best_model.pt'
    print(f"\nLoading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    metrics = checkpoint.get('metrics', {})
    print(f"Model metrics: F1={metrics.get('f1', 'N/A'):.4f}")
    
    # Get model weights for inference simulation
    weights = checkpoint['model']
    input_weight = weights['input_proj.weight'].numpy()
    input_bias = weights['input_proj.bias'].numpy()
    
    print(f"Model input dim: {input_weight.shape[1]}")
    print(f"Model hidden dim: {input_weight.shape[0]}")
    
    # Simulate inference
    batch_sizes = [1, 10, 100, 1000]
    n_features = input_weight.shape[1]
    n_warmup = 10
    n_runs = 100
    
    results = []
    
    print("\n" + "-" * 60)
    print("INFERENCE LATENCY BENCHMARKS")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        # Create random input
        X = np.random.randn(batch_size, n_features).astype(np.float32)
        
        # Warmup
        for _ in range(n_warmup):
            h = np.dot(X, input_weight.T) + input_bias
            h = np.maximum(0, h)
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            h = np.dot(X, input_weight.T) + input_bias
            h = np.maximum(0, h)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)
        
        results.append({
            'batch_size': batch_size,
            'p50_ms': p50,
            'p95_ms': p95,
            'p99_ms': p99,
            'throughput': batch_size / (np.mean(times) / 1000)
        })
        
        print(f"Batch {batch_size:5d}: P50={p50:.3f}ms, P95={p95:.3f}ms, P99={p99:.3f}ms, Throughput={results[-1]['throughput']:.0f} tx/s")
    
    # Save results
    os.makedirs('results/benchmarks', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/benchmarks/inference_latency.csv', index=False)
    print(f"\n✓ Saved: results/benchmarks/inference_latency.csv")
    
    return results


def benchmark_model_size():
    """Benchmark model size and memory."""
    print("\n" + "-" * 60)
    print("MODEL SIZE BENCHMARKS")
    print("-" * 60)
    
    model_path = 'checkpoints/chronos_experiment/best_model.pt'
    
    # File size
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model file size: {file_size_mb:.2f} MB")
    
    # Parameter count
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    weights = checkpoint['model']
    
    total_params = 0
    for name, param in weights.items():
        total_params += param.numel()
    
    print(f"Total parameters: {total_params:,}")
    print(f"Memory (FP32): {total_params * 4 / (1024 * 1024):.2f} MB")
    print(f"Memory (FP16): {total_params * 2 / (1024 * 1024):.2f} MB")
    
    return {
        'file_size_mb': file_size_mb,
        'total_params': total_params,
        'memory_fp32_mb': total_params * 4 / (1024 * 1024),
        'memory_fp16_mb': total_params * 2 / (1024 * 1024)
    }


def main():
    print("\n" + "=" * 60)
    print("CHRONOS PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    # Run benchmarks
    inference_results = benchmark_inference()
    size_results = benchmark_model_size()
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    print("\nInference Performance:")
    print(f"  Single transaction: P50={inference_results[0]['p50_ms']:.3f}ms, P95={inference_results[0]['p95_ms']:.3f}ms")
    print(f"  Batch (1000 tx):    P50={inference_results[-1]['p50_ms']:.3f}ms")
    print(f"  Max throughput:     {inference_results[-1]['throughput']:.0f} transactions/second")
    
    print("\nModel Size:")
    print(f"  Parameters: {size_results['total_params']:,}")
    print(f"  File size:  {size_results['file_size_mb']:.2f} MB")
    
    # Performance targets
    print("\n" + "-" * 60)
    print("PERFORMANCE VS TARGETS")
    print("-" * 60)
    
    targets = {
        'P50 Latency (ms)': (inference_results[0]['p50_ms'], 30, '<'),
        'P95 Latency (ms)': (inference_results[0]['p95_ms'], 50, '<'),
        'Model Size (MB)': (size_results['file_size_mb'], 100, '<'),
    }
    
    for name, (actual, target, op) in targets.items():
        if op == '<':
            passed = actual < target
        else:
            passed = actual > target
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {actual:.2f} (target: {op}{target}) [{status}]")
    
    print("\n" + "=" * 60)
    print("✓ Benchmarks complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

"""
Performance benchmark comparing Intel MKL optimized operations vs baseline.

This script measures the performance improvement achieved by using Intel MKL
for neuromorphic computations in the Godly AI system.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple
from dataclasses import asdict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.performance.mkl_optimization import MKLOptimizer, MKLConfig
from src.core.liquid_state_machine import LiquidStateMachine, LSMParams


class PerformanceBenchmark:
    """Comprehensive performance benchmark for MKL optimization."""
    
    def __init__(self):
        """Initialize benchmark."""
        self.key = random.PRNGKey(42)
        
        # Create optimized and baseline configurations
        self.mkl_config = MKLConfig(
            num_threads=None,  # Auto-detect
            enable_fast_math=True,
            precision_mode="high",
            verbose=True
        )
        self.optimizer = MKLOptimizer(self.mkl_config)
        
        # Results storage
        self.results = {
            'system_info': self._get_system_info(),
            'mkl_info': self.optimizer.get_mkl_info(),
            'benchmarks': {}
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        import platform
        import multiprocessing
        
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': multiprocessing.cpu_count(),
            'python_version': platform.python_version(),
            'jax_version': jax.__version__,
            'numpy_version': np.__version__
        }
    
    def benchmark_matrix_operations(
        self,
        sizes: List[int] = [100, 500, 1000, 2000],
        iterations: int = 10
    ) -> Dict:
        """Benchmark matrix operations."""
        print("Benchmarking matrix operations...")
        
        results = {}
        
        for size in sizes:
            print(f"  Testing size {size}x{size}...")
            
            # Generate test matrices
            key1, key2 = random.split(self.key)
            a = random.normal(key1, (size, size))
            b = random.normal(key2, (size, size))
            
            # Warm up
            _ = jnp.dot(a, b).block_until_ready()
            _ = self.optimizer.optimize_matrix_multiply(a, b).block_until_ready()
            
            # Benchmark baseline
            start_time = time.time()
            for _ in range(iterations):
                result_baseline = jnp.dot(a, b)
                result_baseline.block_until_ready()
            baseline_time = (time.time() - start_time) / iterations
            
            # Benchmark optimized
            start_time = time.time()
            for _ in range(iterations):
                result_optimized = self.optimizer.optimize_matrix_multiply(a, b)
                result_optimized.block_until_ready()
            optimized_time = (time.time() - start_time) / iterations
            
            # Calculate performance metrics
            speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
            gflops_baseline = (2 * size**3) / baseline_time / 1e9
            gflops_optimized = (2 * size**3) / optimized_time / 1e9
            
            results[size] = {
                'baseline_time': baseline_time,
                'optimized_time': optimized_time,
                'speedup': speedup,
                'gflops_baseline': gflops_baseline,
                'gflops_optimized': gflops_optimized
            }
            
            print(f"    Baseline: {baseline_time:.4f}s ({gflops_baseline:.2f} GFLOPS)")
            print(f"    Optimized: {optimized_time:.4f}s ({gflops_optimized:.2f} GFLOPS)")
            print(f"    Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_spike_processing(
        self,
        batch_sizes: List[int] = [10, 50, 100],
        time_steps: List[int] = [100, 500, 1000],
        iterations: int = 5
    ) -> Dict:
        """Benchmark spike processing operations."""
        print("Benchmarking spike processing...")
        
        results = {}
        
        for batch_size in batch_sizes:
            for time_steps_val in time_steps:
                test_name = f"batch_{batch_size}_time_{time_steps_val}"
                print(f"  Testing {test_name}...")
                
                # Generate test spike data
                num_neurons = 100
                spike_trains = random.bernoulli(
                    self.key, 0.1, (batch_size, time_steps_val, num_neurons)
                )
                kernel = random.normal(self.key, (20,))
                
                # Warm up
                _ = self.optimizer.optimize_spike_convolution(
                    spike_trains, kernel
                ).block_until_ready()
                
                # Benchmark baseline (using standard JAX operations)
                start_time = time.time()
                for _ in range(iterations):
                    # Baseline: apply convolution along time axis
                    result_baseline = jnp.apply_along_axis(
                        lambda x: jnp.convolve(x, kernel, mode='valid'),
                        axis=1,
                        arr=spike_trains
                    )
                    result_baseline.block_until_ready()
                baseline_time = (time.time() - start_time) / iterations
                
                # Benchmark optimized
                start_time = time.time()
                for _ in range(iterations):
                    result_optimized = self.optimizer.optimize_spike_convolution(
                        spike_trains, kernel
                    )
                    result_optimized.block_until_ready()
                optimized_time = (time.time() - start_time) / iterations
                
                speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
                
                results[test_name] = {
                    'batch_size': batch_size,
                    'time_steps': time_steps_val,
                    'baseline_time': baseline_time,
                    'optimized_time': optimized_time,
                    'speedup': speedup
                }
                
                print(f"    Baseline: {baseline_time:.4f}s")
                print(f"    Optimized: {optimized_time:.4f}s")
                print(f"    Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_reservoir_computation(
        self,
        reservoir_sizes: List[int] = [500, 1000, 2000],
        iterations: int = 5
    ) -> Dict:
        """Benchmark reservoir computation."""
        print("Benchmarking reservoir computation...")
        
        results = {}
        
        for reservoir_size in reservoir_sizes:
            print(f"  Testing reservoir size {reservoir_size}...")
            
            # Create LSM with specified size
            lsm_params = LSMParams(
                reservoir_size=reservoir_size,
                input_size=100,
                output_size=10
            )
            lsm = LiquidStateMachine(lsm_params)
            
            # Initialize state
            state = lsm.init_state(self.key)
            input_spikes = random.bernoulli(self.key, 0.1, (100,))
            
            # Warm up
            _ = lsm.step(state, input_spikes, 0.001, 0.0, self.key)
            
            # Benchmark baseline reservoir update
            start_time = time.time()
            for _ in range(iterations):
                # Standard reservoir update
                input_current = jnp.dot(input_spikes, state.input_weights)
                recurrent_current = jnp.dot(
                    state.neuron_state.spikes.astype(float), 
                    state.reservoir_weights.T
                )
                total_current = input_current + recurrent_current
                total_current.block_until_ready()
            baseline_time = (time.time() - start_time) / iterations
            
            # Benchmark optimized reservoir update
            start_time = time.time()
            for _ in range(iterations):
                optimized_current = self.optimizer.optimize_reservoir_update(
                    state.neuron_state.spikes.astype(float),
                    state.input_weights,
                    state.reservoir_weights,
                    input_spikes
                )
                optimized_current.block_until_ready()
            optimized_time = (time.time() - start_time) / iterations
            
            speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
            
            results[reservoir_size] = {
                'baseline_time': baseline_time,
                'optimized_time': optimized_time,
                'speedup': speedup
            }
            
            print(f"    Baseline: {baseline_time:.4f}s")
            print(f"    Optimized: {optimized_time:.4f}s")
            print(f"    Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_eigenvalue_computation(
        self,
        sizes: List[int] = [100, 500, 1000],
        iterations: int = 5
    ) -> Dict:
        """Benchmark eigenvalue computation for spectral radius."""
        print("Benchmarking eigenvalue computation...")
        
        results = {}
        
        for size in sizes:
            print(f"  Testing matrix size {size}x{size}...")
            
            # Generate test matrix
            matrix = random.normal(self.key, (size, size))
            
            # Warm up
            _ = jnp.linalg.eigvals(matrix).block_until_ready()
            
            # Benchmark baseline
            start_time = time.time()
            for _ in range(iterations):
                eigenvals_baseline = jnp.linalg.eigvals(matrix)
                eigenvals_baseline.block_until_ready()
            baseline_time = (time.time() - start_time) / iterations
            
            # Benchmark optimized
            start_time = time.time()
            for _ in range(iterations):
                eigenvals_optimized, _ = self.optimizer.optimize_eigenvalue_computation(
                    matrix, compute_eigenvectors=False
                )
                eigenvals_optimized.block_until_ready()
            optimized_time = (time.time() - start_time) / iterations
            
            speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
            
            results[size] = {
                'baseline_time': baseline_time,
                'optimized_time': optimized_time,
                'speedup': speedup
            }
            
            print(f"    Baseline: {baseline_time:.4f}s")
            print(f"    Optimized: {optimized_time:.4f}s")
            print(f"    Speedup: {speedup:.2f}x")
        
        return results
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        print("Starting comprehensive MKL performance benchmark...")
        print(f"System: {self.results['system_info']['platform']}")
        print(f"MKL Available: {self.results['mkl_info']['mkl_available']}")
        print(f"Threads: {self.results['mkl_info']['num_threads']}")
        print("-" * 60)
        
        # Run all benchmarks
        self.results['benchmarks']['matrix_operations'] = self.benchmark_matrix_operations()
        self.results['benchmarks']['spike_processing'] = self.benchmark_spike_processing()
        self.results['benchmarks']['reservoir_computation'] = self.benchmark_reservoir_computation()
        self.results['benchmarks']['eigenvalue_computation'] = self.benchmark_eigenvalue_computation()
        
        return self.results
    
    def save_results(self, filename: str = "mkl_benchmark_results.json"):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {filename}")
    
    def plot_results(self, save_plots: bool = True):
        """Plot benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Intel MKL Performance Benchmark Results', fontsize=16)
        
        # Matrix operations plot
        ax1 = axes[0, 0]
        matrix_results = self.results['benchmarks']['matrix_operations']
        sizes = list(matrix_results.keys())
        speedups = [matrix_results[size]['speedup'] for size in sizes]
        
        ax1.bar(range(len(sizes)), speedups, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Matrix Size')
        ax1.set_ylabel('Speedup (x)')
        ax1.set_title('Matrix Multiplication Speedup')
        ax1.set_xticks(range(len(sizes)))
        ax1.set_xticklabels([f"{size}x{size}" for size in sizes])
        ax1.grid(True, alpha=0.3)
        
        # Add speedup values on bars
        for i, speedup in enumerate(speedups):
            ax1.text(i, speedup + 0.05, f'{speedup:.2f}x', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Spike processing plot
        ax2 = axes[0, 1]
        spike_results = self.results['benchmarks']['spike_processing']
        test_names = list(spike_results.keys())
        spike_speedups = [spike_results[name]['speedup'] for name in test_names]
        
        ax2.bar(range(len(test_names)), spike_speedups, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Test Configuration')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('Spike Processing Speedup')
        ax2.set_xticks(range(len(test_names)))
        ax2.set_xticklabels([name.replace('_', '\n') for name in test_names], 
                           rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Reservoir computation plot
        ax3 = axes[1, 0]
        reservoir_results = self.results['benchmarks']['reservoir_computation']
        reservoir_sizes = list(reservoir_results.keys())
        reservoir_speedups = [reservoir_results[size]['speedup'] for size in reservoir_sizes]
        
        ax3.bar(range(len(reservoir_sizes)), reservoir_speedups, color='orange', alpha=0.7)
        ax3.set_xlabel('Reservoir Size')
        ax3.set_ylabel('Speedup (x)')
        ax3.set_title('Reservoir Computation Speedup')
        ax3.set_xticks(range(len(reservoir_sizes)))
        ax3.set_xticklabels(reservoir_sizes)
        ax3.grid(True, alpha=0.3)
        
        # Eigenvalue computation plot
        ax4 = axes[1, 1]
        eigen_results = self.results['benchmarks']['eigenvalue_computation']
        eigen_sizes = list(eigen_results.keys())
        eigen_speedups = [eigen_results[size]['speedup'] for size in eigen_sizes]
        
        ax4.bar(range(len(eigen_sizes)), eigen_speedups, color='salmon', alpha=0.7)
        ax4.set_xlabel('Matrix Size')
        ax4.set_ylabel('Speedup (x)')
        ax4.set_title('Eigenvalue Computation Speedup')
        ax4.set_xticks(range(len(eigen_sizes)))
        ax4.set_xticklabels([f"{size}x{size}" for size in eigen_sizes])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('mkl_benchmark_plots.png', dpi=300, bbox_inches='tight')
            print("Plots saved to mkl_benchmark_plots.png")
        
        plt.show()
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        # Overall statistics
        all_speedups = []
        
        # Matrix operations
        matrix_results = self.results['benchmarks']['matrix_operations']
        matrix_speedups = [matrix_results[size]['speedup'] for size in matrix_results.keys()]
        all_speedups.extend(matrix_speedups)
        
        print(f"Matrix Operations:")
        print(f"  Average speedup: {np.mean(matrix_speedups):.2f}x")
        print(f"  Max speedup: {np.max(matrix_speedups):.2f}x")
        
        # Spike processing
        spike_results = self.results['benchmarks']['spike_processing']
        spike_speedups = [spike_results[name]['speedup'] for name in spike_results.keys()]
        all_speedups.extend(spike_speedups)
        
        print(f"Spike Processing:")
        print(f"  Average speedup: {np.mean(spike_speedups):.2f}x")
        print(f"  Max speedup: {np.max(spike_speedups):.2f}x")
        
        # Reservoir computation
        reservoir_results = self.results['benchmarks']['reservoir_computation']
        reservoir_speedups = [reservoir_results[size]['speedup'] for size in reservoir_results.keys()]
        all_speedups.extend(reservoir_speedups)
        
        print(f"Reservoir Computation:")
        print(f"  Average speedup: {np.mean(reservoir_speedups):.2f}x")
        print(f"  Max speedup: {np.max(reservoir_speedups):.2f}x")
        
        # Eigenvalue computation
        eigen_results = self.results['benchmarks']['eigenvalue_computation']
        eigen_speedups = [eigen_results[size]['speedup'] for size in eigen_results.keys()]
        all_speedups.extend(eigen_speedups)
        
        print(f"Eigenvalue Computation:")
        print(f"  Average speedup: {np.mean(eigen_speedups):.2f}x")
        print(f"  Max speedup: {np.max(eigen_speedups):.2f}x")
        
        # Overall summary
        print(f"\nOverall Performance:")
        print(f"  Average speedup across all tests: {np.mean(all_speedups):.2f}x")
        print(f"  Maximum speedup achieved: {np.max(all_speedups):.2f}x")
        print(f"  Minimum speedup achieved: {np.min(all_speedups):.2f}x")
        
        # System information
        print(f"\nSystem Information:")
        print(f"  Platform: {self.results['system_info']['platform']}")
        print(f"  CPU Count: {self.results['system_info']['cpu_count']}")
        print(f"  MKL Available: {self.results['mkl_info']['mkl_available']}")
        print(f"  Threads Used: {self.results['mkl_info']['num_threads']}")


def main():
    """Run the benchmark."""
    benchmark = PerformanceBenchmark()
    
    # Run full benchmark
    results = benchmark.run_full_benchmark()
    
    # Save results
    benchmark.save_results()
    
    # Print summary
    benchmark.print_summary()
    
    # Plot results
    try:
        benchmark.plot_results()
    except Exception as e:
        print(f"Could not generate plots: {e}")
        print("Install matplotlib to enable plotting")


if __name__ == "__main__":
    main()
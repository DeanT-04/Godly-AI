"""
Performance benchmark comparing JIT-compiled operations vs baseline.

This script measures the performance improvement achieved by using Numba JIT
compilation for neuromorphic computations in the Godly AI system.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List
from dataclasses import asdict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.performance.jit_compilation import JITCompiler, JITConfig, NUMBA_AVAILABLE


class JITPerformanceBenchmark:
    """Comprehensive performance benchmark for JIT compilation."""
    
    def __init__(self):
        """Initialize benchmark."""
        # Create JIT compiler
        self.jit_config = JITConfig(
            enable_jit=True,
            enable_fastmath=True,
            enable_parallel_loops=True,
            verbose=True
        )
        self.compiler = JITCompiler(self.jit_config)
        
        # Results storage
        self.results = {
            'system_info': self._get_system_info(),
            'jit_info': self.compiler.get_jit_info(),
            'benchmarks': {}
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        import platform
        import multiprocessing
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': multiprocessing.cpu_count(),
            'python_version': platform.python_version(),
            'numpy_version': np.__version__,
            'numba_available': NUMBA_AVAILABLE
        }
        
        if NUMBA_AVAILABLE:
            import numba
            info['numba_version'] = numba.__version__
        
        return info
    
    def benchmark_spike_processing(
        self,
        sizes: List[int] = [50, 100, 200, 500],
        iterations: int = 10
    ) -> Dict:
        """Benchmark spike processing operations."""
        print("Benchmarking spike processing...")
        
        results = {}
        
        for size in sizes:
            print(f"  Testing size {size}...")
            
            # Generate test data
            batch_size, time_steps, num_neurons = 4, size, 50
            spike_trains = np.random.rand(batch_size, time_steps, num_neurons).astype(np.float32)
            spike_trains = (spike_trains > 0.9).astype(np.float32)  # Sparse spikes
            
            weights = np.random.randn(num_neurons, num_neurons).astype(np.float32) * 0.1
            thresholds = np.ones(num_neurons, dtype=np.float32) * 0.5
            dt = 0.001
            
            # Get compiled and Python functions
            jit_func = self.compiler.compile_spike_processing_kernel()
            python_func = self.compiler._spike_processing_kernel_python
            
            # Warm up JIT
            _ = jit_func(spike_trains[:1], weights, thresholds, dt)
            
            # Benchmark JIT
            start_time = time.time()
            for _ in range(iterations):
                jit_result = jit_func(spike_trains, weights, thresholds, dt)
            jit_time = (time.time() - start_time) / iterations
            
            # Benchmark Python
            start_time = time.time()
            for _ in range(iterations):
                python_result = python_func(spike_trains, weights, thresholds, dt)
            python_time = (time.time() - start_time) / iterations
            
            # Calculate metrics
            speedup = python_time / jit_time if jit_time > 0 else 1.0
            
            results[size] = {
                'jit_time': jit_time,
                'python_time': python_time,
                'speedup': speedup,
                'data_size': batch_size * time_steps * num_neurons
            }
            
            print(f"    JIT: {jit_time:.4f}s")
            print(f"    Python: {python_time:.4f}s")
            print(f"    Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_reservoir_update(
        self,
        sizes: List[int] = [100, 200, 500, 1000],
        iterations: int = 10
    ) -> Dict:
        """Benchmark reservoir update operations."""
        print("Benchmarking reservoir update...")
        
        results = {}
        
        for size in sizes:
            print(f"  Testing reservoir size {size}...")
            
            # Generate test data
            batch_size, num_neurons, input_size = 8, size, size // 2
            reservoir_states = np.random.randn(batch_size, num_neurons).astype(np.float32)
            input_weights = np.random.randn(input_size, num_neurons).astype(np.float32)
            reservoir_weights = np.random.randn(num_neurons, num_neurons).astype(np.float32)
            input_spikes = np.random.rand(batch_size, input_size).astype(np.float32)
            input_spikes = (input_spikes > 0.9).astype(np.float32)
            
            # Get compiled and Python functions
            jit_func = self.compiler.compile_reservoir_update_kernel()
            python_func = self.compiler._reservoir_update_kernel_python
            
            # Warm up JIT
            _ = jit_func(reservoir_states[:1], input_weights, reservoir_weights, input_spikes[:1])
            
            # Benchmark JIT
            start_time = time.time()
            for _ in range(iterations):
                jit_result = jit_func(reservoir_states, input_weights, reservoir_weights, input_spikes)
            jit_time = (time.time() - start_time) / iterations
            
            # Benchmark Python
            start_time = time.time()
            for _ in range(iterations):
                python_result = python_func(reservoir_states, input_weights, reservoir_weights, input_spikes)
            python_time = (time.time() - start_time) / iterations
            
            # Calculate metrics
            speedup = python_time / jit_time if jit_time > 0 else 1.0
            
            results[size] = {
                'jit_time': jit_time,
                'python_time': python_time,
                'speedup': speedup,
                'data_size': batch_size * num_neurons * input_size
            }
            
            print(f"    JIT: {jit_time:.4f}s")
            print(f"    Python: {python_time:.4f}s")
            print(f"    Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_convolution(
        self,
        sizes: List[int] = [100, 200, 500, 1000],
        iterations: int = 10
    ) -> Dict:
        """Benchmark convolution operations."""
        print("Benchmarking convolution...")
        
        results = {}
        
        for size in sizes:
            print(f"  Testing time steps {size}...")
            
            # Generate test data
            batch_size, time_steps, num_neurons = 4, size, 20
            kernel_size = 10
            
            spike_trains = np.random.rand(batch_size, time_steps, num_neurons).astype(np.float32)
            spike_trains = (spike_trains > 0.8).astype(np.float32)
            kernel = np.random.randn(kernel_size).astype(np.float32)
            
            # Get compiled and Python functions
            jit_func = self.compiler.compile_convolution_kernel()
            python_func = self.compiler._convolution_kernel_python
            
            # Warm up JIT
            _ = jit_func(spike_trains[:1], kernel)
            
            # Benchmark JIT
            start_time = time.time()
            for _ in range(iterations):
                jit_result = jit_func(spike_trains, kernel)
            jit_time = (time.time() - start_time) / iterations
            
            # Benchmark Python
            start_time = time.time()
            for _ in range(iterations):
                python_result = python_func(spike_trains, kernel)
            python_time = (time.time() - start_time) / iterations
            
            # Calculate metrics
            speedup = python_time / jit_time if jit_time > 0 else 1.0
            
            results[size] = {
                'jit_time': jit_time,
                'python_time': python_time,
                'speedup': speedup,
                'data_size': batch_size * time_steps * num_neurons
            }
            
            print(f"    JIT: {jit_time:.4f}s")
            print(f"    Python: {python_time:.4f}s")
            print(f"    Speedup: {speedup:.2f}x")
        
        return results
    
    def benchmark_matrix_multiply(
        self,
        sizes: List[int] = [100, 200, 500, 1000],
        iterations: int = 10
    ) -> Dict:
        """Benchmark matrix multiplication operations."""
        print("Benchmarking matrix multiplication...")
        
        results = {}
        
        for size in sizes:
            print(f"  Testing matrix size {size}x{size}...")
            
            # Generate test matrices
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            # Get compiled and Python functions
            jit_func = self.compiler.compile_matrix_multiply_kernel()
            python_func = self.compiler._matrix_multiply_kernel_python
            
            # Warm up JIT
            _ = jit_func(a[:10, :10], b[:10, :10])
            
            # Benchmark JIT
            start_time = time.time()
            for _ in range(iterations):
                jit_result = jit_func(a, b)
            jit_time = (time.time() - start_time) / iterations
            
            # Benchmark Python (NumPy)
            start_time = time.time()
            for _ in range(iterations):
                python_result = python_func(a, b)
            python_time = (time.time() - start_time) / iterations
            
            # Calculate metrics
            speedup = python_time / jit_time if jit_time > 0 else 1.0
            gflops_jit = (2 * size**3) / jit_time / 1e9
            gflops_python = (2 * size**3) / python_time / 1e9
            
            results[size] = {
                'jit_time': jit_time,
                'python_time': python_time,
                'speedup': speedup,
                'gflops_jit': gflops_jit,
                'gflops_python': gflops_python,
                'data_size': size * size
            }
            
            print(f"    JIT: {jit_time:.4f}s ({gflops_jit:.2f} GFLOPS)")
            print(f"    Python: {python_time:.4f}s ({gflops_python:.2f} GFLOPS)")
            print(f"    Speedup: {speedup:.2f}x")
        
        return results
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        print("Starting comprehensive JIT performance benchmark...")
        print(f"System: {self.results['system_info']['platform']}")
        print(f"Numba Available: {self.results['jit_info']['numba_available']}")
        print(f"JIT Enabled: {self.results['jit_info']['jit_enabled']}")
        print("-" * 60)
        
        # Run all benchmarks
        self.results['benchmarks']['spike_processing'] = self.benchmark_spike_processing()
        self.results['benchmarks']['reservoir_update'] = self.benchmark_reservoir_update()
        self.results['benchmarks']['convolution'] = self.benchmark_convolution()
        self.results['benchmarks']['matrix_multiply'] = self.benchmark_matrix_multiply()
        
        return self.results
    
    def save_results(self, filename: str = "jit_benchmark_results.json"):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {filename}")
    
    def plot_results(self, save_plots: bool = True):
        """Plot benchmark results."""
        if not NUMBA_AVAILABLE:
            print("Numba not available, skipping plots")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Numba JIT Performance Benchmark Results', fontsize=16)
        
        # Spike processing plot
        ax1 = axes[0, 0]
        spike_results = self.results['benchmarks']['spike_processing']
        sizes = list(spike_results.keys())
        speedups = [spike_results[size]['speedup'] for size in sizes]
        
        ax1.bar(range(len(sizes)), speedups, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Speedup (x)')
        ax1.set_title('Spike Processing Speedup')
        ax1.set_xticks(range(len(sizes)))
        ax1.set_xticklabels(sizes)
        ax1.grid(True, alpha=0.3)
        
        # Add speedup values on bars
        for i, speedup in enumerate(speedups):
            ax1.text(i, speedup + 0.05, f'{speedup:.2f}x', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Reservoir update plot
        ax2 = axes[0, 1]
        reservoir_results = self.results['benchmarks']['reservoir_update']
        reservoir_sizes = list(reservoir_results.keys())
        reservoir_speedups = [reservoir_results[size]['speedup'] for size in reservoir_sizes]
        
        ax2.bar(range(len(reservoir_sizes)), reservoir_speedups, color='lightgreen', alpha=0.7)
        ax2.set_xlabel('Reservoir Size')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('Reservoir Update Speedup')
        ax2.set_xticks(range(len(reservoir_sizes)))
        ax2.set_xticklabels(reservoir_sizes)
        ax2.grid(True, alpha=0.3)
        
        # Convolution plot
        ax3 = axes[1, 0]
        conv_results = self.results['benchmarks']['convolution']
        conv_sizes = list(conv_results.keys())
        conv_speedups = [conv_results[size]['speedup'] for size in conv_sizes]
        
        ax3.bar(range(len(conv_sizes)), conv_speedups, color='orange', alpha=0.7)
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Speedup (x)')
        ax3.set_title('Convolution Speedup')
        ax3.set_xticks(range(len(conv_sizes)))
        ax3.set_xticklabels(conv_sizes)
        ax3.grid(True, alpha=0.3)
        
        # Matrix multiplication plot
        ax4 = axes[1, 1]
        matrix_results = self.results['benchmarks']['matrix_multiply']
        matrix_sizes = list(matrix_results.keys())
        matrix_speedups = [matrix_results[size]['speedup'] for size in matrix_sizes]
        
        ax4.bar(range(len(matrix_sizes)), matrix_speedups, color='salmon', alpha=0.7)
        ax4.set_xlabel('Matrix Size')
        ax4.set_ylabel('Speedup (x)')
        ax4.set_title('Matrix Multiplication Speedup')
        ax4.set_xticks(range(len(matrix_sizes)))
        ax4.set_xticklabels([f"{size}x{size}" for size in matrix_sizes])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('jit_benchmark_plots.png', dpi=300, bbox_inches='tight')
            print("Plots saved to jit_benchmark_plots.png")
        
        plt.show()
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("JIT BENCHMARK SUMMARY")
        print("="*60)
        
        if not NUMBA_AVAILABLE:
            print("Numba not available - JIT compilation disabled")
            print("All operations used Python fallbacks")
            return
        
        # Overall statistics
        all_speedups = []
        
        # Spike processing
        spike_results = self.results['benchmarks']['spike_processing']
        spike_speedups = [spike_results[size]['speedup'] for size in spike_results.keys()]
        all_speedups.extend(spike_speedups)
        
        print(f"Spike Processing:")
        print(f"  Average speedup: {np.mean(spike_speedups):.2f}x")
        print(f"  Max speedup: {np.max(spike_speedups):.2f}x")
        
        # Reservoir update
        reservoir_results = self.results['benchmarks']['reservoir_update']
        reservoir_speedups = [reservoir_results[size]['speedup'] for size in reservoir_results.keys()]
        all_speedups.extend(reservoir_speedups)
        
        print(f"Reservoir Update:")
        print(f"  Average speedup: {np.mean(reservoir_speedups):.2f}x")
        print(f"  Max speedup: {np.max(reservoir_speedups):.2f}x")
        
        # Convolution
        conv_results = self.results['benchmarks']['convolution']
        conv_speedups = [conv_results[size]['speedup'] for size in conv_results.keys()]
        all_speedups.extend(conv_speedups)
        
        print(f"Convolution:")
        print(f"  Average speedup: {np.mean(conv_speedups):.2f}x")
        print(f"  Max speedup: {np.max(conv_speedups):.2f}x")
        
        # Matrix multiplication
        matrix_results = self.results['benchmarks']['matrix_multiply']
        matrix_speedups = [matrix_results[size]['speedup'] for size in matrix_results.keys()]
        all_speedups.extend(matrix_speedups)
        
        print(f"Matrix Multiplication:")
        print(f"  Average speedup: {np.mean(matrix_speedups):.2f}x")
        print(f"  Max speedup: {np.max(matrix_speedups):.2f}x")
        
        # Overall summary
        print(f"\nOverall Performance:")
        print(f"  Average speedup across all tests: {np.mean(all_speedups):.2f}x")
        print(f"  Maximum speedup achieved: {np.max(all_speedups):.2f}x")
        print(f"  Minimum speedup achieved: {np.min(all_speedups):.2f}x")
        
        # System information
        print(f"\nSystem Information:")
        print(f"  Platform: {self.results['system_info']['platform']}")
        print(f"  CPU Count: {self.results['system_info']['cpu_count']}")
        print(f"  Numba Available: {self.results['jit_info']['numba_available']}")
        print(f"  JIT Enabled: {self.results['jit_info']['jit_enabled']}")


def main():
    """Run the benchmark."""
    benchmark = JITPerformanceBenchmark()
    
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
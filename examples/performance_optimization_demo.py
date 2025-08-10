"""
Performance Optimization Demo

This example demonstrates how to use all three performance optimization
components together for maximum performance in neuromorphic computations.
"""

import time
import numpy as np
import jax.numpy as jnp
from jax import random

# Import performance optimization modules
from src.performance import (
    MKLOptimizer, MKLConfig,
    ParallelProcessor, ParallelConfig,
    JITCompiler, JITConfig,
    enable_mkl_optimization,
    enable_parallel_processing,
    enable_jit_compilation
)


def generate_test_data():
    """Generate test data for performance demonstrations."""
    key = random.PRNGKey(42)
    
    # Spike train data
    batch_size, time_steps, num_neurons = 10, 1000, 100
    spike_trains = random.bernoulli(key, 0.1, (batch_size, time_steps, num_neurons))
    
    # Matrix data
    matrix_size = 1000
    matrix_a = random.normal(key, (matrix_size, matrix_size))
    matrix_b = random.normal(key, (matrix_size, matrix_size))
    
    # Reservoir data
    reservoir_size, input_size = 500, 100
    reservoir_states = random.normal(key, (reservoir_size,))
    input_weights = random.normal(key, (input_size, reservoir_size))
    reservoir_weights = random.normal(key, (reservoir_size, reservoir_size))
    input_spikes = random.bernoulli(key, 0.1, (input_size,))
    
    return {
        'spike_trains': spike_trains,
        'matrix_a': matrix_a,
        'matrix_b': matrix_b,
        'reservoir_states': reservoir_states,
        'input_weights': input_weights,
        'reservoir_weights': reservoir_weights,
        'input_spikes': input_spikes
    }


def demo_mkl_optimization():
    """Demonstrate Intel MKL optimization."""
    print("=" * 60)
    print("INTEL MKL OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Configure MKL optimizer
    config = MKLConfig(
        num_threads=4,
        enable_fast_math=True,
        verbose=True
    )
    optimizer = MKLOptimizer(config)
    
    # Display MKL information
    info = optimizer.get_mkl_info()
    print(f"MKL Available: {info['mkl_available']}")
    print(f"Threads: {info['num_threads']}")
    print(f"Thread Affinity: {info['thread_affinity']}")
    print()
    
    # Generate test data
    data = generate_test_data()
    
    # Matrix multiplication benchmark
    print("Matrix Multiplication Benchmark:")
    
    # Baseline
    start_time = time.time()
    baseline_result = jnp.dot(data['matrix_a'], data['matrix_b'])
    baseline_result.block_until_ready()
    baseline_time = time.time() - start_time
    
    # Optimized
    start_time = time.time()
    optimized_result = optimizer.optimize_matrix_multiply(
        data['matrix_a'], data['matrix_b']
    )
    optimized_result.block_until_ready()
    optimized_time = time.time() - start_time
    
    speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
    
    print(f"  Baseline: {baseline_time:.4f}s")
    print(f"  Optimized: {optimized_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print()
    
    # Reservoir update benchmark
    print("Reservoir Update Benchmark:")
    
    # Baseline
    start_time = time.time()
    input_current = jnp.dot(data['input_spikes'], data['input_weights'])
    recurrent_current = jnp.dot(data['reservoir_states'], data['reservoir_weights'].T)
    baseline_reservoir = input_current + recurrent_current
    baseline_reservoir.block_until_ready()
    baseline_time = time.time() - start_time
    
    # Optimized
    start_time = time.time()
    optimized_reservoir = optimizer.optimize_reservoir_update(
        data['reservoir_states'],
        data['input_weights'],
        data['reservoir_weights'],
        data['input_spikes']
    )
    optimized_reservoir.block_until_ready()
    optimized_time = time.time() - start_time
    
    speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
    
    print(f"  Baseline: {baseline_time:.4f}s")
    print(f"  Optimized: {optimized_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print()


def demo_parallel_processing():
    """Demonstrate parallel processing."""
    print("=" * 60)
    print("PARALLEL PROCESSING DEMO")
    print("=" * 60)
    
    # Configure parallel processor
    config = ParallelConfig(
        num_threads=4,
        parallel_strategy="thread",
        enable_load_balancing=True,
        verbose=True
    )
    processor = ParallelProcessor(config)
    
    # Display parallel processing information
    info = processor.get_parallel_info()
    print(f"Threads: {info['num_threads']}")
    print(f"Strategy: {info['parallel_strategy']}")
    print(f"Load Balancing: {info['load_balancing']}")
    print()
    
    # Generate test data
    key = random.PRNGKey(42)
    spike_batches = [
        random.bernoulli(key, 0.1, (5, 100, 50)) for _ in range(8)
    ]
    
    # Define processing function
    def sum_spikes(batch):
        return jnp.sum(batch, axis=1)  # Sum over time dimension
    
    print("Spike Processing Benchmark:")
    
    # Sequential processing
    start_time = time.time()
    sequential_results = [sum_spikes(batch) for batch in spike_batches]
    sequential_time = time.time() - start_time
    
    # Parallel processing
    start_time = time.time()
    parallel_results = processor.parallel_spike_processing(spike_batches, sum_spikes)
    parallel_time = time.time() - start_time
    
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    
    print(f"  Sequential: {sequential_time:.4f}s")
    print(f"  Parallel: {parallel_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print()
    
    # Multi-modal reasoning demo
    print("Multi-Modal Reasoning Demo:")
    
    # Define reasoning cores
    def visual_core(data):
        return jnp.sum(data, axis=-1)
    
    def audio_core(data):
        return jnp.mean(data, axis=-1)
    
    def text_core(data):
        return jnp.max(data, axis=-1)
    
    reasoning_cores = [visual_core, audio_core, text_core]
    
    # Create input data with different sizes (different computational loads)
    input_data_list = [
        random.normal(key, (100, 50)),  # Large visual data
        random.normal(key, (20, 30)),   # Small audio data
        random.normal(key, (80, 40))    # Medium text data
    ]
    
    # Sequential processing
    start_time = time.time()
    sequential_reasoning = [core(data) for core, data in zip(reasoning_cores, input_data_list)]
    sequential_time = time.time() - start_time
    
    # Parallel processing with load balancing
    start_time = time.time()
    parallel_reasoning = processor.parallel_multi_modal_reasoning(
        reasoning_cores, input_data_list, load_balance=True
    )
    parallel_time = time.time() - start_time
    
    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    
    print(f"  Sequential: {sequential_time:.4f}s")
    print(f"  Parallel (load balanced): {parallel_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print()


def demo_jit_compilation():
    """Demonstrate JIT compilation."""
    print("=" * 60)
    print("JIT COMPILATION DEMO")
    print("=" * 60)
    
    # Configure JIT compiler
    config = JITConfig(
        enable_jit=True,
        cache_compiled_functions=True,
        enable_fastmath=True,
        enable_parallel_loops=True,
        verbose=True
    )
    compiler = JITCompiler(config)
    
    # Display JIT information
    info = compiler.get_jit_info()
    print(f"Numba Available: {info['numba_available']}")
    print(f"JIT Enabled: {info['jit_enabled']}")
    print(f"Cache Enabled: {info['cache_enabled']}")
    print()
    
    # Generate test data (convert to NumPy for JIT)
    data = generate_test_data()
    spike_trains_np = np.array(data['spike_trains'], dtype=np.float32)
    matrix_a_np = np.array(data['matrix_a'], dtype=np.float32)
    matrix_b_np = np.array(data['matrix_b'], dtype=np.float32)
    
    # Matrix multiplication benchmark
    print("Matrix Multiplication Benchmark:")
    
    # Get compiled and Python functions
    jit_func = compiler.compile_matrix_multiply_kernel()
    python_func = compiler._matrix_multiply_kernel_python
    
    # First call (includes compilation time)
    start_time = time.time()
    jit_result_first = jit_func(matrix_a_np, matrix_b_np)
    first_call_time = time.time() - start_time
    
    # Subsequent call (cached)
    start_time = time.time()
    jit_result_cached = jit_func(matrix_a_np, matrix_b_np)
    cached_call_time = time.time() - start_time
    
    # Python baseline
    start_time = time.time()
    python_result = python_func(matrix_a_np, matrix_b_np)
    python_time = time.time() - start_time
    
    compilation_overhead = first_call_time - cached_call_time
    speedup = python_time / cached_call_time if cached_call_time > 0 else 1.0
    
    print(f"  First call (with compilation): {first_call_time:.4f}s")
    print(f"  Cached call: {cached_call_time:.4f}s")
    print(f"  Python baseline: {python_time:.4f}s")
    print(f"  Compilation overhead: {compilation_overhead:.4f}s")
    print(f"  Runtime speedup: {speedup:.2f}x")
    print()
    
    # Spike processing benchmark
    print("Spike Processing Benchmark:")
    
    # Create spike processing data
    weights = np.random.randn(100, 100).astype(np.float32) * 0.1
    thresholds = np.ones(100, dtype=np.float32) * 0.5
    dt = 0.001
    
    # Get compiled and Python functions
    spike_jit_func = compiler.compile_spike_processing_kernel()
    spike_python_func = compiler._spike_processing_kernel_python
    
    # JIT benchmark (after warm-up)
    _ = spike_jit_func(spike_trains_np[:1], weights, thresholds, dt)  # Warm up
    
    start_time = time.time()
    jit_spike_result = spike_jit_func(spike_trains_np, weights, thresholds, dt)
    jit_time = time.time() - start_time
    
    # Python benchmark
    start_time = time.time()
    python_spike_result = spike_python_func(spike_trains_np, weights, thresholds, dt)
    python_time = time.time() - start_time
    
    speedup = python_time / jit_time if jit_time > 0 else 1.0
    
    print(f"  JIT: {jit_time:.4f}s")
    print(f"  Python: {python_time:.4f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print()


def demo_integrated_optimization():
    """Demonstrate using all optimizations together."""
    print("=" * 60)
    print("INTEGRATED OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Enable all optimizations
    mkl_optimizer = enable_mkl_optimization()
    parallel_processor = enable_parallel_processing()
    jit_compiler = enable_jit_compilation()
    
    print("All optimizations enabled!")
    print(f"MKL: {mkl_optimizer.is_mkl_available()}")
    print(f"Parallel: {parallel_processor.get_parallel_info()['num_threads']} threads")
    print(f"JIT: {jit_compiler.is_numba_available()}")
    print()
    
    # Generate complex workload
    key = random.PRNGKey(42)
    
    # Multiple spike batches for parallel processing
    spike_batches = [
        random.bernoulli(key, 0.1, (8, 500, 100)) for _ in range(4)
    ]
    
    # Large matrices for MKL optimization
    large_matrices = [
        (random.normal(key, (800, 800)), random.normal(key, (800, 800)))
        for _ in range(2)
    ]
    
    print("Complex Workload Benchmark:")
    print("- 4 spike batches (8x500x100 each)")
    print("- 2 matrix multiplications (800x800 each)")
    print("- Parallel + MKL + JIT optimizations")
    print()
    
    # Define complex processing function
    def complex_processing(spike_batch):
        # Spike processing with convolution
        kernel = jnp.ones(10) / 10  # Simple averaging kernel
        convolved = mkl_optimizer.optimize_spike_convolution(spike_batch, kernel)
        
        # Matrix operation on processed spikes
        batch_mean = jnp.mean(convolved, axis=1)  # [batch, neurons]
        weight_matrix = random.normal(key, (100, 50))
        result = mkl_optimizer.optimize_matrix_multiply(batch_mean, weight_matrix)
        
        return result
    
    # Baseline: sequential processing
    start_time = time.time()
    baseline_results = []
    for batch in spike_batches:
        result = complex_processing(batch)
        baseline_results.append(result)
    
    # Also do matrix multiplications
    for a, b in large_matrices:
        result = jnp.dot(a, b)
        result.block_until_ready()
    
    baseline_time = time.time() - start_time
    
    # Optimized: parallel + MKL + JIT
    start_time = time.time()
    
    # Parallel spike processing
    optimized_results = parallel_processor.parallel_spike_processing(
        spike_batches, complex_processing
    )
    
    # Optimized matrix multiplications
    for a, b in large_matrices:
        result = mkl_optimizer.optimize_matrix_multiply(a, b)
        result.block_until_ready()
    
    optimized_time = time.time() - start_time
    
    speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0
    
    print(f"  Baseline (sequential): {baseline_time:.4f}s")
    print(f"  Optimized (parallel+MKL+JIT): {optimized_time:.4f}s")
    print(f"  Overall speedup: {speedup:.2f}x")
    print()
    
    print("Performance optimization demo completed!")


def main():
    """Run all performance optimization demos."""
    print("GODLY AI PERFORMANCE OPTIMIZATION DEMO")
    print("=" * 60)
    print()
    
    try:
        # Run individual demos
        demo_mkl_optimization()
        demo_parallel_processing()
        demo_jit_compilation()
        demo_integrated_optimization()
        
        print("All demos completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
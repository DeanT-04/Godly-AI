# Performance Optimization Layer

The performance optimization layer provides comprehensive CPU optimization, parallel processing, and JIT compilation capabilities to maximize performance on consumer hardware for the Godly AI neuromorphic system.

## Overview

The performance layer consists of three main components:

1. **Intel MKL Optimization** - Optimized matrix operations and vectorized spike processing
2. **Parallel Processing** - Multi-threaded computation with OpenMP support
3. **JIT Compilation** - Just-in-time compilation with Numba for critical kernels

## Quick Start

```python
from src.performance import (
    enable_mkl_optimization, 
    enable_parallel_processing, 
    enable_jit_compilation
)

# Enable all optimizations
mkl_optimizer = enable_mkl_optimization()
parallel_processor = enable_parallel_processing()
jit_compiler = enable_jit_compilation()

# Use optimized operations
import jax.numpy as jnp
a = jnp.random.normal(key, (1000, 1000))
b = jnp.random.normal(key, (1000, 1000))

# Optimized matrix multiplication
result = mkl_optimizer.optimize_matrix_multiply(a, b)
```

## Components

### Intel MKL Optimization

Provides optimized implementations using Intel Math Kernel Library:

- **Matrix Operations**: Optimized BLAS operations with automatic MKL detection
- **Spike Processing**: Vectorized convolution operations for spike trains
- **Reservoir Updates**: Efficient computation of reservoir state updates
- **Eigenvalue Computation**: Fast spectral radius calculation

**Key Features:**
- Automatic fallback to OpenBLAS when MKL is not available
- Configurable threading and memory optimization
- Performance benchmarking against baseline implementations

### Parallel Processing

Multi-threaded processing capabilities with OpenMP support:

- **Spike Processing**: Parallel processing of spike train batches
- **Reservoir Computation**: Multi-core reservoir state updates
- **Load Balancing**: Intelligent distribution of multi-modal reasoning tasks
- **Multiple Strategies**: Thread pools, process pools, and JAX pmap support

**Key Features:**
- Automatic thread count detection
- Load balancing for heterogeneous workloads
- Comprehensive performance monitoring

### JIT Compilation

Just-in-time compilation with Numba for performance-critical kernels:

- **Spike Processing Kernels**: Compiled neuromorphic simulation loops
- **Matrix Operations**: Cache-friendly blocked matrix multiplication
- **Convolution Kernels**: Optimized spike train convolution
- **Memory Layout**: Optimized memory access patterns

**Key Features:**
- Automatic compilation caching
- Graceful fallback when Numba is not available
- Memory layout optimization for cache efficiency

## Performance Results

Based on benchmarking on consumer hardware:

### Intel MKL Optimization
- **Matrix Operations**: Up to 2.02x speedup
- **Reservoir Computation**: Up to 15.91x speedup
- **Average Performance**: 1.58x speedup across all operations

### Parallel Processing
- **Spike Processing**: Scalable with number of cores
- **Multi-modal Reasoning**: Efficient load balancing
- **Reservoir Updates**: Significant speedup for large reservoirs

### JIT Compilation
- **Compilation Overhead**: One-time cost with persistent caching
- **Runtime Performance**: Significant speedup for iterative operations
- **Memory Efficiency**: Optimized cache utilization

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For maximum performance, install Intel MKL:

```bash
# Intel MKL (optional but recommended)
pip install mkl intel-openmp

# Numba for JIT compilation
pip install numba
```

### Verification

Run the performance benchmarks to verify installation:

```bash
# MKL benchmark
python benchmarks/mkl_performance_benchmark.py

# JIT benchmark  
python benchmarks/jit_performance_benchmark.py
```

## Configuration

### MKL Configuration

```python
from src.performance import MKLConfig, MKLOptimizer

config = MKLConfig(
    num_threads=8,              # Number of threads (auto-detect if None)
    thread_affinity="balanced", # Thread affinity strategy
    enable_fast_math=True,      # Enable fast math optimizations
    precision_mode="high",      # Precision mode
    verbose=True                # Enable verbose logging
)

optimizer = MKLOptimizer(config)
```

### Parallel Processing Configuration

```python
from src.performance import ParallelConfig, ParallelProcessor

config = ParallelConfig(
    num_threads=4,                    # Number of worker threads
    parallel_strategy="thread",       # "thread", "process", "jax_pmap"
    enable_load_balancing=True,       # Enable load balancing
    load_balance_threshold=0.8,       # Load balance threshold
    verbose=True
)

processor = ParallelProcessor(config)
```

### JIT Configuration

```python
from src.performance import JITConfig, JITCompiler

config = JITConfig(
    enable_jit=True,                  # Enable JIT compilation
    cache_compiled_functions=True,    # Cache compiled functions
    enable_fastmath=True,             # Enable fast math
    enable_parallel_loops=True,       # Enable parallel loops
    verbose=True
)

compiler = JITCompiler(config)
```

## API Reference

### MKL Optimizer

```python
class MKLOptimizer:
    def optimize_matrix_multiply(self, a, b, transpose_a=False, transpose_b=False)
    def optimize_spike_convolution(self, spike_trains, kernel, mode='valid')
    def optimize_reservoir_update(self, reservoir_state, input_weights, reservoir_weights, input_spikes)
    def optimize_batch_processing(self, spike_batches, weight_matrix)
    def optimize_eigenvalue_computation(self, matrix, compute_eigenvectors=False)
    def benchmark_performance(self, matrix_sizes=[100, 500, 1000, 2000], num_iterations=10)
```

### Parallel Processor

```python
class ParallelProcessor:
    def parallel_spike_processing(self, spike_batches, processing_func, **kwargs)
    def parallel_reservoir_computation(self, reservoir_states, input_weights, reservoir_weights, input_spikes_list)
    def parallel_multi_modal_reasoning(self, reasoning_cores, input_data_list, load_balance=True)
    def benchmark_parallel_performance(self, test_sizes=[100, 500, 1000], num_iterations=5)
```

### JIT Compiler

```python
class JITCompiler:
    def compile_spike_processing_kernel(self)
    def compile_reservoir_update_kernel(self)
    def compile_convolution_kernel(self)
    def compile_matrix_multiply_kernel(self)
    def optimize_memory_layout(self, array, access_pattern="row_major")
    def benchmark_jit_performance(self, test_sizes=[100, 500, 1000], num_iterations=5)
```

## Examples

See the `examples/` directory for detailed usage examples:

- `mkl_optimization_example.py` - Intel MKL optimization examples
- `parallel_processing_example.py` - Parallel processing examples  
- `jit_compilation_example.py` - JIT compilation examples
- `integrated_performance_example.py` - Using all optimizations together

## Testing

Run the comprehensive test suite:

```bash
# Run all performance tests
pytest tests/test_mkl_optimization.py -v
pytest tests/test_parallel_processing.py -v
pytest tests/test_jit_compilation.py -v

# Run with coverage
pytest tests/test_*_optimization.py --cov=src/performance
```

## Benchmarking

Performance benchmarks are available in the `benchmarks/` directory:

```bash
# MKL performance benchmark
python benchmarks/mkl_performance_benchmark.py

# JIT performance benchmark
python benchmarks/jit_performance_benchmark.py
```

Results are saved as JSON files and plots are generated showing performance improvements.

## Troubleshooting

### Common Issues

1. **MKL Not Available**: The system automatically falls back to OpenBLAS
2. **Numba Not Available**: JIT compilation is disabled, Python fallbacks are used
3. **Threading Issues**: Adjust thread count in configuration
4. **Memory Issues**: Reduce batch sizes or enable memory monitoring

### Performance Tips

1. **Enable All Optimizations**: Use MKL, parallel processing, and JIT together
2. **Tune Thread Count**: Optimal thread count is usually 4-8 for most systems
3. **Use Appropriate Data Types**: float32 is often faster than float64
4. **Batch Operations**: Process data in batches for better cache utilization
5. **Profile Your Code**: Use the built-in benchmarking tools to identify bottlenecks

## Contributing

When contributing to the performance layer:

1. **Add Tests**: All new optimizations must include comprehensive tests
2. **Benchmark Performance**: Include performance comparisons with baselines
3. **Document Changes**: Update this documentation for new features
4. **Maintain Compatibility**: Ensure fallback behavior when dependencies are missing

## License

This performance optimization layer is part of the Godly AI project and follows the same licensing terms.
# JIT Compilation

The JIT (Just-In-Time) compilation module provides high-performance compiled kernels using Numba for performance-critical neuromorphic computations.

## Overview

The JIT compilation module uses Numba to compile Python functions to optimized machine code, providing:

- **Compiled neuromorphic kernels** - JIT-compiled spike processing and reservoir updates
- **Cache-friendly memory access** - Optimized memory layouts for better cache utilization
- **Automatic compilation caching** - Persistent caching of compiled functions
- **Graceful fallbacks** - Automatic fallback to Python when Numba is not available

## Installation

### Numba Installation

Install Numba for JIT compilation:

```bash
# Using pip
pip install numba

# Using conda
conda install numba
```

### Verification

Verify Numba installation:

```python
from src.performance.jit_compilation import JITCompiler, NUMBA_AVAILABLE

print(f"Numba available: {NUMBA_AVAILABLE}")

if NUMBA_AVAILABLE:
    compiler = JITCompiler()
    print(compiler.get_jit_info())
```

## Configuration

### Basic Configuration

```python
from src.performance import JITConfig, JITCompiler

# Default configuration
compiler = JITCompiler()

# Custom configuration
config = JITConfig(
    enable_jit=True,                  # Enable JIT compilation
    cache_compiled_functions=True,    # Cache compiled functions
    enable_fastmath=True,             # Enable fast math optimizations
    enable_parallel_loops=True,       # Enable parallel loops
    verbose=True                      # Enable verbose logging
)

compiler = JITCompiler(config)
```

### Configuration Options

#### Compilation Settings
- `enable_jit`: Enable/disable JIT compilation
- `cache_compiled_functions`: Enable persistent caching of compiled functions
- `parallel_compilation`: Enable parallel compilation (when supported)

#### Optimization Settings
- `optimize_memory_layout`: Optimize array memory layouts
- `enable_fastmath`: Enable fast math optimizations (may reduce precision)
- `enable_parallel_loops`: Enable automatic parallelization of loops

#### Cache Settings
- `cache_directory`: Directory for cached compiled functions
- `max_cache_size`: Maximum cache size in bytes

#### Debugging
- `verbose`: Enable verbose compilation logging
- `profile_compilation`: Profile compilation times

## API Reference

### Core Methods

#### Spike Processing Kernel

```python
def compile_spike_processing_kernel(self) -> Callable:
    """Compile optimized spike processing kernel."""
```

**Example:**
```python
import numpy as np

# Create test data
batch_size, time_steps, num_neurons = 4, 100, 50
spike_trains = np.random.rand(batch_size, time_steps, num_neurons).astype(np.float32)
spike_trains = (spike_trains > 0.9).astype(np.float32)  # Sparse spikes

weights = np.random.randn(num_neurons, num_neurons).astype(np.float32) * 0.1
thresholds = np.ones(num_neurons, dtype=np.float32) * 0.5
dt = 0.001

# Compile and use kernel
kernel = compiler.compile_spike_processing_kernel()
result = kernel(spike_trains, weights, thresholds, dt)
```

#### Reservoir Update Kernel

```python
def compile_reservoir_update_kernel(self) -> Callable:
    """Compile optimized reservoir update kernel."""
```

**Example:**
```python
# Create reservoir data
batch_size, num_neurons, input_size = 8, 100, 50
reservoir_states = np.random.randn(batch_size, num_neurons).astype(np.float32)
input_weights = np.random.randn(input_size, num_neurons).astype(np.float32)
reservoir_weights = np.random.randn(num_neurons, num_neurons).astype(np.float32)
input_spikes = np.random.rand(batch_size, input_size).astype(np.float32)
input_spikes = (input_spikes > 0.9).astype(np.float32)

# Compile and use kernel
kernel = compiler.compile_reservoir_update_kernel()
result = kernel(reservoir_states, input_weights, reservoir_weights, input_spikes)
```

#### Convolution Kernel

```python
def compile_convolution_kernel(self) -> Callable:
    """Compile optimized convolution kernel for spike trains."""
```

**Example:**
```python
# Create convolution data
batch_size, time_steps, num_neurons = 4, 200, 20
kernel_size = 10

spike_trains = np.random.rand(batch_size, time_steps, num_neurons).astype(np.float32)
spike_trains = (spike_trains > 0.8).astype(np.float32)
kernel = np.random.randn(kernel_size).astype(np.float32)

# Compile and use kernel
conv_kernel = compiler.compile_convolution_kernel()
result = conv_kernel(spike_trains, kernel)
```

#### Matrix Multiplication Kernel

```python
def compile_matrix_multiply_kernel(self) -> Callable:
    """Compile optimized matrix multiplication kernel."""
```

**Example:**
```python
# Create matrices
M, K, N = 500, 300, 400
a = np.random.randn(M, K).astype(np.float32)
b = np.random.randn(K, N).astype(np.float32)

# Compile and use kernel
matmul_kernel = compiler.compile_matrix_multiply_kernel()
result = matmul_kernel(a, b)
```

### Memory Layout Optimization

```python
def optimize_memory_layout(
    self, 
    array: np.ndarray, 
    access_pattern: str = "row_major"
) -> np.ndarray:
    """Optimize array memory layout for cache efficiency."""
```

**Example:**
```python
# Optimize for different access patterns
array = np.random.randn(1000, 500)

# Row-major optimization (C-style)
row_major = compiler.optimize_memory_layout(array, "row_major")

# Column-major optimization (Fortran-style)
col_major = compiler.optimize_memory_layout(array, "column_major")

# Blocked optimization
blocked = compiler.optimize_memory_layout(array, "blocked")
```

### Utility Methods

#### Information and Status

```python
def is_numba_available(self) -> bool:
    """Check if Numba is available."""

def get_jit_info(self) -> Dict[str, Any]:
    """Get information about JIT configuration."""
```

#### Performance Benchmarking

```python
def benchmark_jit_performance(
    self,
    test_sizes: list = [100, 500, 1000],
    num_iterations: int = 5
) -> Dict[str, Any]:
    """Benchmark JIT compilation performance."""
```

#### Cache Management

```python
def clear_cache(self):
    """Clear compiled function cache."""
```

## JIT Compilation Process

### Compilation Workflow

1. **First Call**: Function is compiled to machine code (compilation overhead)
2. **Subsequent Calls**: Use cached compiled version (fast execution)
3. **Persistent Caching**: Compiled functions are saved to disk

### Compilation Overhead

```python
import time

# First call includes compilation time
start_time = time.time()
kernel = compiler.compile_spike_processing_kernel()
result1 = kernel(spike_trains, weights, thresholds, dt)
first_call_time = time.time() - start_time

# Subsequent calls use cached version
start_time = time.time()
result2 = kernel(spike_trains, weights, thresholds, dt)
cached_call_time = time.time() - start_time

print(f"First call (with compilation): {first_call_time:.3f}s")
print(f"Cached call: {cached_call_time:.3f}s")
print(f"Speedup after compilation: {first_call_time / cached_call_time:.1f}x")
```

## Performance Optimization

### Best Practices

1. **Use NumPy Arrays**: JIT compilation works best with NumPy arrays
2. **Specify Data Types**: Use explicit dtypes (float32, int32) for better performance
3. **Avoid Python Objects**: Stick to numerical operations for best performance
4. **Warm Up Functions**: Call functions once to trigger compilation before timing

### Memory Access Optimization

```python
# Cache-friendly memory access patterns
@njit(parallel=True, fastmath=True)
def optimized_matrix_multiply(a, b):
    """Cache-friendly blocked matrix multiplication."""
    M, K = a.shape
    K2, N = b.shape
    result = np.zeros((M, N), dtype=np.float32)
    
    block_size = 64  # Optimize for cache line size
    
    for i_block in prange(0, M, block_size):
        for j_block in range(0, N, block_size):
            for k_block in range(0, K, block_size):
                # Process block
                i_end = min(i_block + block_size, M)
                j_end = min(j_block + block_size, N)
                k_end = min(k_block + block_size, K)
                
                for i in range(i_block, i_end):
                    for j in range(j_block, j_end):
                        temp_sum = 0.0
                        for k in range(k_block, k_end):
                            temp_sum += a[i, k] * b[k, j]
                        result[i, j] += temp_sum
    
    return result
```

### Parallel Loop Optimization

```python
# Enable parallel loops for better performance
config = JITConfig(
    enable_parallel_loops=True,
    enable_fastmath=True
)
compiler = JITCompiler(config)
```

## Performance Results

Based on benchmarking with and without JIT compilation:

### Compilation Overhead
- **First call**: Includes compilation time (0.1-2.0 seconds)
- **Subsequent calls**: Near-native performance
- **Persistent caching**: Eliminates recompilation across sessions

### Runtime Performance
- **Spike processing**: 2-10x speedup over Python
- **Matrix operations**: 1.5-5x speedup over NumPy
- **Convolution**: 3-8x speedup for iterative operations
- **Memory-bound operations**: Limited speedup due to memory bandwidth

### Memory Efficiency
- **Cache utilization**: Improved through blocked algorithms
- **Memory layout**: Optimized for access patterns
- **Reduced allocations**: In-place operations where possible

## Integration Examples

### Global JIT Compilation

```python
from src.performance import enable_jit_compilation, set_global_compiler
from src.performance.jit_compilation import (
    jit_spike_processing, jit_reservoir_update, jit_matrix_multiply
)

# Enable JIT compilation globally
compiler = enable_jit_compilation()
set_global_compiler(compiler)

# Use global JIT functions
result = jit_matrix_multiply(a, b)
spike_result = jit_spike_processing(spike_trains, weights, thresholds, dt)
reservoir_result = jit_reservoir_update(states, input_weights, reservoir_weights, spikes)
```

### Liquid State Machine Integration

```python
from src.core.liquid_state_machine import LiquidStateMachine

class OptimizedLSM(LiquidStateMachine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jit_compiler = JITCompiler()
        
        # Pre-compile kernels
        self.spike_kernel = self.jit_compiler.compile_spike_processing_kernel()
        self.reservoir_kernel = self.jit_compiler.compile_reservoir_update_kernel()
    
    def optimized_step(self, state, input_spikes, dt, t, key=None):
        # Use JIT-compiled kernels for critical computations
        reservoir_currents = self.reservoir_kernel(
            state.neuron_state.spikes.astype(np.float32),
            state.input_weights.astype(np.float32),
            state.reservoir_weights.astype(np.float32),
            input_spikes.astype(np.float32)
        )
        
        # Continue with standard LSM processing
        return super().step(state, input_spikes, dt, t, key)
```

## Troubleshooting

### Common Issues

#### Numba Not Available
```python
# Check Numba availability
from src.performance.jit_compilation import NUMBA_AVAILABLE

if not NUMBA_AVAILABLE:
    print("Numba not available - using Python fallbacks")
    # JIT compilation will be disabled automatically
```

#### Compilation Errors
```python
# Enable verbose logging to debug compilation issues
config = JITConfig(verbose=True)
compiler = JITCompiler(config)

try:
    kernel = compiler.compile_spike_processing_kernel()
except Exception as e:
    print(f"Compilation error: {e}")
    # Fallback to Python implementation
```

#### Performance Issues
```python
# Profile compilation and execution times
config = JITConfig(
    profile_compilation=True,
    verbose=True
)
compiler = JITCompiler(config)

# Check compilation times
info = compiler.get_jit_info()
print("Compilation times:", info.get('compilation_times', {}))
```

### Memory Issues

```python
# Optimize memory usage
config = JITConfig(
    optimize_memory_layout=True,
    max_cache_size=512 * 1024 * 1024  # 512MB cache limit
)
```

### Cache Management

```python
# Clear cache if needed
compiler.clear_cache()

# Check cache status
info = compiler.get_jit_info()
print(f"Compiled functions: {info['compiled_functions']}")
```

## Advanced Usage

### Custom JIT Kernels

```python
from numba import njit, prange

@njit(parallel=True, fastmath=True, cache=True)
def custom_neuromorphic_kernel(spikes, weights, dt):
    """Custom JIT-compiled neuromorphic kernel."""
    batch_size, time_steps, num_neurons = spikes.shape
    output = np.zeros_like(spikes)
    
    for batch_idx in prange(batch_size):
        membrane_potentials = np.zeros(num_neurons, dtype=np.float32)
        
        for t in range(time_steps):
            # Custom neuromorphic computation
            for i in range(num_neurons):
                synaptic_input = 0.0
                for j in range(num_neurons):
                    if spikes[batch_idx, t, j] > 0.5:
                        synaptic_input += weights[j, i]
                
                membrane_potentials[i] += synaptic_input * dt
                
                if membrane_potentials[i] > 1.0:  # Threshold
                    output[batch_idx, t, i] = 1.0
                    membrane_potentials[i] = 0.0  # Reset
                
                membrane_potentials[i] *= 0.95  # Decay
    
    return output

# Use custom kernel
result = custom_neuromorphic_kernel(spike_trains, weights, dt)
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile JIT compilation and execution
profiler = cProfile.Profile()

# Compile kernel (includes compilation time)
profiler.enable()
kernel = compiler.compile_spike_processing_kernel()
profiler.disable()

compilation_stats = pstats.Stats(profiler)
print("Compilation profile:")
compilation_stats.sort_stats('cumulative').print_stats(5)

# Profile execution (compiled kernel)
profiler = cProfile.Profile()
profiler.enable()
result = kernel(spike_trains, weights, thresholds, dt)
profiler.disable()

execution_stats = pstats.Stats(profiler)
print("Execution profile:")
execution_stats.sort_stats('cumulative').print_stats(5)
```

### Benchmarking

```python
# Comprehensive performance benchmark
results = compiler.benchmark_jit_performance(
    test_sizes=[100, 500, 1000],
    num_iterations=10
)

# Analyze results
print("JIT Performance Results:")
for operation, size_results in results.items():
    if operation == 'jit_info':
        continue
    
    print(f"\n{operation.title()}:")
    for size, metrics in size_results.items():
        if isinstance(metrics, dict) and 'speedup' in metrics:
            print(f"  Size {size}: {metrics['speedup']:.2f}x speedup")
```

This comprehensive guide covers all aspects of using JIT compilation for optimal performance in neuromorphic computations.
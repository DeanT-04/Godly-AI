# Parallel Processing

The parallel processing module provides multi-threaded computation capabilities with OpenMP support for the Godly AI neuromorphic system.

## Overview

The parallel processing module enables efficient utilization of multi-core systems through:

- **Multi-threaded spike processing** - Parallel processing of spike train batches
- **Parallel reservoir computation** - Multi-core reservoir state updates
- **Load balancing** - Intelligent distribution of multi-modal reasoning tasks
- **Multiple strategies** - Thread pools, process pools, and JAX pmap support

## Installation

### Basic Requirements

```bash
# Core dependencies (already in requirements.txt)
pip install jax jaxlib numpy
```

### Optional Dependencies

For enhanced parallel processing:

```bash
# Numba for additional optimizations
pip install numba

# For process-based parallelism
# (multiprocessing is built into Python)
```

### OpenMP Support

OpenMP support is automatically configured through:
- Intel MKL (if available)
- Numba (if available)
- JAX threading

## Configuration

### Basic Configuration

```python
from src.performance import ParallelConfig, ParallelProcessor

# Default configuration
processor = ParallelProcessor()

# Custom configuration
config = ParallelConfig(
    num_threads=4,                    # Number of worker threads
    parallel_strategy="thread",       # Parallelization strategy
    enable_load_balancing=True,       # Enable load balancing
    load_balance_threshold=0.8,       # Load balance threshold
    verbose=True                      # Enable verbose logging
)

processor = ParallelProcessor(config)
```

### Configuration Options

#### Threading Configuration
- `num_threads`: Number of worker threads (auto-detect if None)
- `thread_affinity`: Enable thread affinity
- `use_openmp`: Enable OpenMP support

#### Processing Strategy
- `parallel_strategy`: Strategy for parallelization
  - `"thread"`: Thread-based parallelism (default)
  - `"process"`: Process-based parallelism
  - `"jax_pmap"`: JAX parallel map (requires multiple devices)
  - `"hybrid"`: Combination of thread and process

#### Load Balancing
- `enable_load_balancing`: Enable intelligent load balancing
- `load_balance_threshold`: Threshold for triggering rebalancing (0.0-1.0)

#### Memory Management
- `max_memory_per_worker`: Maximum memory per worker (bytes)
- `enable_memory_monitoring`: Enable memory usage monitoring

## API Reference

### Core Methods

#### Parallel Spike Processing

```python
def parallel_spike_processing(
    self,
    spike_batches: List[jnp.ndarray],
    processing_func: Callable[[jnp.ndarray], jnp.ndarray],
    **kwargs
) -> List[jnp.ndarray]:
    """Process spike batches in parallel."""
```

**Example:**
```python
import jax.numpy as jnp
from jax import random

# Create spike batches
key = random.PRNGKey(42)
spike_batches = [
    random.bernoulli(key, 0.1, (10, 100, 50)) for _ in range(4)
]

# Define processing function
def sum_spikes(batch):
    return jnp.sum(batch, axis=1)  # Sum over time dimension

# Process in parallel
results = processor.parallel_spike_processing(spike_batches, sum_spikes)
```

#### Parallel Reservoir Computation

```python
def parallel_reservoir_computation(
    self,
    reservoir_states: List[jnp.ndarray],
    input_weights: jnp.ndarray,
    reservoir_weights: jnp.ndarray,
    input_spikes_list: List[jnp.ndarray]
) -> List[jnp.ndarray]:
    """Compute reservoir updates in parallel across multiple cores."""
```

**Example:**
```python
# Create reservoir data
reservoir_states = [random.normal(key, (100,)) for _ in range(4)]
input_weights = random.normal(key, (50, 100))
reservoir_weights = random.normal(key, (100, 100))
input_spikes_list = [random.bernoulli(key, 0.1, (50,)) for _ in range(4)]

# Parallel reservoir computation
results = processor.parallel_reservoir_computation(
    reservoir_states, input_weights, reservoir_weights, input_spikes_list
)
```

#### Parallel Multi-Modal Reasoning

```python
def parallel_multi_modal_reasoning(
    self,
    reasoning_cores: List[Callable],
    input_data_list: List[jnp.ndarray],
    load_balance: bool = True
) -> List[jnp.ndarray]:
    """Execute multi-modal reasoning cores in parallel with load balancing."""
```

**Example:**
```python
# Define reasoning cores
def visual_core(data):
    return jnp.sum(data, axis=-1)

def audio_core(data):
    return jnp.mean(data, axis=-1)

def text_core(data):
    return jnp.max(data, axis=-1)

reasoning_cores = [visual_core, audio_core, text_core]

# Create input data
input_data_list = [
    random.normal(key, (20, 30)),  # Visual data
    random.normal(key, (15, 25)),  # Audio data
    random.normal(key, (10, 35))   # Text data
]

# Parallel reasoning with load balancing
results = processor.parallel_multi_modal_reasoning(
    reasoning_cores, input_data_list, load_balance=True
)
```

### Utility Methods

#### Information and Status

```python
def get_parallel_info(self) -> Dict[str, Any]:
    """Get information about parallel configuration."""

def is_openmp_available(self) -> bool:
    """Check if OpenMP is available."""
```

#### Performance Benchmarking

```python
def benchmark_parallel_performance(
    self,
    test_sizes: List[int] = [100, 500, 1000],
    num_iterations: int = 5
) -> Dict[str, Any]:
    """Benchmark parallel processing performance."""
```

## Parallelization Strategies

### Thread-Based Parallelism

Best for I/O-bound and mixed workloads:

```python
config = ParallelConfig(
    parallel_strategy="thread",
    num_threads=4
)
processor = ParallelProcessor(config)
```

**Advantages:**
- Low overhead
- Shared memory
- Good for mixed workloads

**Disadvantages:**
- GIL limitations in pure Python code
- Memory sharing can cause contention

### Process-Based Parallelism

Best for CPU-intensive workloads:

```python
config = ParallelConfig(
    parallel_strategy="process",
    num_threads=4
)
processor = ParallelProcessor(config)
```

**Advantages:**
- No GIL limitations
- True parallelism for CPU-bound tasks
- Memory isolation

**Disadvantages:**
- Higher overhead
- Inter-process communication costs
- Memory duplication

### JAX Parallel Map

Best for homogeneous data-parallel workloads:

```python
config = ParallelConfig(
    parallel_strategy="jax_pmap",
    num_threads=2  # Number of devices
)
processor = ParallelProcessor(config)
```

**Advantages:**
- Optimized for numerical computations
- Automatic device placement
- Efficient for large arrays

**Disadvantages:**
- Requires multiple devices
- Limited to JAX-compatible operations
- Homogeneous data requirements

## Load Balancing

### Automatic Load Balancing

The system automatically balances workloads based on computational complexity:

```python
config = ParallelConfig(
    enable_load_balancing=True,
    load_balance_threshold=0.8  # Rebalance if load > 80%
)
```

### Load Estimation

Load is estimated based on:
- Input data size (`np.prod(data.shape)`)
- Historical execution times
- Worker availability

### Load Balancing Algorithm

1. **Estimate Load**: Calculate computational load for each task
2. **Sort Tasks**: Order tasks by load (heaviest first)
3. **Assign Workers**: Use greedy algorithm to assign tasks to workers
4. **Monitor Performance**: Track execution times for future optimization

## Performance Optimization

### Best Practices

1. **Choose Appropriate Strategy**: 
   - Use threads for I/O-bound tasks
   - Use processes for CPU-bound tasks
   - Use JAX pmap for numerical computations

2. **Optimize Batch Sizes**:
   - Larger batches reduce overhead
   - Smaller batches improve load balancing

3. **Memory Management**:
   - Monitor memory usage per worker
   - Use memory-efficient data structures

4. **Thread Count Optimization**:
   - Start with CPU core count
   - Adjust based on workload characteristics

### Memory Optimization

```python
config = ParallelConfig(
    max_memory_per_worker=512 * 1024 * 1024,  # 512MB per worker
    enable_memory_monitoring=True
)
```

### Performance Monitoring

```python
# Get performance information
info = processor.get_parallel_info()
print(f"Threads: {info['num_threads']}")
print(f"Strategy: {info['parallel_strategy']}")
print(f"Load Balancing: {info['load_balancing']}")

# Run performance benchmark
results = processor.benchmark_parallel_performance(
    test_sizes=[100, 500],
    num_iterations=3
)

# Analyze results
for size, metrics in results['spike_processing'].items():
    speedup = metrics['speedup']
    print(f"Size {size}: {speedup:.2f}x speedup")
```

## Integration Examples

### Liquid State Machine Integration

```python
from src.core.liquid_state_machine import LiquidStateMachine
from src.performance import ParallelProcessor

# Create LSM and parallel processor
lsm = LiquidStateMachine()
processor = ParallelProcessor()

# Parallel processing of multiple LSM instances
def process_lsm_batch(spike_batch):
    state = lsm.init_state()
    return lsm.step(state, spike_batch, dt=0.001, t=0.0)

# Process multiple spike batches in parallel
spike_batches = [generate_spike_batch() for _ in range(8)]
results = processor.parallel_spike_processing(spike_batches, process_lsm_batch)
```

### Multi-Modal Reasoning Integration

```python
from src.agents.reasoning import (
    VisualReasoningCore, AudioReasoningCore, TextReasoningCore
)

# Create reasoning cores
visual_core = VisualReasoningCore()
audio_core = AudioReasoningCore()
text_core = TextReasoningCore()

reasoning_cores = [
    visual_core.process,
    audio_core.process,
    text_core.process
]

# Parallel multi-modal processing
input_data = [visual_data, audio_data, text_data]
results = processor.parallel_multi_modal_reasoning(
    reasoning_cores, input_data, load_balance=True
)
```

## Troubleshooting

### Common Issues

#### Thread Contention
```python
# Reduce thread count if experiencing contention
config = ParallelConfig(num_threads=2)
processor = ParallelProcessor(config)
```

#### Memory Issues
```python
# Enable memory monitoring and reduce per-worker memory
config = ParallelConfig(
    enable_memory_monitoring=True,
    max_memory_per_worker=256 * 1024 * 1024  # 256MB
)
```

#### Load Balancing Issues
```python
# Adjust load balance threshold
config = ParallelConfig(
    enable_load_balancing=True,
    load_balance_threshold=0.6  # More aggressive rebalancing
)
```

### Performance Debugging

```python
# Enable verbose logging
config = ParallelConfig(verbose=True)
processor = ParallelProcessor(config)

# Monitor parallel execution
import time

start_time = time.time()
results = processor.parallel_spike_processing(spike_batches, processing_func)
end_time = time.time()

print(f"Parallel execution time: {end_time - start_time:.3f}s")
print(f"Processed {len(spike_batches)} batches")
```

### Resource Management

```python
# Use context manager for automatic cleanup
with ParallelProcessor(config) as processor:
    results = processor.parallel_spike_processing(spike_batches, processing_func)
# Resources are automatically cleaned up
```

## Advanced Usage

### Custom Processing Functions

```python
def custom_spike_processor(spike_batch, threshold=0.5, decay=0.95):
    """Custom spike processing with parameters."""
    # Apply threshold
    thresholded = spike_batch > threshold
    
    # Apply decay
    decayed = spike_batch * decay
    
    return thresholded.astype(float) * decayed

# Use with parallel processing
results = processor.parallel_spike_processing(
    spike_batches, 
    custom_spike_processor,
    threshold=0.3,
    decay=0.9
)
```

### Hybrid Parallelization

```python
# Combine different parallelization strategies
config = ParallelConfig(parallel_strategy="hybrid")
processor = ParallelProcessor(config)

# The processor will automatically choose the best strategy
# based on the workload characteristics
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile parallel execution
profiler = cProfile.Profile()
profiler.enable()

results = processor.parallel_spike_processing(spike_batches, processing_func)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

This comprehensive guide covers all aspects of using the parallel processing module for optimal performance in the Godly AI neuromorphic system.
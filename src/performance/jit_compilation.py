"""
Numba JIT compilation for neuromorphic computations.

This module provides JIT-compiled versions of performance-critical functions
with optimized memory access patterns for cache efficiency.
"""

import warnings
from typing import Optional, Dict, Any, Callable, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import logging
from functools import wraps
import time

# Configure logging
logger = logging.getLogger(__name__)

# Try to import Numba
try:
    import numba
    from numba import jit, njit, prange, types
    from numba.typed import Dict as NumbaDict
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators if Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if args else decorator
    
    def prange(*args, **kwargs):
        return range(*args, **kwargs)


@dataclass
class JITConfig:
    """Configuration for JIT compilation."""
    
    # Compilation settings
    enable_jit: bool = True
    cache_compiled_functions: bool = True
    parallel_compilation: bool = True
    
    # Optimization settings
    optimize_memory_layout: bool = True
    enable_fastmath: bool = True
    enable_parallel_loops: bool = True
    
    # Cache settings
    cache_directory: Optional[str] = None
    max_cache_size: int = 1024 * 1024 * 1024  # 1GB
    
    # Debugging
    verbose: bool = False
    profile_compilation: bool = False


class JITCompiler:
    """
    Numba JIT compiler for neuromorphic computations.
    
    Provides JIT-compiled versions of performance-critical functions
    with optimized memory access patterns and cache efficiency.
    """
    
    def __init__(self, config: Optional[JITConfig] = None):
        """Initialize JIT compiler with configuration."""
        self.config = config or JITConfig()
        self._compiled_functions = {}
        self._compilation_times = {}
        
        # Initialize JIT compilation
        self._initialize_jit()
    
    def _initialize_jit(self) -> None:
        """Initialize JIT compilation settings."""
        if not NUMBA_AVAILABLE:
            logger.warning("Numba not available, JIT compilation disabled")
            self.config.enable_jit = False
            return
        
        # Configure Numba settings
        if self.config.cache_compiled_functions:
            numba.config.CACHE_DIR = self.config.cache_directory or numba.config.CACHE_DIR
        
        if self.config.enable_fastmath:
            # Enable fast math optimizations
            pass  # This is set per-function with fastmath=True
        
        if self.config.verbose:
            logger.info(f"JIT compilation initialized with Numba {numba.__version__}")
    
    def is_numba_available(self) -> bool:
        """Check if Numba is available."""
        return NUMBA_AVAILABLE
    
    def get_jit_info(self) -> Dict[str, Any]:
        """Get information about JIT configuration."""
        info = {
            'numba_available': NUMBA_AVAILABLE,
            'jit_enabled': self.config.enable_jit,
            'cache_enabled': self.config.cache_compiled_functions,
            'parallel_enabled': self.config.enable_parallel_loops,
            'fastmath_enabled': self.config.enable_fastmath,
            'compiled_functions': len(self._compiled_functions)
        }
        
        if NUMBA_AVAILABLE:
            info['numba_version'] = numba.__version__
            info['compilation_times'] = self._compilation_times.copy()
        
        return info
    
    def compile_spike_processing_kernel(self) -> Callable:
        """Compile optimized spike processing kernel."""
        if not self.config.enable_jit:
            return self._spike_processing_kernel_python
        
        if 'spike_processing' in self._compiled_functions:
            return self._compiled_functions['spike_processing']
        
        # Compile with Numba
        start_time = time.time()
        
        @njit(
            parallel=self.config.enable_parallel_loops,
            fastmath=self.config.enable_fastmath,
            cache=self.config.cache_compiled_functions
        )
        def spike_processing_kernel(
            spike_trains: np.ndarray,
            weights: np.ndarray,
            thresholds: np.ndarray,
            dt: float
        ) -> np.ndarray:
            """
            Optimized spike processing kernel.
            
            Args:
                spike_trains: Input spike trains [batch, time, neurons]
                weights: Connection weights [neurons, neurons]
                thresholds: Neuron thresholds [neurons]
                dt: Time step
                
            Returns:
                Processed spike outputs [batch, time, neurons]
            """
            batch_size, time_steps, num_neurons = spike_trains.shape
            output = np.zeros_like(spike_trains, dtype=np.float32)
            
            # Process each batch in parallel
            for batch_idx in prange(batch_size):
                # Initialize membrane potentials
                membrane_potentials = np.zeros(num_neurons, dtype=np.float32)
                
                # Process each time step
                for t in range(time_steps):
                    # Get current input spikes
                    input_spikes = spike_trains[batch_idx, t, :]
                    
                    # Compute synaptic currents
                    synaptic_current = np.zeros(num_neurons, dtype=np.float32)
                    for i in range(num_neurons):
                        for j in range(num_neurons):
                            if input_spikes[j] > 0.5:  # Spike detected
                                synaptic_current[i] += weights[j, i]
                    
                    # Update membrane potentials
                    for i in range(num_neurons):
                        membrane_potentials[i] += synaptic_current[i] * dt
                        
                        # Check for spike generation
                        if membrane_potentials[i] >= thresholds[i]:
                            output[batch_idx, t, i] = 1.0
                            membrane_potentials[i] = 0.0  # Reset
                        else:
                            output[batch_idx, t, i] = 0.0
                        
                        # Membrane potential decay
                        membrane_potentials[i] *= 0.95  # Simple decay
            
            return output
        
        compilation_time = time.time() - start_time
        self._compilation_times['spike_processing'] = compilation_time
        self._compiled_functions['spike_processing'] = spike_processing_kernel
        
        if self.config.verbose:
            logger.info(f"Compiled spike processing kernel in {compilation_time:.3f}s")
        
        return spike_processing_kernel
    
    def _spike_processing_kernel_python(
        self,
        spike_trains: np.ndarray,
        weights: np.ndarray,
        thresholds: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Python fallback for spike processing kernel."""
        batch_size, time_steps, num_neurons = spike_trains.shape
        output = np.zeros_like(spike_trains, dtype=np.float32)
        
        for batch_idx in range(batch_size):
            membrane_potentials = np.zeros(num_neurons, dtype=np.float32)
            
            for t in range(time_steps):
                input_spikes = spike_trains[batch_idx, t, :]
                synaptic_current = np.dot(input_spikes, weights)
                
                membrane_potentials += synaptic_current * dt
                
                # Generate spikes
                spike_mask = membrane_potentials >= thresholds
                output[batch_idx, t, :] = spike_mask.astype(np.float32)
                membrane_potentials[spike_mask] = 0.0  # Reset
                membrane_potentials *= 0.95  # Decay
        
        return output
    
    def compile_reservoir_update_kernel(self) -> Callable:
        """Compile optimized reservoir update kernel."""
        if not self.config.enable_jit:
            return self._reservoir_update_kernel_python
        
        if 'reservoir_update' in self._compiled_functions:
            return self._compiled_functions['reservoir_update']
        
        start_time = time.time()
        
        @njit(
            parallel=self.config.enable_parallel_loops,
            fastmath=self.config.enable_fastmath,
            cache=self.config.cache_compiled_functions
        )
        def reservoir_update_kernel(
            reservoir_states: np.ndarray,
            input_weights: np.ndarray,
            reservoir_weights: np.ndarray,
            input_spikes: np.ndarray
        ) -> np.ndarray:
            """
            Optimized reservoir state update kernel.
            
            Args:
                reservoir_states: Current reservoir states [batch, neurons]
                input_weights: Input weights [input_size, neurons]
                reservoir_weights: Reservoir weights [neurons, neurons]
                input_spikes: Input spikes [batch, input_size]
                
            Returns:
                Updated reservoir currents [batch, neurons]
            """
            batch_size, num_neurons = reservoir_states.shape
            input_size = input_spikes.shape[1]
            output = np.zeros_like(reservoir_states, dtype=np.float32)
            
            # Process each batch in parallel
            for batch_idx in prange(batch_size):
                # Compute input contribution
                for neuron_idx in range(num_neurons):
                    input_current = 0.0
                    for input_idx in range(input_size):
                        input_current += input_spikes[batch_idx, input_idx] * input_weights[input_idx, neuron_idx]
                    
                    # Compute recurrent contribution
                    recurrent_current = 0.0
                    for other_neuron in range(num_neurons):
                        recurrent_current += reservoir_states[batch_idx, other_neuron] * reservoir_weights[other_neuron, neuron_idx]
                    
                    output[batch_idx, neuron_idx] = input_current + recurrent_current
            
            return output
        
        compilation_time = time.time() - start_time
        self._compilation_times['reservoir_update'] = compilation_time
        self._compiled_functions['reservoir_update'] = reservoir_update_kernel
        
        if self.config.verbose:
            logger.info(f"Compiled reservoir update kernel in {compilation_time:.3f}s")
        
        return reservoir_update_kernel
    
    def _reservoir_update_kernel_python(
        self,
        reservoir_states: np.ndarray,
        input_weights: np.ndarray,
        reservoir_weights: np.ndarray,
        input_spikes: np.ndarray
    ) -> np.ndarray:
        """Python fallback for reservoir update kernel."""
        # Compute input contribution
        input_current = np.dot(input_spikes, input_weights)
        
        # Compute recurrent contribution
        recurrent_current = np.dot(reservoir_states, reservoir_weights.T)
        
        return input_current + recurrent_current
    
    def compile_convolution_kernel(self) -> Callable:
        """Compile optimized convolution kernel for spike trains."""
        if not self.config.enable_jit:
            return self._convolution_kernel_python
        
        if 'convolution' in self._compiled_functions:
            return self._compiled_functions['convolution']
        
        start_time = time.time()
        
        @njit(
            parallel=self.config.enable_parallel_loops,
            fastmath=self.config.enable_fastmath,
            cache=self.config.cache_compiled_functions
        )
        def convolution_kernel(
            spike_trains: np.ndarray,
            kernel: np.ndarray
        ) -> np.ndarray:
            """
            Optimized convolution kernel for spike trains.
            
            Args:
                spike_trains: Input spike trains [batch, time, neurons]
                kernel: Convolution kernel [kernel_size]
                
            Returns:
                Convolved spike trains [batch, output_time, neurons]
            """
            batch_size, time_steps, num_neurons = spike_trains.shape
            kernel_size = kernel.shape[0]
            output_time = time_steps - kernel_size + 1
            
            if output_time <= 0:
                return np.zeros((batch_size, 1, num_neurons), dtype=np.float32)
            
            output = np.zeros((batch_size, output_time, num_neurons), dtype=np.float32)
            
            # Process each batch and neuron in parallel
            for batch_idx in prange(batch_size):
                for neuron_idx in prange(num_neurons):
                    for out_t in range(output_time):
                        conv_sum = 0.0
                        for k in range(kernel_size):
                            conv_sum += spike_trains[batch_idx, out_t + k, neuron_idx] * kernel[k]
                        output[batch_idx, out_t, neuron_idx] = conv_sum
            
            return output
        
        compilation_time = time.time() - start_time
        self._compilation_times['convolution'] = compilation_time
        self._compiled_functions['convolution'] = convolution_kernel
        
        if self.config.verbose:
            logger.info(f"Compiled convolution kernel in {compilation_time:.3f}s")
        
        return convolution_kernel
    
    def _convolution_kernel_python(
        self,
        spike_trains: np.ndarray,
        kernel: np.ndarray
    ) -> np.ndarray:
        """Python fallback for convolution kernel."""
        batch_size, time_steps, num_neurons = spike_trains.shape
        kernel_size = kernel.shape[0]
        output_time = time_steps - kernel_size + 1
        
        if output_time <= 0:
            return np.zeros((batch_size, 1, num_neurons), dtype=np.float32)
        
        output = np.zeros((batch_size, output_time, num_neurons), dtype=np.float32)
        
        for batch_idx in range(batch_size):
            for neuron_idx in range(num_neurons):
                neuron_data = spike_trains[batch_idx, :, neuron_idx]
                convolved = np.convolve(neuron_data, kernel, mode='valid')
                output[batch_idx, :, neuron_idx] = convolved
        
        return output
    
    def compile_matrix_multiply_kernel(self) -> Callable:
        """Compile optimized matrix multiplication kernel."""
        if not self.config.enable_jit:
            return self._matrix_multiply_kernel_python
        
        if 'matrix_multiply' in self._compiled_functions:
            return self._compiled_functions['matrix_multiply']
        
        start_time = time.time()
        
        @njit(
            parallel=self.config.enable_parallel_loops,
            fastmath=self.config.enable_fastmath,
            cache=self.config.cache_compiled_functions
        )
        def matrix_multiply_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """
            Optimized matrix multiplication kernel with cache-friendly access patterns.
            
            Args:
                a: First matrix [M, K]
                b: Second matrix [K, N]
                
            Returns:
                Matrix product [M, N]
            """
            M, K = a.shape
            K2, N = b.shape
            
            if K != K2:
                raise ValueError("Matrix dimensions don't match for multiplication")
            
            result = np.zeros((M, N), dtype=np.float32)
            
            # Cache-friendly blocked matrix multiplication
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
        
        compilation_time = time.time() - start_time
        self._compilation_times['matrix_multiply'] = compilation_time
        self._compiled_functions['matrix_multiply'] = matrix_multiply_kernel
        
        if self.config.verbose:
            logger.info(f"Compiled matrix multiply kernel in {compilation_time:.3f}s")
        
        return matrix_multiply_kernel
    
    def _matrix_multiply_kernel_python(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Python fallback for matrix multiplication."""
        return np.dot(a, b)
    
    def benchmark_jit_performance(
        self,
        test_sizes: list = [100, 500, 1000],
        num_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark JIT compilation performance.
        
        Args:
            test_sizes: List of test data sizes
            num_iterations: Number of iterations per test
            
        Returns:
            Performance benchmark results
        """
        results = {
            'jit_info': self.get_jit_info(),
            'spike_processing': {},
            'reservoir_update': {},
            'convolution': {},
            'matrix_multiply': {}
        }
        
        for size in test_sizes:
            # Spike processing benchmark
            spike_trains = np.random.rand(4, size, 50).astype(np.float32)
            spike_trains = (spike_trains > 0.9).astype(np.float32)  # Sparse spikes
            weights = np.random.randn(50, 50).astype(np.float32) * 0.1
            thresholds = np.ones(50, dtype=np.float32) * 0.5
            dt = 0.001
            
            # Get compiled function
            jit_func = self.compile_spike_processing_kernel()
            python_func = self._spike_processing_kernel_python
            
            # Warm up JIT
            _ = jit_func(spike_trains[:1], weights, thresholds, dt)
            
            # Benchmark JIT
            start_time = time.time()
            for _ in range(num_iterations):
                jit_result = jit_func(spike_trains, weights, thresholds, dt)
            jit_time = (time.time() - start_time) / num_iterations
            
            # Benchmark Python
            start_time = time.time()
            for _ in range(num_iterations):
                python_result = python_func(spike_trains, weights, thresholds, dt)
            python_time = (time.time() - start_time) / num_iterations
            
            speedup = python_time / jit_time if jit_time > 0 else 1.0
            
            results['spike_processing'][size] = {
                'jit_time': jit_time,
                'python_time': python_time,
                'speedup': speedup
            }
            
            # Matrix multiplication benchmark
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            jit_matmul = self.compile_matrix_multiply_kernel()
            python_matmul = self._matrix_multiply_kernel_python
            
            # Warm up
            _ = jit_matmul(a[:10, :10], b[:10, :10])
            
            # Benchmark JIT
            start_time = time.time()
            for _ in range(num_iterations):
                jit_result = jit_matmul(a, b)
            jit_time = (time.time() - start_time) / num_iterations
            
            # Benchmark Python
            start_time = time.time()
            for _ in range(num_iterations):
                python_result = python_matmul(a, b)
            python_time = (time.time() - start_time) / num_iterations
            
            speedup = python_time / jit_time if jit_time > 0 else 1.0
            
            results['matrix_multiply'][size] = {
                'jit_time': jit_time,
                'python_time': python_time,
                'speedup': speedup
            }
        
        return results
    
    def optimize_memory_layout(self, array: np.ndarray, access_pattern: str = "row_major") -> np.ndarray:
        """
        Optimize array memory layout for cache efficiency.
        
        Args:
            array: Input array
            access_pattern: Access pattern ("row_major", "column_major", "blocked")
            
        Returns:
            Optimized array
        """
        if not self.config.optimize_memory_layout:
            return array
        
        if access_pattern == "row_major":
            return np.ascontiguousarray(array)
        elif access_pattern == "column_major":
            return np.asfortranarray(array)
        elif access_pattern == "blocked":
            # For blocked access patterns, ensure contiguous memory
            return np.ascontiguousarray(array)
        else:
            return array
    
    def clear_cache(self):
        """Clear compiled function cache."""
        self._compiled_functions.clear()
        self._compilation_times.clear()
        
        if NUMBA_AVAILABLE and hasattr(numba, 'clear_cache'):
            numba.clear_cache()


def enable_jit_compilation(config: Optional[JITConfig] = None) -> JITCompiler:
    """
    Enable JIT compilation globally.
    
    Args:
        config: JIT compilation configuration
        
    Returns:
        JIT compiler instance
    """
    compiler = JITCompiler(config)
    
    if compiler.is_numba_available():
        # Pre-compile commonly used functions
        compiler.compile_spike_processing_kernel()
        compiler.compile_reservoir_update_kernel()
        compiler.compile_matrix_multiply_kernel()
    
    return compiler


# Global compiler instance
_global_compiler: Optional[JITCompiler] = None


def get_global_compiler() -> Optional[JITCompiler]:
    """Get the global JIT compiler instance."""
    return _global_compiler


def set_global_compiler(compiler: JITCompiler) -> None:
    """Set the global JIT compiler instance."""
    global _global_compiler
    _global_compiler = compiler


# Convenience functions for JIT-compiled operations
def jit_spike_processing(
    spike_trains: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray,
    dt: float
) -> np.ndarray:
    """JIT-compiled spike processing using global compiler."""
    compiler = get_global_compiler()
    if compiler is not None:
        kernel = compiler.compile_spike_processing_kernel()
        return kernel(spike_trains, weights, thresholds, dt)
    else:
        # Fallback to numpy
        return np.zeros_like(spike_trains)


def jit_reservoir_update(
    reservoir_states: np.ndarray,
    input_weights: np.ndarray,
    reservoir_weights: np.ndarray,
    input_spikes: np.ndarray
) -> np.ndarray:
    """JIT-compiled reservoir update using global compiler."""
    compiler = get_global_compiler()
    if compiler is not None:
        kernel = compiler.compile_reservoir_update_kernel()
        return kernel(reservoir_states, input_weights, reservoir_weights, input_spikes)
    else:
        # Fallback to numpy
        input_current = np.dot(input_spikes, input_weights)
        recurrent_current = np.dot(reservoir_states, reservoir_weights.T)
        return input_current + recurrent_current


def jit_matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """JIT-compiled matrix multiplication using global compiler."""
    compiler = get_global_compiler()
    if compiler is not None:
        kernel = compiler.compile_matrix_multiply_kernel()
        return kernel(a, b)
    else:
        return np.dot(a, b)
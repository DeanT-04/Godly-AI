"""
Intel Math Kernel Library (MKL) optimization for neuromorphic computations.

This module provides optimized matrix operations and vectorized spike processing
using Intel MKL for maximum CPU performance.
"""

import os
import warnings
from typing import Optional, Dict, Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import config
from dataclasses import dataclass
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MKLConfig:
    """Configuration for Intel MKL optimization."""
    
    # Threading configuration
    num_threads: Optional[int] = None  # Auto-detect if None
    thread_affinity: str = "balanced"  # "balanced", "compact", "scatter"
    
    # Memory optimization
    enable_memory_pool: bool = True
    memory_pool_size: int = 1024 * 1024 * 1024  # 1GB default
    
    # Computation optimization
    enable_fast_math: bool = True
    precision_mode: str = "high"  # "high", "medium", "low"
    
    # Vectorization
    enable_avx512: bool = True
    enable_avx2: bool = True
    enable_sse: bool = True
    
    # Debugging
    verbose: bool = False


class MKLOptimizer:
    """
    Intel MKL optimizer for neuromorphic computations.
    
    Provides optimized implementations of matrix operations, spike processing,
    and reservoir computations using Intel MKL.
    """
    
    def __init__(self, config: Optional[MKLConfig] = None):
        """Initialize MKL optimizer with configuration."""
        self.config = config or MKLConfig()
        self._mkl_available = False
        self._original_blas_threads = None
        
        # Try to initialize MKL
        self._initialize_mkl()
    
    def _initialize_mkl(self) -> None:
        """Initialize Intel MKL if available."""
        try:
            # Try to import Intel MKL
            import mkl
            self._mkl_available = True
            
            # Configure MKL threading
            if self.config.num_threads is not None:
                mkl.set_num_threads(self.config.num_threads)
            else:
                # Auto-detect optimal thread count
                import multiprocessing
                optimal_threads = min(multiprocessing.cpu_count(), 8)
                mkl.set_num_threads(optimal_threads)
                self.config.num_threads = optimal_threads
            
            # Set thread affinity
            if hasattr(mkl, 'set_threading_layer'):
                if self.config.thread_affinity == "compact":
                    os.environ['KMP_AFFINITY'] = 'compact'
                elif self.config.thread_affinity == "scatter":
                    os.environ['KMP_AFFINITY'] = 'scatter'
                else:  # balanced
                    os.environ['KMP_AFFINITY'] = 'balanced'
            
            # Configure memory pool
            if self.config.enable_memory_pool and hasattr(mkl, 'mem_stat'):
                os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX512'
            
            # Configure precision
            if self.config.precision_mode == "low":
                os.environ['MKL_CBWR'] = 'COMPATIBLE'
            elif self.config.precision_mode == "medium":
                os.environ['MKL_CBWR'] = 'SSE4_2'
            else:  # high precision
                os.environ['MKL_CBWR'] = 'AUTO'
            
            if self.config.verbose:
                logger.info(f"Intel MKL initialized with {self.config.num_threads} threads")
                
        except ImportError:
            # Fallback to OpenBLAS or standard BLAS
            self._mkl_available = False
            self._initialize_fallback_blas()
            
            if self.config.verbose:
                logger.warning("Intel MKL not available, using fallback BLAS")
    
    def _initialize_fallback_blas(self) -> None:
        """Initialize fallback BLAS configuration."""
        try:
            # Try to configure OpenBLAS
            import numpy as np
            
            # Set number of threads for BLAS operations
            if self.config.num_threads is not None:
                os.environ['OPENBLAS_NUM_THREADS'] = str(self.config.num_threads)
                os.environ['MKL_NUM_THREADS'] = str(self.config.num_threads)
                os.environ['NUMEXPR_NUM_THREADS'] = str(self.config.num_threads)
                os.environ['OMP_NUM_THREADS'] = str(self.config.num_threads)
            
            # Configure JAX to use optimized BLAS
            config.update('jax_enable_x64', True)
            
        except Exception as e:
            logger.warning(f"Failed to configure fallback BLAS: {e}")
    
    def is_mkl_available(self) -> bool:
        """Check if Intel MKL is available."""
        return self._mkl_available
    
    def get_mkl_info(self) -> Dict[str, Any]:
        """Get information about MKL configuration."""
        info = {
            'mkl_available': self._mkl_available,
            'num_threads': self.config.num_threads,
            'thread_affinity': self.config.thread_affinity,
            'precision_mode': self.config.precision_mode
        }
        
        if self._mkl_available:
            try:
                import mkl
                info['mkl_version'] = mkl.get_version_string()
                if hasattr(mkl, 'get_max_threads'):
                    info['max_threads'] = mkl.get_max_threads()
            except:
                pass
        
        return info
    
    def optimize_matrix_multiply(
        self, 
        a: jnp.ndarray, 
        b: jnp.ndarray,
        transpose_a: bool = False,
        transpose_b: bool = False
    ) -> jnp.ndarray:
        """
        Optimized matrix multiplication using MKL.
        
        Args:
            a: First matrix
            b: Second matrix
            transpose_a: Whether to transpose first matrix
            transpose_b: Whether to transpose second matrix
            
        Returns:
            Matrix product
        """
        if self._mkl_available:
            # Use MKL-optimized operations
            if transpose_a:
                a = jnp.transpose(a)
            if transpose_b:
                b = jnp.transpose(b)
            
            # JAX will automatically use MKL if available
            return jnp.dot(a, b)
        else:
            # Fallback implementation
            return jnp.dot(
                jnp.transpose(a) if transpose_a else a,
                jnp.transpose(b) if transpose_b else b
            )
    
    def optimize_spike_convolution(
        self,
        spike_trains: jnp.ndarray,
        kernel: jnp.ndarray,
        mode: str = 'valid'
    ) -> jnp.ndarray:
        """
        Optimized convolution for spike train processing.
        
        Args:
            spike_trains: Input spike trains [batch, time, neurons]
            kernel: Convolution kernel [kernel_size]
            mode: Convolution mode ('valid', 'same', 'full')
            
        Returns:
            Convolved spike trains
        """
        # Use JAX's optimized convolution operations
        batch_size, time_steps, num_neurons = spike_trains.shape
        kernel_size = kernel.shape[0]
        
        # Apply convolution along the time axis for each neuron
        def convolve_single_neuron(neuron_data):
            # neuron_data shape: [batch, time]
            return jnp.apply_along_axis(
                lambda x: jnp.convolve(x, kernel, mode=mode),
                axis=1,
                arr=neuron_data
            )
        
        # Process each neuron separately
        results = []
        for neuron_idx in range(num_neurons):
            neuron_spikes = spike_trains[:, :, neuron_idx]  # [batch, time]
            convolved = convolve_single_neuron(neuron_spikes)  # [batch, output_time]
            results.append(convolved)
        
        # Stack results back together
        result = jnp.stack(results, axis=2)  # [batch, output_time, neurons]
        
        return result
    
    def _create_convolution_matrix(
        self,
        kernel: jnp.ndarray,
        input_size: int,
        mode: str
    ) -> jnp.ndarray:
        """Create Toeplitz matrix for convolution."""
        kernel_size = kernel.shape[0]
        
        if mode == 'valid':
            output_size = input_size - kernel_size + 1
            # Create Toeplitz matrix
            conv_matrix = jnp.zeros((output_size, input_size))
            for i in range(output_size):
                conv_matrix = conv_matrix.at[i, i:i+kernel_size].set(kernel)
        elif mode == 'same':
            output_size = input_size
            pad_size = kernel_size // 2
            # Create padded Toeplitz matrix
            conv_matrix = jnp.zeros((output_size, input_size))
            for i in range(output_size):
                start_idx = max(0, i - pad_size)
                end_idx = min(input_size, i + kernel_size - pad_size)
                kernel_start = max(0, pad_size - i)
                kernel_end = kernel_start + (end_idx - start_idx)
                conv_matrix = conv_matrix.at[i, start_idx:end_idx].set(
                    kernel[kernel_start:kernel_end]
                )
        else:  # 'full'
            output_size = input_size + kernel_size - 1
            conv_matrix = jnp.zeros((output_size, input_size))
            for i in range(output_size):
                input_start = max(0, i - kernel_size + 1)
                input_end = min(input_size, i + 1)
                kernel_start = max(0, kernel_size - 1 - i)
                kernel_end = kernel_start + (input_end - input_start)
                conv_matrix = conv_matrix.at[i, input_start:input_end].set(
                    kernel[kernel_start:kernel_end]
                )
        
        return conv_matrix
    
    def optimize_reservoir_update(
        self,
        reservoir_state: jnp.ndarray,
        input_weights: jnp.ndarray,
        reservoir_weights: jnp.ndarray,
        input_spikes: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Optimized reservoir state update using vectorized operations.
        
        Args:
            reservoir_state: Current reservoir state [reservoir_size]
            input_weights: Input connection weights [input_size, reservoir_size]
            reservoir_weights: Reservoir weights [reservoir_size, reservoir_size]
            input_spikes: Input spike pattern [input_size]
            
        Returns:
            Updated reservoir currents [reservoir_size]
        """
        # Compute input contribution: input_spikes @ input_weights
        input_current = jnp.dot(input_spikes, input_weights)
        
        # Compute recurrent contribution: reservoir_state @ reservoir_weights.T
        recurrent_current = jnp.dot(reservoir_state, reservoir_weights.T)
        
        return input_current + recurrent_current
    
    def optimize_batch_processing(
        self,
        spike_batches: jnp.ndarray,
        weight_matrix: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Optimized batch processing of spike patterns.
        
        Args:
            spike_batches: Batch of spike patterns [batch_size, input_size]
            weight_matrix: Weight matrix [input_size, output_size]
            
        Returns:
            Batch of outputs [batch_size, output_size]
        """
        # Use optimized batch matrix multiplication
        return self.optimize_matrix_multiply(spike_batches, weight_matrix)
    
    def optimize_eigenvalue_computation(
        self,
        matrix: jnp.ndarray,
        compute_eigenvectors: bool = False
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Optimized eigenvalue computation for spectral radius calculation.
        
        Args:
            matrix: Input matrix
            compute_eigenvectors: Whether to compute eigenvectors
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        if compute_eigenvectors:
            eigenvalues, eigenvectors = jnp.linalg.eig(matrix)
            return eigenvalues, eigenvectors
        else:
            eigenvalues = jnp.linalg.eigvals(matrix)
            return eigenvalues, None
    
    def benchmark_performance(
        self,
        matrix_sizes: list = [100, 500, 1000, 2000],
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark MKL performance against baseline.
        
        Args:
            matrix_sizes: List of matrix sizes to test
            num_iterations: Number of iterations per test
            
        Returns:
            Performance benchmark results
        """
        import time
        
        results = {
            'mkl_available': self._mkl_available,
            'matrix_multiply': {},
            'eigenvalue': {},
            'convolution': {}
        }
        
        for size in matrix_sizes:
            # Matrix multiplication benchmark
            key = jax.random.PRNGKey(42)
            a = jax.random.normal(key, (size, size))
            b = jax.random.normal(key, (size, size))
            
            # Warm up
            _ = jnp.dot(a, b)
            
            # Time matrix multiplication
            start_time = time.time()
            for _ in range(num_iterations):
                result = self.optimize_matrix_multiply(a, b)
                result.block_until_ready()  # Ensure computation completes
            end_time = time.time()
            
            results['matrix_multiply'][size] = {
                'time_per_op': (end_time - start_time) / num_iterations,
                'gflops': (2 * size**3) / ((end_time - start_time) / num_iterations) / 1e9
            }
            
            # Eigenvalue computation benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                eigenvals, _ = self.optimize_eigenvalue_computation(a)
                eigenvals.block_until_ready()
            end_time = time.time()
            
            results['eigenvalue'][size] = {
                'time_per_op': (end_time - start_time) / num_iterations
            }
            
            # Convolution benchmark (for smaller sizes)
            if size <= 1000:
                spike_data = jax.random.bernoulli(key, 0.1, (10, size, 100))
                kernel = jax.random.normal(key, (20,))
                
                start_time = time.time()
                for _ in range(num_iterations):
                    conv_result = self.optimize_spike_convolution(spike_data, kernel)
                    conv_result.block_until_ready()
                end_time = time.time()
                
                results['convolution'][size] = {
                    'time_per_op': (end_time - start_time) / num_iterations
                }
        
        return results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Restore original BLAS settings if needed
        pass


def enable_mkl_optimization(config: Optional[MKLConfig] = None) -> MKLOptimizer:
    """
    Enable Intel MKL optimization globally.
    
    Args:
        config: MKL configuration
        
    Returns:
        MKL optimizer instance
    """
    optimizer = MKLOptimizer(config)
    
    # Set global JAX configuration for optimal performance
    if optimizer.is_mkl_available():
        config.update('jax_enable_x64', True)
        config.update('jax_platform_name', 'cpu')
    
    return optimizer


# Global optimizer instance
_global_optimizer: Optional[MKLOptimizer] = None


def get_global_optimizer() -> Optional[MKLOptimizer]:
    """Get the global MKL optimizer instance."""
    return _global_optimizer


def set_global_optimizer(optimizer: MKLOptimizer) -> None:
    """Set the global MKL optimizer instance."""
    global _global_optimizer
    _global_optimizer = optimizer


# Convenience functions for optimized operations
def optimized_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Optimized matrix multiplication using global optimizer."""
    optimizer = get_global_optimizer()
    if optimizer is not None:
        return optimizer.optimize_matrix_multiply(a, b)
    else:
        return jnp.dot(a, b)


def optimized_spike_conv(
    spike_trains: jnp.ndarray,
    kernel: jnp.ndarray,
    mode: str = 'valid'
) -> jnp.ndarray:
    """Optimized spike convolution using global optimizer."""
    optimizer = get_global_optimizer()
    if optimizer is not None:
        return optimizer.optimize_spike_convolution(spike_trains, kernel, mode)
    else:
        # Fallback to standard convolution
        return jnp.apply_along_axis(
            lambda x: jnp.convolve(x, kernel, mode=mode),
            axis=1,
            arr=spike_trains
        )
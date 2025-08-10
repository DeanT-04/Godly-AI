"""
Tests for Intel MKL optimization module.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import time

from src.performance.mkl_optimization import (
    MKLOptimizer, MKLConfig, enable_mkl_optimization,
    optimized_matmul, optimized_spike_conv
)


class TestMKLOptimizer:
    """Test suite for MKL optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = MKLConfig(
            num_threads=2,
            verbose=False
        )
        self.optimizer = MKLOptimizer(self.config)
        self.key = random.PRNGKey(42)
    
    def test_initialization(self):
        """Test MKL optimizer initialization."""
        assert self.optimizer is not None
        assert isinstance(self.optimizer.config, MKLConfig)
        
        # Test configuration
        assert self.optimizer.config.num_threads == 2
        assert not self.optimizer.config.verbose
    
    def test_mkl_info(self):
        """Test MKL information retrieval."""
        info = self.optimizer.get_mkl_info()
        
        assert 'mkl_available' in info
        assert 'num_threads' in info
        assert 'thread_affinity' in info
        assert 'precision_mode' in info
        
        assert isinstance(info['mkl_available'], bool)
        assert info['num_threads'] == 2
    
    def test_matrix_multiplication(self):
        """Test optimized matrix multiplication."""
        # Create test matrices
        size = 100
        a = random.normal(self.key, (size, size))
        b = random.normal(self.key, (size, size))
        
        # Test basic multiplication
        result = self.optimizer.optimize_matrix_multiply(a, b)
        expected = jnp.dot(a, b)
        
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        
        # Test with transpose
        result_t = self.optimizer.optimize_matrix_multiply(
            a, b, transpose_a=True
        )
        expected_t = jnp.dot(a.T, b)
        
        np.testing.assert_allclose(result_t, expected_t, rtol=1e-6)
    
    def test_spike_convolution(self):
        """Test optimized spike convolution."""
        # Create test spike trains
        batch_size, time_steps, num_neurons = 5, 100, 20
        spike_trains = random.bernoulli(
            self.key, 0.1, (batch_size, time_steps, num_neurons)
        )
        
        # Create convolution kernel
        kernel_size = 10
        kernel = random.normal(self.key, (kernel_size,))
        
        # Test convolution
        result = self.optimizer.optimize_spike_convolution(
            spike_trains, kernel, mode='valid'
        )
        
        expected_time = time_steps - kernel_size + 1
        assert result.shape == (batch_size, expected_time, num_neurons)
        
        # Test different modes
        result_same = self.optimizer.optimize_spike_convolution(
            spike_trains, kernel, mode='same'
        )
        assert result_same.shape == (batch_size, time_steps, num_neurons)
        
        result_full = self.optimizer.optimize_spike_convolution(
            spike_trains, kernel, mode='full'
        )
        expected_full_time = time_steps + kernel_size - 1
        assert result_full.shape == (batch_size, expected_full_time, num_neurons)
    
    def test_reservoir_update(self):
        """Test optimized reservoir state update."""
        # Create test data
        reservoir_size = 100
        input_size = 50
        
        reservoir_state = random.normal(self.key, (reservoir_size,))
        input_weights = random.normal(self.key, (input_size, reservoir_size))
        reservoir_weights = random.normal(self.key, (reservoir_size, reservoir_size))
        input_spikes = random.bernoulli(self.key, 0.1, (input_size,))
        
        # Test reservoir update
        result = self.optimizer.optimize_reservoir_update(
            reservoir_state, input_weights, reservoir_weights, input_spikes
        )
        
        assert result.shape == (reservoir_size,)
        
        # Verify computation manually
        input_current = jnp.dot(input_spikes, input_weights)
        recurrent_current = jnp.dot(reservoir_state, reservoir_weights.T)
        expected = input_current + recurrent_current
        
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_batch_processing(self):
        """Test optimized batch processing."""
        batch_size = 32
        input_size = 100
        output_size = 50
        
        spike_batches = random.bernoulli(
            self.key, 0.1, (batch_size, input_size)
        )
        weight_matrix = random.normal(self.key, (input_size, output_size))
        
        result = self.optimizer.optimize_batch_processing(
            spike_batches, weight_matrix
        )
        
        assert result.shape == (batch_size, output_size)
        
        # Verify against standard implementation
        expected = jnp.dot(spike_batches, weight_matrix)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_eigenvalue_computation(self):
        """Test optimized eigenvalue computation."""
        size = 50
        matrix = random.normal(self.key, (size, size))
        
        # Test eigenvalues only
        eigenvals, eigenvecs = self.optimizer.optimize_eigenvalue_computation(
            matrix, compute_eigenvectors=False
        )
        
        assert eigenvals.shape == (size,)
        assert eigenvecs is None
        
        # Test with eigenvectors
        eigenvals_with_vecs, eigenvecs_with_vecs = self.optimizer.optimize_eigenvalue_computation(
            matrix, compute_eigenvectors=True
        )
        
        assert eigenvals_with_vecs.shape == (size,)
        assert eigenvecs_with_vecs.shape == (size, size)
        
        # Verify eigenvalue computation
        expected_eigenvals = jnp.linalg.eigvals(matrix)
        
        # Sort both arrays for comparison (eigenvalues may be in different order)
        eigenvals_sorted = jnp.sort(eigenvals)
        expected_sorted = jnp.sort(expected_eigenvals)
        
        np.testing.assert_allclose(eigenvals_sorted, expected_sorted, rtol=1e-6)
    
    def test_convolution_matrix_creation(self):
        """Test convolution matrix creation."""
        kernel = jnp.array([1.0, 2.0, 3.0])
        input_size = 5
        
        # Test valid mode
        conv_matrix = self.optimizer._create_convolution_matrix(
            kernel, input_size, 'valid'
        )
        expected_output_size = input_size - len(kernel) + 1
        assert conv_matrix.shape == (expected_output_size, input_size)
        
        # Test same mode
        conv_matrix_same = self.optimizer._create_convolution_matrix(
            kernel, input_size, 'same'
        )
        assert conv_matrix_same.shape == (input_size, input_size)
        
        # Test full mode
        conv_matrix_full = self.optimizer._create_convolution_matrix(
            kernel, input_size, 'full'
        )
        expected_full_size = input_size + len(kernel) - 1
        assert conv_matrix_full.shape == (expected_full_size, input_size)
    
    def test_performance_benchmark(self):
        """Test performance benchmarking."""
        # Run a quick benchmark with small matrices
        results = self.optimizer.benchmark_performance(
            matrix_sizes=[50, 100],
            num_iterations=3
        )
        
        assert 'mkl_available' in results
        assert 'matrix_multiply' in results
        assert 'eigenvalue' in results
        assert 'convolution' in results
        
        # Check that we have results for each matrix size
        for size in [50, 100]:
            assert size in results['matrix_multiply']
            assert 'time_per_op' in results['matrix_multiply'][size]
            assert 'gflops' in results['matrix_multiply'][size]
            
            assert size in results['eigenvalue']
            assert 'time_per_op' in results['eigenvalue'][size]
            
            assert size in results['convolution']
            assert 'time_per_op' in results['convolution'][size]
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with MKLOptimizer(self.config) as optimizer:
            assert optimizer is not None
            
            # Test basic operation within context
            a = random.normal(self.key, (10, 10))
            b = random.normal(self.key, (10, 10))
            result = optimizer.optimize_matrix_multiply(a, b)
            
            assert result.shape == (10, 10)


class TestGlobalOptimizer:
    """Test global optimizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = random.PRNGKey(42)
    
    def test_enable_mkl_optimization(self):
        """Test enabling MKL optimization globally."""
        config = MKLConfig(num_threads=1, verbose=False)
        optimizer = enable_mkl_optimization(config)
        
        assert optimizer is not None
        assert isinstance(optimizer, MKLOptimizer)
    
    def test_optimized_functions(self):
        """Test global optimized functions."""
        # Enable optimization
        config = MKLConfig(num_threads=1, verbose=False)
        optimizer = enable_mkl_optimization(config)
        
        # Set global optimizer
        from src.performance.mkl_optimization import set_global_optimizer
        set_global_optimizer(optimizer)
        
        # Test optimized matrix multiplication
        a = random.normal(self.key, (20, 20))
        b = random.normal(self.key, (20, 20))
        
        result = optimized_matmul(a, b)
        expected = jnp.dot(a, b)
        
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        
        # Test optimized spike convolution
        spike_trains = random.bernoulli(self.key, 0.1, (5, 50, 10))
        kernel = random.normal(self.key, (5,))
        
        result = optimized_spike_conv(spike_trains, kernel)
        
        # Should have valid convolution shape
        expected_time = spike_trains.shape[1] - kernel.shape[0] + 1
        assert result.shape == (5, expected_time, 10)


class TestMKLConfig:
    """Test MKL configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MKLConfig()
        
        assert config.num_threads is None
        assert config.thread_affinity == "balanced"
        assert config.enable_memory_pool is True
        assert config.memory_pool_size == 1024 * 1024 * 1024
        assert config.enable_fast_math is True
        assert config.precision_mode == "high"
        assert config.enable_avx512 is True
        assert config.enable_avx2 is True
        assert config.enable_sse is True
        assert config.verbose is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MKLConfig(
            num_threads=4,
            thread_affinity="compact",
            enable_memory_pool=False,
            precision_mode="medium",
            verbose=True
        )
        
        assert config.num_threads == 4
        assert config.thread_affinity == "compact"
        assert config.enable_memory_pool is False
        assert config.precision_mode == "medium"
        assert config.verbose is True


if __name__ == "__main__":
    pytest.main([__file__])
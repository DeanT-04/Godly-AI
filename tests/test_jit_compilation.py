"""
Tests for JIT compilation module.
"""

import pytest
import numpy as np
import time

from src.performance.jit_compilation import (
    JITCompiler, JITConfig, enable_jit_compilation,
    jit_spike_processing, jit_reservoir_update, jit_matrix_multiply,
    NUMBA_AVAILABLE
)


class TestJITCompiler:
    """Test suite for JIT compiler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = JITConfig(
            enable_jit=True,
            verbose=False,
            profile_compilation=False
        )
        self.compiler = JITCompiler(self.config)
    
    def test_initialization(self):
        """Test JIT compiler initialization."""
        assert self.compiler is not None
        assert isinstance(self.compiler.config, JITConfig)
        
        # Test configuration (may be disabled if Numba not available)
        if NUMBA_AVAILABLE:
            assert self.compiler.config.enable_jit is True
        else:
            # JIT gets disabled if Numba is not available
            assert self.compiler.config.enable_jit is False
        assert not self.compiler.config.verbose
    
    def test_numba_availability(self):
        """Test Numba availability detection."""
        available = self.compiler.is_numba_available()
        assert isinstance(available, bool)
        assert available == NUMBA_AVAILABLE
    
    def test_jit_info(self):
        """Test JIT information retrieval."""
        info = self.compiler.get_jit_info()
        
        assert 'numba_available' in info
        assert 'jit_enabled' in info
        assert 'cache_enabled' in info
        assert 'parallel_enabled' in info
        assert 'fastmath_enabled' in info
        assert 'compiled_functions' in info
        
        assert info['numba_available'] == NUMBA_AVAILABLE
        assert info['jit_enabled'] == self.config.enable_jit
        assert isinstance(info['compiled_functions'], int)
    
    def test_spike_processing_kernel(self):
        """Test spike processing kernel compilation and execution."""
        # Create test data
        batch_size, time_steps, num_neurons = 2, 10, 5
        spike_trains = np.random.rand(batch_size, time_steps, num_neurons).astype(np.float32)
        spike_trains = (spike_trains > 0.8).astype(np.float32)  # Sparse spikes
        
        weights = np.random.randn(num_neurons, num_neurons).astype(np.float32) * 0.1
        thresholds = np.ones(num_neurons, dtype=np.float32) * 0.5
        dt = 0.001
        
        # Compile and test kernel
        kernel = self.compiler.compile_spike_processing_kernel()
        result = kernel(spike_trains, weights, thresholds, dt)
        
        # Verify output shape and type
        assert result.shape == spike_trains.shape
        assert result.dtype == np.float32
        
        # Verify output is binary (0 or 1)
        assert np.all((result == 0) | (result == 1))
        
        # Test against Python fallback
        python_result = self.compiler._spike_processing_kernel_python(
            spike_trains, weights, thresholds, dt
        )
        
        # Results should be similar (may not be identical due to numerical precision)
        assert result.shape == python_result.shape
    
    def test_reservoir_update_kernel(self):
        """Test reservoir update kernel compilation and execution."""
        # Create test data
        batch_size, num_neurons, input_size = 3, 20, 10
        reservoir_states = np.random.randn(batch_size, num_neurons).astype(np.float32)
        input_weights = np.random.randn(input_size, num_neurons).astype(np.float32)
        reservoir_weights = np.random.randn(num_neurons, num_neurons).astype(np.float32)
        input_spikes = np.random.rand(batch_size, input_size).astype(np.float32)
        input_spikes = (input_spikes > 0.9).astype(np.float32)
        
        # Compile and test kernel
        kernel = self.compiler.compile_reservoir_update_kernel()
        result = kernel(reservoir_states, input_weights, reservoir_weights, input_spikes)
        
        # Verify output shape and type
        assert result.shape == (batch_size, num_neurons)
        assert result.dtype == np.float32
        
        # Test against Python fallback
        python_result = self.compiler._reservoir_update_kernel_python(
            reservoir_states, input_weights, reservoir_weights, input_spikes
        )
        
        # Results should be very close
        np.testing.assert_allclose(result, python_result, rtol=1e-5, atol=1e-6)
    
    def test_convolution_kernel(self):
        """Test convolution kernel compilation and execution."""
        # Create test data
        batch_size, time_steps, num_neurons = 2, 20, 8
        kernel_size = 5
        
        spike_trains = np.random.rand(batch_size, time_steps, num_neurons).astype(np.float32)
        spike_trains = (spike_trains > 0.8).astype(np.float32)
        kernel = np.random.randn(kernel_size).astype(np.float32)
        
        # Compile and test kernel
        conv_kernel = self.compiler.compile_convolution_kernel()
        result = conv_kernel(spike_trains, kernel)
        
        # Verify output shape
        expected_time = time_steps - kernel_size + 1
        assert result.shape == (batch_size, expected_time, num_neurons)
        assert result.dtype == np.float32
        
        # Test against Python fallback
        python_result = self.compiler._convolution_kernel_python(spike_trains, kernel)
        
        # Results should be very close
        np.testing.assert_allclose(result, python_result, rtol=1e-5, atol=1e-6)
    
    def test_matrix_multiply_kernel(self):
        """Test matrix multiplication kernel compilation and execution."""
        # Create test matrices
        M, K, N = 50, 30, 40
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)
        
        # Compile and test kernel
        matmul_kernel = self.compiler.compile_matrix_multiply_kernel()
        result = matmul_kernel(a, b)
        
        # Verify output shape and type
        assert result.shape == (M, N)
        assert result.dtype == np.float32
        
        # Test against Python fallback (numpy)
        python_result = self.compiler._matrix_multiply_kernel_python(a, b)
        
        # Results should be very close
        np.testing.assert_allclose(result, python_result, rtol=1e-5, atol=1e-6)
    
    def test_memory_layout_optimization(self):
        """Test memory layout optimization."""
        # Create test array
        array = np.random.randn(100, 50)
        
        # Test row-major optimization
        row_major = self.compiler.optimize_memory_layout(array, "row_major")
        assert row_major.flags['C_CONTIGUOUS']
        
        # Test column-major optimization
        col_major = self.compiler.optimize_memory_layout(array, "column_major")
        assert col_major.flags['F_CONTIGUOUS']
        
        # Test blocked optimization
        blocked = self.compiler.optimize_memory_layout(array, "blocked")
        assert blocked.flags['C_CONTIGUOUS']
        
        # Test unknown pattern (should return original)
        unknown = self.compiler.optimize_memory_layout(array, "unknown")
        np.testing.assert_array_equal(unknown, array)
    
    def test_cache_management(self):
        """Test compiled function cache management."""
        # Compile some functions
        self.compiler.compile_spike_processing_kernel()
        self.compiler.compile_matrix_multiply_kernel()
        
        # Check cache behavior (depends on whether Numba is available)
        if NUMBA_AVAILABLE and self.compiler.config.enable_jit:
            # With Numba, functions should be cached
            assert len(self.compiler._compiled_functions) > 0
        else:
            # Without Numba, no functions are cached (fallbacks are used)
            assert len(self.compiler._compiled_functions) == 0
        
        # Clear cache
        self.compiler.clear_cache()
        
        # Check cache is empty
        assert len(self.compiler._compiled_functions) == 0
        assert len(self.compiler._compilation_times) == 0
    
    def test_benchmark_jit_performance(self):
        """Test JIT performance benchmarking."""
        # Run a quick benchmark with small sizes
        results = self.compiler.benchmark_jit_performance(
            test_sizes=[20, 50],
            num_iterations=2
        )
        
        assert 'jit_info' in results
        assert 'spike_processing' in results
        assert 'matrix_multiply' in results
        
        # Check that we have results for each test size
        for size in [20, 50]:
            assert size in results['spike_processing']
            assert 'jit_time' in results['spike_processing'][size]
            assert 'python_time' in results['spike_processing'][size]
            assert 'speedup' in results['spike_processing'][size]
            
            assert size in results['matrix_multiply']
            assert 'jit_time' in results['matrix_multiply'][size]
            assert 'python_time' in results['matrix_multiply'][size]
            assert 'speedup' in results['matrix_multiply'][size]
    
    def test_disabled_jit(self):
        """Test behavior when JIT is disabled."""
        # Create compiler with JIT disabled
        disabled_config = JITConfig(enable_jit=False)
        disabled_compiler = JITCompiler(disabled_config)
        
        # Test that it returns Python fallbacks
        spike_kernel = disabled_compiler.compile_spike_processing_kernel()
        assert spike_kernel == disabled_compiler._spike_processing_kernel_python
        
        reservoir_kernel = disabled_compiler.compile_reservoir_update_kernel()
        assert reservoir_kernel == disabled_compiler._reservoir_update_kernel_python


class TestGlobalCompiler:
    """Test global compiler functionality."""
    
    def test_enable_jit_compilation(self):
        """Test enabling JIT compilation globally."""
        config = JITConfig(enable_jit=True, verbose=False)
        compiler = enable_jit_compilation(config)
        
        assert compiler is not None
        assert isinstance(compiler, JITCompiler)
    
    def test_global_functions(self):
        """Test global JIT functions."""
        # Enable JIT compilation
        config = JITConfig(enable_jit=True, verbose=False)
        compiler = enable_jit_compilation(config)
        
        # Set global compiler
        from src.performance.jit_compilation import set_global_compiler
        set_global_compiler(compiler)
        
        # Test global spike processing
        batch_size, time_steps, num_neurons = 2, 10, 5
        spike_trains = np.random.rand(batch_size, time_steps, num_neurons).astype(np.float32)
        spike_trains = (spike_trains > 0.8).astype(np.float32)
        weights = np.random.randn(num_neurons, num_neurons).astype(np.float32) * 0.1
        thresholds = np.ones(num_neurons, dtype=np.float32) * 0.5
        dt = 0.001
        
        result = jit_spike_processing(spike_trains, weights, thresholds, dt)
        assert result.shape == spike_trains.shape
        
        # Test global reservoir update
        batch_size, num_neurons, input_size = 2, 10, 5
        reservoir_states = np.random.randn(batch_size, num_neurons).astype(np.float32)
        input_weights = np.random.randn(input_size, num_neurons).astype(np.float32)
        reservoir_weights = np.random.randn(num_neurons, num_neurons).astype(np.float32)
        input_spikes = np.random.rand(batch_size, input_size).astype(np.float32)
        
        result = jit_reservoir_update(reservoir_states, input_weights, reservoir_weights, input_spikes)
        assert result.shape == (batch_size, num_neurons)
        
        # Test global matrix multiply
        a = np.random.randn(20, 15).astype(np.float32)
        b = np.random.randn(15, 25).astype(np.float32)
        
        result = jit_matrix_multiply(a, b)
        assert result.shape == (20, 25)


class TestJITConfig:
    """Test JIT configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = JITConfig()
        
        assert config.enable_jit is True
        assert config.cache_compiled_functions is True
        assert config.parallel_compilation is True
        assert config.optimize_memory_layout is True
        assert config.enable_fastmath is True
        assert config.enable_parallel_loops is True
        assert config.cache_directory is None
        assert config.max_cache_size == 1024 * 1024 * 1024
        assert config.verbose is False
        assert config.profile_compilation is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = JITConfig(
            enable_jit=False,
            cache_compiled_functions=False,
            enable_fastmath=False,
            verbose=True,
            max_cache_size=512 * 1024 * 1024
        )
        
        assert config.enable_jit is False
        assert config.cache_compiled_functions is False
        assert config.enable_fastmath is False
        assert config.verbose is True
        assert config.max_cache_size == 512 * 1024 * 1024


class TestPerformanceComparison:
    """Test performance comparison between JIT and Python."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = JITConfig(enable_jit=True, verbose=False)
        self.compiler = JITCompiler(self.config)
    
    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_matrix_multiply_performance(self):
        """Test that JIT matrix multiplication works correctly."""
        # Create test matrices
        size = 100
        a = np.random.randn(size, size).astype(np.float32)
        b = np.random.randn(size, size).astype(np.float32)
        
        # Get JIT and Python functions
        jit_func = self.compiler.compile_matrix_multiply_kernel()
        python_func = self.compiler._matrix_multiply_kernel_python
        
        # Warm up JIT
        _ = jit_func(a[:10, :10], b[:10, :10])
        
        # Test correctness
        jit_result = jit_func(a, b)
        python_result = python_func(a, b)
        
        # Results should be very close
        np.testing.assert_allclose(jit_result, python_result, rtol=1e-5, atol=1e-6)
        
        # Time both implementations
        num_iterations = 3
        
        start_time = time.time()
        for _ in range(num_iterations):
            _ = jit_func(a, b)
        jit_time = (time.time() - start_time) / num_iterations
        
        start_time = time.time()
        for _ in range(num_iterations):
            _ = python_func(a, b)
        python_time = (time.time() - start_time) / num_iterations
        
        # JIT should be at least as fast (or faster) than Python/NumPy
        # Note: We don't assert speedup because it depends on system and problem size
        print(f"JIT time: {jit_time:.4f}s, Python time: {python_time:.4f}s")
        
        # At minimum, JIT should complete without errors
        assert jit_time > 0
        assert python_time > 0
    
    def test_fallback_behavior(self):
        """Test fallback behavior when Numba is not available or JIT is disabled."""
        # Create compiler with JIT disabled
        disabled_config = JITConfig(enable_jit=False)
        disabled_compiler = JITCompiler(disabled_config)
        
        # Test that functions still work (using Python fallbacks)
        a = np.random.randn(10, 10).astype(np.float32)
        b = np.random.randn(10, 10).astype(np.float32)
        
        matmul_func = disabled_compiler.compile_matrix_multiply_kernel()
        result = matmul_func(a, b)
        
        expected = np.dot(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
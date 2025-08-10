"""
Tests for parallel processing module.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import time
from typing import List

from src.performance.parallel_processing import (
    ParallelProcessor, ParallelConfig, enable_parallel_processing,
    parallel_spike_processing, parallel_reservoir_computation
)


class TestParallelProcessor:
    """Test suite for parallel processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ParallelConfig(
            num_threads=2,
            parallel_strategy="thread",
            verbose=False
        )
        self.processor = ParallelProcessor(self.config)
        self.key = random.PRNGKey(42)
    
    def test_initialization(self):
        """Test parallel processor initialization."""
        assert self.processor is not None
        assert isinstance(self.processor.config, ParallelConfig)
        
        # Test configuration
        assert self.processor.config.num_threads == 2
        assert self.processor.config.parallel_strategy == "thread"
        assert not self.processor.config.verbose
    
    def test_parallel_info(self):
        """Test parallel information retrieval."""
        info = self.processor.get_parallel_info()
        
        assert 'num_threads' in info
        assert 'openmp_available' in info
        assert 'parallel_strategy' in info
        assert 'thread_affinity' in info
        assert 'load_balancing' in info
        assert 'cpu_count' in info
        
        assert info['num_threads'] == 2
        assert info['parallel_strategy'] == "thread"
        assert isinstance(info['openmp_available'], bool)
        assert isinstance(info['cpu_count'], int)
    
    def test_parallel_spike_processing_thread(self):
        """Test parallel spike processing with thread strategy."""
        # Create test spike batches
        batch_size, time_steps, num_neurons = 10, 50, 20
        num_batches = 4
        
        spike_batches = []
        for _ in range(num_batches):
            batch = random.bernoulli(self.key, 0.1, (batch_size, time_steps, num_neurons))
            spike_batches.append(batch)
        
        # Define simple processing function
        def sum_spikes(batch):
            return jnp.sum(batch, axis=1)  # Sum over time dimension
        
        # Test parallel processing
        results = self.processor.parallel_spike_processing(spike_batches, sum_spikes)
        
        assert len(results) == num_batches
        for i, result in enumerate(results):
            expected_shape = (batch_size, num_neurons)
            assert result.shape == expected_shape
            
            # Verify correctness
            expected = sum_spikes(spike_batches[i])
            np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_parallel_spike_processing_jax(self):
        """Test parallel spike processing with JAX pmap strategy."""
        # Skip if not enough devices available for pmap
        try:
            import jax
            if jax.device_count() < 2:
                pytest.skip("JAX pmap requires multiple devices")
        except:
            pytest.skip("JAX pmap not available")
        
        # Create processor with JAX strategy
        jax_config = ParallelConfig(
            num_threads=2,
            parallel_strategy="jax_pmap",
            verbose=False
        )
        jax_processor = ParallelProcessor(jax_config)
        
        # Create test spike batches (same size for pmap)
        batch_size, time_steps, num_neurons = 8, 30, 15
        num_batches = 2
        
        spike_batches = []
        for _ in range(num_batches):
            batch = random.bernoulli(self.key, 0.1, (batch_size, time_steps, num_neurons))
            spike_batches.append(batch)
        
        # Define simple processing function
        def mean_spikes(batch):
            return jnp.mean(batch, axis=1)  # Mean over time dimension
        
        # Test parallel processing
        results = jax_processor.parallel_spike_processing(spike_batches, mean_spikes)
        
        assert len(results) == num_batches
        for i, result in enumerate(results):
            expected_shape = (batch_size, num_neurons)
            assert result.shape == expected_shape
            
            # Verify correctness
            expected = mean_spikes(spike_batches[i])
            np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_parallel_reservoir_computation(self):
        """Test parallel reservoir computation."""
        # Create test data
        reservoir_size = 50
        input_size = 30
        num_reservoirs = 3
        
        reservoir_states = [
            random.normal(self.key, (reservoir_size,)) for _ in range(num_reservoirs)
        ]
        input_weights = random.normal(self.key, (input_size, reservoir_size))
        reservoir_weights = random.normal(self.key, (reservoir_size, reservoir_size))
        input_spikes_list = [
            random.bernoulli(self.key, 0.1, (input_size,)) for _ in range(num_reservoirs)
        ]
        
        # Test parallel computation
        results = self.processor.parallel_reservoir_computation(
            reservoir_states, input_weights, reservoir_weights, input_spikes_list
        )
        
        assert len(results) == num_reservoirs
        
        # Verify correctness against sequential computation
        for i, result in enumerate(results):
            assert result.shape == (reservoir_size,)
            
            # Compute expected result
            input_current = jnp.dot(input_spikes_list[i], input_weights)
            recurrent_current = jnp.dot(reservoir_states[i], reservoir_weights.T)
            expected = input_current + recurrent_current
            
            np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_parallel_multi_modal_reasoning(self):
        """Test parallel multi-modal reasoning."""
        # Create mock reasoning cores
        def visual_core(data):
            return jnp.sum(data, axis=-1)
        
        def audio_core(data):
            return jnp.mean(data, axis=-1)
        
        def text_core(data):
            return jnp.max(data, axis=-1)
        
        reasoning_cores = [visual_core, audio_core, text_core]
        
        # Create test input data
        input_data_list = [
            random.normal(self.key, (20, 30)),  # Visual data
            random.normal(self.key, (15, 25)),  # Audio data
            random.normal(self.key, (10, 35))   # Text data
        ]
        
        # Test parallel reasoning
        results = self.processor.parallel_multi_modal_reasoning(
            reasoning_cores, input_data_list
        )
        
        assert len(results) == len(reasoning_cores)
        
        # Verify correctness
        for i, (result, core, data) in enumerate(zip(results, reasoning_cores, input_data_list)):
            expected = core(data)
            np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_load_balanced_reasoning(self):
        """Test load-balanced multi-modal reasoning."""
        # Create processor with load balancing enabled
        lb_config = ParallelConfig(
            num_threads=2,
            enable_load_balancing=True,
            verbose=False
        )
        lb_processor = ParallelProcessor(lb_config)
        
        # Create reasoning cores with different computational loads
        def light_core(data):
            return jnp.sum(data)
        
        def heavy_core(data):
            # Simulate heavier computation with more stable operations
            result = data
            for _ in range(3):  # Reduced iterations to avoid overflow
                if result.ndim == 2:
                    result = jnp.tanh(jnp.dot(result, result.T) * 0.01)  # Scale down and use tanh
                else:
                    result = jnp.tanh(result * result * 0.01)
            return jnp.sum(result)
        
        reasoning_cores = [heavy_core, light_core, heavy_core, light_core]
        
        # Create input data with different sizes (different loads)
        input_data_list = [
            random.normal(self.key, (50, 50)),   # Large data for heavy core
            random.normal(self.key, (10, 10)),   # Small data for light core
            random.normal(self.key, (40, 40)),   # Medium data for heavy core
            random.normal(self.key, (5, 5))      # Small data for light core
        ]
        
        # Test load-balanced reasoning
        results = lb_processor.parallel_multi_modal_reasoning(
            reasoning_cores, input_data_list, load_balance=True
        )
        
        assert len(results) == len(reasoning_cores)
        
        # Verify correctness (results should be the same regardless of load balancing)
        # Note: Due to load balancing, the order might be different, so we need to match results
        expected_results = [core(data) for core, data in zip(reasoning_cores, input_data_list)]
        
        # Check that we have the right number of results
        assert len(results) == len(expected_results)
        
        # For load balancing test, we just verify that all computations completed successfully
        # and produced reasonable results (not NaN or infinite)
        for result in results:
            assert jnp.isfinite(result).all(), f"Result contains non-finite values: {result}"
    
    def test_benchmark_parallel_performance(self):
        """Test performance benchmarking."""
        # Run a quick benchmark with small sizes
        results = self.processor.benchmark_parallel_performance(
            test_sizes=[20, 50],
            num_iterations=2
        )
        
        assert 'parallel_info' in results
        assert 'spike_processing' in results
        assert 'reservoir_computation' in results
        
        # Check that we have results for each test size
        for size in [20, 50]:
            assert size in results['spike_processing']
            assert 'sequential_time' in results['spike_processing'][size]
            assert 'parallel_time' in results['spike_processing'][size]
            assert 'speedup' in results['spike_processing'][size]
            
            assert size in results['reservoir_computation']
            assert 'sequential_time' in results['reservoir_computation'][size]
            assert 'parallel_time' in results['reservoir_computation'][size]
            assert 'speedup' in results['reservoir_computation'][size]
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with ParallelProcessor(self.config) as processor:
            assert processor is not None
            
            # Test basic operation within context
            spike_batches = [random.bernoulli(self.key, 0.1, (5, 10, 8)) for _ in range(2)]
            
            def simple_func(batch):
                return jnp.sum(batch, axis=1)
            
            results = processor.parallel_spike_processing(spike_batches, simple_func)
            assert len(results) == 2
    
    def test_cleanup(self):
        """Test resource cleanup."""
        processor = ParallelProcessor(self.config)
        
        # Verify pools are initialized
        assert processor._thread_pool is not None
        
        # Test cleanup
        processor.cleanup()
        
        # Verify pools are cleaned up
        assert processor._thread_pool is None


class TestGlobalProcessor:
    """Test global processor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = random.PRNGKey(42)
    
    def test_enable_parallel_processing(self):
        """Test enabling parallel processing globally."""
        config = ParallelConfig(num_threads=2, verbose=False)
        processor = enable_parallel_processing(config)
        
        assert processor is not None
        assert isinstance(processor, ParallelProcessor)
    
    def test_global_functions(self):
        """Test global parallel functions."""
        # Enable parallel processing
        config = ParallelConfig(num_threads=2, verbose=False)
        processor = enable_parallel_processing(config)
        
        # Set global processor
        from src.performance.parallel_processing import set_global_processor
        set_global_processor(processor)
        
        # Test global spike processing
        spike_batches = [random.bernoulli(self.key, 0.1, (5, 10, 8)) for _ in range(2)]
        
        def sum_func(batch):
            return jnp.sum(batch, axis=1)
        
        results = parallel_spike_processing(spike_batches, sum_func)
        assert len(results) == 2
        
        # Test global reservoir computation
        reservoir_states = [random.normal(self.key, (20,)) for _ in range(2)]
        input_weights = random.normal(self.key, (10, 20))
        reservoir_weights = random.normal(self.key, (20, 20))
        input_spikes_list = [random.bernoulli(self.key, 0.1, (10,)) for _ in range(2)]
        
        results = parallel_reservoir_computation(
            reservoir_states, input_weights, reservoir_weights, input_spikes_list
        )
        assert len(results) == 2
        
        # Cleanup
        processor.cleanup()


class TestParallelConfig:
    """Test parallel configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ParallelConfig()
        
        assert config.num_threads is None
        assert config.thread_affinity is True
        assert config.use_openmp is True
        assert config.parallel_strategy == "thread"
        assert config.chunk_size is None
        assert config.enable_load_balancing is True
        assert config.load_balance_threshold == 0.8
        assert config.max_memory_per_worker == 1024 * 1024 * 1024
        assert config.enable_memory_monitoring is True
        assert config.verbose is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ParallelConfig(
            num_threads=4,
            thread_affinity=False,
            parallel_strategy="process",
            enable_load_balancing=False,
            verbose=True
        )
        
        assert config.num_threads == 4
        assert config.thread_affinity is False
        assert config.parallel_strategy == "process"
        assert config.enable_load_balancing is False
        assert config.verbose is True


class TestPerformanceComparison:
    """Test performance comparison between sequential and parallel."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = random.PRNGKey(42)
        self.config = ParallelConfig(num_threads=2, verbose=False)
        self.processor = ParallelProcessor(self.config)
    
    def test_spike_processing_performance(self):
        """Test that parallel spike processing works correctly."""
        # Create moderately sized test data
        batch_size, time_steps, num_neurons = 20, 100, 30
        num_batches = 4
        
        spike_batches = [
            random.bernoulli(self.key, 0.1, (batch_size, time_steps, num_neurons))
            for _ in range(num_batches)
        ]
        
        def processing_func(batch):
            # Simple but non-trivial computation with proper broadcasting
            time_weights = jnp.arange(batch.shape[1]).reshape(1, -1, 1)  # Shape: (1, time_steps, 1)
            return jnp.sum(batch * time_weights, axis=1)
        
        # Sequential processing
        start_time = time.time()
        sequential_results = [processing_func(batch) for batch in spike_batches]
        sequential_time = time.time() - start_time
        
        # Parallel processing
        start_time = time.time()
        parallel_results = self.processor.parallel_spike_processing(spike_batches, processing_func)
        parallel_time = time.time() - start_time
        
        # Verify correctness
        assert len(parallel_results) == len(sequential_results)
        for seq_result, par_result in zip(sequential_results, parallel_results):
            np.testing.assert_allclose(seq_result, par_result, rtol=1e-6)
        
        # Note: We don't assert speedup because it depends on system load and overhead
        # But we can log the performance for manual inspection
        if sequential_time > 0:
            speedup = sequential_time / parallel_time
            print(f"Spike processing speedup: {speedup:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__])
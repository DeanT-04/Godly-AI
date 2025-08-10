"""
OpenMP-based parallel processing for neuromorphic computations.

This module provides multi-threaded spike processing, parallel reservoir computation,
and load balancing for multi-modal reasoning cores.
"""

import os
import warnings
from typing import Optional, Dict, Any, List, Callable, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, pmap, vmap
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import threading

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    
    # Threading configuration
    num_threads: Optional[int] = None  # Auto-detect if None
    thread_affinity: bool = True
    use_openmp: bool = True
    
    # Processing strategy
    parallel_strategy: str = "thread"  # "thread", "process", "jax_pmap"
    chunk_size: Optional[int] = None  # Auto-calculate if None
    
    # Load balancing
    enable_load_balancing: bool = True
    load_balance_threshold: float = 0.8  # Rebalance if load > 80%
    
    # Memory management
    max_memory_per_worker: int = 1024 * 1024 * 1024  # 1GB per worker
    enable_memory_monitoring: bool = True
    
    # Debugging
    verbose: bool = False


class ParallelProcessor:
    """
    OpenMP-based parallel processor for neuromorphic computations.
    
    Provides multi-threaded processing capabilities for spike trains,
    reservoir computations, and multi-modal reasoning cores.
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize parallel processor with configuration."""
        self.config = config or ParallelConfig()
        self._openmp_available = False
        self._thread_pool = None
        self._process_pool = None
        
        # Initialize parallel processing
        self._initialize_parallel_processing()
    
    def _initialize_parallel_processing(self) -> None:
        """Initialize parallel processing capabilities."""
        # Determine number of threads
        if self.config.num_threads is None:
            self.config.num_threads = min(multiprocessing.cpu_count(), 8)
        
        # Try to configure OpenMP
        self._configure_openmp()
        
        # Initialize thread/process pools
        self._initialize_pools()
        
        if self.config.verbose:
            logger.info(f"Parallel processing initialized with {self.config.num_threads} threads")
            logger.info(f"OpenMP available: {self._openmp_available}")
            logger.info(f"Strategy: {self.config.parallel_strategy}")
    
    def _configure_openmp(self) -> None:
        """Configure OpenMP if available."""
        try:
            # Set OpenMP environment variables
            os.environ['OMP_NUM_THREADS'] = str(self.config.num_threads)
            
            if self.config.thread_affinity:
                os.environ['OMP_PROC_BIND'] = 'true'
                os.environ['OMP_PLACES'] = 'cores'
            
            # Try to import OpenMP-enabled libraries
            try:
                import numba
                if hasattr(numba, 'set_num_threads'):
                    numba.set_num_threads(self.config.num_threads)
                    self._openmp_available = True
            except ImportError:
                pass
            
            # Check if JAX can use multiple threads
            if hasattr(jax.config, 'update'):
                jax.config.update('jax_platform_name', 'cpu')
                
        except Exception as e:
            logger.warning(f"Failed to configure OpenMP: {e}")
    
    def _initialize_pools(self) -> None:
        """Initialize thread and process pools."""
        if self.config.parallel_strategy in ["thread", "hybrid"]:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.config.num_threads,
                thread_name_prefix="neuromorphic_worker"
            )
        
        if self.config.parallel_strategy in ["process", "hybrid"]:
            self._process_pool = ProcessPoolExecutor(
                max_workers=min(self.config.num_threads, multiprocessing.cpu_count())
            )
    
    def is_openmp_available(self) -> bool:
        """Check if OpenMP is available."""
        return self._openmp_available
    
    def get_parallel_info(self) -> Dict[str, Any]:
        """Get information about parallel configuration."""
        return {
            'num_threads': self.config.num_threads,
            'openmp_available': self._openmp_available,
            'parallel_strategy': self.config.parallel_strategy,
            'thread_affinity': self.config.thread_affinity,
            'load_balancing': self.config.enable_load_balancing,
            'cpu_count': multiprocessing.cpu_count()
        }
    
    def parallel_spike_processing(
        self,
        spike_batches: List[jnp.ndarray],
        processing_func: Callable[[jnp.ndarray], jnp.ndarray],
        **kwargs
    ) -> List[jnp.ndarray]:
        """
        Process spike batches in parallel.
        
        Args:
            spike_batches: List of spike train batches
            processing_func: Function to apply to each batch
            **kwargs: Additional arguments for processing function
            
        Returns:
            List of processed spike batches
        """
        if self.config.parallel_strategy == "jax_pmap":
            return self._parallel_spike_processing_jax(spike_batches, processing_func, **kwargs)
        elif self.config.parallel_strategy == "thread":
            return self._parallel_spike_processing_thread(spike_batches, processing_func, **kwargs)
        elif self.config.parallel_strategy == "process":
            return self._parallel_spike_processing_process(spike_batches, processing_func, **kwargs)
        else:
            # Fallback to sequential processing
            return [processing_func(batch, **kwargs) for batch in spike_batches]
    
    def _parallel_spike_processing_jax(
        self,
        spike_batches: List[jnp.ndarray],
        processing_func: Callable,
        **kwargs
    ) -> List[jnp.ndarray]:
        """Process spike batches using JAX pmap."""
        # Stack batches for parallel processing
        if len(spike_batches) == 0:
            return []
        
        # Pad batches to same size if needed
        max_batch_size = max(batch.shape[0] for batch in spike_batches)
        padded_batches = []
        
        for batch in spike_batches:
            if batch.shape[0] < max_batch_size:
                padding = jnp.zeros((max_batch_size - batch.shape[0],) + batch.shape[1:])
                padded_batch = jnp.concatenate([batch, padding], axis=0)
            else:
                padded_batch = batch
            padded_batches.append(padded_batch)
        
        # Stack for parallel processing
        stacked_batches = jnp.stack(padded_batches, axis=0)
        
        # Create parallel processing function
        parallel_func = pmap(processing_func, axis_name='batch')
        
        # Process in parallel
        results = parallel_func(stacked_batches)
        
        # Unstack results and remove padding
        result_list = []
        for i, original_batch in enumerate(spike_batches):
            result = results[i]
            # Remove padding if it was added
            if original_batch.shape[0] < max_batch_size:
                result = result[:original_batch.shape[0]]
            result_list.append(result)
        
        return result_list
    
    def _parallel_spike_processing_thread(
        self,
        spike_batches: List[jnp.ndarray],
        processing_func: Callable,
        **kwargs
    ) -> List[jnp.ndarray]:
        """Process spike batches using thread pool."""
        if self._thread_pool is None:
            return [processing_func(batch, **kwargs) for batch in spike_batches]
        
        # Submit tasks to thread pool
        futures = []
        for batch in spike_batches:
            future = self._thread_pool.submit(processing_func, batch, **kwargs)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel spike processing: {e}")
                # Fallback to sequential processing for this batch
                batch_idx = futures.index(future)
                result = processing_func(spike_batches[batch_idx], **kwargs)
                results.append(result)
        
        return results
    
    def _parallel_spike_processing_process(
        self,
        spike_batches: List[jnp.ndarray],
        processing_func: Callable,
        **kwargs
    ) -> List[jnp.ndarray]:
        """Process spike batches using process pool."""
        if self._process_pool is None:
            return [processing_func(batch, **kwargs) for batch in spike_batches]
        
        # Submit tasks to process pool
        futures = []
        for batch in spike_batches:
            future = self._process_pool.submit(processing_func, batch, **kwargs)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)  # 60 second timeout for processes
                results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel spike processing: {e}")
                # Fallback to sequential processing for this batch
                batch_idx = futures.index(future)
                result = processing_func(spike_batches[batch_idx], **kwargs)
                results.append(result)
        
        return results
    
    def parallel_reservoir_computation(
        self,
        reservoir_states: List[jnp.ndarray],
        input_weights: jnp.ndarray,
        reservoir_weights: jnp.ndarray,
        input_spikes_list: List[jnp.ndarray]
    ) -> List[jnp.ndarray]:
        """
        Compute reservoir updates in parallel across multiple cores.
        
        Args:
            reservoir_states: List of reservoir states
            input_weights: Input connection weights
            reservoir_weights: Reservoir connection weights
            input_spikes_list: List of input spike patterns
            
        Returns:
            List of updated reservoir currents
        """
        def reservoir_update_func(args):
            reservoir_state, input_spikes = args
            # Compute input contribution
            input_current = jnp.dot(input_spikes, input_weights)
            # Compute recurrent contribution
            recurrent_current = jnp.dot(reservoir_state, reservoir_weights.T)
            return input_current + recurrent_current
        
        # Prepare arguments for parallel processing
        args_list = list(zip(reservoir_states, input_spikes_list))
        
        if self.config.parallel_strategy == "jax_pmap" and len(args_list) > 1:
            # Use JAX pmap for parallel computation
            def parallel_reservoir_func(states_and_spikes):
                states, spikes = states_and_spikes
                input_current = jnp.dot(spikes, input_weights)
                recurrent_current = jnp.dot(states, reservoir_weights.T)
                return input_current + recurrent_current
            
            # Stack inputs for parallel processing
            states_stack = jnp.stack(reservoir_states, axis=0)
            spikes_stack = jnp.stack(input_spikes_list, axis=0)
            
            # Apply parallel function
            parallel_func = pmap(parallel_reservoir_func, axis_name='batch')
            results = parallel_func((states_stack, spikes_stack))
            
            return [results[i] for i in range(len(reservoir_states))]
        
        elif self._thread_pool is not None:
            # Use thread pool
            futures = [self._thread_pool.submit(reservoir_update_func, args) for args in args_list]
            results = [future.result() for future in futures]
            return results
        
        else:
            # Sequential fallback
            return [reservoir_update_func(args) for args in args_list]
    
    def parallel_multi_modal_reasoning(
        self,
        reasoning_cores: List[Callable],
        input_data_list: List[jnp.ndarray],
        load_balance: bool = True
    ) -> List[jnp.ndarray]:
        """
        Execute multi-modal reasoning cores in parallel with load balancing.
        
        Args:
            reasoning_cores: List of reasoning core functions
            input_data_list: List of input data for each core
            load_balance: Whether to enable load balancing
            
        Returns:
            List of reasoning outputs
        """
        if len(reasoning_cores) != len(input_data_list):
            raise ValueError("Number of reasoning cores must match input data list")
        
        if load_balance and self.config.enable_load_balancing:
            return self._load_balanced_reasoning(reasoning_cores, input_data_list)
        else:
            return self._simple_parallel_reasoning(reasoning_cores, input_data_list)
    
    def _simple_parallel_reasoning(
        self,
        reasoning_cores: List[Callable],
        input_data_list: List[jnp.ndarray]
    ) -> List[jnp.ndarray]:
        """Simple parallel execution of reasoning cores."""
        def reasoning_task(args):
            core_func, input_data = args
            return core_func(input_data)
        
        args_list = list(zip(reasoning_cores, input_data_list))
        
        if self._thread_pool is not None:
            futures = [self._thread_pool.submit(reasoning_task, args) for args in args_list]
            results = [future.result() for future in futures]
            return results
        else:
            return [reasoning_task(args) for args in args_list]
    
    def _load_balanced_reasoning(
        self,
        reasoning_cores: List[Callable],
        input_data_list: List[jnp.ndarray]
    ) -> List[jnp.ndarray]:
        """Load-balanced execution of reasoning cores."""
        # Estimate computational load for each core based on input size
        loads = [np.prod(data.shape) for data in input_data_list]
        total_load = sum(loads)
        
        # Sort cores by load (heaviest first)
        core_data_load = list(zip(reasoning_cores, input_data_list, loads))
        core_data_load.sort(key=lambda x: x[2], reverse=True)
        
        # Distribute cores across available threads
        num_workers = self.config.num_threads
        worker_loads = [0.0] * num_workers
        worker_tasks = [[] for _ in range(num_workers)]
        
        # Assign tasks to workers using greedy algorithm
        for core, data, load in core_data_load:
            # Find worker with minimum current load
            min_worker = min(range(num_workers), key=lambda i: worker_loads[i])
            worker_tasks[min_worker].append((core, data))
            worker_loads[min_worker] += load
        
        # Execute tasks in parallel
        def worker_function(tasks):
            results = []
            for core, data in tasks:
                result = core(data)
                results.append(result)
            return results
        
        if self._thread_pool is not None:
            futures = [
                self._thread_pool.submit(worker_function, tasks)
                for tasks in worker_tasks if tasks
            ]
            
            # Collect results and maintain original order
            worker_results = [future.result() for future in futures]
            
            # Reconstruct original order
            results = [None] * len(reasoning_cores)
            result_idx = 0
            
            for worker_result in worker_results:
                for result in worker_result:
                    # Find original position
                    while results[result_idx] is not None:
                        result_idx += 1
                    results[result_idx] = result
            
            return results
        else:
            # Sequential fallback
            results = []
            for core, data in zip(reasoning_cores, input_data_list):
                results.append(core(data))
            return results
    
    def benchmark_parallel_performance(
        self,
        test_sizes: List[int] = [100, 500, 1000],
        num_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        Benchmark parallel processing performance.
        
        Args:
            test_sizes: List of test data sizes
            num_iterations: Number of iterations per test
            
        Returns:
            Performance benchmark results
        """
        import time
        
        results = {
            'parallel_info': self.get_parallel_info(),
            'spike_processing': {},
            'reservoir_computation': {},
            'multi_modal_reasoning': {}
        }
        
        key = random.PRNGKey(42)
        
        for size in test_sizes:
            # Spike processing benchmark
            spike_batches = [
                random.bernoulli(key, 0.1, (size // 4, size, 50))
                for _ in range(4)
            ]
            
            def simple_spike_func(batch):
                return jnp.sum(batch, axis=1)
            
            # Sequential baseline
            start_time = time.time()
            for _ in range(num_iterations):
                sequential_results = [simple_spike_func(batch) for batch in spike_batches]
            sequential_time = (time.time() - start_time) / num_iterations
            
            # Parallel processing
            start_time = time.time()
            for _ in range(num_iterations):
                parallel_results = self.parallel_spike_processing(spike_batches, simple_spike_func)
            parallel_time = (time.time() - start_time) / num_iterations
            
            speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
            
            results['spike_processing'][size] = {
                'sequential_time': sequential_time,
                'parallel_time': parallel_time,
                'speedup': speedup
            }
            
            # Reservoir computation benchmark
            reservoir_states = [random.normal(key, (size,)) for _ in range(4)]
            input_weights = random.normal(key, (size // 2, size))
            reservoir_weights = random.normal(key, (size, size))
            input_spikes_list = [random.bernoulli(key, 0.1, (size // 2,)) for _ in range(4)]
            
            # Sequential baseline
            start_time = time.time()
            for _ in range(num_iterations):
                sequential_reservoir = []
                for state, spikes in zip(reservoir_states, input_spikes_list):
                    input_current = jnp.dot(spikes, input_weights)
                    recurrent_current = jnp.dot(state, reservoir_weights.T)
                    sequential_reservoir.append(input_current + recurrent_current)
            sequential_reservoir_time = (time.time() - start_time) / num_iterations
            
            # Parallel processing
            start_time = time.time()
            for _ in range(num_iterations):
                parallel_reservoir = self.parallel_reservoir_computation(
                    reservoir_states, input_weights, reservoir_weights, input_spikes_list
                )
            parallel_reservoir_time = (time.time() - start_time) / num_iterations
            
            reservoir_speedup = sequential_reservoir_time / parallel_reservoir_time if parallel_reservoir_time > 0 else 1.0
            
            results['reservoir_computation'][size] = {
                'sequential_time': sequential_reservoir_time,
                'parallel_time': parallel_reservoir_time,
                'speedup': reservoir_speedup
            }
        
        return results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None


def enable_parallel_processing(config: Optional[ParallelConfig] = None) -> ParallelProcessor:
    """
    Enable parallel processing globally.
    
    Args:
        config: Parallel processing configuration
        
    Returns:
        Parallel processor instance
    """
    processor = ParallelProcessor(config)
    
    # Set global JAX configuration for optimal parallel performance
    if hasattr(jax.config, 'update'):
        jax.config.update('jax_platform_name', 'cpu')
        jax.config.update('jax_enable_x64', True)
    
    return processor


# Global processor instance
_global_processor: Optional[ParallelProcessor] = None


def get_global_processor() -> Optional[ParallelProcessor]:
    """Get the global parallel processor instance."""
    return _global_processor


def set_global_processor(processor: ParallelProcessor) -> None:
    """Set the global parallel processor instance."""
    global _global_processor
    _global_processor = processor


# Convenience functions for parallel operations
def parallel_spike_processing(
    spike_batches: List[jnp.ndarray],
    processing_func: Callable,
    **kwargs
) -> List[jnp.ndarray]:
    """Parallel spike processing using global processor."""
    processor = get_global_processor()
    if processor is not None:
        return processor.parallel_spike_processing(spike_batches, processing_func, **kwargs)
    else:
        return [processing_func(batch, **kwargs) for batch in spike_batches]


def parallel_reservoir_computation(
    reservoir_states: List[jnp.ndarray],
    input_weights: jnp.ndarray,
    reservoir_weights: jnp.ndarray,
    input_spikes_list: List[jnp.ndarray]
) -> List[jnp.ndarray]:
    """Parallel reservoir computation using global processor."""
    processor = get_global_processor()
    if processor is not None:
        return processor.parallel_reservoir_computation(
            reservoir_states, input_weights, reservoir_weights, input_spikes_list
        )
    else:
        # Sequential fallback
        results = []
        for state, spikes in zip(reservoir_states, input_spikes_list):
            input_current = jnp.dot(spikes, input_weights)
            recurrent_current = jnp.dot(state, reservoir_weights.T)
            results.append(input_current + recurrent_current)
        return results
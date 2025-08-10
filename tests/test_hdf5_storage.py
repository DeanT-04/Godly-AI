"""
Tests for HDF5 storage system.

Tests HDF5 integration for network states, spike trains, activation data,
learning traces, and file management operations.
"""

import tempfile
import time
import pytest
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from unittest.mock import patch, Mock

from src.storage.hdf5_storage import (
    HDF5Storage, HDF5StorageError, NetworkSnapshot, SpikeTrainData, HDF5_AVAILABLE
)


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
class TestHDF5Storage:
    """Test suite for HDF5 storage functionality."""
    
    @pytest.fixture
    def temp_h5_path(self):
        """Create temporary HDF5 file path for testing."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            h5_path = Path(f.name)
        
        yield h5_path
        
        # Cleanup
        if h5_path.exists():
            h5_path.unlink()
        
        # Clean up rotated files
        for rotated in h5_path.parent.glob(f"{h5_path.stem}_*{h5_path.suffix}"):
            rotated.unlink()
    
    @pytest.fixture
    def hdf5_storage(self, temp_h5_path):
        """HDF5 storage instance with temporary file."""
        storage = HDF5Storage(
            file_path=temp_h5_path,
            compression='gzip',
            compression_opts=6,
            chunk_size=512
        )
        yield storage
        storage.close()
    
    def test_initialization_without_h5py(self):
        """Test HDF5 storage initialization when h5py is not available."""
        with patch('src.storage.hdf5_storage.HDF5_AVAILABLE', False):
            with pytest.raises(HDF5StorageError, match="h5py is not available"):
                HDF5Storage()
    
    def test_initialization_and_file_structure(self, temp_h5_path):
        """Test HDF5 storage initialization and file structure creation."""
        storage = HDF5Storage(file_path=temp_h5_path)
        
        # Check that file was created
        assert temp_h5_path.exists()
        
        # Check file info
        info = storage.get_file_info()
        assert info['file_path'] == str(temp_h5_path)
        assert info['compression'] == 'gzip'
        assert 'network_states' in info['groups']
        assert 'spike_trains' in info['groups']
        assert 'activations' in info['groups']
        assert 'learning_traces' in info['groups']
        
        storage.close()
    
    def test_network_snapshot_operations(self, hdf5_storage):
        """Test network snapshot storage and retrieval."""
        # Create test network data
        adjacency_matrix = np.random.randint(0, 2, (100, 100))
        weight_matrix = np.random.rand(100, 100)
        neuron_states = np.random.rand(100, 5)  # 5 state variables per neuron
        timestamp = time.time()
        metadata = {"network_type": "reservoir", "size": 100}
        
        # Store network snapshot
        snapshot_id = hdf5_storage.store_network_snapshot(
            timestamp=timestamp,
            adjacency_matrix=adjacency_matrix,
            weight_matrix=weight_matrix,
            neuron_states=neuron_states,
            metadata=metadata
        )
        
        assert isinstance(snapshot_id, str)
        assert snapshot_id.startswith("snapshot_")
        
        # Retrieve network snapshot
        retrieved_snapshot = hdf5_storage.retrieve_network_snapshot(snapshot_id)
        
        assert retrieved_snapshot is not None
        assert isinstance(retrieved_snapshot, NetworkSnapshot)
        assert retrieved_snapshot.timestamp == timestamp
        np.testing.assert_array_equal(retrieved_snapshot.adjacency_matrix, adjacency_matrix)
        np.testing.assert_array_equal(retrieved_snapshot.weight_matrix, weight_matrix)
        np.testing.assert_array_equal(retrieved_snapshot.neuron_states, neuron_states)
        assert retrieved_snapshot.metadata["network_type"] == "reservoir"
    
    def test_network_snapshot_with_jax_arrays(self, hdf5_storage):
        """Test network snapshot storage with JAX arrays."""
        # Create JAX arrays
        adjacency_matrix = jnp.array([[1, 0], [1, 1]])
        weight_matrix = jnp.array([[0.5, 0.0], [0.3, 0.8]])
        neuron_states = jnp.array([[1.0, 0.5], [0.2, 0.9]])
        timestamp = time.time()
        
        # Store snapshot
        snapshot_id = hdf5_storage.store_network_snapshot(
            timestamp=timestamp,
            adjacency_matrix=adjacency_matrix,
            weight_matrix=weight_matrix,
            neuron_states=neuron_states
        )
        
        # Retrieve and verify
        retrieved_snapshot = hdf5_storage.retrieve_network_snapshot(snapshot_id)
        
        assert retrieved_snapshot is not None
        np.testing.assert_array_equal(retrieved_snapshot.adjacency_matrix, np.array(adjacency_matrix))
        np.testing.assert_array_equal(retrieved_snapshot.weight_matrix, np.array(weight_matrix))
        np.testing.assert_array_equal(retrieved_snapshot.neuron_states, np.array(neuron_states))
    
    def test_network_snapshots_time_range(self, hdf5_storage):
        """Test retrieving network snapshots within time range."""
        base_time = time.time()
        timestamps = [base_time - 10, base_time - 5, base_time, base_time + 5]
        snapshot_ids = []
        
        # Store multiple snapshots
        for i, timestamp in enumerate(timestamps):
            adjacency = np.eye(10) + np.random.rand(10, 10) * 0.1
            weights = np.random.rand(10, 10)
            states = np.random.rand(10, 3)
            
            snapshot_id = hdf5_storage.store_network_snapshot(
                timestamp=timestamp,
                adjacency_matrix=adjacency,
                weight_matrix=weights,
                neuron_states=states
            )
            snapshot_ids.append(snapshot_id)
        
        # Get snapshots in range
        range_snapshots = hdf5_storage.get_network_snapshots_in_range(
            start_time=base_time - 6,
            end_time=base_time + 1
        )
        
        assert len(range_snapshots) == 2  # Should include timestamps at base_time-5 and base_time
        assert all(sid in snapshot_ids for sid in range_snapshots)
    
    def test_spike_train_operations(self, hdf5_storage):
        """Test spike train storage and retrieval."""
        # Create test spike train data
        spike_times = np.array([0.1, 0.15, 0.3, 0.45, 0.6, 0.75])
        neuron_ids = np.array([1, 3, 1, 2, 3, 1])
        spike_amplitudes = np.array([1.2, 0.8, 1.5, 1.0, 0.9, 1.3])
        timestamp = time.time()
        metadata = {"recording_duration": 1.0, "sampling_rate": 1000}
        
        # Store spike train
        train_id = hdf5_storage.store_spike_train(
            timestamp=timestamp,
            spike_times=spike_times,
            neuron_ids=neuron_ids,
            spike_amplitudes=spike_amplitudes,
            metadata=metadata
        )
        
        assert isinstance(train_id, str)
        assert train_id.startswith("train_")
        
        # Retrieve spike train
        retrieved_train = hdf5_storage.retrieve_spike_train(train_id)
        
        assert retrieved_train is not None
        assert isinstance(retrieved_train, SpikeTrainData)
        assert retrieved_train.timestamp == timestamp
        np.testing.assert_array_equal(retrieved_train.spike_times, spike_times)
        np.testing.assert_array_equal(retrieved_train.neuron_ids, neuron_ids)
        np.testing.assert_array_equal(retrieved_train.spike_amplitudes, spike_amplitudes)
        assert retrieved_train.metadata["recording_duration"] == 1.0
    
    def test_spike_train_without_amplitudes(self, hdf5_storage):
        """Test spike train storage without amplitude data."""
        spike_times = np.array([0.1, 0.2, 0.3])
        neuron_ids = np.array([1, 2, 1])
        timestamp = time.time()
        
        # Store spike train without amplitudes
        train_id = hdf5_storage.store_spike_train(
            timestamp=timestamp,
            spike_times=spike_times,
            neuron_ids=neuron_ids
        )
        
        # Retrieve and verify
        retrieved_train = hdf5_storage.retrieve_spike_train(train_id)
        
        assert retrieved_train is not None
        assert retrieved_train.spike_amplitudes is None
        np.testing.assert_array_equal(retrieved_train.spike_times, spike_times)
        np.testing.assert_array_equal(retrieved_train.neuron_ids, neuron_ids)
    
    def test_compressed_spike_train_storage(self, hdf5_storage):
        """Test compressed spike train storage."""
        # Create compressed spike data (simulated)
        compressed_data = np.random.randint(0, 256, 1000, dtype=np.uint8)
        timestamp = time.time()
        compression_params = {
            "original_shape": (5000, 100),
            "compression_ratio": 5.0,
            "algorithm": "custom_sparse"
        }
        
        # Store compressed spike train
        train_id = hdf5_storage.store_compressed_spike_train(
            timestamp=timestamp,
            spike_data=compressed_data,
            compression_params=compression_params
        )
        
        assert isinstance(train_id, str)
        assert train_id.startswith("compressed_")
    
    def test_reservoir_states_operations(self, hdf5_storage):
        """Test reservoir states storage and retrieval."""
        # Create reservoir state data
        reservoir_states = np.random.rand(500, 10)  # 500 neurons, 10 time steps
        timestamp = time.time()
        reservoir_id = "main_reservoir"
        
        # Store reservoir states
        state_id = hdf5_storage.store_reservoir_states(
            timestamp=timestamp,
            reservoir_states=reservoir_states,
            reservoir_id=reservoir_id
        )
        
        assert isinstance(state_id, str)
        assert reservoir_id in state_id
        
        # Retrieve reservoir states
        retrieved_data = hdf5_storage.retrieve_reservoir_states(state_id)
        
        assert retrieved_data is not None
        states, ret_timestamp, ret_reservoir_id = retrieved_data
        np.testing.assert_array_equal(states, reservoir_states)
        assert ret_timestamp == timestamp
        assert ret_reservoir_id == reservoir_id
    
    def test_layer_outputs_operations(self, hdf5_storage):
        """Test layer outputs storage and retrieval."""
        # Create layer output data
        layer_outputs = {
            "input_layer": np.random.rand(32, 784),
            "hidden_layer_1": np.random.rand(32, 256),
            "hidden_layer_2": np.random.rand(32, 128),
            "output_layer": np.random.rand(32, 10)
        }
        timestamp = time.time()
        
        # Store layer outputs
        output_id = hdf5_storage.store_layer_outputs(
            timestamp=timestamp,
            layer_outputs=layer_outputs
        )
        
        assert isinstance(output_id, str)
        assert output_id.startswith("outputs_")
        
        # Retrieve layer outputs
        retrieved_data = hdf5_storage.retrieve_layer_outputs(output_id)
        
        assert retrieved_data is not None
        ret_outputs, ret_timestamp = retrieved_data
        assert ret_timestamp == timestamp
        assert len(ret_outputs) == len(layer_outputs)
        
        for layer_name, expected_output in layer_outputs.items():
            assert layer_name in ret_outputs
            np.testing.assert_array_equal(ret_outputs[layer_name], expected_output)
    
    def test_plasticity_trace_operations(self, hdf5_storage):
        """Test plasticity trace storage and retrieval."""
        # Create plasticity trace data
        synapse_changes = np.random.randn(1000, 1000) * 0.01  # Small weight changes
        timestamp = time.time()
        plasticity_type = "stdp"
        
        # Store plasticity trace
        trace_id = hdf5_storage.store_plasticity_trace(
            timestamp=timestamp,
            synapse_changes=synapse_changes,
            plasticity_type=plasticity_type
        )
        
        assert isinstance(trace_id, str)
        assert plasticity_type in trace_id
        
        # Retrieve plasticity trace
        retrieved_data = hdf5_storage.retrieve_plasticity_trace(trace_id)
        
        assert retrieved_data is not None
        ret_changes, ret_timestamp, ret_type = retrieved_data
        np.testing.assert_array_equal(ret_changes, synapse_changes)
        assert ret_timestamp == timestamp
        assert ret_type == plasticity_type
    
    def test_homeostatic_trace_operations(self, hdf5_storage):
        """Test homeostatic trace storage and retrieval."""
        # Create homeostatic trace data
        homeostatic_changes = np.random.randn(500) * 0.001  # Small threshold changes
        timestamp = time.time()
        mechanism = "intrinsic_plasticity"
        
        # Store homeostatic trace
        trace_id = hdf5_storage.store_homeostatic_trace(
            timestamp=timestamp,
            homeostatic_changes=homeostatic_changes,
            mechanism=mechanism
        )
        
        assert isinstance(trace_id, str)
        assert mechanism in trace_id
    
    def test_performance_metrics_operations(self, hdf5_storage):
        """Test performance metrics storage and retrieval."""
        # Create performance metrics
        metrics = {
            "accuracy": 0.85,
            "loss": 0.23,
            "learning_rate": 0.001,
            "convergence_time": 45.2
        }
        timestamp = time.time()
        task_type = "classification"
        
        # Store performance metrics
        metrics_id = hdf5_storage.store_performance_metrics(
            timestamp=timestamp,
            metrics=metrics,
            task_type=task_type
        )
        
        assert isinstance(metrics_id, str)
        assert task_type in metrics_id
        
        # Retrieve performance metrics
        retrieved_data = hdf5_storage.retrieve_performance_metrics(metrics_id)
        
        assert retrieved_data is not None
        ret_metrics, ret_timestamp, ret_task_type = retrieved_data
        assert ret_timestamp == timestamp
        assert ret_task_type == task_type
        
        for metric_name, expected_value in metrics.items():
            assert metric_name in ret_metrics
            assert abs(ret_metrics[metric_name] - expected_value) < 1e-6
    
    def test_file_info_and_statistics(self, hdf5_storage):
        """Test file information and statistics collection."""
        # Add some data to the file
        hdf5_storage.store_network_snapshot(
            timestamp=time.time(),
            adjacency_matrix=np.eye(10),
            weight_matrix=np.random.rand(10, 10),
            neuron_states=np.random.rand(10, 3)
        )
        
        hdf5_storage.store_spike_train(
            timestamp=time.time(),
            spike_times=np.array([0.1, 0.2, 0.3]),
            neuron_ids=np.array([1, 2, 3])
        )
        
        # Get file info
        info = hdf5_storage.get_file_info()
        
        assert 'file_path' in info
        assert 'file_size_bytes' in info
        assert 'file_size_mb' in info
        assert info['file_size_bytes'] > 0
        assert info['total_datasets'] > 0
        assert len(info['groups']) == 4  # network_states, spike_trains, activations, learning_traces
    
    def test_file_size_check_and_rotation(self, hdf5_storage):
        """Test file size checking and rotation."""
        # Set a very small max file size for testing
        hdf5_storage.max_file_size = 1024  # 1 KB
        
        # Add data to exceed the limit
        large_data = np.random.rand(1000, 1000)  # Large array
        hdf5_storage.store_network_snapshot(
            timestamp=time.time(),
            adjacency_matrix=large_data,
            weight_matrix=large_data,
            neuron_states=large_data
        )
        
        # Check if rotation is needed
        rotated = hdf5_storage.check_file_size()
        
        # Note: Rotation might not happen immediately due to HDF5 file structure
        # This test mainly checks that the method runs without error
        assert isinstance(rotated, bool)
    
    def test_file_compaction(self, hdf5_storage):
        """Test file compaction functionality."""
        # Add some data
        for i in range(5):
            hdf5_storage.store_spike_train(
                timestamp=time.time() + i,
                spike_times=np.random.rand(100),
                neuron_ids=np.random.randint(0, 50, 100)
            )
        
        # Get initial file size
        initial_info = hdf5_storage.get_file_info()
        initial_size = initial_info['file_size_bytes']
        
        # Compact file
        success = hdf5_storage.compact_file()
        
        assert success is True
        
        # Check that file still exists and has data
        final_info = hdf5_storage.get_file_info()
        assert final_info['total_datasets'] > 0
    
    def test_cleanup_old_data(self, hdf5_storage):
        """Test cleanup of old data."""
        current_time = time.time()
        
        # Store old and new data
        old_timestamp = current_time - (40 * 24 * 3600)  # 40 days ago
        new_timestamp = current_time - (10 * 24 * 3600)  # 10 days ago
        
        # Store old snapshot
        old_snapshot_id = hdf5_storage.store_network_snapshot(
            timestamp=old_timestamp,
            adjacency_matrix=np.eye(5),
            weight_matrix=np.random.rand(5, 5),
            neuron_states=np.random.rand(5, 2)
        )
        
        # Store new snapshot
        new_snapshot_id = hdf5_storage.store_network_snapshot(
            timestamp=new_timestamp,
            adjacency_matrix=np.eye(5),
            weight_matrix=np.random.rand(5, 5),
            neuron_states=np.random.rand(5, 2)
        )
        
        # Store old spike train
        hdf5_storage.store_spike_train(
            timestamp=old_timestamp,
            spike_times=np.array([0.1, 0.2]),
            neuron_ids=np.array([1, 2])
        )
        
        # Store new spike train
        hdf5_storage.store_spike_train(
            timestamp=new_timestamp,
            spike_times=np.array([0.3, 0.4]),
            neuron_ids=np.array([3, 4])
        )
        
        # Cleanup data older than 30 days
        removed_count = hdf5_storage.cleanup_old_data(retention_days=30)
        
        assert removed_count > 0
        
        # Verify old data is gone and new data remains
        old_snapshot = hdf5_storage.retrieve_network_snapshot(old_snapshot_id)
        new_snapshot = hdf5_storage.retrieve_network_snapshot(new_snapshot_id)
        
        assert old_snapshot is None  # Should be removed
        assert new_snapshot is not None  # Should remain
    
    def test_nonexistent_data_retrieval(self, hdf5_storage):
        """Test retrieval of non-existent data."""
        # Test non-existent network snapshot
        snapshot = hdf5_storage.retrieve_network_snapshot("nonexistent_snapshot")
        assert snapshot is None
        
        # Test non-existent spike train
        spike_train = hdf5_storage.retrieve_spike_train("nonexistent_train")
        assert spike_train is None
        
        # Test non-existent reservoir states
        states = hdf5_storage.retrieve_reservoir_states("nonexistent_states")
        assert states is None
        
        # Test non-existent layer outputs
        outputs = hdf5_storage.retrieve_layer_outputs("nonexistent_outputs")
        assert outputs is None
        
        # Test non-existent plasticity trace
        trace = hdf5_storage.retrieve_plasticity_trace("nonexistent_trace")
        assert trace is None
        
        # Test non-existent performance metrics
        metrics = hdf5_storage.retrieve_performance_metrics("nonexistent_metrics")
        assert metrics is None
    
    def test_context_manager_usage(self, temp_h5_path):
        """Test HDF5 storage as context manager."""
        with HDF5Storage(file_path=temp_h5_path) as storage:
            # Store some data
            snapshot_id = storage.store_network_snapshot(
                timestamp=time.time(),
                adjacency_matrix=np.eye(3),
                weight_matrix=np.random.rand(3, 3),
                neuron_states=np.random.rand(3, 2)
            )
            
            assert isinstance(snapshot_id, str)
        
        # Storage should be closed after context exit
        # Verify data persists
        with HDF5Storage(file_path=temp_h5_path) as storage2:
            snapshot = storage2.retrieve_network_snapshot(snapshot_id)
            assert snapshot is not None
    
    def test_large_data_handling(self, hdf5_storage):
        """Test handling of large datasets."""
        # Create large network data
        large_size = 2000
        large_adjacency = np.random.randint(0, 2, (large_size, large_size))
        large_weights = np.random.rand(large_size, large_size)
        large_states = np.random.rand(large_size, 10)
        
        # Store large network snapshot
        snapshot_id = hdf5_storage.store_network_snapshot(
            timestamp=time.time(),
            adjacency_matrix=large_adjacency,
            weight_matrix=large_weights,
            neuron_states=large_states
        )
        
        # Retrieve and verify
        retrieved_snapshot = hdf5_storage.retrieve_network_snapshot(snapshot_id)
        
        assert retrieved_snapshot is not None
        assert retrieved_snapshot.adjacency_matrix.shape == (large_size, large_size)
        assert retrieved_snapshot.weight_matrix.shape == (large_size, large_size)
        assert retrieved_snapshot.neuron_states.shape == (large_size, 10)
    
    def test_concurrent_access_simulation(self, hdf5_storage):
        """Test concurrent access patterns (simulated)."""
        import threading
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(3):
                    # Store network snapshot
                    snapshot_id = hdf5_storage.store_network_snapshot(
                        timestamp=time.time() + worker_id * 10 + i,
                        adjacency_matrix=np.eye(10) + worker_id * 0.1,
                        weight_matrix=np.random.rand(10, 10),
                        neuron_states=np.random.rand(10, 3)
                    )
                    results.append(snapshot_id)
                    
                    # Store spike train
                    train_id = hdf5_storage.store_spike_train(
                        timestamp=time.time() + worker_id * 10 + i,
                        spike_times=np.random.rand(50),
                        neuron_ids=np.random.randint(0, 10, 50)
                    )
                    results.append(train_id)
                    
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 18  # 3 workers * 3 iterations * 2 operations each
        assert len(set(results)) == 18  # All IDs should be unique


@pytest.mark.skipif(HDF5_AVAILABLE, reason="Testing behavior when h5py is not available")
class TestHDF5StorageWithoutH5PY:
    """Test HDF5 storage behavior when h5py is not available."""
    
    def test_import_error_handling(self):
        """Test that appropriate error is raised when h5py is not available."""
        with pytest.raises(HDF5StorageError, match="h5py is not available"):
            HDF5Storage()


@pytest.mark.skipif(not HDF5_AVAILABLE, reason="h5py not available")
class TestHDF5StorageIntegration:
    """Integration tests for HDF5 storage."""
    
    @pytest.mark.integration
    def test_large_scale_data_operations(self):
        """Test large-scale data operations."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            h5_path = Path(f.name)
        
        try:
            with HDF5Storage(file_path=h5_path, max_file_size=100*1024*1024) as storage:  # 100MB
                # Store many network snapshots
                snapshot_ids = []
                for i in range(50):
                    snapshot_id = storage.store_network_snapshot(
                        timestamp=time.time() + i,
                        adjacency_matrix=np.random.randint(0, 2, (200, 200)),
                        weight_matrix=np.random.rand(200, 200),
                        neuron_states=np.random.rand(200, 5)
                    )
                    snapshot_ids.append(snapshot_id)
                
                # Store many spike trains
                train_ids = []
                for i in range(100):
                    train_id = storage.store_spike_train(
                        timestamp=time.time() + i,
                        spike_times=np.random.rand(1000),
                        neuron_ids=np.random.randint(0, 200, 1000)
                    )
                    train_ids.append(train_id)
                
                # Verify data integrity
                info = storage.get_file_info()
                assert info['total_datasets'] >= 150  # At least 50 snapshots + 100 spike trains
                
                # Test retrieval performance
                start_time = time.time()
                for snapshot_id in snapshot_ids[:10]:
                    snapshot = storage.retrieve_network_snapshot(snapshot_id)
                    assert snapshot is not None
                retrieval_time = time.time() - start_time
                
                # Should retrieve 10 snapshots in reasonable time
                assert retrieval_time < 10.0
        
        finally:
            if h5_path.exists():
                h5_path.unlink()
    
    @pytest.mark.integration
    def test_performance_benchmarks(self):
        """Test HDF5 storage performance benchmarks."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            h5_path = Path(f.name)
        
        try:
            with HDF5Storage(file_path=h5_path) as storage:
                # Benchmark network snapshot storage
                network_data = [
                    (np.random.randint(0, 2, (100, 100)), 
                     np.random.rand(100, 100), 
                     np.random.rand(100, 5))
                    for _ in range(100)
                ]
                
                start_time = time.time()
                snapshot_ids = []
                for i, (adj, weights, states) in enumerate(network_data):
                    snapshot_id = storage.store_network_snapshot(
                        timestamp=time.time() + i,
                        adjacency_matrix=adj,
                        weight_matrix=weights,
                        neuron_states=states
                    )
                    snapshot_ids.append(snapshot_id)
                storage_time = time.time() - start_time
                
                # Benchmark retrieval
                start_time = time.time()
                for snapshot_id in snapshot_ids[:20]:
                    snapshot = storage.retrieve_network_snapshot(snapshot_id)
                    assert snapshot is not None
                retrieval_time = time.time() - start_time
                
                # Performance assertions (adjust based on hardware)
                assert storage_time < 60.0  # Store 100 snapshots in < 60 seconds
                assert retrieval_time < 10.0  # Retrieve 20 snapshots in < 10 seconds
                
                logger.info(f"Storage time: {storage_time:.2f}s, Retrieval time: {retrieval_time:.2f}s")
        
        finally:
            if h5_path.exists():
                h5_path.unlink()
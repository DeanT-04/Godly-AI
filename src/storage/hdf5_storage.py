"""
HDF5 integration for large-scale data storage in the Godly AI System.

This module provides HDF5-based storage for network states, spike trains,
activation data, and other large-scale neuromorphic data with compression
and efficient chunking.
"""

import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager
import threading

import numpy as np
import jax.numpy as jnp

try:
    import h5py
    HDF5_AVAILABLE = True
    H5File = h5py.File
    H5Group = h5py.Group
    H5Dataset = h5py.Dataset
except ImportError:
    h5py = None
    HDF5_AVAILABLE = False
    # Create dummy types for type annotations
    H5File = object
    H5Group = object
    H5Dataset = object

logger = logging.getLogger(__name__)


@dataclass
class NetworkSnapshot:
    """Represents a network state snapshot."""
    timestamp: float
    adjacency_matrix: np.ndarray
    weight_matrix: np.ndarray
    neuron_states: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class SpikeTrainData:
    """Represents spike train data with metadata."""
    timestamp: float
    spike_times: np.ndarray
    neuron_ids: np.ndarray
    spike_amplitudes: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


class HDF5StorageError(Exception):
    """Raised when HDF5 storage operations fail."""
    pass


class HDF5Storage:
    """
    HDF5-based storage system for large-scale neuromorphic data.
    
    Provides efficient storage for network states, spike trains, activation data,
    and other high-volume neuromorphic data with compression and chunking.
    """
    
    def __init__(
        self,
        file_path: Union[str, Path] = "godly_ai_data.h5",
        compression: str = "gzip",
        compression_opts: int = 6,
        chunk_size: int = 1024,
        max_file_size: int = 10 * 1024**3,  # 10 GB
        enable_swmr: bool = True
    ):
        """
        Initialize HDF5 storage with configuration.
        
        Args:
            file_path: Path to HDF5 file
            compression: Compression algorithm ('gzip', 'lzf', 'szip')
            compression_opts: Compression level (0-9 for gzip)
            chunk_size: Chunk size for datasets
            max_file_size: Maximum file size before rotation
            enable_swmr: Enable Single Writer Multiple Reader mode
        """
        if not HDF5_AVAILABLE:
            raise HDF5StorageError("h5py is not available. Install with: pip install h5py")
        
        self.file_path = Path(file_path)
        self.compression = compression
        self.compression_opts = compression_opts
        self.chunk_size = chunk_size
        self.max_file_size = max_file_size
        self.enable_swmr = enable_swmr
        
        # Thread-local storage for file handles
        self._local = threading.local()
        
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file structure
        self._initialize_file_structure()
        
        logger.info(f"HDF5 storage initialized: {self.file_path}")
    
    def _get_file_handle(self, mode: str = 'r+') -> H5File:
        """Get thread-local HDF5 file handle."""
        if not hasattr(self._local, 'file_handle') or self._local.file_handle is None:
            try:
                if self.file_path.exists():
                    self._local.file_handle = h5py.File(
                        str(self.file_path), 
                        mode,
                        swmr=self.enable_swmr and mode == 'r'
                    )
                else:
                    self._local.file_handle = h5py.File(
                        str(self.file_path), 
                        'w'
                    )
                    self._initialize_file_structure()
            except Exception as e:
                raise HDF5StorageError(f"Failed to open HDF5 file: {e}")
        
        return self._local.file_handle
    
    @contextmanager
    def _file_context(self, mode: str = 'r+'):
        """Context manager for HDF5 file operations."""
        file_handle = self._get_file_handle(mode)
        try:
            yield file_handle
            file_handle.flush()
        except Exception as e:
            logger.error(f"HDF5 operation failed: {e}")
            raise HDF5StorageError(f"HDF5 operation failed: {e}")
    
    def _initialize_file_structure(self) -> None:
        """Initialize HDF5 file structure with groups."""
        with h5py.File(str(self.file_path), 'a') as f:
            # Create main groups
            if 'network_states' not in f:
                network_group = f.create_group('network_states')
                network_group.create_group('topology')
                network_group.create_group('weights')
                network_group.create_group('neuron_states')
            
            if 'spike_trains' not in f:
                spike_group = f.create_group('spike_trains')
                spike_group.create_group('raw_data')
                spike_group.create_group('compressed')
                spike_group.create_group('metadata')
            
            if 'activations' not in f:
                activation_group = f.create_group('activations')
                activation_group.create_group('reservoir_states')
                activation_group.create_group('layer_outputs')
                activation_group.create_group('attention_maps')
            
            if 'learning_traces' not in f:
                learning_group = f.create_group('learning_traces')
                learning_group.create_group('plasticity')
                learning_group.create_group('homeostatic')
                learning_group.create_group('performance')
            
            # Store metadata
            if 'metadata' not in f.attrs:
                f.attrs['created_at'] = time.time()
                f.attrs['version'] = '1.0'
                f.attrs['compression'] = self.compression
                f.attrs['chunk_size'] = self.chunk_size
    
    def _create_dataset(
        self,
        group: H5Group,
        name: str,
        data: np.ndarray,
        maxshape: Optional[Tuple] = None,
        chunks: Optional[Tuple] = None
    ) -> H5Dataset:
        """Create HDF5 dataset with compression and chunking."""
        if chunks is None:
            # Auto-determine chunk size based on data shape
            chunks = tuple(min(self.chunk_size, dim) for dim in data.shape)
        
        if maxshape is None:
            maxshape = tuple(None if i == 0 else dim for i, dim in enumerate(data.shape))
        
        return group.create_dataset(
            name,
            data=data,
            maxshape=maxshape,
            chunks=chunks,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=True,  # Improve compression
            fletcher32=True  # Add checksum
        )
    
    # Network State Operations
    
    def store_network_snapshot(
        self,
        timestamp: float,
        adjacency_matrix: Union[np.ndarray, jnp.ndarray],
        weight_matrix: Union[np.ndarray, jnp.ndarray],
        neuron_states: Union[np.ndarray, jnp.ndarray],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a complete network state snapshot.
        
        Args:
            timestamp: Snapshot timestamp
            adjacency_matrix: Network topology matrix
            weight_matrix: Synaptic weight matrix
            neuron_states: Current neuron states
            metadata: Optional metadata dictionary
            
        Returns:
            Snapshot identifier
        """
        # Convert JAX arrays to numpy
        if hasattr(adjacency_matrix, 'device'):
            adjacency_matrix = np.array(adjacency_matrix)
        if hasattr(weight_matrix, 'device'):
            weight_matrix = np.array(weight_matrix)
        if hasattr(neuron_states, 'device'):
            neuron_states = np.array(neuron_states)
        
        snapshot_id = f"snapshot_{int(timestamp * 1000)}"
        
        with self._file_context('a') as f:
            # Store topology
            topology_group = f['network_states/topology']
            if snapshot_id not in topology_group:
                topology_group = topology_group.create_group(snapshot_id)
            
            self._create_dataset(topology_group, 'adjacency', adjacency_matrix)
            topology_group.attrs['timestamp'] = timestamp
            topology_group.attrs['shape'] = adjacency_matrix.shape
            
            # Store weights
            weights_group = f['network_states/weights']
            if snapshot_id not in weights_group:
                weights_group = weights_group.create_group(snapshot_id)
            
            self._create_dataset(weights_group, 'weights', weight_matrix)
            weights_group.attrs['timestamp'] = timestamp
            weights_group.attrs['shape'] = weight_matrix.shape
            
            # Store neuron states
            states_group = f['network_states/neuron_states']
            if snapshot_id not in states_group:
                states_group = states_group.create_group(snapshot_id)
            
            self._create_dataset(states_group, 'states', neuron_states)
            states_group.attrs['timestamp'] = timestamp
            states_group.attrs['shape'] = neuron_states.shape
            
            # Store metadata
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        topology_group.attrs[f"meta_{key}"] = value
        
        logger.debug(f"Stored network snapshot: {snapshot_id}")
        return snapshot_id
    
    def retrieve_network_snapshot(self, snapshot_id: str) -> Optional[NetworkSnapshot]:
        """
        Retrieve a network state snapshot.
        
        Args:
            snapshot_id: Snapshot identifier
            
        Returns:
            NetworkSnapshot object or None if not found
        """
        with self._file_context('r') as f:
            try:
                # Load topology
                topology_group = f[f'network_states/topology/{snapshot_id}']
                adjacency_matrix = topology_group['adjacency'][:]
                
                # Load weights
                weights_group = f[f'network_states/weights/{snapshot_id}']
                weight_matrix = weights_group['weights'][:]
                
                # Load neuron states
                states_group = f[f'network_states/neuron_states/{snapshot_id}']
                neuron_states = states_group['states'][:]
                
                # Load metadata
                timestamp = topology_group.attrs['timestamp']
                metadata = {}
                for key in topology_group.attrs:
                    if key.startswith('meta_'):
                        metadata[key[5:]] = topology_group.attrs[key]
                
                return NetworkSnapshot(
                    timestamp=timestamp,
                    adjacency_matrix=adjacency_matrix,
                    weight_matrix=weight_matrix,
                    neuron_states=neuron_states,
                    metadata=metadata
                )
                
            except KeyError:
                logger.warning(f"Network snapshot not found: {snapshot_id}")
                return None
    
    def get_network_snapshots_in_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[str]:
        """
        Get network snapshot IDs within time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
            
        Returns:
            List of snapshot IDs
        """
        snapshot_ids = []
        
        with self._file_context('r') as f:
            topology_group = f['network_states/topology']
            
            for snapshot_id in topology_group.keys():
                snapshot_group = topology_group[snapshot_id]
                timestamp = snapshot_group.attrs['timestamp']
                
                if start_time <= timestamp <= end_time:
                    snapshot_ids.append(snapshot_id)
        
        return sorted(snapshot_ids)
    
    # Spike Train Operations
    
    def store_spike_train(
        self,
        timestamp: float,
        spike_times: Union[np.ndarray, jnp.ndarray],
        neuron_ids: Union[np.ndarray, jnp.ndarray],
        spike_amplitudes: Optional[Union[np.ndarray, jnp.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store spike train data with efficient compression.
        
        Args:
            timestamp: Spike train timestamp
            spike_times: Array of spike times
            neuron_ids: Array of neuron IDs corresponding to spikes
            spike_amplitudes: Optional spike amplitudes
            metadata: Optional metadata dictionary
            
        Returns:
            Spike train identifier
        """
        # Convert JAX arrays to numpy
        if hasattr(spike_times, 'device'):
            spike_times = np.array(spike_times)
        if hasattr(neuron_ids, 'device'):
            neuron_ids = np.array(neuron_ids)
        if spike_amplitudes is not None and hasattr(spike_amplitudes, 'device'):
            spike_amplitudes = np.array(spike_amplitudes)
        
        train_id = f"train_{int(timestamp * 1000)}"
        
        with self._file_context('a') as f:
            # Store raw spike data
            raw_group = f['spike_trains/raw_data']
            if train_id not in raw_group:
                train_group = raw_group.create_group(train_id)
            else:
                train_group = raw_group[train_id]
            
            # Store spike times and neuron IDs
            self._create_dataset(train_group, 'spike_times', spike_times)
            self._create_dataset(train_group, 'neuron_ids', neuron_ids)
            
            if spike_amplitudes is not None:
                self._create_dataset(train_group, 'spike_amplitudes', spike_amplitudes)
            
            # Store metadata
            train_group.attrs['timestamp'] = timestamp
            train_group.attrs['num_spikes'] = len(spike_times)
            train_group.attrs['num_neurons'] = len(np.unique(neuron_ids))
            
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        train_group.attrs[f"meta_{key}"] = value
        
        logger.debug(f"Stored spike train: {train_id}")
        return train_id
    
    def retrieve_spike_train(self, train_id: str) -> Optional[SpikeTrainData]:
        """
        Retrieve spike train data.
        
        Args:
            train_id: Spike train identifier
            
        Returns:
            SpikeTrainData object or None if not found
        """
        with self._file_context('r') as f:
            try:
                train_group = f[f'spike_trains/raw_data/{train_id}']
                
                spike_times = train_group['spike_times'][:]
                neuron_ids = train_group['neuron_ids'][:]
                
                spike_amplitudes = None
                if 'spike_amplitudes' in train_group:
                    spike_amplitudes = train_group['spike_amplitudes'][:]
                
                timestamp = train_group.attrs['timestamp']
                
                # Load metadata
                metadata = {}
                for key in train_group.attrs:
                    if key.startswith('meta_'):
                        metadata[key[5:]] = train_group.attrs[key]
                
                return SpikeTrainData(
                    timestamp=timestamp,
                    spike_times=spike_times,
                    neuron_ids=neuron_ids,
                    spike_amplitudes=spike_amplitudes,
                    metadata=metadata
                )
                
            except KeyError:
                logger.warning(f"Spike train not found: {train_id}")
                return None
    
    def store_compressed_spike_train(
        self,
        timestamp: float,
        spike_data: Union[np.ndarray, jnp.ndarray],
        compression_params: Dict[str, Any]
    ) -> str:
        """
        Store pre-compressed spike train data.
        
        Args:
            timestamp: Spike train timestamp
            spike_data: Compressed spike data
            compression_params: Compression parameters
            
        Returns:
            Compressed spike train identifier
        """
        if hasattr(spike_data, 'device'):
            spike_data = np.array(spike_data)
        
        train_id = f"compressed_{int(timestamp * 1000)}"
        
        with self._file_context('a') as f:
            compressed_group = f['spike_trains/compressed']
            
            # Store compressed data
            dataset = self._create_dataset(compressed_group, train_id, spike_data)
            dataset.attrs['timestamp'] = timestamp
            dataset.attrs['original_shape'] = compression_params.get('original_shape', spike_data.shape)
            dataset.attrs['compression_ratio'] = compression_params.get('compression_ratio', 1.0)
            
            for key, value in compression_params.items():
                if isinstance(value, (str, int, float, bool)):
                    dataset.attrs[f"param_{key}"] = value
        
        logger.debug(f"Stored compressed spike train: {train_id}")
        return train_id 
   
    # Activation Data Operations
    
    def store_reservoir_states(
        self,
        timestamp: float,
        reservoir_states: Union[np.ndarray, jnp.ndarray],
        reservoir_id: str = "default"
    ) -> str:
        """
        Store reservoir activation states.
        
        Args:
            timestamp: State timestamp
            reservoir_states: Reservoir neuron states
            reservoir_id: Reservoir identifier
            
        Returns:
            State identifier
        """
        if hasattr(reservoir_states, 'device'):
            reservoir_states = np.array(reservoir_states)
        
        state_id = f"{reservoir_id}_{int(timestamp * 1000)}"
        
        with self._file_context('a') as f:
            reservoir_group = f['activations/reservoir_states']
            
            # Create timestamped dataset
            dataset = self._create_dataset(reservoir_group, state_id, reservoir_states)
            dataset.attrs['timestamp'] = timestamp
            dataset.attrs['reservoir_id'] = reservoir_id
            dataset.attrs['shape'] = reservoir_states.shape
        
        logger.debug(f"Stored reservoir states: {state_id}")
        return state_id
    
    def retrieve_reservoir_states(
        self,
        state_id: str
    ) -> Optional[Tuple[np.ndarray, float, str]]:
        """
        Retrieve reservoir states.
        
        Args:
            state_id: State identifier
            
        Returns:
            Tuple of (states, timestamp, reservoir_id) or None
        """
        with self._file_context('r') as f:
            try:
                dataset = f[f'activations/reservoir_states/{state_id}']
                states = dataset[:]
                timestamp = dataset.attrs['timestamp']
                reservoir_id = dataset.attrs['reservoir_id']
                
                return states, timestamp, reservoir_id
                
            except KeyError:
                logger.warning(f"Reservoir states not found: {state_id}")
                return None
    
    def store_layer_outputs(
        self,
        timestamp: float,
        layer_outputs: Dict[str, Union[np.ndarray, jnp.ndarray]]
    ) -> str:
        """
        Store neural layer outputs.
        
        Args:
            timestamp: Output timestamp
            layer_outputs: Dictionary of layer_name -> output_array
            
        Returns:
            Output identifier
        """
        output_id = f"outputs_{int(timestamp * 1000)}"
        
        with self._file_context('a') as f:
            outputs_group = f['activations/layer_outputs']
            
            if output_id not in outputs_group:
                output_group = outputs_group.create_group(output_id)
            else:
                output_group = outputs_group[output_id]
            
            # Store each layer's output
            for layer_name, output_array in layer_outputs.items():
                if hasattr(output_array, 'device'):
                    output_array = np.array(output_array)
                
                self._create_dataset(output_group, layer_name, output_array)
            
            output_group.attrs['timestamp'] = timestamp
            output_group.attrs['num_layers'] = len(layer_outputs)
        
        logger.debug(f"Stored layer outputs: {output_id}")
        return output_id
    
    def retrieve_layer_outputs(
        self,
        output_id: str
    ) -> Optional[Tuple[Dict[str, np.ndarray], float]]:
        """
        Retrieve layer outputs.
        
        Args:
            output_id: Output identifier
            
        Returns:
            Tuple of (layer_outputs_dict, timestamp) or None
        """
        with self._file_context('r') as f:
            try:
                output_group = f[f'activations/layer_outputs/{output_id}']
                
                layer_outputs = {}
                for layer_name in output_group.keys():
                    layer_outputs[layer_name] = output_group[layer_name][:]
                
                timestamp = output_group.attrs['timestamp']
                
                return layer_outputs, timestamp
                
            except KeyError:
                logger.warning(f"Layer outputs not found: {output_id}")
                return None
    
    # Learning Trace Operations
    
    def store_plasticity_trace(
        self,
        timestamp: float,
        synapse_changes: Union[np.ndarray, jnp.ndarray],
        plasticity_type: str = "stdp"
    ) -> str:
        """
        Store synaptic plasticity traces.
        
        Args:
            timestamp: Trace timestamp
            synapse_changes: Array of synaptic weight changes
            plasticity_type: Type of plasticity rule
            
        Returns:
            Trace identifier
        """
        if hasattr(synapse_changes, 'device'):
            synapse_changes = np.array(synapse_changes)
        
        trace_id = f"{plasticity_type}_{int(timestamp * 1000)}"
        
        with self._file_context('a') as f:
            plasticity_group = f['learning_traces/plasticity']
            
            dataset = self._create_dataset(plasticity_group, trace_id, synapse_changes)
            dataset.attrs['timestamp'] = timestamp
            dataset.attrs['plasticity_type'] = plasticity_type
            dataset.attrs['num_synapses'] = synapse_changes.size
            dataset.attrs['mean_change'] = float(np.mean(synapse_changes))
            dataset.attrs['std_change'] = float(np.std(synapse_changes))
        
        logger.debug(f"Stored plasticity trace: {trace_id}")
        return trace_id
    
    def retrieve_plasticity_trace(
        self,
        trace_id: str
    ) -> Optional[Tuple[np.ndarray, float, str]]:
        """
        Retrieve plasticity trace.
        
        Args:
            trace_id: Trace identifier
            
        Returns:
            Tuple of (synapse_changes, timestamp, plasticity_type) or None
        """
        with self._file_context('r') as f:
            try:
                dataset = f[f'learning_traces/plasticity/{trace_id}']
                synapse_changes = dataset[:]
                timestamp = dataset.attrs['timestamp']
                plasticity_type = dataset.attrs['plasticity_type']
                
                return synapse_changes, timestamp, plasticity_type
                
            except KeyError:
                logger.warning(f"Plasticity trace not found: {trace_id}")
                return None
    
    def store_homeostatic_trace(
        self,
        timestamp: float,
        homeostatic_changes: Union[np.ndarray, jnp.ndarray],
        mechanism: str = "intrinsic_plasticity"
    ) -> str:
        """
        Store homeostatic plasticity traces.
        
        Args:
            timestamp: Trace timestamp
            homeostatic_changes: Array of homeostatic changes
            mechanism: Homeostatic mechanism type
            
        Returns:
            Trace identifier
        """
        if hasattr(homeostatic_changes, 'device'):
            homeostatic_changes = np.array(homeostatic_changes)
        
        trace_id = f"{mechanism}_{int(timestamp * 1000)}"
        
        with self._file_context('a') as f:
            homeostatic_group = f['learning_traces/homeostatic']
            
            dataset = self._create_dataset(homeostatic_group, trace_id, homeostatic_changes)
            dataset.attrs['timestamp'] = timestamp
            dataset.attrs['mechanism'] = mechanism
            dataset.attrs['num_neurons'] = homeostatic_changes.size
        
        logger.debug(f"Stored homeostatic trace: {trace_id}")
        return trace_id
    
    # Performance and Analytics Operations
    
    def store_performance_metrics(
        self,
        timestamp: float,
        metrics: Dict[str, float],
        task_type: str = "general"
    ) -> str:
        """
        Store performance metrics over time.
        
        Args:
            timestamp: Metrics timestamp
            metrics: Dictionary of metric_name -> value
            task_type: Type of task being measured
            
        Returns:
            Metrics identifier
        """
        metrics_id = f"{task_type}_{int(timestamp * 1000)}"
        
        with self._file_context('a') as f:
            performance_group = f['learning_traces/performance']
            
            # Convert metrics to arrays for storage
            metric_names = list(metrics.keys())
            metric_values = np.array(list(metrics.values()))
            
            if metrics_id not in performance_group:
                metrics_group = performance_group.create_group(metrics_id)
            else:
                metrics_group = performance_group[metrics_id]
            
            # Store metric values
            self._create_dataset(metrics_group, 'values', metric_values)
            
            # Store metric names as attributes
            metrics_group.attrs['timestamp'] = timestamp
            metrics_group.attrs['task_type'] = task_type
            metrics_group.attrs['metric_names'] = metric_names
        
        logger.debug(f"Stored performance metrics: {metrics_id}")
        return metrics_id
    
    def retrieve_performance_metrics(
        self,
        metrics_id: str
    ) -> Optional[Tuple[Dict[str, float], float, str]]:
        """
        Retrieve performance metrics.
        
        Args:
            metrics_id: Metrics identifier
            
        Returns:
            Tuple of (metrics_dict, timestamp, task_type) or None
        """
        with self._file_context('r') as f:
            try:
                metrics_group = f[f'learning_traces/performance/{metrics_id}']
                
                metric_values = metrics_group['values'][:]
                metric_names = metrics_group.attrs['metric_names']
                timestamp = metrics_group.attrs['timestamp']
                task_type = metrics_group.attrs['task_type']
                
                metrics = dict(zip(metric_names, metric_values))
                
                return metrics, timestamp, task_type
                
            except KeyError:
                logger.warning(f"Performance metrics not found: {metrics_id}")
                return None
    
    # File Management Operations
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        Get information about the HDF5 file.
        
        Returns:
            Dictionary with file information
        """
        info = {
            'file_path': str(self.file_path),
            'file_size_bytes': 0,
            'file_size_mb': 0,
            'compression': self.compression,
            'chunk_size': self.chunk_size,
            'groups': [],
            'total_datasets': 0
        }
        
        if self.file_path.exists():
            info['file_size_bytes'] = self.file_path.stat().st_size
            info['file_size_mb'] = info['file_size_bytes'] / (1024 * 1024)
            
            with self._file_context('r') as f:
                info['created_at'] = f.attrs.get('created_at', 0)
                info['version'] = f.attrs.get('version', 'unknown')
                
                def count_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        info['total_datasets'] += 1
                
                f.visititems(count_datasets)
                info['groups'] = list(f.keys())
        
        return info
    
    def compact_file(self) -> bool:
        """
        Compact the HDF5 file to reclaim space.
        
        Returns:
            True if successful
        """
        try:
            # Close current file handles
            self.close()
            
            # Create temporary compacted file
            temp_path = self.file_path.with_suffix('.tmp.h5')
            
            with h5py.File(str(self.file_path), 'r') as src:
                with h5py.File(str(temp_path), 'w') as dst:
                    # Copy all data to new file (automatically compacts)
                    def copy_item(name, obj):
                        if isinstance(obj, h5py.Group):
                            dst.create_group(name)
                            # Copy attributes
                            for key, value in obj.attrs.items():
                                dst[name].attrs[key] = value
                        elif isinstance(obj, h5py.Dataset):
                            # Recreate dataset with current compression settings
                            dst.create_dataset(
                                name,
                                data=obj[:],
                                compression=self.compression,
                                compression_opts=self.compression_opts,
                                chunks=obj.chunks,
                                shuffle=True,
                                fletcher32=True
                            )
                            # Copy attributes
                            for key, value in obj.attrs.items():
                                dst[name].attrs[key] = value
                    
                    src.visititems(copy_item)
                    
                    # Copy root attributes
                    for key, value in src.attrs.items():
                        dst.attrs[key] = value
            
            # Replace original file with compacted version
            self.file_path.unlink()
            temp_path.rename(self.file_path)
            
            logger.info(f"HDF5 file compacted: {self.file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to compact HDF5 file: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def rotate_file(self) -> Path:
        """
        Rotate the current file and create a new one.
        
        Returns:
            Path to the rotated file
        """
        timestamp = int(time.time())
        rotated_path = self.file_path.with_name(
            f"{self.file_path.stem}_{timestamp}{self.file_path.suffix}"
        )
        
        # Close current file
        self.close()
        
        # Move current file to rotated name
        if self.file_path.exists():
            self.file_path.rename(rotated_path)
        
        # Initialize new file
        self._initialize_file_structure()
        
        logger.info(f"HDF5 file rotated: {rotated_path}")
        return rotated_path
    
    def check_file_size(self) -> bool:
        """
        Check if file size exceeds maximum and rotate if needed.
        
        Returns:
            True if file was rotated
        """
        if self.file_path.exists():
            file_size = self.file_path.stat().st_size
            if file_size > self.max_file_size:
                self.rotate_file()
                return True
        return False
    
    def cleanup_old_data(self, retention_days: int = 30) -> int:
        """
        Remove data older than retention period.
        
        Args:
            retention_days: Number of days to retain data
            
        Returns:
            Number of items removed
        """
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        removed_count = 0
        
        with self._file_context('a') as f:
            # Clean up old network snapshots
            topology_group = f['network_states/topology']
            snapshots_to_remove = []
            
            for snapshot_id in topology_group.keys():
                snapshot_group = topology_group[snapshot_id]
                if snapshot_group.attrs['timestamp'] < cutoff_time:
                    snapshots_to_remove.append(snapshot_id)
            
            for snapshot_id in snapshots_to_remove:
                # Remove from all related groups
                for group_path in ['network_states/topology', 'network_states/weights', 'network_states/neuron_states']:
                    if snapshot_id in f[group_path]:
                        del f[f'{group_path}/{snapshot_id}']
                        removed_count += 1
            
            # Clean up old spike trains
            spike_group = f['spike_trains/raw_data']
            trains_to_remove = []
            
            for train_id in spike_group.keys():
                train_group = spike_group[train_id]
                if train_group.attrs['timestamp'] < cutoff_time:
                    trains_to_remove.append(train_id)
            
            for train_id in trains_to_remove:
                del spike_group[train_id]
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} old data items")
        return removed_count
    
    def close(self) -> None:
        """Close HDF5 file handles and perform cleanup."""
        try:
            if hasattr(self._local, 'file_handle') and self._local.file_handle is not None:
                self._local.file_handle.close()
                self._local.file_handle = None
            
            logger.info("HDF5 storage closed")
        except Exception as e:
            logger.error(f"Error closing HDF5 storage: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
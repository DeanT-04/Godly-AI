# HDF5 Storage

HDF5 Storage provides efficient storage for large-scale scientific data with compression, chunking, and hierarchical organization. It's designed for network states, spike trains, activation data, and other high-volume neuromorphic data.

## Overview

HDF5 Storage is optimized for:
- Network state snapshots with large adjacency and weight matrices
- Spike train data with efficient compression
- Neural activation data from reservoirs and layers
- Learning traces and plasticity data
- Performance metrics and analytics
- Long-term data archival with compression

## Features

- **Hierarchical Organization**: Data organized in logical groups and datasets
- **Compression**: Multiple compression algorithms (gzip, lzf, szip)
- **Chunking**: Optimized data access patterns with configurable chunk sizes
- **File Management**: Automatic file rotation, compaction, and cleanup
- **Checksums**: Data integrity verification with Fletcher32 checksums
- **SWMR Support**: Single Writer Multiple Reader for concurrent access

## Configuration

### Basic Configuration

```python
from src.storage import HDF5Storage

# Basic setup
hdf5_store = HDF5Storage(file_path="godly_ai_data.h5")

# Advanced configuration
hdf5_store = HDF5Storage(
    file_path="./data/godly_ai_data.h5",
    compression="gzip",           # Compression algorithm
    compression_opts=6,           # Compression level (0-9)
    chunk_size=1024,             # Chunk size for datasets
    max_file_size=10*1024**3,    # 10GB max file size
    enable_swmr=True             # Enable SWMR mode
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | str/Path | "godly_ai_data.h5" | Path to HDF5 file |
| `compression` | str | "gzip" | Compression algorithm |
| `compression_opts` | int | 6 | Compression level (0-9) |
| `chunk_size` | int | 1024 | Chunk size for datasets |
| `max_file_size` | int | 10GB | Maximum file size before rotation |
| `enable_swmr` | bool | True | Enable SWMR mode |

### Compression Options

| Algorithm | Speed | Ratio | Description |
|-----------|-------|-------|-------------|
| `gzip` | Medium | High | General purpose, good compression |
| `lzf` | Fast | Medium | Fast compression/decompression |
| `szip` | Medium | Medium | Scientific data optimized |

## File Structure

HDF5 Storage organizes data in a hierarchical structure:

```
godly_ai_data.h5
├── network_states/
│   ├── topology/
│   │   └── snapshot_<timestamp>/
│   │       └── adjacency
│   ├── weights/
│   │   └── snapshot_<timestamp>/
│   │       └── weights
│   └── neuron_states/
│       └── snapshot_<timestamp>/
│           └── states
├── spike_trains/
│   ├── raw_data/
│   │   └── train_<timestamp>/
│   │       ├── spike_times
│   │       ├── neuron_ids
│   │       └── spike_amplitudes
│   ├── compressed/
│   │   └── compressed_<timestamp>
│   └── metadata/
├── activations/
│   ├── reservoir_states/
│   │   └── <reservoir_id>_<timestamp>
│   ├── layer_outputs/
│   │   └── outputs_<timestamp>/
│   │       ├── input_layer
│   │       ├── hidden_layer_1
│   │       └── output_layer
│   └── attention_maps/
└── learning_traces/
    ├── plasticity/
    │   └── <type>_<timestamp>
    ├── homeostatic/
    │   └── <mechanism>_<timestamp>
    └── performance/
        └── <task_type>_<timestamp>/
            └── values
```

## API Reference

### Network State Operations

#### store_network_snapshot()

Store a complete network state snapshot with topology, weights, and neuron states.

```python
snapshot_id = hdf5_store.store_network_snapshot(
    timestamp=time.time(),
    adjacency_matrix=np.random.randint(0, 2, (1000, 1000)),
    weight_matrix=np.random.rand(1000, 1000),
    neuron_states=np.random.rand(1000, 5),
    metadata={
        "network_type": "liquid_state_machine",
        "num_neurons": 1000,
        "connectivity": 0.1
    }
)
print(f"Stored network snapshot: {snapshot_id}")
```

**Parameters:**
- `timestamp` (float): Snapshot timestamp
- `adjacency_matrix` (np.ndarray | jnp.ndarray): Network topology matrix
- `weight_matrix` (np.ndarray | jnp.ndarray): Synaptic weight matrix
- `neuron_states` (np.ndarray | jnp.ndarray): Current neuron states
- `metadata` (dict, optional): Additional metadata

**Returns:** `str` - Snapshot identifier

#### retrieve_network_snapshot()

Retrieve a complete network state snapshot.

```python
snapshot = hdf5_store.retrieve_network_snapshot(snapshot_id)
if snapshot:
    print(f"Network snapshot at {snapshot.timestamp}")
    print(f"Adjacency matrix shape: {snapshot.adjacency_matrix.shape}")
    print(f"Weight matrix shape: {snapshot.weight_matrix.shape}")
    print(f"Neuron states shape: {snapshot.neuron_states.shape}")
    print(f"Metadata: {snapshot.metadata}")
```

**Parameters:**
- `snapshot_id` (str): Snapshot identifier

**Returns:** `NetworkSnapshot | None` - NetworkSnapshot object or None if not found

#### get_network_snapshots_in_range()

Get network snapshot IDs within a time range.

```python
snapshot_ids = hdf5_store.get_network_snapshots_in_range(
    start_time=time.time() - 3600,  # Last hour
    end_time=time.time()
)

print(f"Found {len(snapshot_ids)} snapshots in time range")
for snapshot_id in snapshot_ids:
    snapshot = hdf5_store.retrieve_network_snapshot(snapshot_id)
    print(f"  {snapshot_id}: {snapshot.timestamp}")
```

**Parameters:**
- `start_time` (float): Start timestamp
- `end_time` (float): End timestamp

**Returns:** `List[str]` - List of snapshot IDs

### Spike Train Operations

#### store_spike_train()

Store spike train data with efficient compression.

```python
train_id = hdf5_store.store_spike_train(
    timestamp=time.time(),
    spike_times=np.array([0.1, 0.15, 0.3, 0.45, 0.6]),
    neuron_ids=np.array([1, 3, 1, 2, 3]),
    spike_amplitudes=np.array([1.2, 0.8, 1.5, 1.0, 0.9]),  # Optional
    metadata={
        "recording_duration": 1.0,
        "sampling_rate": 10000,
        "num_neurons": 100
    }
)
```

**Parameters:**
- `timestamp` (float): Spike train timestamp
- `spike_times` (np.ndarray | jnp.ndarray): Array of spike times
- `neuron_ids` (np.ndarray | jnp.ndarray): Array of neuron IDs
- `spike_amplitudes` (np.ndarray | jnp.ndarray, optional): Spike amplitudes
- `metadata` (dict, optional): Additional metadata

**Returns:** `str` - Spike train identifier

#### retrieve_spike_train()

Retrieve spike train data.

```python
spike_train = hdf5_store.retrieve_spike_train(train_id)
if spike_train:
    print(f"Spike train at {spike_train.timestamp}")
    print(f"Number of spikes: {len(spike_train.spike_times)}")
    print(f"Neuron IDs: {np.unique(spike_train.neuron_ids)}")
    if spike_train.spike_amplitudes is not None:
        print(f"Amplitude range: {np.min(spike_train.spike_amplitudes):.2f} - {np.max(spike_train.spike_amplitudes):.2f}")
```

**Parameters:**
- `train_id` (str): Spike train identifier

**Returns:** `SpikeTrainData | None` - SpikeTrainData object or None if not found

#### store_compressed_spike_train()

Store pre-compressed spike train data.

```python
# Example: Store compressed sparse spike data
compressed_data = compress_spike_data(spike_matrix)  # Custom compression
train_id = hdf5_store.store_compressed_spike_train(
    timestamp=time.time(),
    spike_data=compressed_data,
    compression_params={
        "original_shape": (10000, 1000),
        "compression_ratio": 10.0,
        "algorithm": "sparse_encoding"
    }
)
```

**Parameters:**
- `timestamp` (float): Spike train timestamp
- `spike_data` (np.ndarray | jnp.ndarray): Compressed spike data
- `compression_params` (dict): Compression parameters

**Returns:** `str` - Compressed spike train identifier

### Activation Data Operations

#### store_reservoir_states()

Store reservoir activation states.

```python
state_id = hdf5_store.store_reservoir_states(
    timestamp=time.time(),
    reservoir_states=np.random.rand(500, 100),  # 500 neurons, 100 time steps
    reservoir_id="main_reservoir"
)
```

**Parameters:**
- `timestamp` (float): State timestamp
- `reservoir_states` (np.ndarray | jnp.ndarray): Reservoir neuron states
- `reservoir_id` (str): Reservoir identifier

**Returns:** `str` - State identifier

#### retrieve_reservoir_states()

Retrieve reservoir states.

```python
result = hdf5_store.retrieve_reservoir_states(state_id)
if result:
    states, timestamp, reservoir_id = result
    print(f"Reservoir {reservoir_id} states at {timestamp}")
    print(f"States shape: {states.shape}")
```

**Returns:** `Tuple[np.ndarray, float, str] | None` - (states, timestamp, reservoir_id) or None

#### store_layer_outputs()

Store neural layer outputs from multiple layers.

```python
layer_outputs = {
    "input_layer": np.random.rand(32, 784),
    "hidden_layer_1": np.random.rand(32, 256),
    "hidden_layer_2": np.random.rand(32, 128),
    "output_layer": np.random.rand(32, 10)
}

output_id = hdf5_store.store_layer_outputs(
    timestamp=time.time(),
    layer_outputs=layer_outputs
)
```

**Parameters:**
- `timestamp` (float): Output timestamp
- `layer_outputs` (dict): Dictionary of layer_name -> output_array

**Returns:** `str` - Output identifier

#### retrieve_layer_outputs()

Retrieve layer outputs.

```python
result = hdf5_store.retrieve_layer_outputs(output_id)
if result:
    layer_outputs, timestamp = result
    for layer_name, output in layer_outputs.items():
        print(f"{layer_name}: {output.shape}")
```

**Returns:** `Tuple[Dict[str, np.ndarray], float] | None` - (layer_outputs, timestamp) or None

### Learning Trace Operations

#### store_plasticity_trace()

Store synaptic plasticity traces.

```python
trace_id = hdf5_store.store_plasticity_trace(
    timestamp=time.time(),
    synapse_changes=np.random.randn(1000, 1000) * 0.01,  # Small weight changes
    plasticity_type="stdp"
)
```

**Parameters:**
- `timestamp` (float): Trace timestamp
- `synapse_changes` (np.ndarray | jnp.ndarray): Array of synaptic weight changes
- `plasticity_type` (str): Type of plasticity rule

**Returns:** `str` - Trace identifier

#### retrieve_plasticity_trace()

Retrieve plasticity trace.

```python
result = hdf5_store.retrieve_plasticity_trace(trace_id)
if result:
    synapse_changes, timestamp, plasticity_type = result
    print(f"Plasticity trace ({plasticity_type}) at {timestamp}")
    print(f"Mean change: {np.mean(synapse_changes):.6f}")
    print(f"Std change: {np.std(synapse_changes):.6f}")
```

**Returns:** `Tuple[np.ndarray, float, str] | None` - (changes, timestamp, type) or None

#### store_homeostatic_trace()

Store homeostatic plasticity traces.

```python
trace_id = hdf5_store.store_homeostatic_trace(
    timestamp=time.time(),
    homeostatic_changes=np.random.randn(500) * 0.001,  # Threshold changes
    mechanism="intrinsic_plasticity"
)
```

**Parameters:**
- `timestamp` (float): Trace timestamp
- `homeostatic_changes` (np.ndarray | jnp.ndarray): Array of homeostatic changes
- `mechanism` (str): Homeostatic mechanism type

**Returns:** `str` - Trace identifier

#### store_performance_metrics()

Store performance metrics over time.

```python
metrics = {
    "accuracy": 0.85,
    "loss": 0.23,
    "learning_rate": 0.001,
    "convergence_time": 45.2
}

metrics_id = hdf5_store.store_performance_metrics(
    timestamp=time.time(),
    metrics=metrics,
    task_type="classification"
)
```

**Parameters:**
- `timestamp` (float): Metrics timestamp
- `metrics` (dict): Dictionary of metric_name -> value
- `task_type` (str): Type of task being measured

**Returns:** `str` - Metrics identifier

#### retrieve_performance_metrics()

Retrieve performance metrics.

```python
result = hdf5_store.retrieve_performance_metrics(metrics_id)
if result:
    metrics, timestamp, task_type = result
    print(f"Performance metrics for {task_type} at {timestamp}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")
```

**Returns:** `Tuple[Dict[str, float], float, str] | None` - (metrics, timestamp, task_type) or None

### File Management Operations

#### get_file_info()

Get comprehensive information about the HDF5 file.

```python
info = hdf5_store.get_file_info()
print(f"File path: {info['file_path']}")
print(f"File size: {info['file_size_mb']:.1f} MB")
print(f"Compression: {info['compression']}")
print(f"Total datasets: {info['total_datasets']}")
print(f"Groups: {info['groups']}")
print(f"Created at: {time.ctime(info['created_at'])}")
```

**Returns:** `dict` - File information dictionary

#### compact_file()

Compact the HDF5 file to reclaim space and optimize layout.

```python
success = hdf5_store.compact_file()
if success:
    print("File compacted successfully")
    
    # Check new file size
    info = hdf5_store.get_file_info()
    print(f"New file size: {info['file_size_mb']:.1f} MB")
```

**Returns:** `bool` - True if successful

**Note:** This operation creates a temporary file and replaces the original.

#### rotate_file()

Rotate the current file and create a new one.

```python
rotated_path = hdf5_store.rotate_file()
print(f"File rotated to: {rotated_path}")

# New file is automatically created and initialized
info = hdf5_store.get_file_info()
print(f"New file: {info['file_path']}")
```

**Returns:** `Path` - Path to the rotated file

#### check_file_size()

Check if file size exceeds maximum and rotate if needed.

```python
rotated = hdf5_store.check_file_size()
if rotated:
    print("File was rotated due to size limit")
```

**Returns:** `bool` - True if file was rotated

#### cleanup_old_data()

Remove data older than retention period.

```python
removed_count = hdf5_store.cleanup_old_data(retention_days=30)
print(f"Removed {removed_count} old data items")

# Custom retention for different data types
removed_count = hdf5_store.cleanup_old_data(retention_days=7)  # Keep only 1 week
```

**Parameters:**
- `retention_days` (int): Number of days to retain data

**Returns:** `int` - Number of items removed

## Data Models

### NetworkSnapshot

```python
@dataclass
class NetworkSnapshot:
    timestamp: float
    adjacency_matrix: np.ndarray
    weight_matrix: np.ndarray
    neuron_states: np.ndarray
    metadata: Dict[str, Any]
```

### SpikeTrainData

```python
@dataclass
class SpikeTrainData:
    timestamp: float
    spike_times: np.ndarray
    neuron_ids: np.ndarray
    spike_amplitudes: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
```

## Performance Optimization

### Chunking Strategy

```python
# Optimize chunk size based on access patterns
hdf5_store = HDF5Storage(
    chunk_size=2048,  # Larger chunks for sequential access
    compression="lzf"  # Faster compression for real-time data
)

# For random access patterns
hdf5_store = HDF5Storage(
    chunk_size=512,   # Smaller chunks for random access
    compression="gzip"  # Better compression for archival
)
```

### Compression Trade-offs

```python
# Fast compression for real-time data
real_time_store = HDF5Storage(
    compression="lzf",
    compression_opts=None  # No compression level for lzf
)

# High compression for archival data
archive_store = HDF5Storage(
    compression="gzip",
    compression_opts=9  # Maximum compression
)

# Balanced compression
balanced_store = HDF5Storage(
    compression="gzip",
    compression_opts=6  # Good balance of speed/ratio
)
```

### Batch Operations

```python
def store_multiple_snapshots(snapshots_data):
    """Store multiple network snapshots efficiently."""
    with hdf5_store._file_context('a') as f:
        for snapshot_data in snapshots_data:
            snapshot_id = hdf5_store.store_network_snapshot(**snapshot_data)
            
        # Flush once at the end
        f.flush()
```

## Error Handling

### HDF5 Availability

```python
from src.storage.hdf5_storage import HDF5StorageError, HDF5_AVAILABLE

if not HDF5_AVAILABLE:
    print("HDF5 not available, install with: pip install h5py")
    exit(1)

try:
    hdf5_store = HDF5Storage()
except HDF5StorageError as e:
    print(f"Failed to initialize HDF5 storage: {e}")
```

### File Corruption

```python
def verify_file_integrity():
    """Verify HDF5 file integrity."""
    try:
        info = hdf5_store.get_file_info()
        if info['total_datasets'] == 0:
            logger.warning("HDF5 file appears to be empty")
        return True
    except Exception as e:
        logger.error(f"HDF5 file corruption detected: {e}")
        return False

# Automatic recovery
if not verify_file_integrity():
    # Restore from backup or create new file
    backup_files = list(Path("./backups").glob("*.h5"))
    if backup_files:
        latest_backup = max(backup_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Restoring from backup: {latest_backup}")
        # Copy backup to main file location
```

## Best Practices

### 1. Organize Data Hierarchically

```python
def store_experiment_data(experiment_id, trial_data):
    """Store experiment data with proper organization."""
    base_timestamp = time.time()
    
    for trial_id, trial in enumerate(trial_data):
        # Store network snapshots
        snapshot_id = hdf5_store.store_network_snapshot(
            timestamp=base_timestamp + trial_id,
            adjacency_matrix=trial['network']['adjacency'],
            weight_matrix=trial['network']['weights'],
            neuron_states=trial['network']['states'],
            metadata={
                "experiment_id": experiment_id,
                "trial_id": trial_id,
                "condition": trial['condition']
            }
        )
        
        # Store corresponding spike trains
        for spike_train in trial['spike_trains']:
            train_id = hdf5_store.store_spike_train(
                timestamp=base_timestamp + trial_id + spike_train['offset'],
                spike_times=spike_train['times'],
                neuron_ids=spike_train['neurons'],
                metadata={
                    "experiment_id": experiment_id,
                    "trial_id": trial_id,
                    "snapshot_id": snapshot_id
                }
            )
```

### 2. Monitor File Size

```python
def setup_file_monitoring():
    """Set up automatic file size monitoring."""
    import threading
    
    def monitor():
        while True:
            # Check file size every hour
            time.sleep(3600)
            
            info = hdf5_store.get_file_info()
            if info['file_size_mb'] > 8000:  # > 8GB
                logger.warning(f"HDF5 file size: {info['file_size_mb']:.1f} MB")
                
                # Automatic rotation at 9GB
                if info['file_size_mb'] > 9000:
                    rotated_path = hdf5_store.rotate_file()
                    logger.info(f"File rotated to: {rotated_path}")
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
```

### 3. Efficient Data Access Patterns

```python
def analyze_network_evolution(start_time, end_time, sample_interval=60):
    """Analyze network evolution efficiently."""
    # Get snapshots in time range
    snapshot_ids = hdf5_store.get_network_snapshots_in_range(start_time, end_time)
    
    # Sample snapshots to reduce memory usage
    sampled_ids = snapshot_ids[::sample_interval]
    
    evolution_data = []
    for snapshot_id in sampled_ids:
        snapshot = hdf5_store.retrieve_network_snapshot(snapshot_id)
        if snapshot:
            # Extract key metrics without loading full matrices
            metrics = {
                'timestamp': snapshot.timestamp,
                'num_connections': np.sum(snapshot.adjacency_matrix),
                'mean_weight': np.mean(snapshot.weight_matrix),
                'active_neurons': np.sum(snapshot.neuron_states > 0.5)
            }
            evolution_data.append(metrics)
    
    return evolution_data
```

### 4. Data Lifecycle Management

```python
def implement_data_lifecycle():
    """Implement comprehensive data lifecycle management."""
    
    # Daily cleanup of old data
    def daily_cleanup():
        # Remove data older than 90 days
        removed = hdf5_store.cleanup_old_data(retention_days=90)
        logger.info(f"Daily cleanup: removed {removed} old items")
    
    # Weekly compaction
    def weekly_compaction():
        success = hdf5_store.compact_file()
        if success:
            info = hdf5_store.get_file_info()
            logger.info(f"Weekly compaction: file size now {info['file_size_mb']:.1f} MB")
    
    # Monthly archival
    def monthly_archival():
        # Rotate file for long-term storage
        if hdf5_store.get_file_info()['file_size_mb'] > 5000:  # > 5GB
            archive_path = hdf5_store.rotate_file()
            logger.info(f"Monthly archival: created {archive_path}")
    
    # Schedule tasks
    import schedule
    schedule.every().day.at("02:00").do(daily_cleanup)
    schedule.every().sunday.at("03:00").do(weekly_compaction)
    schedule.every().month.do(monthly_archival)
```

## Integration Examples

### With Liquid State Machine

```python
from src.core.liquid_state_machine import LiquidStateMachine

class PersistentLSM(LiquidStateMachine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hdf5_store = HDF5Storage()
        self.snapshot_interval = 100  # Snapshot every 100 steps
        self.step_count = 0
    
    def step(self, input_data):
        # Normal LSM step
        output = super().step(input_data)
        self.step_count += 1
        
        # Periodic snapshots
        if self.step_count % self.snapshot_interval == 0:
            self.save_snapshot()
        
        return output
    
    def save_snapshot(self):
        """Save current network state."""
        snapshot_id = self.hdf5_store.store_network_snapshot(
            timestamp=time.time(),
            adjacency_matrix=self.adjacency_matrix,
            weight_matrix=self.weight_matrix,
            neuron_states=self.neuron_states,
            metadata={
                "step_count": self.step_count,
                "network_type": "liquid_state_machine"
            }
        )
        return snapshot_id
    
    def restore_snapshot(self, snapshot_id):
        """Restore network state from snapshot."""
        snapshot = self.hdf5_store.retrieve_network_snapshot(snapshot_id)
        if snapshot:
            self.adjacency_matrix = snapshot.adjacency_matrix
            self.weight_matrix = snapshot.weight_matrix
            self.neuron_states = snapshot.neuron_states
            self.step_count = snapshot.metadata.get('step_count', 0)
            return True
        return False
```

### With Spike Processing Pipeline

```python
def process_spike_data_pipeline(spike_data_stream):
    """Process continuous spike data with HDF5 storage."""
    
    batch_size = 1000
    spike_batch = []
    
    for spike_event in spike_data_stream:
        spike_batch.append(spike_event)
        
        if len(spike_batch) >= batch_size:
            # Process batch
            processed_spikes = process_spike_batch(spike_batch)
            
            # Store in HDF5
            train_id = hdf5_store.store_spike_train(
                timestamp=time.time(),
                spike_times=processed_spikes['times'],
                neuron_ids=processed_spikes['neurons'],
                spike_amplitudes=processed_spikes['amplitudes'],
                metadata={
                    "batch_size": batch_size,
                    "processing_version": "1.0"
                }
            )
            
            # Clear batch
            spike_batch = []
            
            # Optional: Store analysis results
            if 'patterns' in processed_spikes:
                metrics = {
                    'pattern_count': len(processed_spikes['patterns']),
                    'synchrony_index': processed_spikes['synchrony'],
                    'burst_rate': processed_spikes['burst_rate']
                }
                
                hdf5_store.store_performance_metrics(
                    timestamp=time.time(),
                    metrics=metrics,
                    task_type="spike_analysis"
                )
```
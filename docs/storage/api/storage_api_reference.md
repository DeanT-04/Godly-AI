# Storage API Reference

Complete API reference for all storage systems in the Godly AI System.

## Module Structure

```
src.storage/
├── __init__.py              # Main storage module
├── redis_storage.py         # Redis storage implementation
├── sqlite_storage.py        # SQLite storage implementation
└── hdf5_storage.py         # HDF5 storage implementation
```

## Import Structure

```python
# Import all available storage classes
from src.storage import RedisStorage, SQLiteStorage, HDF5Storage

# Import specific storage class
from src.storage.redis_storage import RedisStorage
from src.storage.sqlite_storage import SQLiteStorage
from src.storage.hdf5_storage import HDF5Storage

# Import exceptions
from src.storage.redis_storage import RedisConnectionError
from src.storage.sqlite_storage import SQLiteStorageError
from src.storage.hdf5_storage import HDF5StorageError
```

## Redis Storage API

### Class: RedisStorage

High-performance storage for real-time memory operations with automatic expiration.

#### Constructor

```python
RedisStorage(
    host: str = 'localhost',
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    max_connections: int = 10,
    compression_threshold: int = 1024,
    retry_attempts: int = 3,
    retry_delay: float = 0.1
)
```

#### Methods

##### Working Memory Operations

**store_working_memory_pattern()**
```python
store_working_memory_pattern(
    pattern_hash: str,
    pattern: Union[np.ndarray, jnp.ndarray],
    timestamp: float,
    metadata: Optional[Dict[str, Any]] = None
) -> bool
```
Store working memory pattern with compression and 1-hour TTL.

**retrieve_working_memory_pattern()**
```python
retrieve_working_memory_pattern(
    pattern_hash: str
) -> Optional[Tuple[np.ndarray, float, Dict[str, Any]]]
```
Retrieve working memory pattern by hash.

**update_working_memory_access()**
```python
update_working_memory_access(pattern_hash: str) -> bool
```
Update access timestamp for pattern.

##### Spike Train Operations

**cache_spike_train()**
```python
cache_spike_train(
    timestamp: float,
    spike_data: Union[np.ndarray, jnp.ndarray],
    neuron_ids: Optional[List[int]] = None,
    ttl: int = 300
) -> bool
```
Cache spike train data with configurable TTL.

**retrieve_spike_train()**
```python
retrieve_spike_train(
    timestamp: float
) -> Optional[Tuple[np.ndarray, List[int]]]
```
Retrieve cached spike train data.

**get_recent_spike_trains()**
```python
get_recent_spike_trains(
    time_window: float = 1.0
) -> List[Tuple[float, np.ndarray, List[int]]]
```
Get all spike trains within time window.

##### Attention Weight Operations

**store_attention_weights()**
```python
store_attention_weights(
    context: str,
    weights: Union[np.ndarray, jnp.ndarray],
    timestamp: float
) -> bool
```
Store attention weights with 10-minute TTL.

**retrieve_attention_weights()**
```python
retrieve_attention_weights(
    context: str
) -> Optional[Tuple[np.ndarray, float]]
```
Retrieve attention weights for context.

##### Utility Operations

**get_memory_usage()**
```python
get_memory_usage() -> Dict[str, int]
```
Get memory usage statistics by data type.

**clear_expired_data()**
```python
clear_expired_data() -> int
```
Manually clear expired data, returns count.

**flush_all_data()**
```python
flush_all_data() -> bool
```
Flush all data from Redis database.

**close()**
```python
close() -> None
```
Close Redis connection pool.

## SQLite Storage API

### Class: SQLiteStorage

Persistent storage for structured data with ACID transactions and migrations.

#### Constructor

```python
SQLiteStorage(
    db_path: Union[str, Path] = "godly_ai.db",
    backup_interval: int = 3600,
    max_backups: int = 10,
    enable_wal: bool = True,
    connection_timeout: float = 30.0
)
```

#### Data Classes

**Episode**
```python
@dataclass
class Episode:
    id: Optional[int] = None
    timestamp: float = 0.0
    experience_data: bytes = b''
    performance_score: float = 0.0
    context_hash: str = ''
    metadata: Dict[str, Any] = None
```

**Concept**
```python
@dataclass
class Concept:
    id: Optional[int] = None
    name: str = ''
    embedding: bytes = b''
    creation_time: float = 0.0
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: Dict[str, Any] = None
```

**LearningEvent**
```python
@dataclass
class LearningEvent:
    id: Optional[int] = None
    task_type: str = ''
    performance_delta: float = 0.0
    strategy_used: str = ''
    timestamp: float = 0.0
    parameters: Dict[str, Any] = None
```

#### Methods

##### Episode Operations

**store_episode()**
```python
store_episode(
    experience_data: Union[np.ndarray, jnp.ndarray, Dict[str, Any]],
    performance_score: float,
    context_hash: str,
    timestamp: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> int
```
Store episodic memory entry, returns episode ID.

**retrieve_episode()**
```python
retrieve_episode(episode_id: int) -> Optional[Episode]
```
Retrieve episode by ID.

**get_episodes_by_context()**
```python
get_episodes_by_context(
    context_hash: str,
    limit: int = 100,
    min_score: Optional[float] = None
) -> List[Episode]
```
Get episodes by context with optional filtering.

**get_recent_episodes()**
```python
get_recent_episodes(
    time_window: float = 3600.0,
    limit: int = 100
) -> List[Episode]
```
Get recent episodes within time window.

##### Concept Operations

**store_concept()**
```python
store_concept(
    name: str,
    embedding: Union[np.ndarray, jnp.ndarray],
    metadata: Optional[Dict[str, Any]] = None,
    creation_time: Optional[float] = None
) -> int
```
Store semantic concept, returns concept ID.

**retrieve_concept()**
```python
retrieve_concept(concept_id: int) -> Optional[Concept]
```
Retrieve concept by ID.

**retrieve_concept_by_name()**
```python
retrieve_concept_by_name(name: str) -> Optional[Concept]
```
Retrieve concept by name (updates access count).

**search_concepts()**
```python
search_concepts(
    query: str,
    limit: int = 10
) -> List[Concept]
```
Search concepts by name pattern.

**add_concept_relationship()**
```python
add_concept_relationship(
    source_id: int,
    target_id: int,
    relationship_type: str,
    strength: float = 1.0
) -> bool
```
Add relationship between concepts.

**get_concept_relationships()**
```python
get_concept_relationships(
    concept_id: int,
    relationship_type: Optional[str] = None
) -> List[Tuple[int, str, float]]
```
Get relationships for concept.

##### Learning Event Operations

**store_learning_event()**
```python
store_learning_event(
    task_type: str,
    performance_delta: float,
    strategy_used: str,
    parameters: Optional[Dict[str, Any]] = None,
    timestamp: Optional[float] = None
) -> int
```
Store learning event, returns event ID.

**get_learning_events_by_task()**
```python
get_learning_events_by_task(
    task_type: str,
    limit: int = 100,
    min_performance_delta: Optional[float] = None
) -> List[LearningEvent]
```
Get learning events by task type.

**get_best_strategies()**
```python
get_best_strategies(
    task_type: str,
    limit: int = 5
) -> List[Tuple[str, float, int]]
```
Get best performing strategies for task.

##### Database Management

**create_backup()**
```python
create_backup(
    backup_path: Optional[Union[str, Path]] = None
) -> Path
```
Create database backup, returns backup path.

**vacuum_database()**
```python
vacuum_database() -> bool
```
Vacuum database to reclaim space.

**get_database_stats()**
```python
get_database_stats() -> Dict[str, Any]
```
Get database statistics and information.

**auto_backup_if_needed()**
```python
auto_backup_if_needed() -> Optional[Path]
```
Create backup if interval elapsed.

**close()**
```python
close() -> None
```
Close database connections.

## HDF5 Storage API

### Class: HDF5Storage

Efficient storage for large-scale scientific data with compression and chunking.

#### Constructor

```python
HDF5Storage(
    file_path: Union[str, Path] = "godly_ai_data.h5",
    compression: str = "gzip",
    compression_opts: int = 6,
    chunk_size: int = 1024,
    max_file_size: int = 10 * 1024**3,
    enable_swmr: bool = True
)
```

#### Data Classes

**NetworkSnapshot**
```python
@dataclass
class NetworkSnapshot:
    timestamp: float
    adjacency_matrix: np.ndarray
    weight_matrix: np.ndarray
    neuron_states: np.ndarray
    metadata: Dict[str, Any]
```

**SpikeTrainData**
```python
@dataclass
class SpikeTrainData:
    timestamp: float
    spike_times: np.ndarray
    neuron_ids: np.ndarray
    spike_amplitudes: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
```

#### Methods

##### Network State Operations

**store_network_snapshot()**
```python
store_network_snapshot(
    timestamp: float,
    adjacency_matrix: Union[np.ndarray, jnp.ndarray],
    weight_matrix: Union[np.ndarray, jnp.ndarray],
    neuron_states: Union[np.ndarray, jnp.ndarray],
    metadata: Optional[Dict[str, Any]] = None
) -> str
```
Store complete network state snapshot.

**retrieve_network_snapshot()**
```python
retrieve_network_snapshot(snapshot_id: str) -> Optional[NetworkSnapshot]
```
Retrieve network snapshot by ID.

**get_network_snapshots_in_range()**
```python
get_network_snapshots_in_range(
    start_time: float,
    end_time: float
) -> List[str]
```
Get snapshot IDs within time range.

##### Spike Train Operations

**store_spike_train()**
```python
store_spike_train(
    timestamp: float,
    spike_times: Union[np.ndarray, jnp.ndarray],
    neuron_ids: Union[np.ndarray, jnp.ndarray],
    spike_amplitudes: Optional[Union[np.ndarray, jnp.ndarray]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str
```
Store spike train data with compression.

**retrieve_spike_train()**
```python
retrieve_spike_train(train_id: str) -> Optional[SpikeTrainData]
```
Retrieve spike train data by ID.

**store_compressed_spike_train()**
```python
store_compressed_spike_train(
    timestamp: float,
    spike_data: Union[np.ndarray, jnp.ndarray],
    compression_params: Dict[str, Any]
) -> str
```
Store pre-compressed spike train data.

##### Activation Data Operations

**store_reservoir_states()**
```python
store_reservoir_states(
    timestamp: float,
    reservoir_states: Union[np.ndarray, jnp.ndarray],
    reservoir_id: str = "default"
) -> str
```
Store reservoir activation states.

**retrieve_reservoir_states()**
```python
retrieve_reservoir_states(
    state_id: str
) -> Optional[Tuple[np.ndarray, float, str]]
```
Retrieve reservoir states.

**store_layer_outputs()**
```python
store_layer_outputs(
    timestamp: float,
    layer_outputs: Dict[str, Union[np.ndarray, jnp.ndarray]]
) -> str
```
Store neural layer outputs.

**retrieve_layer_outputs()**
```python
retrieve_layer_outputs(
    output_id: str
) -> Optional[Tuple[Dict[str, np.ndarray], float]]
```
Retrieve layer outputs.

##### Learning Trace Operations

**store_plasticity_trace()**
```python
store_plasticity_trace(
    timestamp: float,
    synapse_changes: Union[np.ndarray, jnp.ndarray],
    plasticity_type: str = "stdp"
) -> str
```
Store synaptic plasticity traces.

**retrieve_plasticity_trace()**
```python
retrieve_plasticity_trace(
    trace_id: str
) -> Optional[Tuple[np.ndarray, float, str]]
```
Retrieve plasticity trace.

**store_homeostatic_trace()**
```python
store_homeostatic_trace(
    timestamp: float,
    homeostatic_changes: Union[np.ndarray, jnp.ndarray],
    mechanism: str = "intrinsic_plasticity"
) -> str
```
Store homeostatic plasticity traces.

**store_performance_metrics()**
```python
store_performance_metrics(
    timestamp: float,
    metrics: Dict[str, float],
    task_type: str = "general"
) -> str
```
Store performance metrics.

**retrieve_performance_metrics()**
```python
retrieve_performance_metrics(
    metrics_id: str
) -> Optional[Tuple[Dict[str, float], float, str]]
```
Retrieve performance metrics.

##### File Management Operations

**get_file_info()**
```python
get_file_info() -> Dict[str, Any]
```
Get comprehensive file information.

**compact_file()**
```python
compact_file() -> bool
```
Compact file to reclaim space.

**rotate_file()**
```python
rotate_file() -> Path
```
Rotate current file and create new one.

**check_file_size()**
```python
check_file_size() -> bool
```
Check size and rotate if needed.

**cleanup_old_data()**
```python
cleanup_old_data(retention_days: int = 30) -> int
```
Remove data older than retention period.

**close()**
```python
close() -> None
```
Close HDF5 file handles.

## Exception Classes

### RedisConnectionError

```python
class RedisConnectionError(Exception):
    """Raised when Redis connection fails."""
    pass
```

### SQLiteStorageError

```python
class SQLiteStorageError(Exception):
    """Raised when SQLite storage operations fail."""
    pass
```

### HDF5StorageError

```python
class HDF5StorageError(Exception):
    """Raised when HDF5 storage operations fail."""
    pass
```

## Context Manager Support

All storage classes support context manager protocol:

```python
# Redis Storage
with RedisStorage() as redis_store:
    redis_store.store_working_memory_pattern(...)

# SQLite Storage
with SQLiteStorage() as sqlite_store:
    episode_id = sqlite_store.store_episode(...)

# HDF5 Storage
with HDF5Storage() as hdf5_store:
    snapshot_id = hdf5_store.store_network_snapshot(...)
```

## Type Hints

All methods include comprehensive type hints for better IDE support and type checking:

```python
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import jax.numpy as jnp
from pathlib import Path
```

## Thread Safety

- **RedisStorage**: Thread-safe with connection pooling
- **SQLiteStorage**: Thread-safe with thread-local connections
- **HDF5Storage**: Thread-safe with thread-local file handles

## Performance Characteristics

| Operation | Redis | SQLite | HDF5 |
|-----------|-------|--------|------|
| Small data (<1KB) | Excellent | Good | Fair |
| Medium data (1KB-1MB) | Good | Good | Good |
| Large data (>1MB) | Fair | Poor | Excellent |
| Query flexibility | Limited | Excellent | Limited |
| Persistence | TTL-based | Permanent | Permanent |
| Compression | Automatic | None | Configurable |
| Concurrent access | Excellent | Good | Good |

## Memory Usage

- **Redis**: In-memory with optional persistence
- **SQLite**: Disk-based with memory caching
- **HDF5**: Disk-based with chunked access

## Best Practices

1. **Use appropriate storage for data type**:
   - Redis: Temporary, frequently accessed data
   - SQLite: Structured, queryable data
   - HDF5: Large arrays and scientific data

2. **Handle exceptions gracefully**:
   ```python
   try:
       result = storage.operation()
   except StorageError as e:
       logger.error(f"Storage operation failed: {e}")
       # Implement fallback strategy
   ```

3. **Use context managers for automatic cleanup**:
   ```python
   with StorageClass() as storage:
       # Operations here
       pass
   # Automatic cleanup
   ```

4. **Monitor resource usage**:
   ```python
   # Check Redis memory
   stats = redis_store.get_memory_usage()
   
   # Check SQLite database size
   db_stats = sqlite_store.get_database_stats()
   
   # Check HDF5 file size
   file_info = hdf5_store.get_file_info()
   ```

5. **Implement regular maintenance**:
   ```python
   # Daily maintenance routine
   redis_store.clear_expired_data()
   sqlite_store.auto_backup_if_needed()
   hdf5_store.cleanup_old_data(retention_days=30)
   ```
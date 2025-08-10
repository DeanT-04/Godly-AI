# Storage and Persistence Systems

The Godly AI System implements a comprehensive multi-tier storage architecture designed to handle different types of data with varying persistence requirements, access patterns, and performance needs.

## Architecture Overview

The storage system consists of three main components:

1. **Redis Storage** - Real-time, high-frequency data with TTL-based expiration
2. **SQLite Storage** - Structured persistent data with relational capabilities
3. **HDF5 Storage** - Large-scale scientific data with compression and chunking

## Storage Hierarchy

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis Storage │    │  SQLite Storage │    │   HDF5 Storage  │
│                 │    │                 │    │                 │
│ • Working Memory│    │ • Episodes      │    │ • Network States│
│ • Spike Caches  │    │ • Concepts      │    │ • Spike Trains  │
│ • Attention     │    │ • Learning Events│   │ • Activations   │
│ • Real-time     │    │ • Relationships │    │ • Learning Traces│
│                 │    │                 │    │                 │
│ TTL: Minutes    │    │ Persistent      │    │ Long-term Archive│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Installation

```bash
# Install required dependencies
pip install redis sqlite3 h5py numpy jax

# Optional: Install Redis server
# On Ubuntu: sudo apt-get install redis-server
# On macOS: brew install redis
# On Windows: Download from https://redis.io/download
```

### Basic Usage

```python
from src.storage import RedisStorage, SQLiteStorage, HDF5Storage
import numpy as np
import time

# Initialize storage systems
redis_store = RedisStorage(host='localhost', port=6379)
sqlite_store = SQLiteStorage(db_path='godly_ai.db')
hdf5_store = HDF5Storage(file_path='godly_ai_data.h5')

# Store working memory pattern in Redis
pattern = np.random.rand(100, 50)
redis_store.store_working_memory_pattern(
    pattern_hash="pattern_123",
    pattern=pattern,
    timestamp=time.time(),
    metadata={"source": "visual_cortex"}
)

# Store episode in SQLite
episode_id = sqlite_store.store_episode(
    experience_data={"observation": [1, 2, 3], "action": "move_forward"},
    performance_score=0.85,
    context_hash="navigation_task"
)

# Store network snapshot in HDF5
snapshot_id = hdf5_store.store_network_snapshot(
    timestamp=time.time(),
    adjacency_matrix=np.random.randint(0, 2, (1000, 1000)),
    weight_matrix=np.random.rand(1000, 1000),
    neuron_states=np.random.rand(1000, 5)
)

# Clean up
redis_store.close()
sqlite_store.close()
hdf5_store.close()
```

## Storage Components

### [Redis Storage](redis_storage.md)
- **Purpose**: Real-time memory operations with automatic expiration
- **Use Cases**: Working memory patterns, spike train caching, attention weights
- **Features**: Compression, connection pooling, retry mechanisms
- **Performance**: Optimized for high-frequency read/write operations

### [SQLite Storage](sqlite_storage.md)
- **Purpose**: Structured persistent data with relational capabilities
- **Use Cases**: Episodes, concepts, learning events, knowledge graphs
- **Features**: ACID transactions, migrations, backup/restore
- **Performance**: Optimized with indexing and connection management

### [HDF5 Storage](hdf5_storage.md)
- **Purpose**: Large-scale scientific data with efficient compression
- **Use Cases**: Network states, spike trains, activation data, learning traces
- **Features**: Chunking, compression, file rotation, cleanup
- **Performance**: Optimized for large array operations and scientific computing

## Data Flow Patterns

### Typical Data Lifecycle

1. **Real-time Processing** → Redis Storage (working memory, attention)
2. **Episode Completion** → SQLite Storage (structured episode data)
3. **Batch Processing** → HDF5 Storage (large-scale analysis data)
4. **Long-term Archive** → HDF5 Storage (compressed historical data)

### Cross-Storage Operations

```python
# Example: Moving data through storage tiers
def process_learning_episode(experience_data, performance_score):
    # 1. Store in Redis for immediate access
    pattern_hash = redis_store.store_working_memory_pattern(
        pattern_hash=f"episode_{time.time()}",
        pattern=experience_data['neural_activity'],
        timestamp=time.time()
    )
    
    # 2. Store structured data in SQLite
    episode_id = sqlite_store.store_episode(
        experience_data=experience_data,
        performance_score=performance_score,
        context_hash=experience_data['context']
    )
    
    # 3. Archive large-scale data in HDF5
    if len(experience_data['spike_trains']) > 10000:
        train_id = hdf5_store.store_spike_train(
            timestamp=time.time(),
            spike_times=experience_data['spike_trains']['times'],
            neuron_ids=experience_data['spike_trains']['neurons']
        )
    
    return episode_id
```

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_password

# SQLite Configuration
SQLITE_DB_PATH=./data/godly_ai.db
SQLITE_BACKUP_INTERVAL=3600
SQLITE_MAX_BACKUPS=10

# HDF5 Configuration
HDF5_FILE_PATH=./data/godly_ai_data.h5
HDF5_COMPRESSION=gzip
HDF5_COMPRESSION_LEVEL=6
HDF5_MAX_FILE_SIZE=10737418240  # 10GB
```

### Configuration Files

```python
# config/storage.yaml
redis:
  host: localhost
  port: 6379
  db: 0
  max_connections: 10
  compression_threshold: 1024

sqlite:
  db_path: "./data/godly_ai.db"
  backup_interval: 3600
  max_backups: 10
  enable_wal: true

hdf5:
  file_path: "./data/godly_ai_data.h5"
  compression: "gzip"
  compression_opts: 6
  chunk_size: 1024
  max_file_size: 10737418240
```

## Performance Considerations

### Redis Storage
- **Memory Usage**: Monitor Redis memory usage and configure appropriate eviction policies
- **Network Latency**: Use connection pooling for high-frequency operations
- **Data Size**: Enable compression for large patterns (>1KB)

### SQLite Storage
- **Concurrent Access**: Use WAL mode for better concurrency
- **Index Optimization**: Ensure proper indexing on frequently queried columns
- **Backup Strategy**: Regular backups with automatic cleanup

### HDF5 Storage
- **Chunk Size**: Optimize chunk size based on access patterns
- **Compression**: Balance compression ratio vs. access speed
- **File Size**: Implement file rotation for very large datasets

## Monitoring and Maintenance

### Health Checks

```python
def check_storage_health():
    """Check health of all storage systems."""
    health_status = {}
    
    # Redis health
    try:
        redis_store.redis_client.ping()
        health_status['redis'] = 'healthy'
    except Exception as e:
        health_status['redis'] = f'error: {e}'
    
    # SQLite health
    try:
        stats = sqlite_store.get_database_stats()
        health_status['sqlite'] = f"healthy ({stats['total_episodes']} episodes)"
    except Exception as e:
        health_status['sqlite'] = f'error: {e}'
    
    # HDF5 health
    try:
        info = hdf5_store.get_file_info()
        health_status['hdf5'] = f"healthy ({info['file_size_mb']:.1f} MB)"
    except Exception as e:
        health_status['hdf5'] = f'error: {e}'
    
    return health_status
```

### Maintenance Tasks

```python
def daily_maintenance():
    """Perform daily maintenance tasks."""
    # Clean up expired Redis data
    redis_store.clear_expired_data()
    
    # Backup SQLite database
    sqlite_store.auto_backup_if_needed()
    
    # Clean up old HDF5 data (30 days retention)
    hdf5_store.cleanup_old_data(retention_days=30)
    
    # Compact HDF5 file if needed
    if hdf5_store.check_file_size():
        hdf5_store.compact_file()
```

## Error Handling

### Common Error Patterns

```python
from src.storage import RedisConnectionError, SQLiteStorageError, HDF5StorageError

def robust_storage_operation():
    """Example of robust error handling."""
    try:
        # Attempt Redis operation
        result = redis_store.store_working_memory_pattern(...)
        return result
    except RedisConnectionError:
        # Fallback to SQLite for critical data
        logger.warning("Redis unavailable, falling back to SQLite")
        return sqlite_store.store_episode(...)
    except SQLiteStorageError:
        # Final fallback to HDF5
        logger.error("SQLite failed, using HDF5 as last resort")
        return hdf5_store.store_network_snapshot(...)
    except HDF5StorageError:
        # All storage systems failed
        logger.critical("All storage systems failed!")
        raise
```

## Testing

### Unit Tests

```bash
# Run all storage tests
python -m pytest tests/test_*_storage.py -v

# Run specific storage system tests
python -m pytest tests/test_redis_storage.py -v
python -m pytest tests/test_sqlite_storage.py -v
python -m pytest tests/test_hdf5_storage.py -v

# Run integration tests
python -m pytest tests/test_*_storage.py -m integration -v
```

### Performance Tests

```bash
# Run performance benchmarks
python -m pytest tests/test_*_storage.py -m performance -v

# Generate performance reports
python scripts/benchmark_storage.py --output reports/storage_performance.html
```

## Troubleshooting

### Common Issues

1. **Redis Connection Errors**
   - Check Redis server status: `redis-cli ping`
   - Verify connection parameters
   - Check firewall settings

2. **SQLite Lock Errors**
   - Enable WAL mode: `PRAGMA journal_mode=WAL`
   - Check file permissions
   - Ensure proper connection cleanup

3. **HDF5 File Corruption**
   - Use `h5fsck` to check file integrity
   - Restore from backup if available
   - Enable checksums: `fletcher32=True`

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for storage systems
logger = logging.getLogger('src.storage')
logger.setLevel(logging.DEBUG)
```

## API Reference

- [Redis Storage API](api/redis_storage.md)
- [SQLite Storage API](api/sqlite_storage.md)
- [HDF5 Storage API](api/hdf5_storage.md)

## Examples

- [Basic Usage Examples](examples/basic_usage.md)
- [Advanced Patterns](examples/advanced_patterns.md)
- [Performance Optimization](examples/performance_optimization.md)
- [Data Migration](examples/data_migration.md)
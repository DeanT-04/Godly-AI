# Redis Storage

Redis Storage provides high-performance, real-time memory operations with automatic expiration and compression capabilities. It's designed for frequently accessed data that doesn't require long-term persistence.

## Overview

Redis Storage is optimized for:
- Working memory patterns with TTL-based expiration
- Spike train caching for high-frequency neural data
- Attention weight distributions
- Real-time data that needs fast access

## Features

- **Automatic Compression**: Data larger than threshold is automatically compressed
- **Connection Pooling**: Thread-safe connection management with retry logic
- **TTL Management**: Automatic expiration of data based on type
- **Error Recovery**: Robust error handling with exponential backoff
- **Memory Monitoring**: Built-in memory usage statistics

## Configuration

### Basic Configuration

```python
from src.storage import RedisStorage

# Basic setup
redis_store = RedisStorage(
    host='localhost',
    port=6379,
    db=0,
    password=None
)

# Advanced configuration
redis_store = RedisStorage(
    host='redis.example.com',
    port=6379,
    db=1,
    password='secure_password',
    max_connections=20,
    compression_threshold=2048,  # Compress data > 2KB
    retry_attempts=5,
    retry_delay=0.2
)
```

### Connection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | 'localhost' | Redis server hostname |
| `port` | int | 6379 | Redis server port |
| `db` | int | 0 | Redis database number |
| `password` | str | None | Redis password |
| `max_connections` | int | 10 | Connection pool size |
| `compression_threshold` | int | 1024 | Min size for compression (bytes) |
| `retry_attempts` | int | 3 | Number of retry attempts |
| `retry_delay` | float | 0.1 | Initial retry delay (seconds) |

## API Reference

### Working Memory Operations

#### store_working_memory_pattern()

Store a working memory pattern with automatic compression and TTL.

```python
success = redis_store.store_working_memory_pattern(
    pattern_hash="visual_pattern_123",
    pattern=np.random.rand(100, 50),
    timestamp=time.time(),
    metadata={"source": "visual_cortex", "confidence": 0.95}
)
```

**Parameters:**
- `pattern_hash` (str): Unique identifier for the pattern
- `pattern` (np.ndarray | jnp.ndarray): Neural pattern data
- `timestamp` (float): Pattern creation timestamp
- `metadata` (dict, optional): Additional metadata

**Returns:** `bool` - True if stored successfully

**TTL:** 1 hour (3600 seconds)

#### retrieve_working_memory_pattern()

Retrieve a working memory pattern by hash.

```python
result = redis_store.retrieve_working_memory_pattern("visual_pattern_123")
if result:
    pattern, timestamp, metadata = result
    print(f"Pattern shape: {pattern.shape}")
    print(f"Created at: {timestamp}")
    print(f"Metadata: {metadata}")
```

**Parameters:**
- `pattern_hash` (str): Pattern identifier

**Returns:** `tuple | None` - (pattern, timestamp, metadata) or None if not found

#### update_working_memory_access()

Update the last accessed timestamp for a pattern.

```python
success = redis_store.update_working_memory_access("visual_pattern_123")
```

### Spike Train Operations

#### cache_spike_train()

Cache spike train data for high-frequency access.

```python
success = redis_store.cache_spike_train(
    timestamp=time.time(),
    spike_data=np.random.randint(0, 2, (1000, 100)),
    neuron_ids=list(range(100)),
    ttl=300  # 5 minutes
)
```

**Parameters:**
- `timestamp` (float): Spike train timestamp
- `spike_data` (np.ndarray | jnp.ndarray): Spike train array
- `neuron_ids` (list, optional): List of neuron IDs
- `ttl` (int): Time to live in seconds

**Returns:** `bool` - True if cached successfully

#### retrieve_spike_train()

Retrieve cached spike train data.

```python
result = redis_store.retrieve_spike_train(timestamp)
if result:
    spike_data, neuron_ids = result
```

**Returns:** `tuple | None` - (spike_data, neuron_ids) or None if not found

#### get_recent_spike_trains()

Get all spike trains within a time window.

```python
recent_trains = redis_store.get_recent_spike_trains(time_window=1.0)
for timestamp, spike_data, neuron_ids in recent_trains:
    print(f"Train at {timestamp}: {len(neuron_ids)} neurons")
```

**Parameters:**
- `time_window` (float): Time window in seconds from current time

**Returns:** `list` - List of (timestamp, spike_data, neuron_ids) tuples

### Attention Weight Operations

#### store_attention_weights()

Store attention weight distribution.

```python
weights = np.random.rand(256)
weights = weights / np.sum(weights)  # Normalize

success = redis_store.store_attention_weights(
    context="visual_attention",
    weights=weights,
    timestamp=time.time()
)
```

**Parameters:**
- `context` (str): Context identifier
- `weights` (np.ndarray | jnp.ndarray): Attention weight array
- `timestamp` (float): Weight computation timestamp

**Returns:** `bool` - True if stored successfully

**TTL:** 10 minutes (600 seconds)

#### retrieve_attention_weights()

Retrieve attention weights for a context.

```python
result = redis_store.retrieve_attention_weights("visual_attention")
if result:
    weights, timestamp = result
    print(f"Attention sum: {np.sum(weights)}")
```

**Returns:** `tuple | None` - (weights, timestamp) or None if not found

### Utility Operations

#### get_memory_usage()

Get memory usage statistics by data type.

```python
stats = redis_store.get_memory_usage()
print(f"Working memory patterns: {stats['working_memory']}")
print(f"Spike trains: {stats['spike_trains']}")
print(f"Attention weights: {stats['attention_weights']}")
print(f"Total items: {stats['total']}")
```

**Returns:** `dict` - Memory usage statistics

#### clear_expired_data()

Manually clear expired data and return count.

```python
cleared_count = redis_store.clear_expired_data()
print(f"Cleared {cleared_count} expired items")
```

**Returns:** `int` - Number of items cleared

#### flush_all_data()

Flush all data from the Redis database.

```python
success = redis_store.flush_all_data()
```

**Warning:** This operation is irreversible!

## Data Compression

Redis Storage automatically compresses data that exceeds the compression threshold:

```python
# Data smaller than threshold (1024 bytes) - not compressed
small_pattern = np.random.rand(10, 10)  # ~800 bytes
redis_store.store_working_memory_pattern("small", small_pattern, time.time())

# Data larger than threshold - automatically compressed
large_pattern = np.random.rand(100, 100)  # ~80KB
redis_store.store_working_memory_pattern("large", large_pattern, time.time())
```

### Compression Settings

```python
# Configure compression threshold
redis_store = RedisStorage(
    compression_threshold=2048,  # Compress data > 2KB
    # Uses zlib compression with default level
)
```

## Error Handling

### Connection Errors

```python
from src.storage.redis_storage import RedisConnectionError

try:
    redis_store = RedisStorage(host='nonexistent-host')
except RedisConnectionError as e:
    print(f"Failed to connect to Redis: {e}")
```

### Retry Mechanism

Redis Storage includes automatic retry with exponential backoff:

```python
# Configure retry behavior
redis_store = RedisStorage(
    retry_attempts=5,      # Try up to 5 times
    retry_delay=0.1       # Start with 0.1s delay
)

# Retry delays: 0.1s, 0.2s, 0.4s, 0.8s, 1.6s
```

### Graceful Degradation

```python
def store_with_fallback(pattern_hash, pattern, timestamp):
    """Store pattern with fallback handling."""
    try:
        return redis_store.store_working_memory_pattern(
            pattern_hash, pattern, timestamp
        )
    except RedisConnectionError:
        # Fallback to local cache or other storage
        logger.warning("Redis unavailable, using fallback storage")
        return fallback_store.store_pattern(pattern_hash, pattern)
```

## Performance Optimization

### Connection Pooling

```python
# Optimize for high-concurrency scenarios
redis_store = RedisStorage(
    max_connections=50,    # Larger pool for more concurrent operations
    retry_attempts=3,      # Fewer retries for faster failure detection
    retry_delay=0.05      # Shorter delays for high-frequency operations
)
```

### Batch Operations

```python
def store_multiple_patterns(patterns):
    """Store multiple patterns efficiently."""
    pipeline = redis_store.redis_client.pipeline()
    
    for pattern_hash, pattern_data in patterns.items():
        # Use pipeline for batch operations
        serialized = redis_store._serialize_array(pattern_data)
        pipeline.hset(f"wm:{pattern_hash}", "pattern", serialized)
        pipeline.expire(f"wm:{pattern_hash}", 3600)
    
    pipeline.execute()
```

### Memory Management

```python
def monitor_redis_memory():
    """Monitor Redis memory usage."""
    stats = redis_store.get_memory_usage()
    
    if stats['total'] > 10000:  # Too many items
        # Clear expired data
        cleared = redis_store.clear_expired_data()
        logger.info(f"Cleared {cleared} expired items")
    
    # Check Redis server memory
    info = redis_store.redis_client.info('memory')
    used_memory_mb = info['used_memory'] / (1024 * 1024)
    
    if used_memory_mb > 1000:  # > 1GB
        logger.warning(f"Redis using {used_memory_mb:.1f} MB")
```

## Best Practices

### 1. Use Appropriate TTLs

```python
# Short-lived working memory
redis_store.store_working_memory_pattern(
    "temp_pattern", pattern, time.time()
)  # Default: 1 hour TTL

# Very short-lived spike caches
redis_store.cache_spike_train(
    timestamp, spike_data, neuron_ids, ttl=60  # 1 minute
)

# Medium-lived attention weights
redis_store.store_attention_weights(
    "context", weights, timestamp
)  # Default: 10 minutes TTL
```

### 2. Monitor Memory Usage

```python
def setup_memory_monitoring():
    """Set up periodic memory monitoring."""
    import threading
    import time
    
    def monitor():
        while True:
            stats = redis_store.get_memory_usage()
            if stats['total'] > 5000:
                redis_store.clear_expired_data()
            time.sleep(300)  # Check every 5 minutes
    
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
```

### 3. Handle Network Partitions

```python
def robust_pattern_retrieval(pattern_hash):
    """Retrieve pattern with network partition handling."""
    try:
        return redis_store.retrieve_working_memory_pattern(pattern_hash)
    except RedisConnectionError:
        # Network partition - use cached local copy
        return local_cache.get(pattern_hash)
```

### 4. Use Context Managers

```python
# Ensure proper cleanup
with RedisStorage() as redis_store:
    redis_store.store_working_memory_pattern(
        "pattern", np.random.rand(50, 50), time.time()
    )
# Connection automatically closed
```

## Monitoring and Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Redis operations will be logged
redis_store.store_working_memory_pattern("debug", pattern, time.time())
```

### Performance Metrics

```python
import time

def benchmark_redis_operations():
    """Benchmark Redis storage operations."""
    patterns = [np.random.rand(100, 50) for _ in range(100)]
    
    # Benchmark storage
    start_time = time.time()
    for i, pattern in enumerate(patterns):
        redis_store.store_working_memory_pattern(f"bench_{i}", pattern, time.time())
    store_time = time.time() - start_time
    
    # Benchmark retrieval
    start_time = time.time()
    for i in range(100):
        redis_store.retrieve_working_memory_pattern(f"bench_{i}")
    retrieve_time = time.time() - start_time
    
    print(f"Store time: {store_time:.2f}s ({100/store_time:.1f} ops/sec)")
    print(f"Retrieve time: {retrieve_time:.2f}s ({100/retrieve_time:.1f} ops/sec)")
```

## Integration Examples

### With Working Memory System

```python
from src.memory.working import WorkingMemory

class WorkingMemoryWithRedis(WorkingMemory):
    def __init__(self):
        super().__init__()
        self.redis_store = RedisStorage()
    
    def store_pattern(self, pattern_hash, pattern):
        # Store in Redis for fast access
        self.redis_store.store_working_memory_pattern(
            pattern_hash, pattern, time.time()
        )
        # Also store locally for immediate access
        super().store_pattern(pattern_hash, pattern)
    
    def retrieve_pattern(self, pattern_hash):
        # Try Redis first
        result = self.redis_store.retrieve_working_memory_pattern(pattern_hash)
        if result:
            return result[0]  # Return just the pattern
        
        # Fallback to local storage
        return super().retrieve_pattern(pattern_hash)
```

### With Spike Processing

```python
def process_spike_stream(spike_stream):
    """Process continuous spike stream with Redis caching."""
    for timestamp, spike_data, neuron_ids in spike_stream:
        # Cache recent spikes for pattern detection
        redis_store.cache_spike_train(timestamp, spike_data, neuron_ids, ttl=30)
        
        # Get recent spike history for analysis
        recent_trains = redis_store.get_recent_spike_trains(time_window=1.0)
        
        if len(recent_trains) >= 10:  # Enough history
            # Analyze patterns in recent spike trains
            patterns = analyze_spike_patterns(recent_trains)
            
            # Store detected patterns
            for pattern_hash, pattern in patterns.items():
                redis_store.store_working_memory_pattern(
                    pattern_hash, pattern, timestamp
                )
```
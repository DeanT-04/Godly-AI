# Basic Usage Examples

This document provides basic usage examples for all three storage systems in the Godly AI System.

## Setup and Initialization

```python
import numpy as np
import time
from src.storage import RedisStorage, SQLiteStorage, HDF5Storage

# Initialize all storage systems
redis_store = RedisStorage(host='localhost', port=6379)
sqlite_store = SQLiteStorage(db_path='./data/godly_ai.db')
hdf5_store = HDF5Storage(file_path='./data/godly_ai_data.h5')

print("All storage systems initialized successfully!")
```

## Redis Storage Examples

### Working Memory Patterns

```python
# Store a working memory pattern
pattern = np.random.rand(100, 50)
pattern_hash = "visual_pattern_001"

success = redis_store.store_working_memory_pattern(
    pattern_hash=pattern_hash,
    pattern=pattern,
    timestamp=time.time(),
    metadata={"source": "visual_cortex", "confidence": 0.95}
)

if success:
    print(f"Stored pattern: {pattern_hash}")

# Retrieve the pattern
result = redis_store.retrieve_working_memory_pattern(pattern_hash)
if result:
    retrieved_pattern, timestamp, metadata = result
    print(f"Retrieved pattern shape: {retrieved_pattern.shape}")
    print(f"Metadata: {metadata}")
```

### Spike Train Caching

```python
# Cache spike train data
spike_times = np.array([0.1, 0.15, 0.3, 0.45, 0.6])
neuron_ids = np.array([1, 3, 1, 2, 3])
spike_data = np.random.randint(0, 2, (1000, 100))

success = redis_store.cache_spike_train(
    timestamp=time.time(),
    spike_data=spike_data,
    neuron_ids=list(range(100)),
    ttl=300  # 5 minutes
)

print(f"Spike train cached: {success}")

# Get recent spike trains
recent_trains = redis_store.get_recent_spike_trains(time_window=60.0)
print(f"Found {len(recent_trains)} recent spike trains")
```

### Attention Weights

```python
# Store attention weights
attention_weights = np.random.rand(256)
attention_weights = attention_weights / np.sum(attention_weights)  # Normalize

success = redis_store.store_attention_weights(
    context="visual_attention",
    weights=attention_weights,
    timestamp=time.time()
)

# Retrieve attention weights
result = redis_store.retrieve_attention_weights("visual_attention")
if result:
    weights, timestamp = result
    print(f"Attention weights sum: {np.sum(weights):.6f}")
```

## SQLite Storage Examples

### Episode Storage

```python
# Store an episode
experience_data = {
    "observation": np.random.rand(84, 84, 3),
    "action": "move_forward",
    "reward": 1.0,
    "next_observation": np.random.rand(84, 84, 3)
}

episode_id = sqlite_store.store_episode(
    experience_data=experience_data,
    performance_score=0.85,
    context_hash="navigation_task_level_1",
    metadata={"difficulty": "medium", "episode_length": 150}
)

print(f"Stored episode with ID: {episode_id}")

# Retrieve the episode
episode = sqlite_store.retrieve_episode(episode_id)
if episode:
    print(f"Episode {episode.id}: score={episode.performance_score}")
    print(f"Context: {episode.context_hash}")
```

### Concept Management

```python
# Store a concept
concept_embedding = np.random.rand(512)  # 512-dimensional embedding

concept_id = sqlite_store.store_concept(
    name="neural_network",
    embedding=concept_embedding,
    metadata={
        "category": "machine_learning",
        "confidence": 0.95,
        "source": "wikipedia"
    }
)

print(f"Stored concept with ID: {concept_id}")

# Retrieve concept by name
concept = sqlite_store.retrieve_concept_by_name("neural_network")
if concept:
    print(f"Concept: {concept.name}")
    print(f"Access count: {concept.access_count}")
    
    # Deserialize embedding
    embedding = sqlite_store._deserialize_array(concept.embedding)
    print(f"Embedding shape: {embedding.shape}")
```

### Learning Events

```python
# Store a learning event
event_id = sqlite_store.store_learning_event(
    task_type="image_classification",
    performance_delta=0.15,  # 15% improvement
    strategy_used="adam_optimizer",
    parameters={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    }
)

print(f"Stored learning event with ID: {event_id}")

# Get best strategies for a task
best_strategies = sqlite_store.get_best_strategies(
    task_type="image_classification",
    limit=5
)

for strategy, avg_delta, usage_count in best_strategies:
    print(f"{strategy}: {avg_delta:.3f} avg improvement ({usage_count} uses)")
```

### Concept Relationships

```python
# Create relationships between concepts
concept1_id = sqlite_store.store_concept("deep_learning", np.random.rand(512))
concept2_id = sqlite_store.store_concept("neural_network", np.random.rand(512))
concept3_id = sqlite_store.store_concept("machine_learning", np.random.rand(512))

# Add relationships
sqlite_store.add_concept_relationship(
    source_id=concept1_id,
    target_id=concept2_id,
    relationship_type="is_type_of",
    strength=0.9
)

sqlite_store.add_concept_relationship(
    source_id=concept2_id,
    target_id=concept3_id,
    relationship_type="is_part_of",
    strength=0.8
)

# Get relationships
relationships = sqlite_store.get_concept_relationships(concept1_id)
for target_id, rel_type, strength in relationships:
    print(f"Relationship: {rel_type} -> {target_id} (strength: {strength})")
```

## HDF5 Storage Examples

### Network Snapshots

```python
# Store a network snapshot
adjacency_matrix = np.random.randint(0, 2, (1000, 1000))
weight_matrix = np.random.rand(1000, 1000)
neuron_states = np.random.rand(1000, 5)

snapshot_id = hdf5_store.store_network_snapshot(
    timestamp=time.time(),
    adjacency_matrix=adjacency_matrix,
    weight_matrix=weight_matrix,
    neuron_states=neuron_states,
    metadata={
        "network_type": "liquid_state_machine",
        "num_neurons": 1000,
        "connectivity": 0.1
    }
)

print(f"Stored network snapshot: {snapshot_id}")

# Retrieve the snapshot
snapshot = hdf5_store.retrieve_network_snapshot(snapshot_id)
if snapshot:
    print(f"Network snapshot at {snapshot.timestamp}")
    print(f"Adjacency matrix shape: {snapshot.adjacency_matrix.shape}")
    print(f"Weight matrix shape: {snapshot.weight_matrix.shape}")
    print(f"Neuron states shape: {snapshot.neuron_states.shape}")
```

### Spike Train Storage

```python
# Store spike train data
spike_times = np.array([0.1, 0.15, 0.3, 0.45, 0.6, 0.75])
neuron_ids = np.array([1, 3, 1, 2, 3, 1])
spike_amplitudes = np.array([1.2, 0.8, 1.5, 1.0, 0.9, 1.3])

train_id = hdf5_store.store_spike_train(
    timestamp=time.time(),
    spike_times=spike_times,
    neuron_ids=neuron_ids,
    spike_amplitudes=spike_amplitudes,
    metadata={
        "recording_duration": 1.0,
        "sampling_rate": 10000,
        "num_neurons": 100
    }
)

print(f"Stored spike train: {train_id}")

# Retrieve spike train
spike_train = hdf5_store.retrieve_spike_train(train_id)
if spike_train:
    print(f"Spike train at {spike_train.timestamp}")
    print(f"Number of spikes: {len(spike_train.spike_times)}")
    print(f"Unique neurons: {len(np.unique(spike_train.neuron_ids))}")
```

### Reservoir States

```python
# Store reservoir states
reservoir_states = np.random.rand(500, 100)  # 500 neurons, 100 time steps

state_id = hdf5_store.store_reservoir_states(
    timestamp=time.time(),
    reservoir_states=reservoir_states,
    reservoir_id="main_reservoir"
)

print(f"Stored reservoir states: {state_id}")

# Retrieve reservoir states
result = hdf5_store.retrieve_reservoir_states(state_id)
if result:
    states, timestamp, reservoir_id = result
    print(f"Reservoir {reservoir_id} states at {timestamp}")
    print(f"States shape: {states.shape}")
```

### Layer Outputs

```python
# Store layer outputs from a neural network
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

print(f"Stored layer outputs: {output_id}")

# Retrieve layer outputs
result = hdf5_store.retrieve_layer_outputs(output_id)
if result:
    retrieved_outputs, timestamp = result
    for layer_name, output in retrieved_outputs.items():
        print(f"{layer_name}: {output.shape}")
```

### Performance Metrics

```python
# Store performance metrics
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

print(f"Stored performance metrics: {metrics_id}")

# Retrieve performance metrics
result = hdf5_store.retrieve_performance_metrics(metrics_id)
if result:
    retrieved_metrics, timestamp, task_type = result
    print(f"Performance metrics for {task_type} at {timestamp}")
    for metric_name, value in retrieved_metrics.items():
        print(f"  {metric_name}: {value}")
```

## Cross-Storage Integration

### Complete Learning Episode

```python
def process_complete_learning_episode():
    """Example of using all three storage systems together."""
    
    # 1. Generate episode data
    experience_data = {
        "observation": np.random.rand(84, 84, 3),
        "action": "move_forward",
        "reward": 1.0,
        "neural_activity": np.random.rand(1000, 100)
    }
    
    performance_score = 0.85
    context_hash = "navigation_task"
    timestamp = time.time()
    
    # 2. Store working memory pattern in Redis (fast access)
    pattern_success = redis_store.store_working_memory_pattern(
        pattern_hash=f"episode_{int(timestamp)}",
        pattern=experience_data["neural_activity"],
        timestamp=timestamp,
        metadata={"context": context_hash}
    )
    
    # 3. Store episode in SQLite (structured data)
    episode_id = sqlite_store.store_episode(
        experience_data=experience_data,
        performance_score=performance_score,
        context_hash=context_hash,
        timestamp=timestamp
    )
    
    # 4. Store detailed network state in HDF5 (large-scale data)
    snapshot_id = hdf5_store.store_network_snapshot(
        timestamp=timestamp,
        adjacency_matrix=np.random.randint(0, 2, (1000, 1000)),
        weight_matrix=np.random.rand(1000, 1000),
        neuron_states=experience_data["neural_activity"],
        metadata={"episode_id": episode_id, "context": context_hash}
    )
    
    print(f"Complete episode processed:")
    print(f"  Redis pattern: {pattern_success}")
    print(f"  SQLite episode ID: {episode_id}")
    print(f"  HDF5 snapshot ID: {snapshot_id}")
    
    return episode_id, snapshot_id

# Process an episode
episode_id, snapshot_id = process_complete_learning_episode()
```

### Data Retrieval and Analysis

```python
def analyze_learning_progress():
    """Analyze learning progress using data from all storage systems."""
    
    # 1. Get recent episodes from SQLite
    recent_episodes = sqlite_store.get_recent_episodes(
        time_window=3600.0,  # Last hour
        limit=10
    )
    
    print(f"Analyzing {len(recent_episodes)} recent episodes")
    
    # 2. Get performance trends
    performance_scores = [ep.performance_score for ep in recent_episodes]
    avg_performance = np.mean(performance_scores)
    
    print(f"Average performance: {avg_performance:.3f}")
    
    # 3. Check Redis for active patterns
    memory_stats = redis_store.get_memory_usage()
    print(f"Active working memory patterns: {memory_stats['working_memory']}")
    
    # 4. Get network evolution from HDF5
    current_time = time.time()
    snapshot_ids = hdf5_store.get_network_snapshots_in_range(
        start_time=current_time - 3600,  # Last hour
        end_time=current_time
    )
    
    print(f"Network snapshots in last hour: {len(snapshot_ids)}")
    
    # 5. Analyze network changes
    if len(snapshot_ids) >= 2:
        first_snapshot = hdf5_store.retrieve_network_snapshot(snapshot_ids[0])
        last_snapshot = hdf5_store.retrieve_network_snapshot(snapshot_ids[-1])
        
        if first_snapshot and last_snapshot:
            weight_change = np.mean(np.abs(
                last_snapshot.weight_matrix - first_snapshot.weight_matrix
            ))
            print(f"Average weight change: {weight_change:.6f}")
    
    return {
        "avg_performance": avg_performance,
        "num_episodes": len(recent_episodes),
        "active_patterns": memory_stats['working_memory'],
        "network_snapshots": len(snapshot_ids)
    }

# Run analysis
analysis_results = analyze_learning_progress()
print(f"Analysis complete: {analysis_results}")
```

## Cleanup and Resource Management

```python
def cleanup_storage_systems():
    """Properly clean up all storage systems."""
    
    # Redis cleanup
    print("Cleaning up Redis...")
    expired_count = redis_store.clear_expired_data()
    print(f"Cleared {expired_count} expired Redis items")
    
    # SQLite cleanup
    print("Cleaning up SQLite...")
    backup_path = sqlite_store.auto_backup_if_needed()
    if backup_path:
        print(f"Created backup: {backup_path}")
    
    # HDF5 cleanup
    print("Cleaning up HDF5...")
    removed_count = hdf5_store.cleanup_old_data(retention_days=30)
    print(f"Removed {removed_count} old HDF5 items")
    
    # Check file sizes
    hdf5_info = hdf5_store.get_file_info()
    if hdf5_info['file_size_mb'] > 1000:  # > 1GB
        print(f"HDF5 file size: {hdf5_info['file_size_mb']:.1f} MB")
        if hdf5_store.check_file_size():
            print("HDF5 file rotated due to size")
    
    # Close connections
    redis_store.close()
    sqlite_store.close()
    hdf5_store.close()
    
    print("All storage systems cleaned up and closed")

# Run cleanup
cleanup_storage_systems()
```

## Error Handling Examples

```python
from src.storage.redis_storage import RedisConnectionError
from src.storage.sqlite_storage import SQLiteStorageError
from src.storage.hdf5_storage import HDF5StorageError

def robust_data_storage(data, context):
    """Store data with robust error handling."""
    
    # Try Redis first (fastest)
    try:
        success = redis_store.store_working_memory_pattern(
            pattern_hash=f"{context}_{int(time.time())}",
            pattern=data,
            timestamp=time.time()
        )
        if success:
            print("Data stored in Redis")
            return "redis"
    except RedisConnectionError as e:
        print(f"Redis failed: {e}")
    
    # Fallback to SQLite
    try:
        episode_id = sqlite_store.store_episode(
            experience_data={"data": data},
            performance_score=0.5,
            context_hash=context
        )
        print(f"Data stored in SQLite (episode {episode_id})")
        return "sqlite"
    except SQLiteStorageError as e:
        print(f"SQLite failed: {e}")
    
    # Final fallback to HDF5
    try:
        snapshot_id = hdf5_store.store_network_snapshot(
            timestamp=time.time(),
            adjacency_matrix=np.eye(data.shape[0]),
            weight_matrix=data,
            neuron_states=np.zeros((data.shape[0], 1)),
            metadata={"context": context, "fallback": True}
        )
        print(f"Data stored in HDF5 (snapshot {snapshot_id})")
        return "hdf5"
    except HDF5StorageError as e:
        print(f"HDF5 failed: {e}")
    
    print("All storage systems failed!")
    return None

# Test robust storage
test_data = np.random.rand(100, 100)
storage_used = robust_data_storage(test_data, "test_context")
print(f"Data stored using: {storage_used}")
```

This completes the basic usage examples. Each storage system has its strengths:

- **Redis**: Fast, temporary data with automatic expiration
- **SQLite**: Structured, queryable persistent data
- **HDF5**: Large-scale scientific data with compression

Use them together for a complete storage solution that handles all types of data in the Godly AI System.
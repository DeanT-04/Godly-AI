# SQLite Storage

SQLite Storage provides persistent, structured data storage with ACID transactions, migrations, and backup capabilities. It's designed for episodic memory, semantic concepts, learning events, and relational data.

## Overview

SQLite Storage is optimized for:
- Episodic memory storage with rich metadata
- Semantic concepts with embeddings and relationships
- Learning event tracking for meta-memory
- Knowledge graph relationships
- Structured queries and analytics

## Features

- **ACID Transactions**: Full transaction support with rollback capability
- **Schema Migrations**: Automatic database schema versioning and upgrades
- **Backup & Restore**: Automated backup with configurable retention
- **Indexing**: Optimized indexes for fast queries
- **Thread Safety**: Thread-local connections with proper isolation
- **Data Validation**: Built-in data validation and integrity checks

## Configuration

### Basic Configuration

```python
from src.storage import SQLiteStorage

# Basic setup
sqlite_store = SQLiteStorage(db_path="godly_ai.db")

# Advanced configuration
sqlite_store = SQLiteStorage(
    db_path="./data/godly_ai.db",
    backup_interval=3600,      # Backup every hour
    max_backups=24,           # Keep 24 backups (1 day)
    enable_wal=True,          # Enable WAL mode for better concurrency
    connection_timeout=30.0   # 30 second timeout
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | str/Path | "godly_ai.db" | Path to SQLite database file |
| `backup_interval` | int | 3600 | Backup interval in seconds |
| `max_backups` | int | 10 | Maximum backup files to keep |
| `enable_wal` | bool | True | Enable Write-Ahead Logging |
| `connection_timeout` | float | 30.0 | Connection timeout in seconds |

## Database Schema

### Episodes Table

Stores episodic memory entries with experience data and performance metrics.

```sql
CREATE TABLE episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    experience_data BLOB NOT NULL,
    performance_score REAL NOT NULL,
    context_hash TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at REAL DEFAULT (julianday('now'))
);

-- Indexes for performance
CREATE INDEX idx_episodes_timestamp ON episodes(timestamp);
CREATE INDEX idx_episodes_context_hash ON episodes(context_hash);
CREATE INDEX idx_episodes_performance_score ON episodes(performance_score);
```

### Concepts Table

Stores semantic memory concepts with embeddings and access tracking.

```sql
CREATE TABLE concepts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    embedding BLOB NOT NULL,
    creation_time REAL NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed REAL DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

-- Indexes for performance
CREATE INDEX idx_concepts_name ON concepts(name);
CREATE INDEX idx_concepts_creation_time ON concepts(creation_time);
CREATE INDEX idx_concepts_access_count ON concepts(access_count);
```

### Learning Events Table

Tracks learning events for meta-memory and strategy optimization.

```sql
CREATE TABLE learning_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT NOT NULL,
    performance_delta REAL NOT NULL,
    strategy_used TEXT NOT NULL,
    timestamp REAL NOT NULL,
    parameters TEXT DEFAULT '{}'
);

-- Indexes for performance
CREATE INDEX idx_learning_events_task_type ON learning_events(task_type);
CREATE INDEX idx_learning_events_timestamp ON learning_events(timestamp);
CREATE INDEX idx_learning_events_performance_delta ON learning_events(performance_delta);
```

### Concept Relationships Table

Stores relationships between concepts for knowledge graph functionality.

```sql
CREATE TABLE concept_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_concept_id INTEGER NOT NULL,
    target_concept_id INTEGER NOT NULL,
    relationship_type TEXT NOT NULL,
    strength REAL DEFAULT 1.0,
    created_at REAL DEFAULT (julianday('now')),
    FOREIGN KEY (source_concept_id) REFERENCES concepts (id),
    FOREIGN KEY (target_concept_id) REFERENCES concepts (id),
    UNIQUE(source_concept_id, target_concept_id, relationship_type)
);

-- Indexes for performance
CREATE INDEX idx_concept_relationships_source ON concept_relationships(source_concept_id);
CREATE INDEX idx_concept_relationships_target ON concept_relationships(target_concept_id);
CREATE INDEX idx_concept_relationships_type ON concept_relationships(relationship_type);
```

## API Reference

### Episode Operations

#### store_episode()

Store an episodic memory entry with experience data and performance metrics.

```python
episode_id = sqlite_store.store_episode(
    experience_data={
        "observation": np.random.rand(84, 84, 3),
        "action": "move_forward",
        "reward": 1.0,
        "next_observation": np.random.rand(84, 84, 3)
    },
    performance_score=0.85,
    context_hash="navigation_task_level_1",
    timestamp=time.time(),  # Optional, defaults to current time
    metadata={"difficulty": "medium", "episode_length": 150}
)
print(f"Stored episode with ID: {episode_id}")
```

**Parameters:**
- `experience_data` (np.ndarray | dict): Experience data (array or dictionary)
- `performance_score` (float): Performance score for this episode
- `context_hash` (str): Hash identifying the context
- `timestamp` (float, optional): Episode timestamp
- `metadata` (dict, optional): Additional metadata

**Returns:** `int` - Episode ID

#### retrieve_episode()

Retrieve an episode by ID.

```python
episode = sqlite_store.retrieve_episode(episode_id)
if episode:
    print(f"Episode {episode.id}: score={episode.performance_score}")
    print(f"Context: {episode.context_hash}")
    print(f"Metadata: {episode.metadata}")
```

**Parameters:**
- `episode_id` (int): Episode identifier

**Returns:** `Episode | None` - Episode object or None if not found

#### get_episodes_by_context()

Get episodes by context hash with optional filtering.

```python
episodes = sqlite_store.get_episodes_by_context(
    context_hash="navigation_task_level_1",
    limit=50,
    min_score=0.7  # Only episodes with score >= 0.7
)

for episode in episodes:
    print(f"Episode {episode.id}: {episode.performance_score}")
```

**Parameters:**
- `context_hash` (str): Context identifier
- `limit` (int): Maximum number of episodes to return
- `min_score` (float, optional): Minimum performance score filter

**Returns:** `List[Episode]` - List of Episode objects

#### get_recent_episodes()

Get recent episodes within a time window.

```python
recent_episodes = sqlite_store.get_recent_episodes(
    time_window=3600.0,  # Last hour
    limit=100
)

print(f"Found {len(recent_episodes)} recent episodes")
```

**Parameters:**
- `time_window` (float): Time window in seconds from current time
- `limit` (int): Maximum number of episodes

**Returns:** `List[Episode]` - List of recent Episode objects

### Concept Operations

#### store_concept()

Store a semantic memory concept with embedding.

```python
concept_id = sqlite_store.store_concept(
    name="neural_network",
    embedding=np.random.rand(512),  # 512-dimensional embedding
    metadata={
        "category": "machine_learning",
        "confidence": 0.95,
        "source": "wikipedia"
    },
    creation_time=time.time()  # Optional
)
```

**Parameters:**
- `name` (str): Concept name (must be unique)
- `embedding` (np.ndarray | jnp.ndarray): Concept embedding vector
- `metadata` (dict, optional): Additional metadata
- `creation_time` (float, optional): Creation timestamp

**Returns:** `int` - Concept ID

**Note:** If a concept with the same name exists, it will be updated.

#### retrieve_concept()

Retrieve a concept by ID.

```python
concept = sqlite_store.retrieve_concept(concept_id)
if concept:
    embedding = sqlite_store._deserialize_array(concept.embedding)
    print(f"Concept: {concept.name}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Access count: {concept.access_count}")
```

#### retrieve_concept_by_name()

Retrieve a concept by name (automatically updates access count).

```python
concept = sqlite_store.retrieve_concept_by_name("neural_network")
if concept:
    print(f"Concept accessed {concept.access_count} times")
```

**Parameters:**
- `name` (str): Concept name

**Returns:** `Concept | None` - Concept object or None if not found

#### search_concepts()

Search concepts by name pattern.

```python
# Find all concepts containing "neural"
neural_concepts = sqlite_store.search_concepts("neural", limit=10)

for concept in neural_concepts:
    print(f"Found: {concept.name} (accessed {concept.access_count} times)")
```

**Parameters:**
- `query` (str): Search query (supports SQL LIKE patterns)
- `limit` (int): Maximum number of results

**Returns:** `List[Concept]` - List of matching concepts

#### add_concept_relationship()

Add or update a relationship between concepts.

```python
# Add relationship between concepts
success = sqlite_store.add_concept_relationship(
    source_id=concept1_id,
    target_id=concept2_id,
    relationship_type="similar_to",
    strength=0.8
)

# Add different relationship types
sqlite_store.add_concept_relationship(
    source_id=concept1_id,
    target_id=concept3_id,
    relationship_type="part_of",
    strength=0.9
)
```

**Parameters:**
- `source_id` (int): Source concept ID
- `target_id` (int): Target concept ID
- `relationship_type` (str): Type of relationship
- `strength` (float): Relationship strength (0.0 to 1.0)

**Returns:** `bool` - True if successful

#### get_concept_relationships()

Get relationships for a concept.

```python
# Get all relationships
relationships = sqlite_store.get_concept_relationships(concept_id)

# Get specific relationship type
similar_relationships = sqlite_store.get_concept_relationships(
    concept_id, 
    relationship_type="similar_to"
)

for target_id, rel_type, strength in relationships:
    print(f"Relationship: {rel_type} -> {target_id} (strength: {strength})")
```

**Parameters:**
- `concept_id` (int): Source concept ID
- `relationship_type` (str, optional): Filter by relationship type

**Returns:** `List[Tuple[int, str, float]]` - List of (target_id, type, strength) tuples

### Learning Event Operations

#### store_learning_event()

Store a learning event for meta-memory tracking.

```python
event_id = sqlite_store.store_learning_event(
    task_type="image_classification",
    performance_delta=0.15,  # 15% improvement
    strategy_used="adam_optimizer",
    parameters={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10
    },
    timestamp=time.time()  # Optional
)
```

**Parameters:**
- `task_type` (str): Type of learning task
- `performance_delta` (float): Change in performance
- `strategy_used` (str): Learning strategy identifier
- `parameters` (dict, optional): Strategy parameters
- `timestamp` (float, optional): Event timestamp

**Returns:** `int` - Learning event ID

#### get_learning_events_by_task()

Get learning events by task type.

```python
events = sqlite_store.get_learning_events_by_task(
    task_type="image_classification",
    limit=50,
    min_performance_delta=0.05  # Only improvements >= 5%
)

for event in events:
    print(f"Strategy: {event.strategy_used}, Delta: {event.performance_delta}")
```

**Parameters:**
- `task_type` (str): Task type to filter by
- `limit` (int): Maximum number of events
- `min_performance_delta` (float, optional): Minimum performance improvement

**Returns:** `List[LearningEvent]` - List of learning events

#### get_best_strategies()

Get best performing strategies for a task type.

```python
best_strategies = sqlite_store.get_best_strategies(
    task_type="image_classification",
    limit=5
)

for strategy, avg_delta, usage_count in best_strategies:
    print(f"{strategy}: {avg_delta:.3f} avg improvement ({usage_count} uses)")
```

**Parameters:**
- `task_type` (str): Task type to analyze
- `limit` (int): Maximum number of strategies

**Returns:** `List[Tuple[str, float, int]]` - List of (strategy, avg_delta, count) tuples

### Database Management

#### create_backup()

Create a backup of the database.

```python
backup_path = sqlite_store.create_backup()
print(f"Backup created: {backup_path}")

# Create backup with custom path
custom_backup = sqlite_store.create_backup("./backups/manual_backup.db")
```

**Parameters:**
- `backup_path` (str/Path, optional): Custom backup path

**Returns:** `Path` - Path to backup file

#### vacuum_database()

Vacuum the database to reclaim space and optimize performance.

```python
success = sqlite_store.vacuum_database()
if success:
    print("Database vacuumed successfully")
```

**Returns:** `bool` - True if successful

#### get_database_stats()

Get database statistics and information.

```python
stats = sqlite_store.get_database_stats()
print(f"Episodes: {stats['episodes_count']}")
print(f"Concepts: {stats['concepts_count']}")
print(f"Learning events: {stats['learning_events_count']}")
print(f"Database size: {stats['db_size_mb']:.1f} MB")
print(f"Schema version: {stats['schema_version']}")
```

**Returns:** `dict` - Database statistics

#### auto_backup_if_needed()

Create backup if backup interval has elapsed.

```python
backup_path = sqlite_store.auto_backup_if_needed()
if backup_path:
    print(f"Automatic backup created: {backup_path}")
```

**Returns:** `Path | None` - Backup path if created, None otherwise

## Data Models

### Episode

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

### Concept

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

### LearningEvent

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

## Error Handling

### Transaction Errors

```python
from src.storage.sqlite_storage import SQLiteStorageError

try:
    episode_id = sqlite_store.store_episode(
        experience_data=invalid_data,
        performance_score=0.8,
        context_hash="test"
    )
except SQLiteStorageError as e:
    print(f"Transaction failed: {e}")
```

### Connection Management

```python
# Use context manager for automatic cleanup
with SQLiteStorage("test.db") as storage:
    episode_id = storage.store_episode(
        experience_data={"test": "data"},
        performance_score=0.5,
        context_hash="test_context"
    )
# Connection automatically closed
```

## Performance Optimization

### Batch Operations

```python
def store_multiple_episodes(episodes_data):
    """Store multiple episodes efficiently."""
    with sqlite_store._transaction() as conn:
        for episode_data in episodes_data:
            conn.execute("""
                INSERT INTO episodes (timestamp, experience_data, performance_score, context_hash, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                episode_data['timestamp'],
                pickle.dumps(episode_data['experience']),
                episode_data['score'],
                episode_data['context'],
                json.dumps(episode_data.get('metadata', {}))
            ))
```

### Query Optimization

```python
def get_top_episodes_by_context(context_hash, limit=10):
    """Get top performing episodes for a context."""
    episodes = sqlite_store.get_episodes_by_context(
        context_hash=context_hash,
        limit=limit,
        min_score=0.8  # Pre-filter for better performance
    )
    
    # Sort by performance score (already indexed)
    return sorted(episodes, key=lambda e: e.performance_score, reverse=True)
```

### Index Usage

```python
# Queries that use indexes efficiently:

# 1. Context-based queries (uses idx_episodes_context_hash)
episodes = sqlite_store.get_episodes_by_context("navigation_task")

# 2. Time-based queries (uses idx_episodes_timestamp)
recent = sqlite_store.get_recent_episodes(time_window=3600)

# 3. Concept name searches (uses idx_concepts_name)
concept = sqlite_store.retrieve_concept_by_name("neural_network")

# 4. Learning event analysis (uses idx_learning_events_task_type)
events = sqlite_store.get_learning_events_by_task("classification")
```

## Best Practices

### 1. Use Transactions for Related Operations

```python
def store_episode_with_concepts(episode_data, related_concepts):
    """Store episode and update related concepts atomically."""
    with sqlite_store._transaction() as conn:
        # Store episode
        episode_id = sqlite_store.store_episode(**episode_data)
        
        # Update related concepts
        for concept_name in related_concepts:
            concept = sqlite_store.retrieve_concept_by_name(concept_name)
            if concept:
                # Update access count is handled automatically
                pass
```

### 2. Regular Maintenance

```python
def daily_maintenance():
    """Perform daily database maintenance."""
    # Create backup
    sqlite_store.auto_backup_if_needed()
    
    # Vacuum database weekly
    import datetime
    if datetime.datetime.now().weekday() == 0:  # Monday
        sqlite_store.vacuum_database()
    
    # Check database health
    stats = sqlite_store.get_database_stats()
    if stats['db_size_mb'] > 1000:  # > 1GB
        logger.warning(f"Database size: {stats['db_size_mb']:.1f} MB")
```

### 3. Efficient Concept Management

```python
def build_concept_graph():
    """Build concept relationship graph efficiently."""
    # Get all concepts
    all_concepts = sqlite_store.search_concepts("", limit=10000)
    
    # Build similarity matrix
    concept_embeddings = {}
    for concept in all_concepts:
        embedding = sqlite_store._deserialize_array(concept.embedding)
        concept_embeddings[concept.id] = embedding
    
    # Add relationships based on similarity
    for concept1_id, embedding1 in concept_embeddings.items():
        for concept2_id, embedding2 in concept_embeddings.items():
            if concept1_id != concept2_id:
                similarity = np.dot(embedding1, embedding2)
                if similarity > 0.8:  # High similarity threshold
                    sqlite_store.add_concept_relationship(
                        concept1_id, concept2_id, "similar_to", similarity
                    )
```

### 4. Memory-Efficient Data Handling

```python
def process_large_episode_batch(episode_files):
    """Process large batches of episodes memory-efficiently."""
    for episode_file in episode_files:
        # Load one episode at a time
        episode_data = load_episode_from_file(episode_file)
        
        # Store immediately to avoid memory buildup
        episode_id = sqlite_store.store_episode(**episode_data)
        
        # Clear from memory
        del episode_data
```

## Integration Examples

### With Episodic Memory System

```python
from src.memory.episodic import EpisodicMemory

class PersistentEpisodicMemory(EpisodicMemory):
    def __init__(self):
        super().__init__()
        self.sqlite_store = SQLiteStorage()
    
    def store_episode(self, experience_data, performance_score, context):
        # Store in SQLite for persistence
        episode_id = self.sqlite_store.store_episode(
            experience_data=experience_data,
            performance_score=performance_score,
            context_hash=context
        )
        
        # Also store in memory for fast access
        super().store_episode(experience_data, performance_score, context)
        
        return episode_id
    
    def retrieve_similar_episodes(self, context, min_score=0.7):
        # Query SQLite for similar episodes
        return self.sqlite_store.get_episodes_by_context(
            context_hash=context,
            min_score=min_score,
            limit=10
        )
```

### With Meta-Learning System

```python
def track_learning_progress(task_type, strategy, performance_before, performance_after, parameters):
    """Track learning progress for meta-learning."""
    performance_delta = performance_after - performance_before
    
    # Store learning event
    event_id = sqlite_store.store_learning_event(
        task_type=task_type,
        performance_delta=performance_delta,
        strategy_used=strategy,
        parameters=parameters
    )
    
    # Analyze best strategies periodically
    if event_id % 100 == 0:  # Every 100 events
        best_strategies = sqlite_store.get_best_strategies(task_type, limit=5)
        logger.info(f"Best strategies for {task_type}: {best_strategies}")
    
    return event_id
```
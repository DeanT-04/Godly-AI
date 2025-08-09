# Semantic Memory System

The semantic memory system implements knowledge graphs for concept extraction, relationship modeling, and semantic similarity computation in the Godly AI neuromorphic system.

## Overview

The semantic memory system provides:

- **Concept Extraction**: Automatic formation of concepts from experience patterns using clustering
- **Knowledge Graphs**: NetworkX-based graph structure for representing concepts and relationships
- **Semantic Similarity**: Multi-modal similarity computation (embedding, graph-based, temporal)
- **Content-Addressable Retrieval**: Query-based concept retrieval with similarity ranking
- **Hierarchical Abstraction**: Multi-level concept abstraction hierarchy
- **Temporal Dynamics**: Time-based concept decay and consolidation

## Architecture

### Core Components

1. **Concept**: Individual knowledge units with embeddings, metadata, and properties
2. **ConceptRelation**: Relationships between concepts with strength and confidence
3. **KnowledgeGraph**: Graph structure containing concepts, relations, and indices
4. **SemanticMemory**: Main system orchestrating concept formation and retrieval

### Key Features

- **LSM Integration**: Uses Liquid State Machines for concept encoding
- **Clustering-Based Extraction**: K-means clustering for pattern identification
- **Multi-Modal Similarity**: Combines embedding, graph, and temporal similarities
- **Adaptive Consolidation**: Periodic merging and pruning of knowledge
- **Configurable Parameters**: Extensive parameter set for different use cases

## Usage

### Basic Usage

```python
from src.memory.semantic.semantic_memory import create_semantic_memory
from src.memory.episodic.episodic_memory import Experience
import jax.numpy as jnp
from jax import random

# Create semantic memory
semantic_memory = create_semantic_memory("standard")
key = random.PRNGKey(42)
state = semantic_memory.init_state(key)

# Create sample experiences
experiences = [
    Experience(
        observation=jnp.array([1.0, 0.5, -0.5, 1.0]),
        action=jnp.array([0.5, 0.8]),
        reward=0.8,
        next_observation=jnp.array([1.1, 0.6, -0.4, 1.1]),
        timestamp=time.time(),
        context={"type": "visual"}
    )
    # ... more experiences
]

# Extract concepts
state, concepts = semantic_memory.extract_concepts(state, experiences, key)

# Build knowledge graph
state = semantic_memory.build_knowledge_graph(state, concepts)

# Query knowledge
results = semantic_memory.query_knowledge(state, query_embedding)
```

### Configuration Options

```python
# Small configuration for testing
memory = create_semantic_memory("small")

# Large configuration for complex domains
memory = create_semantic_memory("large")

# Fast learning configuration
memory = create_semantic_memory("fast_learning")

# High abstraction configuration
memory = create_semantic_memory("high_abstraction")
```

### Custom Parameters

```python
from src.memory.semantic.semantic_memory import SemanticMemoryParams, SemanticMemory

params = SemanticMemoryParams(
    min_concept_frequency=3,
    concept_similarity_threshold=0.8,
    embedding_dim=256,
    max_concepts=10000,
    relation_strength_threshold=0.3
)

memory = SemanticMemory(params)
```

## API Reference

### SemanticMemory Class

#### Methods

- `init_state(key)`: Initialize semantic memory state
- `extract_concepts(state, experiences, key)`: Extract concepts from experiences
- `build_knowledge_graph(state, concepts)`: Build relationships between concepts
- `compute_semantic_similarity(state, concept1_id, concept2_id)`: Compute similarity
- `query_knowledge(state, query, max_results)`: Query for relevant concepts
- `get_concept_neighbors(state, concept_id, max_depth)`: Find neighboring concepts
- `consolidate_knowledge(state, key)`: Consolidate and optimize knowledge
- `get_knowledge_statistics(state)`: Get graph statistics

### Data Structures

#### Concept
```python
@dataclass
class Concept:
    concept_id: str                    # Unique identifier
    name: str                         # Human-readable name
    embedding: jnp.ndarray            # Vector representation
    frequency: int                    # Usage frequency
    creation_time: float              # Creation timestamp
    last_updated: float               # Last update time
    confidence: float                 # Confidence score (0-1)
    abstraction_level: int            # Abstraction level
    source_experiences: List[str]     # Contributing experiences
    properties: Dict[str, Any]        # Additional properties
    semantic_tags: Set[str]           # Semantic tags
```

#### ConceptRelation
```python
@dataclass
class ConceptRelation:
    relation_id: str                  # Unique identifier
    source_concept: str               # Source concept ID
    target_concept: str               # Target concept ID
    relation_type: str                # Relationship type
    strength: float                   # Relationship strength (0-1)
    confidence: float                 # Confidence (0-1)
    evidence_count: int               # Supporting evidence count
    creation_time: float              # Creation timestamp
    last_reinforced: float            # Last reinforcement time
    temporal_context: Optional[float] # Temporal context
    properties: Dict[str, Any]        # Additional properties
```

## Parameters

### SemanticMemoryParams

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_concept_frequency` | 3 | Minimum frequency to form concept |
| `concept_similarity_threshold` | 0.8 | Threshold for concept merging |
| `max_concepts` | 10000 | Maximum number of concepts |
| `embedding_dim` | 256 | Dimensionality of embeddings |
| `reservoir_size` | 400 | LSM reservoir size |
| `relation_strength_threshold` | 0.3 | Minimum relation strength |
| `query_similarity_threshold` | 0.6 | Minimum similarity for queries |
| `max_query_results` | 20 | Maximum query results |
| `concept_decay_rate` | 0.005 | Decay rate for unused concepts |
| `minute_timescale` | 60.0 | Operating timescale |

## Implementation Details

### Concept Extraction Process

1. **Pattern Extraction**: Combine observation and action into fixed-size patterns
2. **Clustering**: Use K-means to identify recurring patterns
3. **Concept Formation**: Create concepts from cluster centers
4. **LSM Encoding**: Encode concepts using Liquid State Machine
5. **Validation**: Check against existing concepts for merging

### Knowledge Graph Construction

1. **Similarity Computation**: Calculate pairwise concept similarities
2. **Relationship Inference**: Determine relationship types based on similarity
3. **Graph Building**: Add nodes and edges to NetworkX graph
4. **Index Updates**: Update lookup indices for fast retrieval

### Similarity Computation

The system uses multi-modal similarity:

- **Embedding Similarity** (50%): Cosine similarity between embeddings
- **Graph Similarity** (30%): Jaccard similarity of neighbors + path distance
- **Temporal Similarity** (20%): Exponential decay based on time difference

### Consolidation Process

1. **Concept Merging**: Merge highly similar concepts
2. **Relation Pruning**: Remove weak or decayed relations
3. **Abstraction Updates**: Update hierarchical abstraction levels
4. **Temporal Decay**: Apply decay to unused concepts and relations

## Performance Characteristics

### Time Complexity

- Concept extraction: O(n log n) for clustering
- Graph construction: O(c²) for c concepts
- Similarity computation: O(1) for cached results
- Query processing: O(c) for c concepts
- Consolidation: O(c² + r) for c concepts and r relations

### Memory Usage

- Concepts: ~1KB per concept (embedding + metadata)
- Relations: ~100B per relation
- Graph structure: O(c + r) for NetworkX storage
- Activity history: Configurable buffer size

## Testing

The semantic memory system includes comprehensive tests:

```bash
# Run all semantic memory tests
python -m pytest tests/test_semantic_memory.py -v

# Run specific test categories
python -m pytest tests/test_semantic_memory.py::TestSemanticMemory -v
python -m pytest tests/test_semantic_memory.py::TestSemanticMemoryIntegration -v
```

### Test Coverage

- Parameter configuration
- Data structure creation
- Concept extraction
- Knowledge graph construction
- Similarity computation
- Query processing
- Neighbor finding
- Knowledge consolidation
- Edge cases and error handling

## Examples

See `examples/semantic_memory_demo.py` for a complete demonstration of the semantic memory system capabilities.

## Integration

The semantic memory system integrates with:

- **Episodic Memory**: Processes experiences from episodic memory
- **Liquid State Machine**: Uses LSM for concept encoding
- **Working Memory**: Provides concepts for working memory operations
- **Reasoning Agents**: Supplies knowledge for reasoning processes

## Future Enhancements

Planned improvements include:

- **Advanced Clustering**: More sophisticated pattern recognition
- **Hierarchical Concepts**: Better abstraction hierarchy management
- **Temporal Patterns**: Time-series concept formation
- **Multi-Modal Concepts**: Integration of different sensory modalities
- **Distributed Storage**: Scalable storage for large knowledge graphs
- **Online Learning**: Continuous adaptation without full retraining
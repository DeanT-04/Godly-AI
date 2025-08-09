# Meta-Memory System

The Meta-Memory system implements learning-to-learn capabilities for the Godly AI system, operating on hour+ timescales to store and adapt learning strategies, meta-parameters, and architectural memories.

## Overview

The Meta-Memory system is designed to:

- **Store Learning Experiences**: Track learning episodes with their contexts, strategies, and outcomes
- **Adapt Learning Strategies**: Evolve strategy templates based on performance feedback
- **Optimize Meta-Parameters**: Automatically tune learning parameters based on task characteristics
- **Enable Meta-Learning**: Support learning-to-learn by tracking efficiency improvements over time
- **Consolidate Knowledge**: Periodically consolidate experiences into improved strategy templates

## Key Components

### Core Classes

#### `MetaMemory`
The main meta-memory system that orchestrates all meta-learning capabilities.

```python
from src.memory.meta import MetaMemory, MetaMemoryParams

# Create meta-memory system
params = MetaMemoryParams(
    learning_history_size=1000,
    adaptation_rate=0.1,
    exploration_rate=0.2
)
meta_memory = MetaMemory(params)

# Initialize state
state = meta_memory.init_state()
```

#### `LearningExperience`
Represents a single learning episode with complete context and outcomes.

```python
from src.memory.meta import LearningExperience, LearningStrategy

experience = LearningExperience(
    experience_id="exp_001",
    task_type="classification",
    task_context={"difficulty": 0.7, "dataset": "cifar10"},
    strategy_used=LearningStrategy.GRADIENT_DESCENT,
    meta_parameters={"learning_rate": 0.01, "batch_size": 32},
    initial_performance=0.3,
    final_performance=0.85,
    learning_time=120.0,
    timestamp=time.time(),
    success=True,
    difficulty=0.7,
    transfer_source=None
)
```

#### `StrategyTemplate`
Represents a learning strategy template with its parameters and performance history.

```python
from src.memory.meta import StrategyTemplate, LearningStrategy

template = StrategyTemplate(
    strategy_id="custom_gradient_descent",
    strategy_type=LearningStrategy.GRADIENT_DESCENT,
    meta_parameters={"learning_rate": 0.01, "momentum": 0.9},
    applicable_tasks=["classification", "regression"],
    success_rate=0.85,
    average_efficiency=0.7,
    usage_count=15,
    last_updated=time.time(),
    creation_time=time.time()
)
```

### Learning Strategies

The system supports multiple learning strategy types:

- **`GRADIENT_DESCENT`**: Traditional gradient-based optimization
- **`EVOLUTIONARY`**: Evolutionary algorithms and genetic programming
- **`REINFORCEMENT`**: Reinforcement learning approaches
- **`UNSUPERVISED`**: Unsupervised learning methods
- **`TRANSFER`**: Transfer learning and domain adaptation
- **`META_LEARNING`**: Meta-learning algorithms
- **`CURRICULUM`**: Curriculum learning strategies
- **`SELF_SUPERVISED`**: Self-supervised learning approaches

## Usage Examples

### Basic Learning Experience Storage

```python
from src.memory.meta import create_meta_memory, LearningStrategy

# Create meta-memory system
meta_memory = create_meta_memory("standard")
state = meta_memory.init_state()

# Store a learning experience
state, exp_id = meta_memory.store_learning_experience(
    state=state,
    task="image_classification",
    performance=0.92,
    strategy=LearningStrategy.GRADIENT_DESCENT,
    meta_parameters={"learning_rate": 0.01, "batch_size": 32},
    task_context={"difficulty": 0.6, "dataset": "imagenet"},
    learning_time=3600.0,  # 1 hour
    initial_performance=0.3
)

print(f"Stored experience: {exp_id}")
```

### Strategy Retrieval and Adaptation

```python
# Retrieve best strategy for a new task
strategy_id, meta_params, confidence = meta_memory.retrieve_learning_strategy(
    state=state,
    task_similarity=0.8,  # High similarity to previous tasks
    task_type="classification",
    task_context={"difficulty": 0.7, "dataset": "custom"}
)

print(f"Recommended strategy: {strategy_id}")
print(f"Adapted parameters: {meta_params}")
print(f"Confidence: {confidence:.3f}")
```

### Meta-Parameter Updates

```python
# Update meta-parameters based on performance feedback
performance_improvement = 0.3  # 30% improvement
feedback = (performance_improvement - 0.1) * 2  # Scale to [-1, 1]

state = meta_memory.update_meta_parameters(
    state=state,
    performance_feedback=feedback,
    strategy_id=strategy_id,
    task_type="classification"
)
```

### Learning Statistics

```python
# Get comprehensive learning statistics
stats = meta_memory.get_learning_statistics(state)

print(f"Total experiences: {stats['total_experiences']}")
print(f"Success rate: {stats['success_rate']:.3f}")
print(f"Meta-learning progress: {stats['meta_learning_progress']:.6f}")
print(f"Strategy diversity: {stats['strategy_diversity']}")

# Strategy performance breakdown
for strategy, perf in stats['strategy_performance'].items():
    print(f"{strategy}: {perf['success_rate']:.3f} success rate")
```

## Configuration Options

### Pre-configured Memory Types

```python
from src.memory.meta import create_meta_memory

# Standard configuration
meta_memory = create_meta_memory("standard")

# Fast adaptation for dynamic environments
meta_memory = create_meta_memory("fast_adaptation")

# Conservative adaptation for stable environments
meta_memory = create_meta_memory("conservative")

# High exploration for novel domains
meta_memory = create_meta_memory("exploratory")

# Large capacity for complex systems
meta_memory = create_meta_memory("large_capacity")
```

### Custom Parameters

```python
from src.memory.meta import MetaMemory, MetaMemoryParams

params = MetaMemoryParams(
    learning_history_size=2000,        # Store more experiences
    strategy_cache_size=100,           # More strategy templates
    adaptation_rate=0.15,              # Faster adaptation
    exploration_rate=0.25,             # More exploration
    consolidation_interval=1800.0,     # Consolidate every 30 minutes
    performance_improvement_threshold=0.03,  # Lower success threshold
    success_threshold=0.75             # Higher success requirement
)

meta_memory = MetaMemory(params)
```

## Key Features

### 1. Learning Strategy Evolution

The system automatically evolves learning strategies based on performance:

- **Success Rate Tracking**: Monitors strategy success across different tasks
- **Efficiency Measurement**: Tracks learning efficiency (improvement per unit time)
- **Applicability Learning**: Discovers which strategies work for which task types
- **Parameter Adaptation**: Automatically tunes meta-parameters based on feedback

### 2. Meta-Learning Capabilities

- **Learning-to-Learn**: Tracks improvement in learning efficiency over time
- **Transfer Learning**: Identifies when strategies can transfer between tasks
- **Curriculum Discovery**: Learns optimal task ordering and difficulty progression
- **Strategy Selection**: Intelligently selects strategies based on task similarity

### 3. Memory Consolidation

- **Periodic Consolidation**: Consolidates experiences into improved strategy templates
- **Forgetting Mechanisms**: Removes outdated or ineffective strategies
- **Pattern Recognition**: Identifies patterns in successful learning episodes
- **Knowledge Distillation**: Extracts key insights from learning experiences

### 4. Adaptive Behavior

- **Context Sensitivity**: Adapts strategies based on task context and difficulty
- **Performance Feedback**: Continuously improves based on learning outcomes
- **Exploration vs Exploitation**: Balances trying new strategies vs using proven ones
- **Resource Awareness**: Considers computational constraints in strategy selection

## Integration with Other Systems

The Meta-Memory system integrates with other components of the Godly AI system:

### Working Memory Integration
```python
# Meta-memory can inform working memory attention mechanisms
attention_weights = meta_memory.get_attention_guidance(
    state=state,
    current_task="classification",
    working_memory_patterns=patterns
)
```

### Episodic Memory Integration
```python
# Meta-memory can guide episodic memory consolidation
consolidation_strategy = meta_memory.get_consolidation_strategy(
    state=state,
    episodic_experiences=experiences,
    task_context=context
)
```

### Self-Modifying Architecture Integration
```python
# Meta-memory can inform architectural modifications
architecture_changes = meta_memory.suggest_architecture_changes(
    state=state,
    current_performance=performance,
    task_requirements=requirements
)
```

## Performance Characteristics

- **Memory Efficiency**: Stores only essential learning experiences and strategies
- **Computational Efficiency**: Fast strategy retrieval and parameter adaptation
- **Scalability**: Handles thousands of learning experiences efficiently
- **Robustness**: Graceful degradation under resource constraints
- **Biological Plausibility**: Inspired by meta-cognitive processes in biological systems

## Testing and Validation

The meta-memory system includes comprehensive tests:

```bash
# Run all meta-memory tests
python -m pytest tests/test_meta_memory.py -v

# Run specific test categories
python -m pytest tests/test_meta_memory.py::TestMetaMemory -v
python -m pytest tests/test_meta_memory.py::TestMetaMemoryIntegration -v
```

## Demo and Examples

Run the interactive demo to see the meta-memory system in action:

```bash
python examples/meta_memory_demo.py
```

This demo shows:
- Learning strategy storage and retrieval
- Meta-parameter adaptation over time
- Learning-to-learn capability development
- Strategy template evolution
- Performance improvement tracking

## Requirements Satisfied

This implementation satisfies the following system requirements:

- **Requirement 4.4**: "WHEN learning to learn THEN the system SHALL use meta-memory for architectural memories operating on hour+ timescales"
- **Requirement 4.5**: "WHEN retrieving information THEN the system SHALL support content-addressable retrieval and associative pattern completion"

The meta-memory system operates on hour+ timescales, stores architectural memories (learning strategies and meta-parameters), and supports content-addressable retrieval through strategy similarity matching and context-based selection.
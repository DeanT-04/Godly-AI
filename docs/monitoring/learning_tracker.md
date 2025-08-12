# Learning Tracker

The Learning Tracker provides comprehensive learning progress monitoring, goal management, and performance trend analysis for the Godly AI System.

## Overview

The Learning Tracker consists of:

- **LearningProgressTracker**: Main class for tracking learning events and goals
- **LearningEvent**: Data structure for individual learning events
- **GoalProgress**: Data structure for goal tracking and completion

## Basic Usage

### Recording Learning Events

```python
from src.monitoring.learning_tracker import LearningProgressTracker

# Create tracker
tracker = LearningProgressTracker()

# Record learning events
tracker.record_learning_event(
    task_type="classification",
    performance_score=0.85,
    learning_rate=0.01,
    episode_count=100,
    convergence_metric=0.05,
    # Optional metadata
    batch_size=32,
    optimizer="adam",
    dataset="cifar10"
)

# Record multiple events for different tasks
for i in range(50):
    tracker.record_learning_event(
        task_type="reinforcement_learning",
        performance_score=0.6 + i * 0.008,  # Improving performance
        learning_rate=0.001,
        episode_count=i * 10,
        convergence_metric=0.1 - i * 0.002
    )
```

### Goal Management

```python
# Create learning goals
tracker.create_goal(
    goal_id="accuracy_goal",
    description="Achieve 90% accuracy on classification task",
    target_performance=0.9
)

tracker.create_goal(
    goal_id="convergence_goal", 
    description="Reach convergence in under 1000 episodes",
    target_performance=1000  # Episode count target
)

# Update goal progress
tracker.update_goal_progress("accuracy_goal", 0.87)  # 87% accuracy achieved
tracker.update_goal_progress("convergence_goal", 850)  # 850 episodes completed

# Check goal completion
goal_summary = tracker.get_goal_summary()
print(f"Completed goals: {goal_summary['completed_goals']}")
print(f"Active goals: {goal_summary['active_goals']}")
```

## Advanced Features

### Learning Curve Analysis

```python
# Get learning curve for specific task
timestamps, scores = tracker.get_learning_curve("classification", window_size=100)

# Analyze performance trends
trend = tracker.get_performance_trend("classification")
print(f"Trend slope: {trend['trend']:.4f}")
print(f"Confidence: {trend['confidence']:.2f}")
print(f"Recent average: {trend['recent_avg']:.3f}")
print(f"Improvement rate: {trend['improvement_rate']:.4f}")
```

### Convergence Analysis

```python
# Analyze convergence patterns
convergence = tracker.get_convergence_analysis()
print(f"Total convergences: {convergence['total_convergences']}")
print(f"Convergence rate: {convergence['convergence_rate']:.2f} per hour")

# Task-specific convergence statistics
for task_type, stats in convergence['task_statistics'].items():
    print(f"{task_type}: {stats['count']} convergences")
    print(f"  Average metric: {stats['avg_convergence_metric']:.4f}")
```

### Comprehensive Reporting

```python
# Generate detailed learning report
report = tracker.generate_learning_report()

print(f"Total learning events: {report['total_learning_events']}")
print(f"Overall performance average: {report['overall_performance']['average']:.3f}")
print(f"Average learning rate: {report['learning_dynamics']['average_learning_rate']:.4f}")

# Task-specific analysis
for task_type, analysis in report['task_analysis'].items():
    print(f"\n{task_type}:")
    print(f"  Events: {analysis['event_count']}")
    print(f"  Average performance: {analysis['avg_performance']:.3f}")
    print(f"  Latest score: {analysis['latest_score']:.3f}")
    print(f"  Trend: {analysis['performance_trend']['trend']:.4f}")
```

## Visualization

### Learning Progress Visualization

```python
import matplotlib.pyplot as plt

# Create comprehensive learning progress visualization
fig = tracker.visualize_learning_progress(
    task_types=["classification", "reinforcement_learning"],
    save_path="learning_progress.png"
)

# The visualization includes:
# - Learning curves by task type
# - Performance score distribution
# - Learning rate over time
# - Goal completion progress

plt.show()
```

### Custom Visualizations

```python
# Get data for custom visualizations
timestamps, scores = tracker.get_learning_curve("classification")

# Create custom plot
plt.figure(figsize=(12, 6))
plt.plot(timestamps, scores, 'b-', alpha=0.7, label='Classification Performance')
plt.xlabel('Time')
plt.ylabel('Performance Score')
plt.title('Custom Learning Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Data Export and Import

### Export Learning Data

```python
# Export all learning data to JSON
tracker.export_learning_data("learning_data_export.json")

# The export includes:
# - All learning events
# - Goal progress data
# - Convergence history
# - Comprehensive learning report
```

### Real-time Metrics

```python
# Get real-time metrics for dashboards
metrics = tracker.get_real_time_metrics()

print(f"Current performance: {metrics['current_performance']:.3f}")
print(f"Current learning rate: {metrics['current_learning_rate']:.4f}")
print(f"Recent average: {metrics['recent_avg_performance']:.3f}")
print(f"Performance trend: {metrics['performance_trend']}")
print(f"Active goals: {metrics['active_goals_count']}")
print(f"Total events: {metrics['total_learning_events']}")
```

## Data Structures

### LearningEvent

```python
@dataclass
class LearningEvent:
    timestamp: float
    task_type: str
    performance_score: float
    learning_rate: float
    episode_count: int
    convergence_metric: float
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### GoalProgress

```python
@dataclass
class GoalProgress:
    goal_id: str
    goal_description: str
    creation_time: float
    target_performance: float
    current_performance: float
    completion_percentage: float
    estimated_completion_time: Optional[float] = None
    is_completed: bool = False
    completion_time: Optional[float] = None
```

## Integration Examples

### Integration with Training Systems

```python
from src.training.unsupervised.learning_pipeline import LearningPipeline
from src.monitoring.learning_tracker import LearningProgressTracker

class MonitoredLearningPipeline(LearningPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracker = LearningProgressTracker()
        
        # Create learning goals
        self.tracker.create_goal(
            "convergence", 
            "Achieve convergence", 
            0.01  # Convergence threshold
        )
    
    def train_epoch(self, data):
        # Perform training
        performance = super().train_epoch(data)
        
        # Record learning event
        self.tracker.record_learning_event(
            task_type=self.task_type,
            performance_score=performance['accuracy'],
            learning_rate=self.learning_rate,
            episode_count=self.epoch,
            convergence_metric=performance['loss']
        )
        
        # Update goal progress
        self.tracker.update_goal_progress("convergence", performance['loss'])
        
        return performance
```

### Integration with Storage Systems

```python
from src.storage.hdf5_storage import HDF5Storage

# Store learning data in HDF5
storage = HDF5Storage("learning_data.h5")

def store_learning_event(event):
    storage.store_learning_event(event.to_dict())

# Add callback to tracker
tracker = LearningProgressTracker()
# Note: Callbacks would need to be implemented in the tracker
```

### Integration with Web Dashboard

```python
from flask import Flask, jsonify
from src.monitoring.learning_tracker import LearningProgressTracker

app = Flask(__name__)
tracker = LearningProgressTracker()

@app.route('/api/learning/metrics')
def get_learning_metrics():
    return jsonify(tracker.get_real_time_metrics())

@app.route('/api/learning/report')
def get_learning_report():
    return jsonify(tracker.generate_learning_report())

@app.route('/api/learning/goals')
def get_goals():
    return jsonify(tracker.get_goal_summary())
```

## Performance Optimization

### Memory Management

```python
# Limit history to prevent excessive memory usage
tracker = LearningProgressTracker(max_history=10000)  # Instead of default 50000

# For high-frequency logging, use smaller history
high_freq_tracker = LearningProgressTracker(max_history=1000)
```

### Batch Processing

```python
# Process multiple events efficiently
events = []
for i in range(1000):
    events.append({
        'task_type': 'batch_task',
        'performance_score': 0.5 + i * 0.0005,
        'learning_rate': 0.01,
        'episode_count': i,
        'convergence_metric': 0.1 - i * 0.0001
    })

# Batch record (if implemented)
for event_data in events:
    tracker.record_learning_event(**event_data)
```

## Best Practices

1. **Consistent Task Types**: Use consistent naming for task types to enable proper grouping and analysis
2. **Meaningful Metadata**: Include relevant metadata for later analysis and debugging
3. **Regular Goal Updates**: Update goal progress regularly to track achievement accurately
4. **Appropriate History Limits**: Balance between data retention and memory usage
5. **Export Regularly**: Export learning data periodically to prevent data loss
6. **Monitor Convergence**: Use convergence metrics to detect when learning has stabilized

## Troubleshooting

### Memory Issues

```python
# Check current memory usage
import sys
print(f"Learning events: {len(tracker.learning_events)}")
print(f"Goals: {len(tracker.goals)}")
print(f"Memory usage: {sys.getsizeof(tracker)} bytes")

# Reduce history if needed
tracker = LearningProgressTracker(max_history=5000)
```

### Performance Issues

```python
# For large datasets, use sampling
if len(tracker.learning_events) > 10000:
    # Get recent subset for analysis
    recent_events = list(tracker.learning_events)[-1000:]
```

### Data Integrity

```python
# Validate learning events
def validate_event(task_type, performance_score, learning_rate, episode_count, convergence_metric):
    assert isinstance(task_type, str), "task_type must be string"
    assert 0 <= performance_score <= 1, "performance_score must be between 0 and 1"
    assert learning_rate > 0, "learning_rate must be positive"
    assert episode_count >= 0, "episode_count must be non-negative"
    assert convergence_metric >= 0, "convergence_metric must be non-negative"

# Use validation before recording
validate_event("test", 0.85, 0.01, 100, 0.05)
tracker.record_learning_event("test", 0.85, 0.01, 100, 0.05)
```

## Advanced Analytics

### Statistical Analysis

```python
import numpy as np
from scipy import stats

# Get performance data for statistical analysis
timestamps, scores = tracker.get_learning_curve("classification")

# Calculate statistics
mean_score = np.mean(scores)
std_score = np.std(scores)
median_score = np.median(scores)

# Perform trend test
slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(scores)), scores)
print(f"Learning trend: slope={slope:.4f}, r²={r_value**2:.3f}, p={p_value:.3f}")
```

### Comparative Analysis

```python
# Compare performance across different task types
task_types = ["classification", "regression", "reinforcement_learning"]

for task_type in task_types:
    trend = tracker.get_performance_trend(task_type)
    print(f"{task_type}:")
    print(f"  Trend: {trend['trend']:.4f}")
    print(f"  Recent avg: {trend['recent_avg']:.3f}")
    print(f"  Improvement rate: {trend['improvement_rate']:.4f}")
```
# Monitoring API Reference

This directory contains detailed API documentation for all monitoring and visualization components.

## API Documentation

### Core Components
- [SystemMonitor API](system_monitor_api.md) - Real-time system monitoring
- [LearningProgressTracker API](learning_tracker_api.md) - Learning progress tracking
- [Visualization API](visualization_api.md) - Neural activity and network visualization

### Data Structures
- [SystemMetrics](data_structures.md#systemmetrics) - System resource metrics
- [PerformanceMetrics](data_structures.md#performancemetrics) - AI performance metrics
- [LearningEvent](data_structures.md#learningevent) - Individual learning events
- [GoalProgress](data_structures.md#goalprogress) - Goal tracking data
- [SpikeData](data_structures.md#spikedata) - Neural spike data
- [NetworkTopology](data_structures.md#networktopology) - Network structure data

## Quick Reference

### SystemMonitor

```python
from src.monitoring.system_monitor import SystemMonitor

monitor = SystemMonitor(collection_interval=1.0, enable_alerts=True)
monitor.start_monitoring()
monitor.record_inference_time(15.5)
monitor.stop_monitoring()
```

### LearningProgressTracker

```python
from src.monitoring.learning_tracker import LearningProgressTracker

tracker = LearningProgressTracker()
tracker.record_learning_event("classification", 0.85, 0.01, 100, 0.05)
tracker.create_goal("accuracy", "90% accuracy", 0.9)
```

### Visualization

```python
from src.monitoring.visualization import NeuralActivityVisualizer, SpikeData

visualizer = NeuralActivityVisualizer()
spike_data = SpikeData([1,2,3], [0.1,0.2,0.3])
fig = visualizer.create_spike_raster_plot(spike_data)
```

## Error Handling

All API methods include comprehensive error handling and validation:

```python
try:
    monitor.record_inference_time(-1.0)  # Invalid negative time
except ValueError as e:
    print(f"Validation error: {e}")

try:
    tracker.update_goal_progress("nonexistent_goal", 0.5)
except KeyError as e:
    print(f"Goal not found: {e}")
```

## Thread Safety

All monitoring components are thread-safe and can be used in multi-threaded applications:

```python
import threading

def worker():
    monitor.record_inference_time(process_data())

# Safe to use from multiple threads
for i in range(4):
    threading.Thread(target=worker).start()
```

## Performance Considerations

- **SystemMonitor**: <1% CPU overhead during continuous monitoring
- **LearningProgressTracker**: O(1) insertion, O(n) for analysis operations
- **Visualization**: Memory usage scales with data size, use sampling for large datasets

## Configuration

Most components accept configuration parameters:

```python
# Low-overhead monitoring
monitor = SystemMonitor(collection_interval=5.0, enable_alerts=False)

# Limited history for memory efficiency
tracker = LearningProgressTracker(max_history=1000)

# Custom visualization settings
visualizer = NeuralActivityVisualizer(figsize=(16, 10))
```
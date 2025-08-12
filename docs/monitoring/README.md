# Monitoring and Visualization Systems

The Godly AI System includes comprehensive monitoring and visualization capabilities for tracking system performance, learning progress, and neural activity patterns.

## Overview

The monitoring system consists of three main components:

1. **System Monitor** - Real-time system resource and performance monitoring
2. **Learning Tracker** - Learning progress tracking and goal management
3. **Visualization** - Neural activity and network topology visualization

## Quick Start

### Basic System Monitoring

```python
from src.monitoring.system_monitor import SystemMonitor

# Create and start system monitor
monitor = SystemMonitor(collection_interval=1.0)
monitor.start_monitoring()

# Record performance metrics
monitor.record_inference_time(15.5)
monitor.record_learning_progress(0.01)
monitor.record_spike_activity(120.0, 0.8)

# Get current status
status = monitor.get_status()
print(f"Monitoring active: {status['monitoring_active']}")

# Stop monitoring
monitor.stop_monitoring()
```

### Learning Progress Tracking

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
    convergence_metric=0.05
)

# Create and track goals
tracker.create_goal("accuracy_goal", "Achieve 90% accuracy", 0.9)
tracker.update_goal_progress("accuracy_goal", 0.7)

# Generate comprehensive report
report = tracker.generate_learning_report()
```

### Neural Activity Visualization

```python
from src.monitoring.visualization import NeuralActivityVisualizer, SpikeData

# Create visualizer
visualizer = NeuralActivityVisualizer()

# Create spike data
spike_data = SpikeData(
    neuron_ids=[1, 2, 3, 1, 2],
    spike_times=[0.1, 0.2, 0.3, 0.4, 0.5],
    amplitudes=[1.0, 1.5, 2.0, 0.8, 1.2]
)

# Generate visualizations
raster_plot = visualizer.create_spike_raster_plot(spike_data)
heatmap = visualizer.create_firing_rate_heatmap(spike_data)
interactive_plot = visualizer.create_interactive_spike_plot(spike_data)
```

## Components

### [System Monitor](system_monitor.md)
Real-time monitoring of system resources and AI performance metrics.

### [Learning Tracker](learning_tracker.md)
Comprehensive learning progress tracking with goal management and trend analysis.

### [Visualization](visualization.md)
Neural activity and network topology visualization tools.

## Configuration

### Environment Variables

```bash
# Optional: Set monitoring configuration
export GODLY_AI_MONITOR_INTERVAL=1.0
export GODLY_AI_MONITOR_LOG_LEVEL=INFO
export GODLY_AI_MONITOR_ALERTS=true
```

### Configuration File

```yaml
# config/monitoring.yaml
monitoring:
  system:
    collection_interval: 1.0
    enable_alerts: true
    cpu_threshold: 80.0
    memory_threshold: 85.0
    disk_threshold: 90.0
  
  learning:
    max_history: 50000
    convergence_threshold: 0.01
  
  visualization:
    default_figsize: [12, 8]
    color_palette: "viridis"
    save_format: "png"
    dpi: 300
```

## Performance Characteristics

- **Monitoring Overhead**: <1% CPU usage during continuous monitoring
- **Memory Efficiency**: <100MB memory increase for visualization operations
- **Real-time Capability**: Sub-second metric collection and processing
- **Scalability**: Handles 10,000+ spike events efficiently

## Integration

The monitoring system integrates seamlessly with:

- **Storage Systems**: Redis, HDF5, SQLite backends
- **Performance Optimization**: JIT compilation, parallel processing
- **Training Systems**: Evolution, self-modification, unsupervised learning
- **Memory Systems**: Working, episodic, semantic, meta-memory

## Testing

Run the comprehensive test suite:

```bash
# Run all monitoring tests
pytest tests/test_system_monitor.py tests/test_learning_tracker.py tests/test_visualization.py -v

# Run with coverage
pytest tests/test_*monitor*.py tests/test_*tracker*.py tests/test_*visualization*.py --cov=src/monitoring --cov-report=html
```

## Examples

See the [examples](../examples/) directory for complete usage examples:

- [Basic Monitoring](../examples/basic_monitoring_demo.py)
- [Learning Tracking](../examples/learning_tracking_demo.py)
- [Visualization Gallery](../examples/visualization_gallery.py)
- [Performance Analysis](../examples/performance_analysis_demo.py)

## API Reference

- [System Monitor API](api/system_monitor_api.md)
- [Learning Tracker API](api/learning_tracker_api.md)
- [Visualization API](api/visualization_api.md)

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Reduce `max_history` parameter in collectors
2. **Slow Visualizations**: Use smaller datasets or increase `bin_size` for heatmaps
3. **Missing Dependencies**: Install with `pip install -r requirements.txt`
4. **Permission Errors**: Ensure write permissions for log and export directories

### Performance Tuning

- Increase `collection_interval` for lower overhead
- Use `neuron_subset` parameter for large spike datasets
- Enable JIT compilation for visualization-heavy workloads
- Use Redis storage for high-frequency metric collection

## Contributing

When contributing to the monitoring system:

1. Add comprehensive tests for new features
2. Update documentation and examples
3. Ensure backward compatibility
4. Follow the existing code style and patterns
5. Test performance impact of changes

## License

This monitoring system is part of the Godly AI System and follows the same licensing terms.
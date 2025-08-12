# System Monitor

The System Monitor provides real-time monitoring of system resources and AI performance metrics with minimal overhead.

## Overview

The System Monitor consists of three main classes:

- **SystemMonitor**: Main coordinator for monitoring operations
- **MetricsCollector**: Collects and aggregates performance metrics
- **ResourceMonitor**: Monitors system resources and generates alerts

## SystemMonitor

### Basic Usage

```python
from src.monitoring.system_monitor import SystemMonitor

# Create monitor with custom settings
monitor = SystemMonitor(
    collection_interval=1.0,  # Collect metrics every second
    log_file="monitoring.log",  # Optional log file
    enable_alerts=True  # Enable resource alerts
)

# Start monitoring
monitor.start_monitoring()

# Record AI performance metrics
monitor.record_inference_time(15.5)  # milliseconds
monitor.record_learning_progress(0.01)  # learning rate
monitor.record_spike_activity(120.0, 0.8)  # Hz, activity level
monitor.record_plasticity_update(5)  # update count
monitor.record_topology_change(2)  # change count

# Get current status
status = monitor.get_status()
print(f"Active: {status['monitoring_active']}")
print(f"Metrics collected: {status['metrics_count']}")

# Export metrics to file
monitor.export_metrics("metrics_export.json")

# Stop monitoring
monitor.stop_monitoring()
```

### Configuration Options

```python
monitor = SystemMonitor(
    collection_interval=0.5,  # Faster collection (higher overhead)
    log_file="/var/log/godly_ai_monitor.log",
    enable_alerts=True
)
```

### Performance Metrics

The SystemMonitor can record various AI system performance metrics:

```python
# Inference performance
monitor.record_inference_time(time_ms)

# Learning dynamics
monitor.record_learning_progress(learning_rate)

# Neural activity
monitor.record_spike_activity(spike_rate_hz, network_activity)

# Plasticity changes
monitor.record_plasticity_update(update_count)

# Topology evolution
monitor.record_topology_change(change_count)
```

## MetricsCollector

### Direct Usage

```python
from src.monitoring.system_monitor import MetricsCollector

collector = MetricsCollector(max_history=10000)

# Collect system metrics
system_metrics = collector.collect_system_metrics()
print(f"CPU: {system_metrics.cpu_percent}%")
print(f"Memory: {system_metrics.memory_percent}%")

# Record performance metrics
collector.record_performance_metric(
    inference_time_ms=12.5,
    learning_rate=0.001,
    spike_rate_hz=95.0,
    network_activity=0.7
)

# Get recent metrics
recent_system = collector.get_recent_system_metrics(100)
recent_performance = collector.get_recent_performance_metrics(100)

# Get summary statistics
summary = collector.get_system_summary()
print(f"Average CPU: {summary['cpu_avg']:.1f}%")
print(f"Average Memory: {summary['memory_avg']:.1f}%")
```

### Callback System

```python
def metrics_callback(data):
    if data['type'] == 'system':
        print(f"System update: CPU {data['data']['cpu_percent']}%")
    elif data['type'] == 'performance':
        print(f"Performance update: {data['data']['inference_time_ms']}ms")

collector.add_callback(metrics_callback)
```

## ResourceMonitor

### Basic Usage

```python
from src.monitoring.system_monitor import ResourceMonitor

# Create monitor with custom thresholds
resource_monitor = ResourceMonitor(
    cpu_threshold=80.0,     # Alert at 80% CPU
    memory_threshold=85.0,  # Alert at 85% memory
    disk_threshold=90.0     # Alert at 90% disk
)

# Check resources against thresholds
alerts = resource_monitor.check_resources(system_metrics)
for alert in alerts:
    print(f"ALERT: {alert['message']} (severity: {alert['severity']})")

# Get recent alerts
recent_alerts = resource_monitor.get_recent_alerts(10)
```

### Alert Callbacks

```python
def alert_handler(alert):
    if alert['severity'] == 'critical':
        # Send notification, scale resources, etc.
        print(f"CRITICAL: {alert['message']}")
    else:
        print(f"WARNING: {alert['message']}")

resource_monitor.add_alert_callback(alert_handler)
```

## Data Structures

### SystemMetrics

```python
@dataclass
class SystemMetrics:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    thread_count: int
```

### PerformanceMetrics

```python
@dataclass
class PerformanceMetrics:
    timestamp: float
    inference_time_ms: float = 0.0
    learning_rate: float = 0.0
    memory_consolidation_time_ms: float = 0.0
    spike_rate_hz: float = 0.0
    network_activity: float = 0.0
    plasticity_updates: int = 0
    topology_changes: int = 0
    goal_generation_rate: float = 0.0
    exploration_efficiency: float = 0.0
```

## Advanced Usage

### Custom Monitoring Loop

```python
import time
from src.monitoring.system_monitor import MetricsCollector, ResourceMonitor

collector = MetricsCollector()
resource_monitor = ResourceMonitor()

while True:
    # Collect system metrics
    metrics = collector.collect_system_metrics()
    
    # Check for alerts
    alerts = resource_monitor.check_resources(metrics)
    
    # Custom processing
    if metrics.cpu_percent > 90:
        print("High CPU usage detected!")
    
    time.sleep(1.0)
```

### Integration with Storage

```python
from src.storage.redis_storage import RedisStorage

# Store metrics in Redis
storage = RedisStorage()
monitor = SystemMonitor()

def store_metrics(data):
    storage.store_metrics(data)

monitor.metrics_collector.add_callback(store_metrics)
```

### Batch Export

```python
# Export large batches of metrics
system_metrics = monitor.metrics_collector.get_recent_system_metrics(10000)
performance_metrics = monitor.metrics_collector.get_recent_performance_metrics(10000)

# Convert to pandas DataFrame for analysis
import pandas as pd
df_system = pd.DataFrame([m.to_dict() for m in system_metrics])
df_performance = pd.DataFrame([m.to_dict() for m in performance_metrics])
```

## Performance Considerations

### Overhead Optimization

```python
# Low overhead configuration
monitor = SystemMonitor(
    collection_interval=5.0,  # Collect less frequently
    enable_alerts=False       # Disable alerts if not needed
)

# Limit history to reduce memory usage
collector = MetricsCollector(max_history=1000)
```

### Thread Safety

The SystemMonitor is thread-safe and can be used in multi-threaded applications:

```python
import threading

def worker_thread():
    while True:
        # Record metrics from worker threads
        monitor.record_inference_time(process_task())
        time.sleep(0.1)

# Start multiple worker threads
for i in range(4):
    thread = threading.Thread(target=worker_thread)
    thread.daemon = True
    thread.start()
```

## Error Handling

```python
try:
    monitor = SystemMonitor()
    monitor.start_monitoring()
    
    # Monitor operations
    monitor.record_inference_time(15.0)
    
except Exception as e:
    print(f"Monitoring error: {e}")
finally:
    monitor.stop_monitoring()
```

## Best Practices

1. **Choose appropriate collection intervals**: Balance between monitoring granularity and system overhead
2. **Set reasonable history limits**: Prevent excessive memory usage with large `max_history` values
3. **Use callbacks for real-time processing**: Avoid polling for better performance
4. **Configure appropriate alert thresholds**: Prevent alert fatigue with well-tuned thresholds
5. **Export metrics regularly**: Prevent data loss by periodically exporting collected metrics
6. **Monitor the monitor**: Track the monitoring system's own resource usage

## Troubleshooting

### High Memory Usage
```python
# Reduce history limits
collector = MetricsCollector(max_history=1000)  # Instead of default 10000
```

### High CPU Usage
```python
# Increase collection interval
monitor = SystemMonitor(collection_interval=5.0)  # Instead of 1.0
```

### Missing Metrics
```python
# Check if monitoring is active
status = monitor.get_status()
if not status['monitoring_active']:
    monitor.start_monitoring()
```

### Permission Errors
```python
# Use relative paths or ensure write permissions
monitor = SystemMonitor(log_file="./monitoring.log")
```
"""
Real-time system monitoring for the Godly AI System.

This module implements comprehensive system monitoring including:
- Performance metrics collection and logging
- Resource usage monitoring (CPU, memory, storage)
- Real-time data collection with minimal overhead
"""

import time
import threading
import logging
import psutil
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import json
from pathlib import Path


@dataclass
class SystemMetrics:
    """Container for system performance metrics."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'network_bytes_sent': self.network_bytes_sent,
            'network_bytes_recv': self.network_bytes_recv,
            'process_count': self.process_count,
            'thread_count': self.thread_count
        }


@dataclass
class PerformanceMetrics:
    """Container for AI system performance metrics."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'inference_time_ms': self.inference_time_ms,
            'learning_rate': self.learning_rate,
            'memory_consolidation_time_ms': self.memory_consolidation_time_ms,
            'spike_rate_hz': self.spike_rate_hz,
            'network_activity': self.network_activity,
            'plasticity_updates': self.plasticity_updates,
            'topology_changes': self.topology_changes,
            'goal_generation_rate': self.goal_generation_rate,
            'exploration_efficiency': self.exploration_efficiency
        }


class MetricsCollector:
    """Collects and aggregates performance metrics with minimal overhead."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.system_metrics: deque = deque(maxlen=max_history)
        self.performance_metrics: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Initialize network counters
        self._last_network_stats = psutil.net_io_counters()
        
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback function to be called when new metrics are collected."""
        self._callbacks.append(callback)
        
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024 * 1024 * 1024)
        
        # Network usage
        network_stats = psutil.net_io_counters()
        network_bytes_sent = network_stats.bytes_sent
        network_bytes_recv = network_stats.bytes_recv
        
        # Process information
        process_count = len(psutil.pids())
        thread_count = threading.active_count()
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_bytes_sent=network_bytes_sent,
            network_bytes_recv=network_bytes_recv,
            process_count=process_count,
            thread_count=thread_count
        )
        
        with self._lock:
            self.system_metrics.append(metrics)
            
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback({'type': 'system', 'data': metrics.to_dict()})
            except Exception as e:
                logging.warning(f"Metrics callback failed: {e}")
                
        return metrics
        
    def record_performance_metric(self, **kwargs) -> None:
        """Record AI system performance metrics."""
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            **kwargs
        )
        
        with self._lock:
            self.performance_metrics.append(metrics)
            
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback({'type': 'performance', 'data': metrics.to_dict()})
            except Exception as e:
                logging.warning(f"Performance callback failed: {e}")
                
    def get_recent_system_metrics(self, count: int = 100) -> List[SystemMetrics]:
        """Get recent system metrics."""
        with self._lock:
            return list(self.system_metrics)[-count:]
            
    def get_recent_performance_metrics(self, count: int = 100) -> List[PerformanceMetrics]:
        """Get recent performance metrics."""
        with self._lock:
            return list(self.performance_metrics)[-count:]
            
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary statistics for system metrics."""
        with self._lock:
            if not self.system_metrics:
                return {}
                
            recent_metrics = list(self.system_metrics)[-100:]  # Last 100 samples
            
            cpu_values = [m.cpu_percent for m in recent_metrics]
            memory_values = [m.memory_percent for m in recent_metrics]
            
            return {
                'cpu_avg': np.mean(cpu_values),
                'cpu_max': np.max(cpu_values),
                'cpu_min': np.min(cpu_values),
                'memory_avg': np.mean(memory_values),
                'memory_max': np.max(memory_values),
                'memory_min': np.min(memory_values),
                'sample_count': len(recent_metrics),
                'time_span_seconds': recent_metrics[-1].timestamp - recent_metrics[0].timestamp if len(recent_metrics) > 1 else 0
            }


class ResourceMonitor:
    """Monitors system resources and provides alerts for resource constraints."""
    
    def __init__(self, 
                 cpu_threshold: float = 80.0,
                 memory_threshold: float = 85.0,
                 disk_threshold: float = 90.0):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
        self._alerts: List[Dict[str, Any]] = []
        self._alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for resource alerts."""
        self._alert_callbacks.append(callback)
        
    def check_resources(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Check resource usage against thresholds and generate alerts."""
        alerts = []
        
        # CPU usage alert
        if metrics.cpu_percent > self.cpu_threshold:
            alert = {
                'type': 'cpu_high',
                'severity': 'warning' if metrics.cpu_percent < 95 else 'critical',
                'message': f'High CPU usage: {metrics.cpu_percent:.1f}%',
                'value': metrics.cpu_percent,
                'threshold': self.cpu_threshold,
                'timestamp': metrics.timestamp
            }
            alerts.append(alert)
            
        # Memory usage alert
        if metrics.memory_percent > self.memory_threshold:
            alert = {
                'type': 'memory_high',
                'severity': 'warning' if metrics.memory_percent < 95 else 'critical',
                'message': f'High memory usage: {metrics.memory_percent:.1f}%',
                'value': metrics.memory_percent,
                'threshold': self.memory_threshold,
                'timestamp': metrics.timestamp
            }
            alerts.append(alert)
            
        # Disk usage alert
        if metrics.disk_usage_percent > self.disk_threshold:
            alert = {
                'type': 'disk_high',
                'severity': 'warning' if metrics.disk_usage_percent < 98 else 'critical',
                'message': f'High disk usage: {metrics.disk_usage_percent:.1f}%',
                'value': metrics.disk_usage_percent,
                'threshold': self.disk_threshold,
                'timestamp': metrics.timestamp
            }
            alerts.append(alert)
            
        # Store alerts and notify callbacks
        for alert in alerts:
            self._alerts.append(alert)
            for callback in self._alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logging.warning(f"Alert callback failed: {e}")
                    
        return alerts
        
    def get_recent_alerts(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent resource alerts."""
        return self._alerts[-count:]


class SystemMonitor:
    """Main system monitoring coordinator."""
    
    def __init__(self, 
                 collection_interval: float = 1.0,
                 log_file: Optional[str] = None,
                 enable_alerts: bool = True):
        self.collection_interval = collection_interval
        self.enable_alerts = enable_alerts
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor() if enable_alerts else None
        
        # Monitoring control
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        if log_file:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            
        # Setup alert logging
        if self.resource_monitor:
            self.resource_monitor.add_alert_callback(self._log_alert)
            
    def _log_alert(self, alert: Dict[str, Any]) -> None:
        """Log resource alerts."""
        self.logger.warning(f"Resource Alert: {alert['message']}")
        
    def start_monitoring(self) -> None:
        """Start continuous system monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("System monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("System monitoring stopped")
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                # Collect system metrics
                metrics = self.metrics_collector.collect_system_metrics()
                
                # Check for resource alerts
                if self.resource_monitor:
                    alerts = self.resource_monitor.check_resources(metrics)
                    if alerts:
                        self.logger.info(f"Generated {len(alerts)} resource alerts")
                        
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                
            time.sleep(self.collection_interval)
            
    def record_inference_time(self, time_ms: float) -> None:
        """Record inference time metric."""
        self.metrics_collector.record_performance_metric(inference_time_ms=time_ms)
        
    def record_learning_progress(self, learning_rate: float) -> None:
        """Record learning progress metric."""
        self.metrics_collector.record_performance_metric(learning_rate=learning_rate)
        
    def record_spike_activity(self, spike_rate_hz: float, network_activity: float) -> None:
        """Record neural spike activity metrics."""
        self.metrics_collector.record_performance_metric(
            spike_rate_hz=spike_rate_hz,
            network_activity=network_activity
        )
        
    def record_plasticity_update(self, update_count: int = 1) -> None:
        """Record plasticity update event."""
        self.metrics_collector.record_performance_metric(plasticity_updates=update_count)
        
    def record_topology_change(self, change_count: int = 1) -> None:
        """Record network topology change event."""
        self.metrics_collector.record_performance_metric(topology_changes=change_count)
        
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status and summary."""
        return {
            'monitoring_active': self._monitoring,
            'collection_interval': self.collection_interval,
            'system_summary': self.metrics_collector.get_system_summary(),
            'recent_alerts': self.resource_monitor.get_recent_alerts(10) if self.resource_monitor else [],
            'metrics_count': len(self.metrics_collector.system_metrics)
        }
        
    def export_metrics(self, filepath: str, format: str = 'json') -> None:
        """Export collected metrics to file."""
        system_metrics = [m.to_dict() for m in self.metrics_collector.get_recent_system_metrics(1000)]
        performance_metrics = [m.to_dict() for m in self.metrics_collector.get_recent_performance_metrics(1000)]
        
        data = {
            'export_timestamp': time.time(),
            'system_metrics': system_metrics,
            'performance_metrics': performance_metrics,
            'alerts': self.resource_monitor.get_recent_alerts(100) if self.resource_monitor else []
        }
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        self.logger.info(f"Metrics exported to {filepath}")
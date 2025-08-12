"""
Tests for the system monitoring functionality.

This module tests:
- Performance metrics collection and logging
- Resource usage monitoring accuracy
- Learning progress tracking
- Monitoring overhead and performance
"""

import pytest
import time
import threading
import tempfile
import json
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.monitoring.system_monitor import (
    SystemMonitor, MetricsCollector, ResourceMonitor,
    SystemMetrics, PerformanceMetrics
)


class TestSystemMetrics:
    """Test SystemMetrics data structure."""
    
    def test_system_metrics_creation(self):
        """Test SystemMetrics creation and serialization."""
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_mb=1024.0,
            memory_available_mb=2048.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150,
            thread_count=10
        )
        
        # Test serialization
        data = metrics.to_dict()
        assert isinstance(data, dict)
        assert data['cpu_percent'] == 50.0
        assert data['memory_percent'] == 60.0
        assert 'timestamp' in data
        
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation and serialization."""
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            inference_time_ms=10.5,
            learning_rate=0.001,
            spike_rate_hz=100.0,
            network_activity=0.8
        )
        
        data = metrics.to_dict()
        assert isinstance(data, dict)
        assert data['inference_time_ms'] == 10.5
        assert data['learning_rate'] == 0.001
        assert data['spike_rate_hz'] == 100.0


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector(max_history=100)
        
    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization."""
        assert self.collector.max_history == 100
        assert len(self.collector.system_metrics) == 0
        assert len(self.collector.performance_metrics) == 0
        
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @patch('psutil.pids')
    @patch('threading.active_count')
    def test_collect_system_metrics(self, mock_thread_count, mock_pids, 
                                  mock_net, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection."""
        # Mock system calls
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(
            percent=60.0, used=1024*1024*1024, available=2048*1024*1024
        )
        mock_disk.return_value = Mock(
            total=1000*1024*1024*1024, used=700*1024*1024*1024, 
            free=300*1024*1024*1024
        )
        mock_net.return_value = Mock(bytes_sent=1000, bytes_recv=2000)
        mock_pids.return_value = list(range(150))
        mock_thread_count.return_value = 10
        
        # Collect metrics
        metrics = self.collector.collect_system_metrics()
        
        # Verify metrics
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 60.0
        assert len(self.collector.system_metrics) == 1
        
    def test_record_performance_metric(self):
        """Test performance metrics recording."""
        self.collector.record_performance_metric(
            inference_time_ms=15.0,
            learning_rate=0.01,
            spike_rate_hz=120.0
        )
        
        assert len(self.collector.performance_metrics) == 1
        metrics = self.collector.performance_metrics[0]
        assert metrics.inference_time_ms == 15.0
        assert metrics.learning_rate == 0.01
        assert metrics.spike_rate_hz == 120.0
        
    def test_callback_functionality(self):
        """Test metrics callback system."""
        callback_data = []
        
        def test_callback(data):
            callback_data.append(data)
            
        self.collector.add_callback(test_callback)
        self.collector.record_performance_metric(inference_time_ms=10.0)
        
        assert len(callback_data) == 1
        assert callback_data[0]['type'] == 'performance'
        assert callback_data[0]['data']['inference_time_ms'] == 10.0
        
    def test_get_system_summary(self):
        """Test system metrics summary generation."""
        # Add some mock metrics
        for i in range(10):
            metrics = SystemMetrics(
                timestamp=time.time() + i,
                cpu_percent=50.0 + i,
                memory_percent=60.0 + i,
                memory_used_mb=1024.0,
                memory_available_mb=2048.0,
                disk_usage_percent=70.0,
                disk_free_gb=100.0,
                network_bytes_sent=1000,
                network_bytes_recv=2000,
                process_count=150,
                thread_count=10
            )
            self.collector.system_metrics.append(metrics)
            
        summary = self.collector.get_system_summary()
        
        assert 'cpu_avg' in summary
        assert 'memory_avg' in summary
        assert summary['sample_count'] == 10
        assert summary['cpu_avg'] == 54.5  # Average of 50-59
        
    def test_max_history_limit(self):
        """Test that metrics collection respects max_history limit."""
        collector = MetricsCollector(max_history=5)
        
        # Add more metrics than the limit
        for i in range(10):
            metrics = SystemMetrics(
                timestamp=time.time() + i,
                cpu_percent=50.0,
                memory_percent=60.0,
                memory_used_mb=1024.0,
                memory_available_mb=2048.0,
                disk_usage_percent=70.0,
                disk_free_gb=100.0,
                network_bytes_sent=1000,
                network_bytes_recv=2000,
                process_count=150,
                thread_count=10
            )
            collector.system_metrics.append(metrics)
            
        assert len(collector.system_metrics) == 5


class TestResourceMonitor:
    """Test ResourceMonitor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = ResourceMonitor(
            cpu_threshold=80.0,
            memory_threshold=85.0,
            disk_threshold=90.0
        )
        
    def test_resource_monitor_initialization(self):
        """Test ResourceMonitor initialization."""
        assert self.monitor.cpu_threshold == 80.0
        assert self.monitor.memory_threshold == 85.0
        assert self.monitor.disk_threshold == 90.0
        
    def test_check_resources_no_alerts(self):
        """Test resource checking with no alerts."""
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,  # Below threshold
            memory_percent=60.0,  # Below threshold
            memory_used_mb=1024.0,
            memory_available_mb=2048.0,
            disk_usage_percent=70.0,  # Below threshold
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150,
            thread_count=10
        )
        
        alerts = self.monitor.check_resources(metrics)
        assert len(alerts) == 0
        
    def test_check_resources_with_alerts(self):
        """Test resource checking with alerts."""
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=90.0,  # Above threshold
            memory_percent=90.0,  # Above threshold
            memory_used_mb=1024.0,
            memory_available_mb=2048.0,
            disk_usage_percent=95.0,  # Above threshold
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150,
            thread_count=10
        )
        
        alerts = self.monitor.check_resources(metrics)
        assert len(alerts) == 3  # CPU, memory, and disk alerts
        
        # Check alert types
        alert_types = [alert['type'] for alert in alerts]
        assert 'cpu_high' in alert_types
        assert 'memory_high' in alert_types
        assert 'disk_high' in alert_types
        
    def test_alert_severity_levels(self):
        """Test alert severity classification."""
        # Critical level metrics
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=98.0,  # Critical level
            memory_percent=97.0,  # Critical level
            memory_used_mb=1024.0,
            memory_available_mb=2048.0,
            disk_usage_percent=99.0,  # Critical level
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150,
            thread_count=10
        )
        
        alerts = self.monitor.check_resources(metrics)
        
        # All alerts should be critical
        for alert in alerts:
            assert alert['severity'] == 'critical'
            
    def test_alert_callbacks(self):
        """Test alert callback system."""
        callback_alerts = []
        
        def alert_callback(alert):
            callback_alerts.append(alert)
            
        self.monitor.add_alert_callback(alert_callback)
        
        # Trigger an alert
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=90.0,
            memory_percent=60.0,
            memory_used_mb=1024.0,
            memory_available_mb=2048.0,
            disk_usage_percent=70.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150,
            thread_count=10
        )
        
        self.monitor.check_resources(metrics)
        
        assert len(callback_alerts) == 1
        assert callback_alerts[0]['type'] == 'cpu_high'


class TestSystemMonitor:
    """Test SystemMonitor main coordinator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test_monitor.log"
        
    def test_system_monitor_initialization(self):
        """Test SystemMonitor initialization."""
        monitor = SystemMonitor(
            collection_interval=0.5,
            log_file=str(self.log_file),
            enable_alerts=True
        )
        
        assert monitor.collection_interval == 0.5
        assert monitor.enable_alerts is True
        assert monitor.resource_monitor is not None
        assert isinstance(monitor.metrics_collector, MetricsCollector)
        
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle."""
        monitor = SystemMonitor(collection_interval=0.1)
        
        # Initially not monitoring
        assert monitor._monitoring is False
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor._monitoring is True
        assert monitor._monitor_thread is not None
        
        # Let it run briefly
        time.sleep(0.3)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor._monitoring is False
        
    def test_performance_metric_recording(self):
        """Test performance metric recording methods."""
        monitor = SystemMonitor()
        
        # Test inference time recording
        monitor.record_inference_time(15.5)
        metrics = monitor.metrics_collector.get_recent_performance_metrics(1)
        assert len(metrics) == 1
        assert metrics[0].inference_time_ms == 15.5
        
        # Test learning progress recording
        monitor.record_learning_progress(0.01)
        metrics = monitor.metrics_collector.get_recent_performance_metrics(1)
        assert metrics[0].learning_rate == 0.01
        
        # Test spike activity recording
        monitor.record_spike_activity(120.0, 0.8)
        metrics = monitor.metrics_collector.get_recent_performance_metrics(1)
        assert metrics[0].spike_rate_hz == 120.0
        assert metrics[0].network_activity == 0.8
        
    def test_get_status(self):
        """Test status reporting."""
        monitor = SystemMonitor()
        status = monitor.get_status()
        
        assert isinstance(status, dict)
        assert 'monitoring_active' in status
        assert 'collection_interval' in status
        assert 'system_summary' in status
        assert 'recent_alerts' in status
        assert 'metrics_count' in status
        
    def test_export_metrics(self):
        """Test metrics export functionality."""
        monitor = SystemMonitor()
        
        # Add some test metrics
        monitor.record_inference_time(10.0)
        monitor.record_learning_progress(0.01)
        
        # Export to file
        export_file = Path(self.temp_dir) / "metrics_export.json"
        monitor.export_metrics(str(export_file))
        
        # Verify export
        assert export_file.exists()
        
        with open(export_file, 'r') as f:
            data = json.load(f)
            
        assert 'export_timestamp' in data
        assert 'system_metrics' in data
        assert 'performance_metrics' in data
        assert len(data['performance_metrics']) >= 2
        
    def test_monitoring_overhead(self):
        """Test that monitoring has minimal performance overhead."""
        monitor = SystemMonitor(collection_interval=0.01)  # Very frequent collection
        
        # Measure time for metric collection
        start_time = time.time()
        
        for _ in range(100):
            monitor.record_inference_time(10.0)
            
        end_time = time.time()
        
        # Should complete quickly (less than 1 second for 100 recordings)
        assert (end_time - start_time) < 1.0
        
    def test_concurrent_access(self):
        """Test thread safety of monitoring system."""
        monitor = SystemMonitor()
        
        def record_metrics():
            for i in range(50):
                monitor.record_inference_time(float(i))
                monitor.record_learning_progress(0.01 * i)
                
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=record_metrics)
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Verify all metrics were recorded
        metrics = monitor.metrics_collector.get_recent_performance_metrics(1000)
        assert len(metrics) == 500  # 5 threads * 50 recordings * 2 metrics each
        
    def test_error_handling(self):
        """Test error handling in monitoring system."""
        monitor = SystemMonitor()
        
        # Test callback error handling
        def failing_callback(data):
            raise Exception("Test error")
            
        monitor.metrics_collector.add_callback(failing_callback)
        
        # Should not raise exception despite callback failure
        monitor.record_inference_time(10.0)
        
        # Metrics should still be recorded
        metrics = monitor.metrics_collector.get_recent_performance_metrics(1)
        assert len(metrics) == 1


class TestMonitoringAccuracy:
    """Test monitoring accuracy and precision."""
    
    def test_timing_accuracy(self):
        """Test timing measurement accuracy."""
        collector = MetricsCollector()
        
        # Record metrics with known timing
        start_time = time.time()
        collector.record_performance_metric(inference_time_ms=100.0)
        time.sleep(0.1)  # 100ms delay
        collector.record_performance_metric(inference_time_ms=200.0)
        end_time = time.time()
        
        metrics = collector.get_recent_performance_metrics(2)
        
        # Check timestamp accuracy (should be within reasonable bounds)
        time_diff = metrics[1].timestamp - metrics[0].timestamp
        assert 0.09 < time_diff < 0.15  # Allow some tolerance
        
    def test_metrics_precision(self):
        """Test metrics precision and data integrity."""
        collector = MetricsCollector()
        
        # Test with precise values
        test_values = [1.23456789, 0.00001, 999.999, -0.5]
        
        for value in test_values:
            collector.record_performance_metric(inference_time_ms=value)
            
        metrics = collector.get_recent_performance_metrics(len(test_values))
        
        # Verify precision is maintained
        for i, metric in enumerate(metrics):
            assert abs(metric.inference_time_ms - test_values[i]) < 1e-10
            
    def test_resource_monitoring_accuracy(self):
        """Test resource monitoring accuracy."""
        monitor = ResourceMonitor()
        
        # Create metrics with known values
        test_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=75.5,
            memory_percent=82.3,
            memory_used_mb=1536.7,
            memory_available_mb=2048.0,
            disk_usage_percent=88.9,
            disk_free_gb=50.25,
            network_bytes_sent=12345,
            network_bytes_recv=67890,
            process_count=200,
            thread_count=15
        )
        
        # Check that values are preserved exactly
        data = test_metrics.to_dict()
        assert data['cpu_percent'] == 75.5
        assert data['memory_percent'] == 82.3
        assert data['disk_usage_percent'] == 88.9
        
        # Test alert threshold accuracy
        alerts = monitor.check_resources(test_metrics)
        
        # Should not trigger CPU alert (75.5 < 80.0)
        # Should not trigger memory alert (82.3 < 85.0)
        # Should not trigger disk alert (88.9 < 90.0)
        assert len(alerts) == 0
        
        # Test edge case - exactly at threshold
        edge_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=80.0,  # Exactly at threshold
            memory_percent=85.0,  # Exactly at threshold
            memory_used_mb=1024.0,
            memory_available_mb=2048.0,
            disk_usage_percent=90.0,  # Exactly at threshold
            disk_free_gb=100.0,
            network_bytes_sent=1000,
            network_bytes_recv=2000,
            process_count=150,
            thread_count=10
        )
        
        edge_alerts = monitor.check_resources(edge_metrics)
        # Should not trigger alerts at exactly threshold value
        assert len(edge_alerts) == 0
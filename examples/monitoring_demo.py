#!/usr/bin/env python3
"""
Comprehensive monitoring and visualization demo for the Godly AI System.

This example demonstrates:
- Real-time system monitoring
- Learning progress tracking
- Neural activity visualization
- Network topology visualization
- Memory state visualization
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import monitoring components
from src.monitoring.system_monitor import SystemMonitor
from src.monitoring.learning_tracker import LearningProgressTracker
from src.monitoring.visualization import (
    NeuralActivityVisualizer, NetworkTopologyVisualizer, MemoryStateVisualizer,
    SpikeData, NetworkTopology
)


def demo_system_monitoring():
    """Demonstrate system monitoring capabilities."""
    print("=== System Monitoring Demo ===")
    
    # Create system monitor
    monitor = SystemMonitor(
        collection_interval=0.5,  # Collect metrics every 500ms
        enable_alerts=True
    )
    
    print("Starting system monitoring...")
    monitor.start_monitoring()
    
    # Simulate AI system activity
    for i in range(10):
        # Simulate inference operations
        inference_time = 10 + np.random.normal(0, 2)  # ~10ms with variation
        monitor.record_inference_time(inference_time)
        
        # Simulate learning progress
        learning_rate = 0.01 * (1 - i * 0.05)  # Decreasing learning rate
        monitor.record_learning_progress(learning_rate)
        
        # Simulate neural activity
        spike_rate = 100 + np.random.normal(0, 20)
        network_activity = 0.5 + 0.3 * np.sin(i * 0.5)
        monitor.record_spike_activity(spike_rate, network_activity)
        
        # Simulate plasticity updates
        if i % 3 == 0:
            monitor.record_plasticity_update(np.random.randint(1, 5))
        
        # Simulate topology changes
        if i % 5 == 0:
            monitor.record_topology_change(np.random.randint(1, 3))
        
        time.sleep(0.1)  # Brief pause
        print(f"  Recorded metrics for iteration {i+1}")
    
    # Get monitoring status
    status = monitor.get_status()
    print(f"\nMonitoring Status:")
    print(f"  Active: {status['monitoring_active']}")
    print(f"  Metrics collected: {status['metrics_count']}")
    print(f"  System summary: {status['system_summary']}")
    
    # Export metrics
    monitor.export_metrics("demo_metrics.json")
    print("  Exported metrics to demo_metrics.json")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("System monitoring stopped.\n")


def demo_learning_tracking():
    """Demonstrate learning progress tracking."""
    print("=== Learning Progress Tracking Demo ===")
    
    # Create learning tracker
    tracker = LearningProgressTracker()
    
    # Create learning goals
    tracker.create_goal(
        goal_id="classification_accuracy",
        description="Achieve 90% accuracy on classification task",
        target_performance=0.9
    )
    
    tracker.create_goal(
        goal_id="convergence_speed",
        description="Converge within 100 episodes",
        target_performance=100
    )
    
    print("Created learning goals:")
    print("  - Classification accuracy: 90%")
    print("  - Convergence speed: 100 episodes")
    
    # Simulate learning progress for different tasks
    task_types = ["classification", "regression", "reinforcement_learning"]
    
    for episode in range(50):
        for task_type in task_types:
            # Simulate improving performance
            base_performance = 0.5 if task_type == "classification" else 0.3
            performance_score = base_performance + episode * 0.008 + np.random.normal(0, 0.02)
            performance_score = min(1.0, max(0.0, performance_score))  # Clamp to [0,1]
            
            # Simulate decreasing learning rate
            learning_rate = 0.01 * np.exp(-episode * 0.02)
            
            # Simulate convergence
            convergence_metric = 0.1 * np.exp(-episode * 0.05) + np.random.normal(0, 0.01)
            convergence_metric = max(0.0, convergence_metric)
            
            # Record learning event
            tracker.record_learning_event(
                task_type=task_type,
                performance_score=performance_score,
                learning_rate=learning_rate,
                episode_count=episode,
                convergence_metric=convergence_metric,
                # Metadata
                batch_size=32,
                optimizer="adam"
            )
            
            # Update goals (using classification task)
            if task_type == "classification":
                tracker.update_goal_progress("classification_accuracy", performance_score)
                tracker.update_goal_progress("convergence_speed", episode)
    
    print(f"Recorded {50 * len(task_types)} learning events")
    
    # Analyze learning progress
    for task_type in task_types:
        trend = tracker.get_performance_trend(task_type)
        print(f"\n{task_type.title()} Analysis:")
        print(f"  Trend slope: {trend['trend']:.4f}")
        print(f"  Confidence: {trend['confidence']:.3f}")
        print(f"  Recent average: {trend['recent_avg']:.3f}")
        print(f"  Improvement rate: {trend['improvement_rate']:.4f}")
    
    # Check goal progress
    goal_summary = tracker.get_goal_summary()
    print(f"\nGoal Summary:")
    print(f"  Total goals: {goal_summary['total_goals']}")
    print(f"  Completed goals: {goal_summary['completed_goals']}")
    print(f"  Active goals: {goal_summary['active_goals']}")
    print(f"  Completion rate: {goal_summary['completion_rate']:.1%}")
    
    # Generate comprehensive report
    report = tracker.generate_learning_report()
    print(f"\nLearning Report:")
    print(f"  Total events: {report['total_learning_events']}")
    print(f"  Overall performance avg: {report['overall_performance']['average']:.3f}")
    print(f"  Average learning rate: {report['learning_dynamics']['average_learning_rate']:.4f}")
    
    # Create learning progress visualization
    print("\nGenerating learning progress visualization...")
    fig = tracker.visualize_learning_progress(
        task_types=task_types,
        save_path="learning_progress_demo.png"
    )
    plt.close(fig)
    print("  Saved learning_progress_demo.png")
    
    # Export learning data
    tracker.export_learning_data("learning_data_demo.json")
    print("  Exported learning_data_demo.json\n")


def demo_neural_visualization():
    """Demonstrate neural activity visualization."""
    print("=== Neural Activity Visualization Demo ===")
    
    # Create neural activity visualizer
    visualizer = NeuralActivityVisualizer()
    
    # Generate realistic spike data
    np.random.seed(42)  # For reproducible demo
    n_neurons = 20
    n_spikes = 500
    duration = 10.0  # seconds
    
    # Create spike data with realistic patterns
    neuron_ids = []
    spike_times = []
    amplitudes = []
    
    for neuron in range(n_neurons):
        # Each neuron has different firing characteristics
        base_rate = 5 + np.random.exponential(10)  # Hz
        n_neuron_spikes = np.random.poisson(base_rate * duration)
        
        # Generate spike times for this neuron
        neuron_spike_times = np.sort(np.random.uniform(0, duration, n_neuron_spikes))
        neuron_amplitudes = np.random.lognormal(0, 0.3, n_neuron_spikes)
        
        neuron_ids.extend([neuron] * n_neuron_spikes)
        spike_times.extend(neuron_spike_times)
        amplitudes.extend(neuron_amplitudes)
    
    spike_data = SpikeData(
        neuron_ids=neuron_ids,
        spike_times=spike_times,
        amplitudes=amplitudes,
        metadata={"n_neurons": n_neurons, "duration": duration, "demo": True}
    )
    
    print(f"Generated spike data: {len(spike_times)} spikes from {n_neurons} neurons")
    
    # Create spike raster plot
    print("Creating spike raster plot...")
    fig = visualizer.create_spike_raster_plot(
        spike_data,
        save_path="spike_raster_demo.png"
    )
    plt.close(fig)
    print("  Saved spike_raster_demo.png")
    
    # Create firing rate heatmap
    print("Creating firing rate heatmap...")
    fig = visualizer.create_firing_rate_heatmap(
        spike_data,
        bin_size=0.2,  # 200ms bins
        save_path="firing_rate_heatmap_demo.png"
    )
    plt.close(fig)
    print("  Saved firing_rate_heatmap_demo.png")
    
    # Create network activity plot
    print("Creating network activity plot...")
    time_points = np.linspace(0, duration, 1000)
    activity_data = {
        'total_activity': np.sin(time_points * 0.5) + 1 + np.random.normal(0, 0.1, 1000),
        'synchronization': np.cos(time_points * 0.3) * 0.3 + 0.5 + np.random.normal(0, 0.05, 1000),
        'complexity': np.abs(np.sin(time_points * 1.2)) + np.random.normal(0, 0.1, 1000)
    }
    
    fig = visualizer.create_network_activity_plot(
        activity_data,
        time_points.tolist(),
        save_path="network_activity_demo.png"
    )
    plt.close(fig)
    print("  Saved network_activity_demo.png")
    
    # Create interactive spike plot
    print("Creating interactive spike plot...")
    interactive_fig = visualizer.create_interactive_spike_plot(
        spike_data,
        save_path="interactive_spikes_demo.html"
    )
    print("  Saved interactive_spikes_demo.html\n")


def demo_network_visualization():
    """Demonstrate network topology visualization."""
    print("=== Network Topology Visualization Demo ===")
    
    # Create network topology visualizer
    net_viz = NetworkTopologyVisualizer()
    
    # Create a sample neural network topology
    n_nodes = 15
    np.random.seed(42)
    
    # Create adjacency matrix with realistic connectivity
    adj_matrix = np.zeros((n_nodes, n_nodes))
    
    # Add connections with distance-based probability
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            # Higher probability for nearby nodes
            distance = abs(i - j)
            connection_prob = 0.3 * np.exp(-distance * 0.2)
            
            if np.random.random() < connection_prob:
                weight = np.random.uniform(0.1, 1.0)
                adj_matrix[i, j] = weight
                adj_matrix[j, i] = weight  # Symmetric connections
    
    # Create node attributes
    node_attributes = {}
    node_labels = {}
    for i in range(n_nodes):
        if i < 3:
            node_type = "input"
        elif i >= n_nodes - 2:
            node_type = "output"
        else:
            node_type = "hidden"
        
        node_attributes[i] = {
            "type": node_type,
            "activity": np.random.uniform(0.2, 1.0),
            "layer": 0 if node_type == "input" else (2 if node_type == "output" else 1)
        }
        node_labels[i] = f"{node_type.title()}_{i}"
    
    topology = NetworkTopology(
        adjacency_matrix=adj_matrix,
        node_labels=node_labels,
        node_attributes=node_attributes
    )
    
    print(f"Created network topology: {n_nodes} nodes, {np.sum(adj_matrix > 0)//2} connections")
    
    # Create topology graph
    print("Creating network topology graph...")
    fig = net_viz.create_topology_graph(
        topology,
        layout='spring',
        save_path="network_topology_demo.png"
    )
    plt.close(fig)
    print("  Saved network_topology_demo.png")
    
    # Create connectivity matrix
    print("Creating connectivity matrix...")
    fig = net_viz.create_connectivity_matrix(
        topology,
        save_path="connectivity_matrix_demo.png"
    )
    plt.close(fig)
    print("  Saved connectivity_matrix_demo.png")
    
    # Create degree distribution
    print("Creating degree distribution...")
    fig = net_viz.create_degree_distribution(
        topology,
        save_path="degree_distribution_demo.png"
    )
    plt.close(fig)
    print("  Saved degree_distribution_demo.png")
    
    # Create interactive network
    print("Creating interactive network...")
    interactive_fig = net_viz.create_interactive_network(
        topology,
        save_path="interactive_network_demo.html"
    )
    print("  Saved interactive_network_demo.html\n")


def demo_memory_visualization():
    """Demonstrate memory state visualization."""
    print("=== Memory State Visualization Demo ===")
    
    # Create memory state visualizer
    mem_viz = MemoryStateVisualizer()
    
    # Generate realistic memory usage data
    duration = 120  # 2 minutes
    time_points = np.linspace(0, duration, 1000)
    
    # Simulate different memory systems with realistic patterns
    memory_data = {
        'working_memory': (
            40 +  # Base usage
            15 * np.sin(time_points * 0.1) +  # Slow oscillation
            10 * np.sin(time_points * 0.5) +  # Faster oscillation
            np.random.normal(0, 3, 1000)  # Noise
        ),
        'episodic_memory': (
            25 +  # Lower base usage
            20 * (1 - np.exp(-time_points * 0.02)) +  # Gradual increase
            5 * np.sin(time_points * 0.3) +  # Oscillation
            np.random.normal(0, 2, 1000)  # Noise
        ),
        'semantic_memory': (
            60 +  # Higher base usage
            10 * np.sin(time_points * 0.05) +  # Very slow oscillation
            np.random.normal(0, 2, 1000)  # Noise
        ),
        'consolidation_activity': (
            np.abs(np.sin(time_points * 0.2)) * 0.8 +  # Periodic activity
            0.2 * np.random.random(1000)  # Random background
        )
    }
    
    # Ensure memory usage stays within realistic bounds
    for key in ['working_memory', 'episodic_memory', 'semantic_memory']:
        memory_data[key] = np.clip(memory_data[key], 0, 100)
    
    memory_data['consolidation_activity'] = np.clip(memory_data['consolidation_activity'], 0, 1)
    
    print("Generated memory usage data for 4 memory systems over 2 minutes")
    
    # Create memory usage plot
    print("Creating memory usage visualization...")
    fig = mem_viz.create_memory_usage_plot(
        memory_data,
        time_points.tolist(),
        save_path="memory_usage_demo.png"
    )
    plt.close(fig)
    print("  Saved memory_usage_demo.png\n")


def main():
    """Run all monitoring and visualization demos."""
    print("Godly AI System - Monitoring and Visualization Demo")
    print("=" * 55)
    
    # Create output directory
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Change to output directory for file creation
    import os
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        # Run all demos
        demo_system_monitoring()
        demo_learning_tracking()
        demo_neural_visualization()
        demo_network_visualization()
        demo_memory_visualization()
        
        print("Demo completed successfully!")
        print(f"All output files saved to: {output_dir.absolute()}")
        print("\nGenerated files:")
        for file_path in output_dir.glob("*"):
            print(f"  - {file_path.name}")
            
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
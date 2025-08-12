"""
Tests for the neural activity and network topology visualization functionality.

This module tests:
- Spike raster plot generation using Matplotlib
- Network topology visualization with interactive graphs
- Memory state visualization and exploration tools
- Visualization accuracy and performance
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import networkx as nx
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from src.monitoring.visualization import (
    NeuralActivityVisualizer, NetworkTopologyVisualizer, MemoryStateVisualizer,
    SpikeData, NetworkTopology
)


class TestSpikeData:
    """Test SpikeData data structure."""
    
    def test_spike_data_creation(self):
        """Test SpikeData creation with basic parameters."""
        spike_data = SpikeData(
            neuron_ids=[1, 2, 3, 1, 2],
            spike_times=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        assert len(spike_data.neuron_ids) == 5
        assert len(spike_data.spike_times) == 5
        assert len(spike_data.amplitudes) == 5  # Should be auto-filled with 1.0
        assert all(amp == 1.0 for amp in spike_data.amplitudes)
        assert spike_data.metadata == {}
        
    def test_spike_data_with_amplitudes(self):
        """Test SpikeData creation with custom amplitudes."""
        spike_data = SpikeData(
            neuron_ids=[1, 2, 3],
            spike_times=[0.1, 0.2, 0.3],
            amplitudes=[0.5, 1.0, 1.5],
            metadata={"experiment": "test"}
        )
        
        assert spike_data.amplitudes == [0.5, 1.0, 1.5]
        assert spike_data.metadata["experiment"] == "test"


class TestNetworkTopology:
    """Test NetworkTopology data structure."""
    
    def test_network_topology_creation(self):
        """Test NetworkTopology creation with adjacency matrix."""
        adj_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        topology = NetworkTopology(adjacency_matrix=adj_matrix)
        
        assert topology.adjacency_matrix.shape == (3, 3)
        assert topology.node_positions == {}
        assert topology.node_labels == {}
        assert topology.node_attributes == {}
        
    def test_network_topology_with_attributes(self):
        """Test NetworkTopology with additional attributes."""
        adj_matrix = np.array([[0, 1], [1, 0]])
        positions = {0: (0, 0), 1: (1, 1)}
        labels = {0: "Node A", 1: "Node B"}
        attributes = {0: {"type": "input"}, 1: {"type": "output"}}
        
        topology = NetworkTopology(
            adjacency_matrix=adj_matrix,
            node_positions=positions,
            node_labels=labels,
            node_attributes=attributes
        )
        
        assert topology.node_positions == positions
        assert topology.node_labels == labels
        assert topology.node_attributes == attributes


class TestNeuralActivityVisualizer:
    """Test NeuralActivityVisualizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = NeuralActivityVisualizer()
        self.temp_dir = tempfile.mkdtemp()
        
    def create_test_spike_data(self, n_neurons=10, n_spikes=100):
        """Create test spike data."""
        np.random.seed(42)  # For reproducible tests
        neuron_ids = np.random.randint(0, n_neurons, n_spikes)
        spike_times = np.sort(np.random.uniform(0, 10, n_spikes))
        amplitudes = np.random.uniform(0.5, 2.0, n_spikes)
        
        return SpikeData(
            neuron_ids=neuron_ids.tolist(),
            spike_times=spike_times.tolist(),
            amplitudes=amplitudes.tolist()
        )
        
    def test_visualizer_initialization(self):
        """Test NeuralActivityVisualizer initialization."""
        assert self.visualizer.figsize == (12, 8)
        assert hasattr(self.visualizer, 'color_palette')
        
    def test_create_spike_raster_plot(self):
        """Test spike raster plot creation."""
        spike_data = self.create_test_spike_data()
        
        fig = self.visualizer.create_spike_raster_plot(spike_data)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1  # May have colorbar as second axis
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Time (s)'
        assert ax.get_ylabel() == 'Neuron ID'
        assert 'Neural Spike Raster Plot' in ax.get_title()
        
        plt.close(fig)  # Clean up
        
    def test_spike_raster_plot_with_time_window(self):
        """Test spike raster plot with time window filtering."""
        spike_data = self.create_test_spike_data()
        
        fig = self.visualizer.create_spike_raster_plot(
            spike_data, 
            time_window=(2.0, 8.0)
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
    def test_spike_raster_plot_with_neuron_subset(self):
        """Test spike raster plot with neuron subset filtering."""
        spike_data = self.create_test_spike_data()
        
        fig = self.visualizer.create_spike_raster_plot(
            spike_data,
            neuron_subset=[0, 1, 2, 3, 4]
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
    def test_spike_raster_plot_save(self):
        """Test spike raster plot saving to file."""
        spike_data = self.create_test_spike_data()
        save_path = Path(self.temp_dir) / "test_raster.png"
        
        fig = self.visualizer.create_spike_raster_plot(
            spike_data,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close(fig)
        
    def test_create_firing_rate_heatmap(self):
        """Test firing rate heatmap creation."""
        spike_data = self.create_test_spike_data()
        
        fig = self.visualizer.create_firing_rate_heatmap(spike_data)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1  # May have colorbar as second axis
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Time (s)'
        assert ax.get_ylabel() == 'Neuron ID'
        assert 'Neural Firing Rate Heatmap' in ax.get_title()
        
        plt.close(fig)
        
    def test_firing_rate_heatmap_with_bin_size(self):
        """Test firing rate heatmap with custom bin size."""
        spike_data = self.create_test_spike_data()
        
        fig = self.visualizer.create_firing_rate_heatmap(
            spike_data,
            bin_size=0.5
        )
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
    def test_create_network_activity_plot(self):
        """Test network activity plot creation."""
        time_points = np.linspace(0, 10, 100)
        activity_data = {
            'total_activity': np.sin(time_points) + 1,
            'synchronization': np.cos(time_points) * 0.5 + 0.5,
            'complexity': np.random.random(100)
        }
        
        fig = self.visualizer.create_network_activity_plot(
            activity_data, 
            time_points.tolist()
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # Should have 4 subplots
        
        plt.close(fig)
        
    def test_create_interactive_spike_plot(self):
        """Test interactive spike plot creation."""
        spike_data = self.create_test_spike_data(n_neurons=5, n_spikes=20)
        
        fig = self.visualizer.create_interactive_spike_plot(spike_data)
        
        # Check that it's a plotly figure
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        assert len(fig.data) > 0
        
    def test_empty_spike_data_handling(self):
        """Test handling of empty spike data."""
        empty_spike_data = SpikeData([], [])
        
        fig = self.visualizer.create_spike_raster_plot(empty_spike_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        fig = self.visualizer.create_firing_rate_heatmap(empty_spike_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestNetworkTopologyVisualizer:
    """Test NetworkTopologyVisualizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = NetworkTopologyVisualizer()
        self.temp_dir = tempfile.mkdtemp()
        
    def create_test_topology(self, n_nodes=10):
        """Create test network topology."""
        # Create a random adjacency matrix
        np.random.seed(42)
        adj_matrix = np.random.random((n_nodes, n_nodes))
        adj_matrix = (adj_matrix > 0.7).astype(float)  # Sparse network
        
        # Make it symmetric for undirected graph
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        np.fill_diagonal(adj_matrix, 0)  # No self-loops
        
        node_attributes = {
            i: {"activity": np.random.random(), "type": f"neuron_{i}"}
            for i in range(n_nodes)
        }
        
        return NetworkTopology(
            adjacency_matrix=adj_matrix,
            node_attributes=node_attributes
        )
        
    def test_visualizer_initialization(self):
        """Test NetworkTopologyVisualizer initialization."""
        assert self.visualizer.figsize == (12, 10)
        
    def test_create_topology_graph(self):
        """Test network topology graph creation."""
        topology = self.create_test_topology()
        
        fig = self.visualizer.create_topology_graph(topology)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        
        ax = fig.axes[0]
        assert ax.get_title() == 'Network Topology Visualization'
        
        plt.close(fig)
        
    def test_topology_graph_with_different_layouts(self):
        """Test topology graph with different layout algorithms."""
        topology = self.create_test_topology(n_nodes=5)
        
        layouts = ['spring', 'circular', 'random']
        
        for layout in layouts:
            fig = self.visualizer.create_topology_graph(topology, layout=layout)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)
            
    def test_create_connectivity_matrix(self):
        """Test connectivity matrix heatmap creation."""
        topology = self.create_test_topology()
        
        fig = self.visualizer.create_connectivity_matrix(topology)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1  # May have colorbar as second axis
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Target Neuron'
        assert ax.get_ylabel() == 'Source Neuron'
        assert ax.get_title() == 'Network Connectivity Matrix'
        
        plt.close(fig)
        
    def test_create_degree_distribution(self):
        """Test degree distribution analysis."""
        topology = self.create_test_topology()
        
        fig = self.visualizer.create_degree_distribution(topology)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3  # Total, in-degree, out-degree
        
        plt.close(fig)
        
    def test_create_interactive_network(self):
        """Test interactive network visualization."""
        topology = self.create_test_topology(n_nodes=5)
        
        fig = self.visualizer.create_interactive_network(topology)
        
        # Check that it's a plotly figure
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
        assert len(fig.data) >= 2  # Should have edge and node traces
        
    def test_topology_graph_save(self):
        """Test topology graph saving to file."""
        topology = self.create_test_topology(n_nodes=5)
        save_path = Path(self.temp_dir) / "test_topology.png"
        
        fig = self.visualizer.create_topology_graph(
            topology,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close(fig)


class TestMemoryStateVisualizer:
    """Test MemoryStateVisualizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = MemoryStateVisualizer()
        self.temp_dir = tempfile.mkdtemp()
        
    def create_test_memory_data(self):
        """Create test memory usage data."""
        time_points = np.linspace(0, 100, 200)
        
        return {
            'working_memory': 50 + 20 * np.sin(time_points * 0.1) + np.random.normal(0, 5, 200),
            'episodic_memory': 30 + 15 * np.cos(time_points * 0.05) + np.random.normal(0, 3, 200),
            'semantic_memory': 70 + 10 * np.sin(time_points * 0.02) + np.random.normal(0, 2, 200),
            'consolidation_activity': np.abs(np.sin(time_points * 0.3)) + np.random.normal(0, 0.1, 200)
        }
        
    def test_visualizer_initialization(self):
        """Test MemoryStateVisualizer initialization."""
        assert self.visualizer.figsize == (12, 8)
        
    def test_create_memory_usage_plot(self):
        """Test memory usage plot creation."""
        memory_data = self.create_test_memory_data()
        time_points = list(range(200))
        
        fig = self.visualizer.create_memory_usage_plot(memory_data, time_points)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # Should have 4 subplots
        
        # Check subplot titles
        titles = [ax.get_title() for ax in fig.axes]
        expected_titles = [
            'Working Memory Usage',
            'Episodic Memory Usage', 
            'Semantic Memory Usage',
            'Memory Consolidation Activity'
        ]
        
        for expected in expected_titles:
            assert any(expected in title for title in titles)
            
        plt.close(fig)
        
    def test_memory_usage_plot_save(self):
        """Test memory usage plot saving to file."""
        memory_data = self.create_test_memory_data()
        time_points = list(range(200))
        save_path = Path(self.temp_dir) / "test_memory.png"
        
        fig = self.visualizer.create_memory_usage_plot(
            memory_data, 
            time_points,
            save_path=str(save_path)
        )
        
        assert save_path.exists()
        plt.close(fig)
        
    def test_partial_memory_data(self):
        """Test memory usage plot with partial data."""
        # Only provide some memory types
        memory_data = {
            'working_memory': [50, 60, 70, 80, 90],
            'episodic_memory': [30, 35, 40, 45, 50]
        }
        time_points = [0, 1, 2, 3, 4]
        
        fig = self.visualizer.create_memory_usage_plot(memory_data, time_points)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestVisualizationPerformance:
    """Test visualization performance and accuracy."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.neural_viz = NeuralActivityVisualizer()
        self.network_viz = NetworkTopologyVisualizer()
        self.memory_viz = MemoryStateVisualizer()
        
    def test_large_spike_data_performance(self):
        """Test performance with large spike datasets."""
        # Create large spike dataset
        n_spikes = 10000
        np.random.seed(42)
        
        spike_data = SpikeData(
            neuron_ids=np.random.randint(0, 100, n_spikes).tolist(),
            spike_times=np.sort(np.random.uniform(0, 100, n_spikes)).tolist(),
            amplitudes=np.random.uniform(0.5, 2.0, n_spikes).tolist()
        )
        
        # Measure time for raster plot
        start_time = time.time()
        fig = self.neural_viz.create_spike_raster_plot(spike_data)
        end_time = time.time()
        
        # Should complete within reasonable time (less than 10 seconds)
        assert (end_time - start_time) < 10.0
        plt.close(fig)
        
    def test_large_network_performance(self):
        """Test performance with large network topologies."""
        # Create large network
        n_nodes = 100
        np.random.seed(42)
        adj_matrix = np.random.random((n_nodes, n_nodes))
        adj_matrix = (adj_matrix > 0.95).astype(float)  # Very sparse
        
        topology = NetworkTopology(adjacency_matrix=adj_matrix)
        
        # Measure time for topology visualization
        start_time = time.time()
        fig = self.network_viz.create_topology_graph(topology)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 15.0
        plt.close(fig)
        
    def test_visualization_accuracy(self):
        """Test visualization data accuracy."""
        # Create known spike data
        spike_data = SpikeData(
            neuron_ids=[1, 2, 3, 1, 2],
            spike_times=[0.1, 0.2, 0.3, 0.4, 0.5],
            amplitudes=[1.0, 1.5, 2.0, 0.5, 1.2]
        )
        
        fig = self.neural_viz.create_spike_raster_plot(spike_data)
        
        # Check that the plot contains the expected data points
        ax = fig.axes[0]
        collections = ax.collections
        
        # Should have scatter plot data
        assert len(collections) > 0
        
        plt.close(fig)
        
    def test_memory_overhead(self):
        """Test memory usage of visualization components."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple visualizations
        for i in range(10):
            spike_data = SpikeData(
                neuron_ids=list(range(50)),
                spike_times=[j * 0.1 for j in range(50)]
            )
            
            fig = self.neural_viz.create_spike_raster_plot(spike_data)
            plt.close(fig)
            
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
        
    def test_concurrent_visualization(self):
        """Test concurrent visualization creation."""
        import threading
        
        results = []
        
        def create_visualization():
            try:
                spike_data = SpikeData(
                    neuron_ids=[1, 2, 3],
                    spike_times=[0.1, 0.2, 0.3]
                )
                fig = self.neural_viz.create_spike_raster_plot(spike_data)
                plt.close(fig)
                results.append(True)
            except Exception as e:
                results.append(False)
                
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_visualization)
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # All visualizations should succeed
        assert all(results)
        assert len(results) == 5


class TestVisualizationEdgeCases:
    """Test visualization edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.neural_viz = NeuralActivityVisualizer()
        self.network_viz = NetworkTopologyVisualizer()
        
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        # Empty spike data
        empty_spikes = SpikeData([], [])
        fig = self.neural_viz.create_spike_raster_plot(empty_spikes)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Empty network
        empty_network = NetworkTopology(np.array([[]]))
        # Should handle gracefully without crashing
        
    def test_single_data_point(self):
        """Test handling of single data points."""
        single_spike = SpikeData([1], [0.5])
        fig = self.neural_viz.create_spike_raster_plot(single_spike)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
    def test_invalid_time_window(self):
        """Test handling of invalid time windows."""
        spike_data = SpikeData([1, 2, 3], [0.1, 0.2, 0.3])
        
        # Time window outside data range
        fig = self.neural_viz.create_spike_raster_plot(
            spike_data,
            time_window=(10.0, 20.0)  # No data in this range
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
    def test_mismatched_data_lengths(self):
        """Test handling of mismatched data lengths."""
        # This should be handled by the SpikeData constructor
        with pytest.raises((ValueError, IndexError, AssertionError)):
            # Different length arrays should cause an issue
            spike_data = SpikeData([1, 2], [0.1, 0.2, 0.3])  # Mismatched lengths
            
    def test_extreme_values(self):
        """Test handling of extreme values."""
        # Very large spike times
        large_times = SpikeData([1, 2], [1e6, 1e6 + 1])
        fig = self.neural_viz.create_spike_raster_plot(large_times)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Very small spike times
        small_times = SpikeData([1, 2], [1e-6, 2e-6])
        fig = self.neural_viz.create_spike_raster_plot(small_times)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
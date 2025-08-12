# Visualization

The Visualization module provides comprehensive tools for visualizing neural activity, network topology, and memory states in the Godly AI System.

## Overview

The visualization system includes three main components:

- **NeuralActivityVisualizer**: Spike raster plots, firing rate heatmaps, and network activity visualizations
- **NetworkTopologyVisualizer**: Network graphs, connectivity matrices, and degree distributions
- **MemoryStateVisualizer**: Memory usage and consolidation activity visualizations

## Neural Activity Visualization

### Basic Spike Visualization

```python
from src.monitoring.visualization import NeuralActivityVisualizer, SpikeData
import numpy as np

# Create visualizer
visualizer = NeuralActivityVisualizer()

# Create spike data
spike_data = SpikeData(
    neuron_ids=[1, 2, 3, 1, 2, 4, 5],
    spike_times=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    amplitudes=[1.0, 1.5, 2.0, 0.8, 1.2, 1.8, 0.9],
    metadata={"experiment": "test_run", "condition": "baseline"}
)

# Create spike raster plot
fig = visualizer.create_spike_raster_plot(spike_data)
fig.show()

# Save to file
fig = visualizer.create_spike_raster_plot(
    spike_data, 
    save_path="spike_raster.png"
)
```

### Advanced Spike Analysis

```python
# Filter by time window
fig = visualizer.create_spike_raster_plot(
    spike_data,
    time_window=(0.2, 0.6),  # Only show spikes between 0.2s and 0.6s
    save_path="filtered_raster.png"
)

# Filter by neuron subset
fig = visualizer.create_spike_raster_plot(
    spike_data,
    neuron_subset=[1, 2, 3],  # Only show specific neurons
    save_path="subset_raster.png"
)

# Create firing rate heatmap
fig = visualizer.create_firing_rate_heatmap(
    spike_data,
    bin_size=0.1,  # 100ms bins
    save_path="firing_rate_heatmap.png"
)
```

### Network Activity Visualization

```python
import numpy as np

# Create network activity data
time_points = np.linspace(0, 10, 1000)
activity_data = {
    'total_activity': np.sin(time_points) + np.random.normal(0, 0.1, 1000),
    'synchronization': np.cos(time_points * 0.5) * 0.5 + 0.5,
    'complexity': np.abs(np.sin(time_points * 2)) + np.random.normal(0, 0.05, 1000)
}

# Create network activity plot
fig = visualizer.create_network_activity_plot(
    activity_data,
    time_points.tolist(),
    save_path="network_activity.png"
)
```

### Interactive Visualizations

```python
# Create interactive spike plot with Plotly
interactive_fig = visualizer.create_interactive_spike_plot(
    spike_data,
    save_path="interactive_spikes.html"
)

# The interactive plot allows:
# - Zooming and panning
# - Hover information for individual spikes
# - Dynamic filtering and selection
```

## Network Topology Visualization

### Basic Network Visualization

```python
from src.monitoring.visualization import NetworkTopologyVisualizer, NetworkTopology
import numpy as np

# Create network topology
adj_matrix = np.array([
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0]
])

topology = NetworkTopology(
    adjacency_matrix=adj_matrix,
    node_labels={0: "Input", 1: "Hidden1", 2: "Hidden2", 3: "Hidden3", 4: "Output"},
    node_attributes={
        0: {"type": "input", "activity": 0.8},
        1: {"type": "hidden", "activity": 0.6},
        2: {"type": "hidden", "activity": 0.7},
        3: {"type": "hidden", "activity": 0.5},
        4: {"type": "output", "activity": 0.9}
    }
)

# Create visualizer
net_viz = NetworkTopologyVisualizer()

# Create topology graph
fig = net_viz.create_topology_graph(
    topology,
    layout='spring',  # Options: 'spring', 'circular', 'random'
    save_path="network_topology.png"
)
```

### Advanced Network Analysis

```python
# Create connectivity matrix heatmap
fig = net_viz.create_connectivity_matrix(
    topology,
    save_path="connectivity_matrix.png"
)

# Analyze degree distributions
fig = net_viz.create_degree_distribution(
    topology,
    save_path="degree_distribution.png"
)

# Create interactive network
interactive_fig = net_viz.create_interactive_network(
    topology,
    save_path="interactive_network.html"
)
```

### Custom Network Layouts

```python
# Define custom node positions
custom_positions = {
    0: (0, 0),    # Input at origin
    1: (1, 1),    # Hidden nodes
    2: (1, -1),
    3: (2, 1),
    4: (2, -1),
    5: (3, 0)     # Output at end
}

topology_with_positions = NetworkTopology(
    adjacency_matrix=adj_matrix,
    node_positions=custom_positions,
    node_labels={i: f"Node_{i}" for i in range(6)}
)

fig = net_viz.create_topology_graph(topology_with_positions)
```

## Memory State Visualization

### Memory Usage Tracking

```python
from src.monitoring.visualization import MemoryStateVisualizer
import numpy as np

# Create memory state data
time_points = np.linspace(0, 100, 1000)
memory_data = {
    'working_memory': 50 + 20 * np.sin(time_points * 0.1) + np.random.normal(0, 5, 1000),
    'episodic_memory': 30 + 15 * np.cos(time_points * 0.05) + np.random.normal(0, 3, 1000),
    'semantic_memory': 70 + 10 * np.sin(time_points * 0.02) + np.random.normal(0, 2, 1000),
    'consolidation_activity': np.abs(np.sin(time_points * 0.3)) + np.random.normal(0, 0.1, 1000)
}

# Create memory visualizer
mem_viz = MemoryStateVisualizer()

# Create memory usage plot
fig = mem_viz.create_memory_usage_plot(
    memory_data,
    time_points.tolist(),
    save_path="memory_usage.png"
)
```

## Data Structures

### SpikeData

```python
@dataclass
class SpikeData:
    neuron_ids: List[int]
    spike_times: List[float]
    amplitudes: Optional[List[float]] = None  # Auto-filled with 1.0 if None
    metadata: Dict[str, Any] = None  # Auto-filled with {} if None
```

### NetworkTopology

```python
@dataclass
class NetworkTopology:
    adjacency_matrix: np.ndarray
    node_positions: Optional[Dict[int, Tuple[float, float]]] = None
    node_labels: Optional[Dict[int, str]] = None
    edge_weights: Optional[np.ndarray] = None
    node_attributes: Optional[Dict[int, Dict[str, Any]]] = None
```

## Advanced Usage Examples

### Large-Scale Spike Data

```python
# Handle large spike datasets efficiently
def create_large_spike_data(n_neurons=1000, n_spikes=100000, duration=60.0):
    np.random.seed(42)
    
    # Generate realistic spike data
    neuron_ids = np.random.randint(0, n_neurons, n_spikes)
    spike_times = np.sort(np.random.uniform(0, duration, n_spikes))
    amplitudes = np.random.lognormal(0, 0.3, n_spikes)  # Log-normal distribution
    
    return SpikeData(
        neuron_ids=neuron_ids.tolist(),
        spike_times=spike_times.tolist(),
        amplitudes=amplitudes.tolist(),
        metadata={"n_neurons": n_neurons, "duration": duration}
    )

# Create and visualize large dataset
large_spike_data = create_large_spike_data()

# Use subsampling for visualization
fig = visualizer.create_spike_raster_plot(
    large_spike_data,
    neuron_subset=list(range(0, 100, 5)),  # Every 5th neuron from first 100
    time_window=(0, 10),  # First 10 seconds
    save_path="large_dataset_sample.png"
)
```

### Dynamic Network Visualization

```python
# Visualize network evolution over time
def create_evolving_network(time_steps=10):
    """Create a network that evolves over time."""
    networks = []
    
    for t in range(time_steps):
        # Network grows over time
        n_nodes = 5 + t
        
        # Create random adjacency matrix
        adj_matrix = np.random.random((n_nodes, n_nodes))
        adj_matrix = (adj_matrix > 0.7).astype(float)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(adj_matrix, 0)
        
        topology = NetworkTopology(
            adjacency_matrix=adj_matrix,
            node_attributes={
                i: {"time_created": t, "activity": np.random.random()}
                for i in range(n_nodes)
            }
        )
        networks.append(topology)
    
    return networks

# Create evolution sequence
network_sequence = create_evolving_network()

# Visualize each time step
for i, network in enumerate(network_sequence):
    fig = net_viz.create_topology_graph(
        network,
        save_path=f"network_evolution_t{i:02d}.png"
    )
```

### Multi-Modal Visualization

```python
# Combine multiple visualization types
def create_comprehensive_analysis(spike_data, topology, memory_data, time_points):
    """Create a comprehensive analysis dashboard."""
    
    # Create figure with subplots
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 15))
    
    # Spike raster plot
    ax1 = plt.subplot(3, 3, 1)
    spike_fig = visualizer.create_spike_raster_plot(spike_data)
    
    # Firing rate heatmap
    ax2 = plt.subplot(3, 3, 2)
    heatmap_fig = visualizer.create_firing_rate_heatmap(spike_data)
    
    # Network topology
    ax3 = plt.subplot(3, 3, 3)
    topology_fig = net_viz.create_topology_graph(topology)
    
    # Connectivity matrix
    ax4 = plt.subplot(3, 3, 4)
    conn_fig = net_viz.create_connectivity_matrix(topology)
    
    # Degree distribution
    ax5 = plt.subplot(3, 3, 5)
    degree_fig = net_viz.create_degree_distribution(topology)
    
    # Memory usage
    ax6 = plt.subplot(3, 3, (6, 9))  # Span multiple subplots
    memory_fig = mem_viz.create_memory_usage_plot(memory_data, time_points)
    
    plt.tight_layout()
    plt.savefig("comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    
    return fig
```

## Performance Optimization

### Efficient Data Handling

```python
# For large datasets, use data reduction techniques
def reduce_spike_data(spike_data, max_spikes=10000):
    """Reduce spike data size while preserving characteristics."""
    if len(spike_data.spike_times) <= max_spikes:
        return spike_data
    
    # Sample uniformly across time
    indices = np.linspace(0, len(spike_data.spike_times)-1, max_spikes, dtype=int)
    
    return SpikeData(
        neuron_ids=[spike_data.neuron_ids[i] for i in indices],
        spike_times=[spike_data.spike_times[i] for i in indices],
        amplitudes=[spike_data.amplitudes[i] for i in indices] if spike_data.amplitudes else None,
        metadata=spike_data.metadata
    )

# Use reduced data for visualization
reduced_data = reduce_spike_data(large_spike_data, max_spikes=5000)
fig = visualizer.create_spike_raster_plot(reduced_data)
```

### Batch Visualization

```python
# Create multiple visualizations efficiently
def batch_visualize_networks(topologies, output_dir="network_plots"):
    """Create visualizations for multiple networks."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, topology in enumerate(topologies):
        # Topology graph
        net_viz.create_topology_graph(
            topology,
            save_path=f"{output_dir}/topology_{i:03d}.png"
        )
        
        # Connectivity matrix
        net_viz.create_connectivity_matrix(
            topology,
            save_path=f"{output_dir}/connectivity_{i:03d}.png"
        )
```

## Integration Examples

### Real-time Monitoring Dashboard

```python
import time
import threading
from collections import deque

class RealTimeVisualizer:
    def __init__(self):
        self.spike_buffer = deque(maxlen=1000)
        self.visualizer = NeuralActivityVisualizer()
        self.running = False
    
    def add_spike(self, neuron_id, spike_time, amplitude=1.0):
        """Add spike to real-time buffer."""
        self.spike_buffer.append((neuron_id, spike_time, amplitude))
    
    def update_visualization(self):
        """Update visualization with current buffer."""
        if not self.spike_buffer:
            return
        
        # Convert buffer to SpikeData
        neuron_ids, spike_times, amplitudes = zip(*self.spike_buffer)
        spike_data = SpikeData(
            neuron_ids=list(neuron_ids),
            spike_times=list(spike_times),
            amplitudes=list(amplitudes)
        )
        
        # Create updated visualization
        fig = self.visualizer.create_spike_raster_plot(
            spike_data,
            save_path="realtime_spikes.png"
        )
        plt.close(fig)  # Free memory
    
    def start_monitoring(self, update_interval=1.0):
        """Start real-time monitoring."""
        self.running = True
        
        def monitor_loop():
            while self.running:
                self.update_visualization()
                time.sleep(update_interval)
        
        thread = threading.Thread(target=monitor_loop)
        thread.daemon = True
        thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.running = False

# Usage
rt_viz = RealTimeVisualizer()
rt_viz.start_monitoring(update_interval=2.0)

# Simulate incoming spikes
for i in range(100):
    rt_viz.add_spike(
        neuron_id=np.random.randint(0, 10),
        spike_time=time.time(),
        amplitude=np.random.uniform(0.5, 2.0)
    )
    time.sleep(0.1)
```

### Web-based Visualization

```python
from flask import Flask, render_template, jsonify
import json
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/api/spike_plot')
def get_spike_plot():
    # Generate spike data
    spike_data = create_test_spike_data()
    
    # Create plot
    fig = visualizer.create_spike_raster_plot(spike_data)
    
    # Convert to base64 for web display
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.read()).decode()
    
    plt.close(fig)
    
    return jsonify({'image': f'data:image/png;base64,{img_str}'})

@app.route('/api/interactive_network')
def get_interactive_network():
    # Create network topology
    topology = create_test_topology()
    
    # Create interactive plot
    fig = net_viz.create_interactive_network(topology)
    
    # Return Plotly JSON
    return fig.to_json()
```

## Best Practices

1. **Memory Management**: Close matplotlib figures after use to prevent memory leaks
2. **Data Reduction**: Use sampling or filtering for large datasets
3. **File Organization**: Use consistent naming conventions for saved plots
4. **Color Schemes**: Use colorblind-friendly palettes for accessibility
5. **Resolution**: Set appropriate DPI for publication-quality figures
6. **Interactive Elements**: Use Plotly for dashboards and exploration
7. **Batch Processing**: Process multiple visualizations efficiently
8. **Error Handling**: Validate input data before visualization

## Troubleshooting

### Memory Issues

```python
# Monitor memory usage
import psutil
import os

def check_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Check before and after visualization
check_memory_usage()
fig = visualizer.create_spike_raster_plot(large_spike_data)
check_memory_usage()
plt.close(fig)  # Important: close figure
check_memory_usage()
```

### Performance Issues

```python
# Profile visualization performance
import time

def profile_visualization(spike_data):
    start_time = time.time()
    fig = visualizer.create_spike_raster_plot(spike_data)
    end_time = time.time()
    
    print(f"Visualization took {end_time - start_time:.2f} seconds")
    print(f"Data size: {len(spike_data.spike_times)} spikes")
    
    plt.close(fig)
    return end_time - start_time
```

### Display Issues

```python
# Set matplotlib backend for different environments
import matplotlib
matplotlib.use('Agg')  # For headless servers
# matplotlib.use('TkAgg')  # For desktop with Tkinter
# matplotlib.use('Qt5Agg')  # For desktop with Qt

import matplotlib.pyplot as plt
```
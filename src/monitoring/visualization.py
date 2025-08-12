"""
Neural activity and network topology visualization for the Godly AI System.

This module implements comprehensive visualization tools including:
- Spike raster plot generation using Matplotlib
- Network topology visualization with interactive graphs
- Memory state visualization and exploration tools
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Union
import time
from dataclasses import dataclass
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns


@dataclass
class SpikeData:
    """Container for spike train data."""
    neuron_ids: List[int]
    spike_times: List[float]
    amplitudes: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.amplitudes is None:
            self.amplitudes = [1.0] * len(self.spike_times)


@dataclass
class NetworkTopology:
    """Container for network topology data."""
    adjacency_matrix: np.ndarray
    node_positions: Optional[Dict[int, Tuple[float, float]]] = None
    node_labels: Optional[Dict[int, str]] = None
    edge_weights: Optional[np.ndarray] = None
    node_attributes: Optional[Dict[int, Dict[str, Any]]] = None
    
    def __post_init__(self):
        if self.node_positions is None:
            self.node_positions = {}
        if self.node_labels is None:
            self.node_labels = {}
        if self.node_attributes is None:
            self.node_attributes = {}


class NeuralActivityVisualizer:
    """Visualizes neural activity patterns including spike trains and network dynamics."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.color_palette = plt.cm.Set3
        
    def create_spike_raster_plot(self, 
                               spike_data: SpikeData,
                               time_window: Optional[Tuple[float, float]] = None,
                               neuron_subset: Optional[List[int]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create spike raster plot showing neural activity over time."""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Filter data based on parameters
        neuron_ids = spike_data.neuron_ids
        spike_times = spike_data.spike_times
        amplitudes = spike_data.amplitudes
        
        # Apply time window filter
        if time_window:
            time_mask = [(t >= time_window[0] and t <= time_window[1]) 
                        for t in spike_times]
            spike_times = [t for i, t in enumerate(spike_times) if time_mask[i]]
            neuron_ids = [n for i, n in enumerate(neuron_ids) if time_mask[i]]
            amplitudes = [a for i, a in enumerate(amplitudes) if time_mask[i]]
            
        # Apply neuron subset filter
        if neuron_subset:
            neuron_mask = [n in neuron_subset for n in neuron_ids]
            spike_times = [t for i, t in enumerate(spike_times) if neuron_mask[i]]
            neuron_ids = [n for i, n in enumerate(neuron_ids) if neuron_mask[i]]
            amplitudes = [a for i, a in enumerate(amplitudes) if neuron_mask[i]]
            
        # Create scatter plot
        if spike_times:
            scatter = ax.scatter(spike_times, neuron_ids, 
                               c=amplitudes, s=20, alpha=0.7,
                               cmap='viridis', edgecolors='black', linewidth=0.5)
            
            # Add colorbar for amplitudes
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Spike Amplitude', rotation=270, labelpad=15)
            
        # Formatting
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron ID')
        ax.set_title('Neural Spike Raster Plot')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        if spike_times:
            stats_text = f'Total Spikes: {len(spike_times)}\n'
            stats_text += f'Active Neurons: {len(set(neuron_ids))}\n'
            stats_text += f'Time Span: {max(spike_times) - min(spike_times):.2f}s\n'
            stats_text += f'Avg Firing Rate: {len(spike_times) / (max(spike_times) - min(spike_times)):.1f} Hz'
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                   
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_firing_rate_heatmap(self, 
                                 spike_data: SpikeData,
                                 bin_size: float = 0.1,
                                 neuron_subset: Optional[List[int]] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Create heatmap showing firing rates over time."""
        
        # Prepare data
        neuron_ids = spike_data.neuron_ids
        spike_times = spike_data.spike_times
        
        if neuron_subset:
            mask = [n in neuron_subset for n in neuron_ids]
            neuron_ids = [n for i, n in enumerate(neuron_ids) if mask[i]]
            spike_times = [t for i, t in enumerate(spike_times) if mask[i]]
            
        if not spike_times:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No spike data available', ha='center', va='center')
            return fig
            
        # Create time bins
        min_time, max_time = min(spike_times), max(spike_times)
        time_bins = np.arange(min_time, max_time + bin_size, bin_size)
        
        # Get unique neurons
        unique_neurons = sorted(set(neuron_ids))
        
        # Create firing rate matrix
        firing_rates = np.zeros((len(unique_neurons), len(time_bins) - 1))
        
        for i, neuron in enumerate(unique_neurons):
            neuron_spikes = [t for n, t in zip(neuron_ids, spike_times) if n == neuron]
            hist, _ = np.histogram(neuron_spikes, bins=time_bins)
            firing_rates[i, :] = hist / bin_size  # Convert to Hz
            
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        
        im = ax.imshow(firing_rates, aspect='auto', cmap='hot', 
                      extent=[min_time, max_time, unique_neurons[0], unique_neurons[-1]])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Firing Rate (Hz)', rotation=270, labelpad=15)
        
        # Formatting
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Neuron ID')
        ax.set_title(f'Neural Firing Rate Heatmap (bin size: {bin_size}s)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_network_activity_plot(self, 
                                   activity_data: Dict[str, List[float]],
                                   time_points: List[float],
                                   save_path: Optional[str] = None) -> plt.Figure:
        """Create plot showing network-level activity metrics over time."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Network Activity Analysis', fontsize=16)
        
        # Overall activity level
        ax1 = axes[0, 0]
        if 'total_activity' in activity_data:
            ax1.plot(time_points, activity_data['total_activity'], 'b-', linewidth=2)
            ax1.set_title('Total Network Activity')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Activity Level')
            ax1.grid(True, alpha=0.3)
            
        # Synchronization measure
        ax2 = axes[0, 1]
        if 'synchronization' in activity_data:
            ax2.plot(time_points, activity_data['synchronization'], 'r-', linewidth=2)
            ax2.set_title('Network Synchronization')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Synchronization Index')
            ax2.grid(True, alpha=0.3)
            
        # Complexity measure
        ax3 = axes[1, 0]
        if 'complexity' in activity_data:
            ax3.plot(time_points, activity_data['complexity'], 'g-', linewidth=2)
            ax3.set_title('Activity Complexity')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Complexity Measure')
            ax3.grid(True, alpha=0.3)
            
        # Phase space plot
        ax4 = axes[1, 1]
        if 'total_activity' in activity_data and 'synchronization' in activity_data:
            ax4.scatter(activity_data['total_activity'], activity_data['synchronization'], 
                       c=time_points, cmap='viridis', alpha=0.6)
            ax4.set_title('Activity Phase Space')
            ax4.set_xlabel('Total Activity')
            ax4.set_ylabel('Synchronization')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_interactive_spike_plot(self, 
                                    spike_data: SpikeData,
                                    save_path: Optional[str] = None) -> go.Figure:
        """Create interactive spike raster plot using Plotly."""
        
        fig = go.Figure()
        
        # Add spike data
        fig.add_trace(go.Scatter(
            x=spike_data.spike_times,
            y=spike_data.neuron_ids,
            mode='markers',
            marker=dict(
                size=6,
                color=spike_data.amplitudes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Spike Amplitude")
            ),
            text=[f'Neuron: {n}, Time: {t:.3f}s, Amp: {a:.2f}' 
                  for n, t, a in zip(spike_data.neuron_ids, spike_data.spike_times, spike_data.amplitudes)],
            hovertemplate='%{text}<extra></extra>',
            name='Spikes'
        ))
        
        # Update layout
        fig.update_layout(
            title='Interactive Neural Spike Raster Plot',
            xaxis_title='Time (s)',
            yaxis_title='Neuron ID',
            hovermode='closest',
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig


class NetworkTopologyVisualizer:
    """Visualizes network topology and connectivity patterns."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        self.figsize = figsize
        
    def create_topology_graph(self, 
                            topology: NetworkTopology,
                            layout: str = 'spring',
                            node_size_attr: Optional[str] = None,
                            edge_width_attr: bool = True,
                            save_path: Optional[str] = None) -> plt.Figure:
        """Create network topology visualization using NetworkX."""
        
        # Create NetworkX graph
        G = nx.from_numpy_array(topology.adjacency_matrix)
        
        # Set node attributes
        for node_id, attrs in topology.node_attributes.items():
            if node_id in G.nodes():
                for attr_name, attr_value in attrs.items():
                    G.nodes[node_id][attr_name] = attr_value
                    
        # Calculate layout
        if topology.node_positions:
            pos = topology.node_positions
        else:
            if layout == 'spring':
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'random':
                pos = nx.random_layout(G)
            else:
                pos = nx.spring_layout(G)
                
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Determine node sizes
        if node_size_attr and topology.node_attributes:
            node_sizes = []
            for node in G.nodes():
                if node in topology.node_attributes and node_size_attr in topology.node_attributes[node]:
                    size = topology.node_attributes[node][node_size_attr] * 300
                else:
                    size = 300
                node_sizes.append(size)
        else:
            node_sizes = 300
            
        # Determine edge widths
        if edge_width_attr and topology.edge_weights is not None:
            edge_widths = []
            for edge in G.edges():
                i, j = edge
                weight = topology.adjacency_matrix[i, j]
                edge_widths.append(max(0.1, weight * 3))
        else:
            edge_widths = 1.0
            
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                              alpha=0.5, edge_color='gray', ax=ax)
        
        # Add labels if available
        if topology.node_labels:
            nx.draw_networkx_labels(G, pos, topology.node_labels, 
                                  font_size=8, ax=ax)
        else:
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
            
        # Add title and formatting
        ax.set_title('Network Topology Visualization')
        ax.axis('off')
        
        # Add network statistics
        stats_text = f'Nodes: {G.number_of_nodes()}\n'
        stats_text += f'Edges: {G.number_of_edges()}\n'
        stats_text += f'Density: {nx.density(G):.3f}\n'
        
        if G.number_of_nodes() > 0:
            try:
                stats_text += f'Avg Clustering: {nx.average_clustering(G):.3f}\n'
                if nx.is_connected(G):
                    stats_text += f'Avg Path Length: {nx.average_shortest_path_length(G):.2f}'
            except:
                pass  # Skip if computation fails
                
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
               
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_connectivity_matrix(self, 
                                 topology: NetworkTopology,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Create connectivity matrix heatmap."""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(topology.adjacency_matrix, cmap='Blues', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Connection Strength', rotation=270, labelpad=15)
        
        # Formatting
        ax.set_xlabel('Target Neuron')
        ax.set_ylabel('Source Neuron')
        ax.set_title('Network Connectivity Matrix')
        
        # Add grid for better readability
        ax.set_xticks(np.arange(topology.adjacency_matrix.shape[1]))
        ax.set_yticks(np.arange(topology.adjacency_matrix.shape[0]))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_degree_distribution(self, 
                                 topology: NetworkTopology,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Create degree distribution analysis."""
        
        # Calculate degrees
        G = nx.from_numpy_array(topology.adjacency_matrix)
        degrees = [d for n, d in G.degree()]
        
        # For directed graphs, calculate in/out degrees; for undirected, use total degree
        if G.is_directed():
            in_degrees = [d for n, d in G.in_degree()]
            out_degrees = [d for n, d in G.out_degree()]
        else:
            in_degrees = degrees.copy()  # For undirected graphs, in-degree = total degree
            out_degrees = degrees.copy()  # For undirected graphs, out-degree = total degree
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Total degree distribution
        axes[0].hist(degrees, bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_title('Total Degree Distribution')
        axes[0].set_xlabel('Degree')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # In-degree distribution
        axes[1].hist(in_degrees, bins=20, alpha=0.7, edgecolor='black', color='orange')
        axes[1].set_title('In-Degree Distribution')
        axes[1].set_xlabel('In-Degree')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        # Out-degree distribution
        axes[2].hist(out_degrees, bins=20, alpha=0.7, edgecolor='black', color='green')
        axes[2].set_title('Out-Degree Distribution')
        axes[2].set_xlabel('Out-Degree')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_interactive_network(self, 
                                 topology: NetworkTopology,
                                 save_path: Optional[str] = None) -> go.Figure:
        """Create interactive network visualization using Plotly."""
        
        G = nx.from_numpy_array(topology.adjacency_matrix)
        
        # Calculate layout
        if topology.node_positions:
            pos = topology.node_positions
        else:
            pos = nx.spring_layout(G, k=1, iterations=50)
            
        # Prepare edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                              line=dict(width=0.5, color='#888'),
                              hoverinfo='none',
                              mode='lines')
                              
        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node information
            adjacencies = list(G.neighbors(node))
            node_text.append(f'Node {node}<br>Connections: {len(adjacencies)}')
            node_info.append(f'Node: {node}, Degree: {len(adjacencies)}')
            
        node_trace = go.Scatter(x=node_x, y=node_y,
                              mode='markers',
                              hoverinfo='text',
                              text=node_info,
                              marker=dict(showscale=True,
                                        colorscale='YlGnBu',
                                        reversescale=True,
                                        color=[],
                                        size=10,
                                        colorbar=dict(
                                            thickness=15,
                                            len=0.5,
                                            x=1.02,
                                            title="Node Degree"
                                        ),
                                        line=dict(width=2)))
                                        
        # Color nodes by degree
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))
            
        node_trace.marker.color = node_adjacencies
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                            title=dict(text='Interactive Network Topology', font=dict(size=16)),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(
                                text="Network topology visualization with interactive nodes",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor="left", yanchor="bottom",
                                font=dict(color="#888", size=12)
                            )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
                       
        if save_path:
            fig.write_html(save_path)
            
        return fig


class MemoryStateVisualizer:
    """Visualizes memory system states and dynamics."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def create_memory_usage_plot(self, 
                               memory_data: Dict[str, List[float]],
                               time_points: List[float],
                               save_path: Optional[str] = None) -> plt.Figure:
        """Create memory usage visualization across different memory types."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Memory System State Visualization', fontsize=16)
        
        # Working memory usage
        ax1 = axes[0, 0]
        if 'working_memory' in memory_data:
            ax1.plot(time_points, memory_data['working_memory'], 'b-', linewidth=2)
            ax1.fill_between(time_points, memory_data['working_memory'], alpha=0.3)
            ax1.set_title('Working Memory Usage')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Usage (%)')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
        # Episodic memory
        ax2 = axes[0, 1]
        if 'episodic_memory' in memory_data:
            ax2.plot(time_points, memory_data['episodic_memory'], 'r-', linewidth=2)
            ax2.fill_between(time_points, memory_data['episodic_memory'], alpha=0.3, color='red')
            ax2.set_title('Episodic Memory Usage')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Usage (%)')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
            
        # Semantic memory
        ax3 = axes[1, 0]
        if 'semantic_memory' in memory_data:
            ax3.plot(time_points, memory_data['semantic_memory'], 'g-', linewidth=2)
            ax3.fill_between(time_points, memory_data['semantic_memory'], alpha=0.3, color='green')
            ax3.set_title('Semantic Memory Usage')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Usage (%)')
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 100)
            
        # Memory consolidation activity
        ax4 = axes[1, 1]
        if 'consolidation_activity' in memory_data:
            ax4.plot(time_points, memory_data['consolidation_activity'], 'purple', linewidth=2)
            ax4.set_title('Memory Consolidation Activity')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Activity Level')
            ax4.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
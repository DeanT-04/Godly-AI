"""
Network Topology Representation and Management

This module provides the core data structures and operations for representing
and manipulating network topologies in the Godly AI system.
"""

from typing import NamedTuple, Dict, List, Tuple, Optional, Any
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
import numpy as np
from enum import Enum


class ConnectionType(Enum):
    """Types of neural connections."""
    EXCITATORY = "excitatory"
    INHIBITORY = "inhibitory"
    MODULATORY = "modulatory"


@dataclass
class NeuronParams:
    """Parameters for individual neurons in the network."""
    neuron_type: str = "lif"  # Type of neuron model
    tau_mem: float = 20e-3    # Membrane time constant
    tau_syn: float = 5e-3     # Synaptic time constant
    v_thresh: float = -50e-3  # Spike threshold
    connection_type: ConnectionType = ConnectionType.EXCITATORY


@dataclass
class ConnectionParams:
    """Parameters for synaptic connections."""
    weight: float = 1.0           # Synaptic weight
    delay: float = 1e-3           # Synaptic delay
    plasticity_enabled: bool = True  # Whether connection can be modified
    connection_type: ConnectionType = ConnectionType.EXCITATORY


class NetworkTopology(NamedTuple):
    """
    Complete network topology representation.
    
    This structure contains all information needed to define the network
    architecture and can be efficiently modified during evolution.
    """
    # Core connectivity
    adjacency_matrix: jnp.ndarray      # [n_neurons, n_neurons] - binary connectivity
    connection_weights: jnp.ndarray    # [n_neurons, n_neurons] - synaptic weights
    connection_delays: jnp.ndarray     # [n_neurons, n_neurons] - synaptic delays
    
    # Neuron properties
    neuron_parameters: Dict[int, NeuronParams]  # Per-neuron parameters
    
    # Connection properties
    connection_parameters: Dict[Tuple[int, int], ConnectionParams]  # Per-connection parameters
    
    # Topology metadata
    n_neurons: int                     # Total number of neurons
    n_connections: int                 # Total number of connections
    spectral_radius: float             # Current spectral radius
    modularity: float                  # Network modularity measure
    
    # Performance tracking
    performance_history: jnp.ndarray   # Recent performance scores
    fitness_score: float               # Current fitness for evolution


class TopologyManager:
    """
    Manager class for network topology operations.
    
    Provides methods for creating, analyzing, and modifying network topologies
    while maintaining biological plausibility and computational efficiency.
    """
    
    def __init__(self, max_neurons: int = 10000, max_connections_per_neuron: int = 1000):
        """Initialize topology manager with capacity constraints."""
        self.max_neurons = max_neurons
        self.max_connections_per_neuron = max_connections_per_neuron
    
    def create_random_topology(
        self,
        n_neurons: int,
        connectivity: float = 0.1,
        excitatory_ratio: float = 0.8,
        key: Optional[jax.random.PRNGKey] = None
    ) -> NetworkTopology:
        """
        Create a random network topology with specified properties.
        
        Args:
            n_neurons: Number of neurons in the network
            connectivity: Fraction of possible connections to create
            excitatory_ratio: Fraction of excitatory neurons
            key: Random key for reproducible generation
            
        Returns:
            NetworkTopology with random connectivity
        """
        if key is None:
            key = random.PRNGKey(0)
        
        key, *subkeys = random.split(key, 5)
        
        # Create adjacency matrix
        connection_prob = connectivity
        adjacency = random.bernoulli(
            subkeys[0], connection_prob, (n_neurons, n_neurons)
        )
        
        # Remove self-connections
        adjacency = adjacency.at[jnp.diag_indices(n_neurons)].set(False)
        
        # Create connection weights
        weights = random.normal(subkeys[1], (n_neurons, n_neurons)) * adjacency
        
        # Set excitatory/inhibitory structure
        n_excitatory = int(excitatory_ratio * n_neurons)
        weights = weights.at[:n_excitatory, :].set(jnp.abs(weights[:n_excitatory, :]))
        weights = weights.at[n_excitatory:, :].set(-jnp.abs(weights[n_excitatory:, :]))
        
        # Create connection delays (1-5ms range)
        delays = random.uniform(subkeys[2], (n_neurons, n_neurons), minval=1e-3, maxval=5e-3)
        delays = delays * adjacency  # Only for existing connections
        
        # Create neuron parameters
        neuron_parameters = {}
        for i in range(n_neurons):
            conn_type = ConnectionType.EXCITATORY if i < n_excitatory else ConnectionType.INHIBITORY
            neuron_parameters[i] = NeuronParams(connection_type=conn_type)
        
        # Create connection parameters
        connection_parameters = {}
        for i in range(n_neurons):
            for j in range(n_neurons):
                if adjacency[i, j]:
                    conn_type = neuron_parameters[i].connection_type
                    connection_parameters[(i, j)] = ConnectionParams(
                        weight=float(weights[i, j]),
                        delay=float(delays[i, j]),
                        connection_type=conn_type
                    )
        
        # Calculate topology metrics
        n_connections = int(jnp.sum(adjacency))
        spectral_radius = self._compute_spectral_radius(weights)
        modularity = self._compute_modularity(adjacency)
        
        # Initialize performance tracking
        performance_history = jnp.zeros(100)  # Track last 100 evaluations
        fitness_score = 0.0
        
        return NetworkTopology(
            adjacency_matrix=adjacency,
            connection_weights=weights,
            connection_delays=delays,
            neuron_parameters=neuron_parameters,
            connection_parameters=connection_parameters,
            n_neurons=n_neurons,
            n_connections=n_connections,
            spectral_radius=spectral_radius,
            modularity=modularity,
            performance_history=performance_history,
            fitness_score=fitness_score
        )
    
    def create_small_world_topology(
        self,
        n_neurons: int,
        k_neighbors: int = 10,
        rewiring_prob: float = 0.1,
        excitatory_ratio: float = 0.8,
        key: Optional[jax.random.PRNGKey] = None
    ) -> NetworkTopology:
        """
        Create a small-world network topology using Watts-Strogatz model.
        
        Args:
            n_neurons: Number of neurons
            k_neighbors: Number of nearest neighbors for initial ring
            rewiring_prob: Probability of rewiring each edge
            excitatory_ratio: Fraction of excitatory neurons
            key: Random key
            
        Returns:
            Small-world network topology
        """
        if key is None:
            key = random.PRNGKey(1)
        
        # Start with regular ring lattice
        adjacency = jnp.zeros((n_neurons, n_neurons), dtype=bool)
        
        # Connect each neuron to k nearest neighbors
        for i in range(n_neurons):
            for j in range(1, k_neighbors // 2 + 1):
                # Connect to neighbors on both sides
                neighbor1 = (i + j) % n_neurons
                neighbor2 = (i - j) % n_neurons
                adjacency = adjacency.at[i, neighbor1].set(True)
                adjacency = adjacency.at[i, neighbor2].set(True)
        
        # Rewire connections with probability rewiring_prob
        key, subkey = random.split(key)
        rewire_mask = random.bernoulli(subkey, rewiring_prob, adjacency.shape)
        
        # For rewired connections, choose new random targets
        key, subkey = random.split(key)
        new_targets = random.randint(subkey, adjacency.shape, 0, n_neurons)
        
        # Apply rewiring (simplified version)
        adjacency_rewired = jnp.where(
            rewire_mask & adjacency,
            False,  # Remove original connection
            adjacency
        )
        
        # Add new random connections where rewiring occurred
        for i in range(n_neurons):
            for j in range(n_neurons):
                if rewire_mask[i, j] and adjacency[i, j]:
                    new_target = new_targets[i, j]
                    if new_target != i:  # Avoid self-connections
                        adjacency_rewired = adjacency_rewired.at[i, new_target].set(True)
        
        # Create weights and other parameters similar to random topology
        key, subkey = random.split(key)
        weights = random.normal(subkey, (n_neurons, n_neurons)) * adjacency_rewired
        
        # Set excitatory/inhibitory structure
        n_excitatory = int(excitatory_ratio * n_neurons)
        weights = weights.at[:n_excitatory, :].set(jnp.abs(weights[:n_excitatory, :]))
        weights = weights.at[n_excitatory:, :].set(-jnp.abs(weights[n_excitatory:, :]))
        
        # Create delays
        key, subkey = random.split(key)
        delays = random.uniform(subkey, (n_neurons, n_neurons), minval=1e-3, maxval=5e-3)
        delays = delays * adjacency_rewired
        
        # Create parameters dictionaries
        neuron_parameters = {}
        for i in range(n_neurons):
            conn_type = ConnectionType.EXCITATORY if i < n_excitatory else ConnectionType.INHIBITORY
            neuron_parameters[i] = NeuronParams(connection_type=conn_type)
        
        connection_parameters = {}
        for i in range(n_neurons):
            for j in range(n_neurons):
                if adjacency_rewired[i, j]:
                    conn_type = neuron_parameters[i].connection_type
                    connection_parameters[(i, j)] = ConnectionParams(
                        weight=float(weights[i, j]),
                        delay=float(delays[i, j]),
                        connection_type=conn_type
                    )
        
        # Calculate metrics
        n_connections = int(jnp.sum(adjacency_rewired))
        spectral_radius = self._compute_spectral_radius(weights)
        modularity = self._compute_modularity(adjacency_rewired)
        
        return NetworkTopology(
            adjacency_matrix=adjacency_rewired,
            connection_weights=weights,
            connection_delays=delays,
            neuron_parameters=neuron_parameters,
            connection_parameters=connection_parameters,
            n_neurons=n_neurons,
            n_connections=n_connections,
            spectral_radius=spectral_radius,
            modularity=modularity,
            performance_history=jnp.zeros(100),
            fitness_score=0.0
        )
    
    def _compute_spectral_radius(self, weights: jnp.ndarray) -> float:
        """Compute spectral radius of weight matrix."""
        try:
            eigenvalues = jnp.linalg.eigvals(weights)
            return float(jnp.max(jnp.abs(eigenvalues)))
        except:
            return 0.0
    
    def _compute_modularity(self, adjacency: jnp.ndarray) -> float:
        """
        Compute network modularity using simplified community detection.
        
        This is a simplified version that estimates modularity based on
        local clustering patterns.
        """
        n_neurons = adjacency.shape[0]
        if n_neurons < 3:
            return 0.0
        
        # Compute local clustering coefficient as proxy for modularity
        clustering_coeffs = []
        
        for i in range(n_neurons):
            neighbors = jnp.where(adjacency[i, :] | adjacency[:, i])[0]
            k = len(neighbors)
            
            if k < 2:
                clustering_coeffs.append(0.0)
                continue
            
            # Count triangles
            triangles = 0
            for j in range(len(neighbors)):
                for l in range(j + 1, len(neighbors)):
                    if adjacency[neighbors[j], neighbors[l]] or adjacency[neighbors[l], neighbors[j]]:
                        triangles += 1
            
            # Clustering coefficient
            possible_triangles = k * (k - 1) // 2
            if possible_triangles > 0:
                clustering_coeffs.append(triangles / possible_triangles)
            else:
                clustering_coeffs.append(0.0)
        
        return float(jnp.mean(jnp.array(clustering_coeffs)))
    
    def analyze_topology(self, topology: NetworkTopology) -> Dict[str, Any]:
        """
        Analyze network topology properties.
        
        Args:
            topology: Network topology to analyze
            
        Returns:
            Dictionary of topology metrics
        """
        analysis = {}
        
        # Basic connectivity metrics
        analysis['n_neurons'] = topology.n_neurons
        analysis['n_connections'] = topology.n_connections
        analysis['connectivity'] = topology.n_connections / (topology.n_neurons * (topology.n_neurons - 1))
        analysis['spectral_radius'] = topology.spectral_radius
        analysis['modularity'] = topology.modularity
        
        # Weight statistics
        nonzero_weights = topology.connection_weights[topology.adjacency_matrix]
        analysis['mean_weight'] = float(jnp.mean(jnp.abs(nonzero_weights)))
        analysis['weight_std'] = float(jnp.std(nonzero_weights))
        
        # Degree distribution
        in_degrees = jnp.sum(topology.adjacency_matrix, axis=0)
        out_degrees = jnp.sum(topology.adjacency_matrix, axis=1)
        analysis['mean_in_degree'] = float(jnp.mean(in_degrees))
        analysis['mean_out_degree'] = float(jnp.mean(out_degrees))
        analysis['degree_std'] = float(jnp.std(in_degrees + out_degrees))
        
        # Excitatory/inhibitory balance
        excitatory_count = sum(1 for params in topology.neuron_parameters.values() 
                             if params.connection_type == ConnectionType.EXCITATORY)
        analysis['excitatory_ratio'] = excitatory_count / topology.n_neurons
        
        # Performance metrics
        analysis['current_fitness'] = topology.fitness_score
        analysis['performance_trend'] = float(jnp.mean(topology.performance_history[-10:]) - 
                                            jnp.mean(topology.performance_history[:10]))
        
        return analysis
    
    def update_performance(
        self, 
        topology: NetworkTopology, 
        performance_score: float
    ) -> NetworkTopology:
        """
        Update topology with new performance measurement.
        
        Args:
            topology: Current topology
            performance_score: New performance score
            
        Returns:
            Updated topology with new performance data
        """
        # Update performance history (rolling buffer)
        new_history = jnp.roll(topology.performance_history, -1)
        new_history = new_history.at[-1].set(performance_score)
        
        # Update fitness score (exponential moving average)
        alpha = 0.1  # Learning rate for fitness update
        new_fitness = alpha * performance_score + (1 - alpha) * topology.fitness_score
        
        return topology._replace(
            performance_history=new_history,
            fitness_score=new_fitness
        )
    
    def validate_topology(self, topology: NetworkTopology) -> bool:
        """
        Validate topology for consistency and biological plausibility.
        
        Args:
            topology: Topology to validate
            
        Returns:
            True if topology is valid
        """
        try:
            # Check basic constraints
            if topology.n_neurons <= 0 or topology.n_neurons > self.max_neurons:
                return False
            
            # Check matrix dimensions
            expected_shape = (topology.n_neurons, topology.n_neurons)
            if (topology.adjacency_matrix.shape != expected_shape or
                topology.connection_weights.shape != expected_shape or
                topology.connection_delays.shape != expected_shape):
                return False
            
            # Check adjacency matrix is binary
            if not jnp.all((topology.adjacency_matrix == 0) | (topology.adjacency_matrix == 1)):
                return False
            
            # Check no self-connections
            if jnp.any(jnp.diag(topology.adjacency_matrix)):
                return False
            
            # Check weight-adjacency consistency (weights should be non-zero where connections exist)
            weight_nonzero = jnp.abs(topology.connection_weights) > 1e-10
            if not jnp.array_equal(weight_nonzero, topology.adjacency_matrix):
                return False
            
            # Check delay-adjacency consistency (delays should be non-zero where connections exist)
            delay_nonzero = topology.connection_delays > 1e-10
            if not jnp.array_equal(delay_nonzero, topology.adjacency_matrix):
                return False
            
            # Check parameter dictionaries
            if len(topology.neuron_parameters) != topology.n_neurons:
                return False
            
            # Check spectral radius is reasonable (allow higher values for complex networks)
            if topology.spectral_radius < 0 or topology.spectral_radius > 10.0:
                return False
            
            return True
            
        except Exception:
            return False
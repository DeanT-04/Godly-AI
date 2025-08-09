"""
Synaptic Pruning and Growth Mechanisms

This module implements biologically-inspired synaptic pruning and growth
mechanisms for network topology optimization in the Godly AI system.
"""

from typing import List, Dict, Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
import numpy as np
from enum import Enum

from ...core.topology.network_topology import NetworkTopology, TopologyManager, ConnectionParams, ConnectionType


class PruningStrategy(Enum):
    """Strategies for synaptic pruning."""
    MAGNITUDE_BASED = "magnitude_based"
    ACTIVITY_BASED = "activity_based"
    AGE_BASED = "age_based"
    CORRELATION_BASED = "correlation_based"
    INFORMATION_THEORETIC = "information_theoretic"
    HOMEOSTATIC = "homeostatic"


class GrowthStrategy(Enum):
    """Strategies for synaptic growth."""
    RANDOM_GROWTH = "random_growth"
    ACTIVITY_DEPENDENT = "activity_dependent"
    CORRELATION_BASED = "correlation_based"
    DISTANCE_BASED = "distance_based"
    FUNCTIONAL_CLUSTERING = "functional_clustering"
    HOMEOSTATIC_GROWTH = "homeostatic_growth"


@dataclass
class PruningParams:
    """Parameters for synaptic pruning."""
    
    # Pruning strategy
    strategy: PruningStrategy = PruningStrategy.MAGNITUDE_BASED
    
    # Magnitude-based pruning
    magnitude_threshold: float = 0.01  # Minimum weight magnitude to keep
    magnitude_percentile: float = 0.1  # Prune bottom 10% by magnitude
    
    # Activity-based pruning
    activity_threshold: float = 0.001  # Minimum activity level
    activity_window: int = 1000        # Window for activity measurement
    
    # Age-based pruning
    max_age_without_use: int = 10000   # Maximum age without activity
    
    # Correlation-based pruning
    correlation_threshold: float = 0.1  # Minimum correlation to keep
    
    # Information-theoretic pruning
    mutual_information_threshold: float = 0.01  # Minimum mutual information
    
    # Homeostatic pruning
    target_firing_rate: float = 0.1    # Target neuron firing rate
    homeostatic_tolerance: float = 0.05 # Tolerance around target
    
    # General pruning parameters
    pruning_rate: float = 0.05         # Fraction of connections to prune per step
    min_connections_per_neuron: int = 1 # Minimum connections to maintain
    preserve_strong_connections: bool = True  # Preserve strongest connections


@dataclass
class GrowthParams:
    """Parameters for synaptic growth."""
    
    # Growth strategy
    strategy: GrowthStrategy = GrowthStrategy.ACTIVITY_DEPENDENT
    
    # Random growth
    random_growth_rate: float = 0.01   # Probability of random connection
    
    # Activity-dependent growth
    activity_threshold: float = 0.1    # Activity level for growth
    correlation_threshold: float = 0.3  # Correlation threshold for growth
    
    # Distance-based growth
    distance_decay: float = 0.1        # Decay factor for distance-based probability
    max_growth_distance: int = 10      # Maximum distance for new connections
    
    # Functional clustering
    cluster_similarity_threshold: float = 0.5  # Similarity for clustering
    
    # Homeostatic growth
    target_connectivity: float = 0.1   # Target network connectivity
    connectivity_tolerance: float = 0.02  # Tolerance around target
    
    # General growth parameters
    growth_rate: float = 0.02          # Rate of new connection formation
    max_connections_per_neuron: int = 1000  # Maximum connections per neuron
    initial_weight_std: float = 0.1    # Standard deviation for new weights
    initial_delay_range: Tuple[float, float] = (1e-3, 5e-3)  # Delay range for new connections


@dataclass
class SynapticState:
    """State information for synaptic connections."""
    
    # Connection activity tracking
    activity_history: Dict[Tuple[int, int], List[float]] = None
    last_activity_time: Dict[Tuple[int, int], float] = None
    
    # Connection age tracking
    connection_ages: Dict[Tuple[int, int], int] = None
    creation_times: Dict[Tuple[int, int], float] = None
    
    # Correlation tracking
    correlation_matrix: Optional[jnp.ndarray] = None
    correlation_history: Dict[Tuple[int, int], List[float]] = None
    
    # Homeostatic tracking
    neuron_firing_rates: Optional[jnp.ndarray] = None
    target_rates: Optional[jnp.ndarray] = None
    
    def __post_init__(self):
        if self.activity_history is None:
            self.activity_history = {}
        if self.last_activity_time is None:
            self.last_activity_time = {}
        if self.connection_ages is None:
            self.connection_ages = {}
        if self.creation_times is None:
            self.creation_times = {}
        if self.correlation_history is None:
            self.correlation_history = {}


class SynapticPruner:
    """
    Implements various synaptic pruning strategies.
    
    This class provides biologically-inspired mechanisms for removing
    weak, unused, or redundant synaptic connections from the network.
    """
    
    def __init__(self, params: Optional[PruningParams] = None):
        """Initialize synaptic pruner."""
        self.params = params or PruningParams()
        self.topology_manager = TopologyManager()
    
    def prune_connections(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        current_time: float = 0.0,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[NetworkTopology, SynapticState]:
        """
        Prune connections based on the configured strategy.
        
        Args:
            topology: Current network topology
            synaptic_state: Current synaptic state
            current_time: Current simulation time
            key: Random key for stochastic pruning
            
        Returns:
            Tuple of (pruned_topology, updated_synaptic_state)
        """
        if key is None:
            key = random.PRNGKey(42)
        
        if self.params.strategy == PruningStrategy.MAGNITUDE_BASED:
            return self._magnitude_based_pruning(topology, synaptic_state, key)
        elif self.params.strategy == PruningStrategy.ACTIVITY_BASED:
            return self._activity_based_pruning(topology, synaptic_state, current_time, key)
        elif self.params.strategy == PruningStrategy.AGE_BASED:
            return self._age_based_pruning(topology, synaptic_state, current_time, key)
        elif self.params.strategy == PruningStrategy.CORRELATION_BASED:
            return self._correlation_based_pruning(topology, synaptic_state, key)
        elif self.params.strategy == PruningStrategy.INFORMATION_THEORETIC:
            return self._information_theoretic_pruning(topology, synaptic_state, key)
        elif self.params.strategy == PruningStrategy.HOMEOSTATIC:
            return self._homeostatic_pruning(topology, synaptic_state, key)
        else:
            return topology, synaptic_state
    
    def _magnitude_based_pruning(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Prune connections based on weight magnitude."""
        
        # Get connection weights
        weights = topology.connection_weights
        adjacency = topology.adjacency_matrix
        
        # Find connections to potentially prune
        existing_connections = []
        connection_weights = []
        
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if adjacency[i, j]:
                    existing_connections.append((i, j))
                    connection_weights.append(abs(weights[i, j]))
        
        if not existing_connections:
            return topology, synaptic_state
        
        connection_weights = jnp.array(connection_weights)
        
        # Determine pruning threshold
        if self.params.magnitude_percentile > 0:
            threshold = jnp.percentile(connection_weights, self.params.magnitude_percentile * 100)
        else:
            threshold = self.params.magnitude_threshold
        
        # Find connections to prune
        connections_to_prune = []
        for i, (source, target) in enumerate(existing_connections):
            if connection_weights[i] < threshold:
                # Check if pruning would violate minimum connectivity
                out_degree = jnp.sum(adjacency[source, :])
                in_degree = jnp.sum(adjacency[:, target])
                
                if (out_degree > self.params.min_connections_per_neuron and
                    in_degree > self.params.min_connections_per_neuron):
                    connections_to_prune.append((source, target))
        
        # Apply pruning
        if connections_to_prune:
            topology = self._remove_connections(topology, connections_to_prune)
            synaptic_state = self._update_state_after_pruning(synaptic_state, connections_to_prune)
        
        return topology, synaptic_state
    
    def _activity_based_pruning(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        current_time: float,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Prune connections based on activity levels."""
        
        connections_to_prune = []
        
        # Check each connection's activity
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if topology.adjacency_matrix[i, j]:
                    connection = (i, j)
                    
                    # Get recent activity
                    if connection in synaptic_state.activity_history:
                        recent_activity = synaptic_state.activity_history[connection]
                        if len(recent_activity) >= self.params.activity_window:
                            avg_activity = jnp.mean(jnp.array(recent_activity[-self.params.activity_window:]))
                            
                            if avg_activity < self.params.activity_threshold:
                                # Check connectivity constraints
                                out_degree = jnp.sum(topology.adjacency_matrix[i, :])
                                in_degree = jnp.sum(topology.adjacency_matrix[:, j])
                                
                                if (out_degree > self.params.min_connections_per_neuron and
                                    in_degree > self.params.min_connections_per_neuron):
                                    connections_to_prune.append(connection)
        
        # Apply pruning
        if connections_to_prune:
            topology = self._remove_connections(topology, connections_to_prune)
            synaptic_state = self._update_state_after_pruning(synaptic_state, connections_to_prune)
        
        return topology, synaptic_state
    
    def _age_based_pruning(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        current_time: float,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Prune connections based on age without use."""
        
        connections_to_prune = []
        
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if topology.adjacency_matrix[i, j]:
                    connection = (i, j)
                    
                    # Check last activity time
                    if connection in synaptic_state.last_activity_time:
                        time_since_activity = current_time - synaptic_state.last_activity_time[connection]
                        
                        if time_since_activity > self.params.max_age_without_use:
                            # Check connectivity constraints
                            out_degree = jnp.sum(topology.adjacency_matrix[i, :])
                            in_degree = jnp.sum(topology.adjacency_matrix[:, j])
                            
                            if (out_degree > self.params.min_connections_per_neuron and
                                in_degree > self.params.min_connections_per_neuron):
                                connections_to_prune.append(connection)
        
        # Apply pruning
        if connections_to_prune:
            topology = self._remove_connections(topology, connections_to_prune)
            synaptic_state = self._update_state_after_pruning(synaptic_state, connections_to_prune)
        
        return topology, synaptic_state
    
    def _correlation_based_pruning(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Prune connections based on correlation between neurons."""
        
        if synaptic_state.correlation_matrix is None:
            return topology, synaptic_state
        
        connections_to_prune = []
        correlation_matrix = synaptic_state.correlation_matrix
        
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if topology.adjacency_matrix[i, j]:
                    correlation = abs(correlation_matrix[i, j])
                    
                    if correlation < self.params.correlation_threshold:
                        # Check connectivity constraints
                        out_degree = jnp.sum(topology.adjacency_matrix[i, :])
                        in_degree = jnp.sum(topology.adjacency_matrix[:, j])
                        
                        if (out_degree > self.params.min_connections_per_neuron and
                            in_degree > self.params.min_connections_per_neuron):
                            connections_to_prune.append((i, j))
        
        # Apply pruning
        if connections_to_prune:
            topology = self._remove_connections(topology, connections_to_prune)
            synaptic_state = self._update_state_after_pruning(synaptic_state, connections_to_prune)
        
        return topology, synaptic_state
    
    def _information_theoretic_pruning(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Prune connections based on mutual information."""
        # Simplified implementation - would need actual mutual information calculation
        return self._correlation_based_pruning(topology, synaptic_state, key)
    
    def _homeostatic_pruning(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Prune connections to maintain homeostatic balance."""
        
        if synaptic_state.neuron_firing_rates is None:
            return topology, synaptic_state
        
        connections_to_prune = []
        firing_rates = synaptic_state.neuron_firing_rates
        target_rate = self.params.target_firing_rate
        tolerance = self.params.homeostatic_tolerance
        
        # Find neurons with excessive firing rates
        overactive_neurons = jnp.where(firing_rates > target_rate + tolerance)[0]
        
        for neuron_id in overactive_neurons:
            # Find outgoing connections to prune
            outgoing_connections = []
            for j in range(topology.n_neurons):
                if topology.adjacency_matrix[neuron_id, j]:
                    outgoing_connections.append((int(neuron_id), j))
            
            if len(outgoing_connections) > self.params.min_connections_per_neuron:
                # Prune weakest connections
                connection_weights = []
                for source, target in outgoing_connections:
                    connection_weights.append(abs(topology.connection_weights[source, target]))
                
                # Sort by weight and prune weakest
                sorted_indices = jnp.argsort(jnp.array(connection_weights))
                n_to_prune = min(len(outgoing_connections) - self.params.min_connections_per_neuron, 
                               max(1, len(outgoing_connections) // 4))  # Prune up to 25%
                
                for idx in sorted_indices[:n_to_prune]:
                    connections_to_prune.append(outgoing_connections[idx])
        
        # Apply pruning
        if connections_to_prune:
            topology = self._remove_connections(topology, connections_to_prune)
            synaptic_state = self._update_state_after_pruning(synaptic_state, connections_to_prune)
        
        return topology, synaptic_state
    
    def _remove_connections(
        self,
        topology: NetworkTopology,
        connections_to_remove: List[Tuple[int, int]]
    ) -> NetworkTopology:
        """Remove specified connections from topology."""
        
        new_adjacency = topology.adjacency_matrix
        new_weights = topology.connection_weights
        new_delays = topology.connection_delays
        new_conn_params = topology.connection_parameters.copy()
        
        for source, target in connections_to_remove:
            new_adjacency = new_adjacency.at[source, target].set(False)
            new_weights = new_weights.at[source, target].set(0.0)
            new_delays = new_delays.at[source, target].set(0.0)
            
            if (source, target) in new_conn_params:
                del new_conn_params[(source, target)]
        
        # Update metrics
        new_n_connections = topology.n_connections - len(connections_to_remove)
        new_spectral_radius = self.topology_manager._compute_spectral_radius(new_weights)
        new_modularity = self.topology_manager._compute_modularity(new_adjacency)
        
        return topology._replace(
            adjacency_matrix=new_adjacency,
            connection_weights=new_weights,
            connection_delays=new_delays,
            connection_parameters=new_conn_params,
            n_connections=new_n_connections,
            spectral_radius=new_spectral_radius,
            modularity=new_modularity
        )
    
    def _update_state_after_pruning(
        self,
        synaptic_state: SynapticState,
        pruned_connections: List[Tuple[int, int]]
    ) -> SynapticState:
        """Update synaptic state after pruning connections."""
        
        # Remove pruned connections from tracking
        for connection in pruned_connections:
            if connection in synaptic_state.activity_history:
                del synaptic_state.activity_history[connection]
            if connection in synaptic_state.last_activity_time:
                del synaptic_state.last_activity_time[connection]
            if connection in synaptic_state.connection_ages:
                del synaptic_state.connection_ages[connection]
            if connection in synaptic_state.creation_times:
                del synaptic_state.creation_times[connection]
            if connection in synaptic_state.correlation_history:
                del synaptic_state.correlation_history[connection]
        
        return synaptic_state


class SynapticGrower:
    """
    Implements various synaptic growth strategies.
    
    This class provides mechanisms for adding new synaptic connections
    to the network based on activity patterns and functional requirements.
    """
    
    def __init__(self, params: Optional[GrowthParams] = None):
        """Initialize synaptic grower."""
        self.params = params or GrowthParams()
        self.topology_manager = TopologyManager()
    
    def grow_connections(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        current_time: float = 0.0,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[NetworkTopology, SynapticState]:
        """
        Grow new connections based on the configured strategy.
        
        Args:
            topology: Current network topology
            synaptic_state: Current synaptic state
            current_time: Current simulation time
            key: Random key for stochastic growth
            
        Returns:
            Tuple of (grown_topology, updated_synaptic_state)
        """
        if key is None:
            key = random.PRNGKey(42)
        
        if self.params.strategy == GrowthStrategy.RANDOM_GROWTH:
            return self._random_growth(topology, synaptic_state, current_time, key)
        elif self.params.strategy == GrowthStrategy.ACTIVITY_DEPENDENT:
            return self._activity_dependent_growth(topology, synaptic_state, current_time, key)
        elif self.params.strategy == GrowthStrategy.CORRELATION_BASED:
            return self._correlation_based_growth(topology, synaptic_state, current_time, key)
        elif self.params.strategy == GrowthStrategy.DISTANCE_BASED:
            return self._distance_based_growth(topology, synaptic_state, current_time, key)
        elif self.params.strategy == GrowthStrategy.FUNCTIONAL_CLUSTERING:
            return self._functional_clustering_growth(topology, synaptic_state, current_time, key)
        elif self.params.strategy == GrowthStrategy.HOMEOSTATIC_GROWTH:
            return self._homeostatic_growth(topology, synaptic_state, current_time, key)
        else:
            return topology, synaptic_state
    
    def _random_growth(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        current_time: float,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Grow connections randomly."""
        
        # Find potential connection sites
        potential_connections = []
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if i != j and not topology.adjacency_matrix[i, j]:
                    # Check connection limits
                    out_degree = jnp.sum(topology.adjacency_matrix[i, :])
                    in_degree = jnp.sum(topology.adjacency_matrix[:, j])
                    
                    if (out_degree < self.params.max_connections_per_neuron and
                        in_degree < self.params.max_connections_per_neuron):
                        potential_connections.append((i, j))
        
        if not potential_connections:
            return topology, synaptic_state
        
        # Determine number of connections to add
        n_to_add = max(1, int(len(potential_connections) * self.params.growth_rate))
        
        # Randomly select connections to add
        key, subkey = random.split(key)
        selected_indices = random.choice(
            subkey, len(potential_connections), (n_to_add,), replace=False
        )
        
        connections_to_add = [potential_connections[i] for i in selected_indices]
        
        # Add connections
        topology, synaptic_state = self._add_connections(
            topology, synaptic_state, connections_to_add, current_time, key
        )
        
        return topology, synaptic_state
    
    def _activity_dependent_growth(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        current_time: float,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Grow connections based on neural activity patterns."""
        
        if synaptic_state.neuron_firing_rates is None:
            return self._random_growth(topology, synaptic_state, current_time, key)
        
        firing_rates = synaptic_state.neuron_firing_rates
        active_neurons = jnp.where(firing_rates > self.params.activity_threshold)[0]
        
        if len(active_neurons) < 2:
            return topology, synaptic_state
        
        # Find potential connections between active neurons
        potential_connections = []
        for i in active_neurons:
            for j in active_neurons:
                if i != j and not topology.adjacency_matrix[i, j]:
                    # Check connection limits
                    out_degree = jnp.sum(topology.adjacency_matrix[i, :])
                    in_degree = jnp.sum(topology.adjacency_matrix[:, j])
                    
                    if (out_degree < self.params.max_connections_per_neuron and
                        in_degree < self.params.max_connections_per_neuron):
                        
                        # Weight by activity levels
                        connection_strength = firing_rates[i] * firing_rates[j]
                        potential_connections.append(((int(i), int(j)), connection_strength))
        
        if not potential_connections:
            return topology, synaptic_state
        
        # Sort by connection strength and select top candidates
        potential_connections.sort(key=lambda x: x[1], reverse=True)
        n_to_add = max(1, int(len(potential_connections) * self.params.growth_rate))
        
        connections_to_add = [conn for conn, _ in potential_connections[:n_to_add]]
        
        # Add connections
        topology, synaptic_state = self._add_connections(
            topology, synaptic_state, connections_to_add, current_time, key
        )
        
        return topology, synaptic_state
    
    def _correlation_based_growth(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        current_time: float,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Grow connections based on neural correlations."""
        
        if synaptic_state.correlation_matrix is None:
            return self._random_growth(topology, synaptic_state, current_time, key)
        
        correlation_matrix = synaptic_state.correlation_matrix
        
        # Find highly correlated neuron pairs without connections
        potential_connections = []
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if i != j and not topology.adjacency_matrix[i, j]:
                    correlation = abs(correlation_matrix[i, j])
                    
                    if correlation > self.params.correlation_threshold:
                        # Check connection limits
                        out_degree = jnp.sum(topology.adjacency_matrix[i, :])
                        in_degree = jnp.sum(topology.adjacency_matrix[:, j])
                        
                        if (out_degree < self.params.max_connections_per_neuron and
                            in_degree < self.params.max_connections_per_neuron):
                            potential_connections.append(((i, j), correlation))
        
        if not potential_connections:
            return topology, synaptic_state
        
        # Sort by correlation and select top candidates
        potential_connections.sort(key=lambda x: x[1], reverse=True)
        n_to_add = max(1, int(len(potential_connections) * self.params.growth_rate))
        
        connections_to_add = [conn for conn, _ in potential_connections[:n_to_add]]
        
        # Add connections
        topology, synaptic_state = self._add_connections(
            topology, synaptic_state, connections_to_add, current_time, key
        )
        
        return topology, synaptic_state
    
    def _distance_based_growth(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        current_time: float,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Grow connections based on spatial distance (simplified)."""
        # Simplified implementation - assumes neuron IDs represent spatial ordering
        return self._random_growth(topology, synaptic_state, current_time, key)
    
    def _functional_clustering_growth(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        current_time: float,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Grow connections to enhance functional clustering."""
        # Simplified implementation
        return self._correlation_based_growth(topology, synaptic_state, current_time, key)
    
    def _homeostatic_growth(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        current_time: float,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Grow connections to maintain homeostatic balance."""
        
        current_connectivity = topology.n_connections / (topology.n_neurons * (topology.n_neurons - 1))
        target_connectivity = self.params.target_connectivity
        tolerance = self.params.connectivity_tolerance
        
        if current_connectivity >= target_connectivity - tolerance:
            return topology, synaptic_state
        
        # Need to add connections to reach target connectivity
        target_connections = int(target_connectivity * topology.n_neurons * (topology.n_neurons - 1))
        connections_to_add_count = target_connections - topology.n_connections
        
        # Find potential connection sites
        potential_connections = []
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if i != j and not topology.adjacency_matrix[i, j]:
                    out_degree = jnp.sum(topology.adjacency_matrix[i, :])
                    in_degree = jnp.sum(topology.adjacency_matrix[:, j])
                    
                    if (out_degree < self.params.max_connections_per_neuron and
                        in_degree < self.params.max_connections_per_neuron):
                        potential_connections.append((i, j))
        
        if not potential_connections:
            return topology, synaptic_state
        
        # Randomly select connections to add
        n_to_add = min(connections_to_add_count, len(potential_connections))
        key, subkey = random.split(key)
        selected_indices = random.choice(
            subkey, len(potential_connections), (n_to_add,), replace=False
        )
        
        connections_to_add = [potential_connections[i] for i in selected_indices]
        
        # Add connections
        topology, synaptic_state = self._add_connections(
            topology, synaptic_state, connections_to_add, current_time, key
        )
        
        return topology, synaptic_state
    
    def _add_connections(
        self,
        topology: NetworkTopology,
        synaptic_state: SynapticState,
        connections_to_add: List[Tuple[int, int]],
        current_time: float,
        key: jax.random.PRNGKey
    ) -> Tuple[NetworkTopology, SynapticState]:
        """Add specified connections to topology."""
        
        if not connections_to_add:
            return topology, synaptic_state
        
        new_adjacency = topology.adjacency_matrix
        new_weights = topology.connection_weights
        new_delays = topology.connection_delays
        new_conn_params = topology.connection_parameters.copy()
        
        for source, target in connections_to_add:
            # Set adjacency
            new_adjacency = new_adjacency.at[source, target].set(True)
            
            # Generate weight based on source neuron type
            key, subkey = random.split(key)
            source_type = topology.neuron_parameters[source].connection_type
            
            if source_type == ConnectionType.EXCITATORY:
                weight = abs(random.normal(subkey, ()) * self.params.initial_weight_std)
            else:
                weight = -abs(random.normal(subkey, ()) * self.params.initial_weight_std)
            
            new_weights = new_weights.at[source, target].set(weight)
            
            # Generate delay
            key, subkey = random.split(key)
            delay = random.uniform(
                subkey, (),
                minval=self.params.initial_delay_range[0],
                maxval=self.params.initial_delay_range[1]
            )
            new_delays = new_delays.at[source, target].set(delay)
            
            # Add connection parameters
            new_conn_params[(source, target)] = ConnectionParams(
                weight=float(weight),
                delay=float(delay),
                connection_type=source_type
            )
            
            # Initialize synaptic state tracking
            connection = (source, target)
            synaptic_state.activity_history[connection] = []
            synaptic_state.last_activity_time[connection] = current_time
            synaptic_state.connection_ages[connection] = 0
            synaptic_state.creation_times[connection] = current_time
            synaptic_state.correlation_history[connection] = []
        
        # Update metrics
        new_n_connections = topology.n_connections + len(connections_to_add)
        new_spectral_radius = self.topology_manager._compute_spectral_radius(new_weights)
        new_modularity = self.topology_manager._compute_modularity(new_adjacency)
        
        updated_topology = topology._replace(
            adjacency_matrix=new_adjacency,
            connection_weights=new_weights,
            connection_delays=new_delays,
            connection_parameters=new_conn_params,
            n_connections=new_n_connections,
            spectral_radius=new_spectral_radius,
            modularity=new_modularity
        )
        
        return updated_topology, synaptic_state


class SynapticPlasticityManager:
    """
    Manages both pruning and growth of synaptic connections.
    
    This class coordinates pruning and growth mechanisms to maintain
    optimal network connectivity and performance.
    """
    
    def __init__(
        self,
        pruning_params: Optional[PruningParams] = None,
        growth_params: Optional[GrowthParams] = None
    ):
        """Initialize synaptic plasticity manager."""
        self.pruner = SynapticPruner(pruning_params)
        self.grower = SynapticGrower(growth_params)
        self.synaptic_state = SynapticState()
    
    def update_synaptic_connections(
        self,
        topology: NetworkTopology,
        current_time: float = 0.0,
        activity_data: Optional[Dict[str, Any]] = None,
        key: Optional[jax.random.PRNGKey] = None
    ) -> NetworkTopology:
        """
        Update synaptic connections through pruning and growth.
        
        Args:
            topology: Current network topology
            current_time: Current simulation time
            activity_data: Neural activity data for decision making
            key: Random key for stochastic operations
            
        Returns:
            Updated topology with modified connections
        """
        if key is None:
            key = random.PRNGKey(42)
        
        # Update synaptic state with new activity data
        if activity_data:
            self._update_synaptic_state(activity_data, current_time)
        
        # Apply pruning
        key, subkey = random.split(key)
        topology, self.synaptic_state = self.pruner.prune_connections(
            topology, self.synaptic_state, current_time, subkey
        )
        
        # Apply growth
        key, subkey = random.split(key)
        topology, self.synaptic_state = self.grower.grow_connections(
            topology, self.synaptic_state, current_time, subkey
        )
        
        return topology
    
    def _update_synaptic_state(self, activity_data: Dict[str, Any], current_time: float):
        """Update synaptic state with new activity data."""
        
        # Update firing rates if provided
        if 'firing_rates' in activity_data:
            self.synaptic_state.neuron_firing_rates = activity_data['firing_rates']
        
        # Update correlation matrix if provided
        if 'correlation_matrix' in activity_data:
            self.synaptic_state.correlation_matrix = activity_data['correlation_matrix']
        
        # Update connection activities if provided
        if 'connection_activities' in activity_data:
            for connection, activity in activity_data['connection_activities'].items():
                if connection not in self.synaptic_state.activity_history:
                    self.synaptic_state.activity_history[connection] = []
                
                self.synaptic_state.activity_history[connection].append(activity)
                
                # Keep history bounded
                if len(self.synaptic_state.activity_history[connection]) > 1000:
                    self.synaptic_state.activity_history[connection] = \
                        self.synaptic_state.activity_history[connection][-1000:]
                
                # Update last activity time if activity is significant
                if activity > 0.01:
                    self.synaptic_state.last_activity_time[connection] = current_time
        
        # Update connection ages
        for connection in self.synaptic_state.connection_ages:
            self.synaptic_state.connection_ages[connection] += 1
    
    def get_plasticity_statistics(self) -> Dict[str, Any]:
        """Get statistics about synaptic plasticity."""
        stats = {
            'total_tracked_connections': len(self.synaptic_state.activity_history),
            'active_connections': sum(1 for history in self.synaptic_state.activity_history.values() 
                                    if history and history[-1] > 0.01),
            'average_connection_age': np.mean(list(self.synaptic_state.connection_ages.values())) 
                                    if self.synaptic_state.connection_ages else 0,
            'pruning_strategy': self.pruner.params.strategy.value,
            'growth_strategy': self.grower.params.strategy.value
        }
        
        if self.synaptic_state.neuron_firing_rates is not None:
            stats['mean_firing_rate'] = float(jnp.mean(self.synaptic_state.neuron_firing_rates))
            stats['firing_rate_std'] = float(jnp.std(self.synaptic_state.neuron_firing_rates))
        
        return stats
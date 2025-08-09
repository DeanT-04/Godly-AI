"""
Topology Mutation Operators

This module implements various mutation operators for evolving network topologies
in the Godly AI system. Includes connection addition/removal, weight mutations,
and structural modifications.
"""

from typing import List, Tuple, Optional, Dict, Any
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .network_topology import NetworkTopology, TopologyManager, NeuronParams, ConnectionParams, ConnectionType


class MutationType(Enum):
    """Types of topology mutations."""
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"
    MODIFY_WEIGHT = "modify_weight"
    ADD_NEURON = "add_neuron"
    REMOVE_NEURON = "remove_neuron"
    REWIRE_CONNECTION = "rewire_connection"
    MODIFY_DELAY = "modify_delay"
    PRUNE_WEAK_CONNECTIONS = "prune_weak_connections"
    DUPLICATE_NEURON = "duplicate_neuron"
    MERGE_NEURONS = "merge_neurons"


@dataclass
class MutationParams:
    """Parameters controlling mutation behavior."""
    
    # Mutation probabilities
    add_connection_prob: float = 0.1
    remove_connection_prob: float = 0.05
    modify_weight_prob: float = 0.3
    add_neuron_prob: float = 0.02
    remove_neuron_prob: float = 0.01
    rewire_connection_prob: float = 0.05
    modify_delay_prob: float = 0.1
    prune_weak_prob: float = 0.05
    duplicate_neuron_prob: float = 0.01
    merge_neurons_prob: float = 0.005
    
    # Mutation magnitudes
    weight_mutation_std: float = 0.1      # Standard deviation for weight changes
    delay_mutation_std: float = 0.5e-3    # Standard deviation for delay changes (0.5ms)
    
    # Constraints
    max_connections_per_neuron: int = 1000
    min_connections_per_neuron: int = 1
    max_total_neurons: int = 10000
    min_total_neurons: int = 10
    
    # Pruning thresholds
    weak_connection_threshold: float = 0.01  # Connections below this weight are "weak"
    
    # Structural constraints
    maintain_excitatory_ratio: bool = True
    target_excitatory_ratio: float = 0.8
    preserve_spectral_radius: bool = True
    target_spectral_radius: float = 0.95


class TopologyMutation:
    """
    Represents a single topology mutation operation.
    
    This class encapsulates all information needed to apply or reverse
    a specific mutation to a network topology.
    """
    
    def __init__(
        self,
        mutation_type: MutationType,
        parameters: Dict[str, Any],
        expected_fitness_change: float = 0.0
    ):
        """
        Initialize mutation operation.
        
        Args:
            mutation_type: Type of mutation to perform
            parameters: Parameters specific to this mutation
            expected_fitness_change: Predicted fitness change from this mutation
        """
        self.mutation_type = mutation_type
        self.parameters = parameters
        self.expected_fitness_change = expected_fitness_change
        self.applied = False
        self.actual_fitness_change = 0.0
    
    def __repr__(self) -> str:
        return f"TopologyMutation({self.mutation_type.value}, {self.parameters})"


class MutationOperator:
    """
    Implements topology mutation operations for network evolution.
    
    This class provides methods to generate and apply various types of
    mutations to network topologies, enabling evolutionary optimization
    of network structure.
    """
    
    def __init__(self, params: Optional[MutationParams] = None):
        """Initialize mutation operator with given parameters."""
        self.params = params or MutationParams()
        self.topology_manager = TopologyManager()
    
    def generate_mutations(
        self,
        topology: NetworkTopology,
        n_mutations: int = 5,
        key: Optional[jax.random.PRNGKey] = None
    ) -> List[TopologyMutation]:
        """
        Generate a list of potential mutations for the given topology.
        
        Args:
            topology: Current network topology
            n_mutations: Number of mutations to generate
            key: Random key for mutation generation
            
        Returns:
            List of potential mutations
        """
        if key is None:
            key = random.PRNGKey(42)
        
        mutations = []
        
        for i in range(n_mutations):
            key, subkey = random.split(key)
            mutation = self._generate_single_mutation(topology, subkey)
            if mutation is not None:
                mutations.append(mutation)
        
        return mutations
    
    def _generate_single_mutation(
        self,
        topology: NetworkTopology,
        key: jax.random.PRNGKey
    ) -> Optional[TopologyMutation]:
        """Generate a single random mutation."""
        
        # Choose mutation type based on probabilities
        mutation_probs = jnp.array([
            self.params.add_connection_prob,
            self.params.remove_connection_prob,
            self.params.modify_weight_prob,
            self.params.add_neuron_prob,
            self.params.remove_neuron_prob,
            self.params.rewire_connection_prob,
            self.params.modify_delay_prob,
            self.params.prune_weak_prob,
            self.params.duplicate_neuron_prob,
            self.params.merge_neurons_prob
        ])
        
        # Normalize probabilities
        mutation_probs = mutation_probs / jnp.sum(mutation_probs)
        
        # Sample mutation type
        key, subkey = random.split(key)
        mutation_idx = random.categorical(subkey, jnp.log(mutation_probs))
        
        mutation_types = list(MutationType)
        chosen_type = mutation_types[int(mutation_idx)]
        
        # Generate specific mutation based on type
        key, subkey = random.split(key)
        
        if chosen_type == MutationType.ADD_CONNECTION:
            return self._generate_add_connection_mutation(topology, subkey)
        elif chosen_type == MutationType.REMOVE_CONNECTION:
            return self._generate_remove_connection_mutation(topology, subkey)
        elif chosen_type == MutationType.MODIFY_WEIGHT:
            return self._generate_modify_weight_mutation(topology, subkey)
        elif chosen_type == MutationType.ADD_NEURON:
            return self._generate_add_neuron_mutation(topology, subkey)
        elif chosen_type == MutationType.REMOVE_NEURON:
            return self._generate_remove_neuron_mutation(topology, subkey)
        elif chosen_type == MutationType.REWIRE_CONNECTION:
            return self._generate_rewire_connection_mutation(topology, subkey)
        elif chosen_type == MutationType.MODIFY_DELAY:
            return self._generate_modify_delay_mutation(topology, subkey)
        elif chosen_type == MutationType.PRUNE_WEAK_CONNECTIONS:
            return self._generate_prune_weak_mutation(topology, subkey)
        elif chosen_type == MutationType.DUPLICATE_NEURON:
            return self._generate_duplicate_neuron_mutation(topology, subkey)
        elif chosen_type == MutationType.MERGE_NEURONS:
            return self._generate_merge_neurons_mutation(topology, subkey)
        
        return None
    
    def _generate_add_connection_mutation(
        self,
        topology: NetworkTopology,
        key: jax.random.PRNGKey
    ) -> Optional[TopologyMutation]:
        """Generate mutation to add a new connection."""
        
        # Find potential connection sites
        possible_connections = []
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if i != j and not topology.adjacency_matrix[i, j]:
                    # Check connection limits
                    out_degree = jnp.sum(topology.adjacency_matrix[i, :])
                    in_degree = jnp.sum(topology.adjacency_matrix[:, j])
                    
                    if (out_degree < self.params.max_connections_per_neuron and
                        in_degree < self.params.max_connections_per_neuron):
                        possible_connections.append((i, j))
        
        if not possible_connections:
            return None
        
        # Choose random connection to add
        key, subkey = random.split(key)
        conn_idx = random.randint(subkey, (), 0, len(possible_connections))
        source, target = possible_connections[int(conn_idx)]
        
        # Generate connection parameters
        key, subkey = random.split(key)
        
        # Determine connection type based on source neuron
        source_type = topology.neuron_parameters[source].connection_type
        if source_type == ConnectionType.EXCITATORY:
            weight = abs(random.normal(subkey, ()) * 0.1)
        else:
            weight = -abs(random.normal(subkey, ()) * 0.1)
        
        key, subkey = random.split(key)
        delay = random.uniform(subkey, (), minval=1e-3, maxval=5e-3)
        
        return TopologyMutation(
            MutationType.ADD_CONNECTION,
            {
                'source': source,
                'target': target,
                'weight': float(weight),
                'delay': float(delay)
            }
        )
    
    def _generate_remove_connection_mutation(
        self,
        topology: NetworkTopology,
        key: jax.random.PRNGKey
    ) -> Optional[TopologyMutation]:
        """Generate mutation to remove an existing connection."""
        
        # Find existing connections
        existing_connections = []
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if topology.adjacency_matrix[i, j]:
                    # Check if removal would violate minimum connectivity
                    out_degree = jnp.sum(topology.adjacency_matrix[i, :])
                    in_degree = jnp.sum(topology.adjacency_matrix[:, j])
                    
                    if (out_degree > self.params.min_connections_per_neuron and
                        in_degree > self.params.min_connections_per_neuron):
                        existing_connections.append((i, j))
        
        if not existing_connections:
            return None
        
        # Choose random connection to remove
        key, subkey = random.split(key)
        conn_idx = random.randint(subkey, (), 0, len(existing_connections))
        source, target = existing_connections[int(conn_idx)]
        
        return TopologyMutation(
            MutationType.REMOVE_CONNECTION,
            {
                'source': source,
                'target': target,
                'old_weight': float(topology.connection_weights[source, target]),
                'old_delay': float(topology.connection_delays[source, target])
            }
        )
    
    def _generate_modify_weight_mutation(
        self,
        topology: NetworkTopology,
        key: jax.random.PRNGKey
    ) -> Optional[TopologyMutation]:
        """Generate mutation to modify connection weight."""
        
        # Find existing connections
        existing_connections = []
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if topology.adjacency_matrix[i, j]:
                    existing_connections.append((i, j))
        
        if not existing_connections:
            return None
        
        # Choose random connection to modify
        key, subkey = random.split(key)
        conn_idx = random.randint(subkey, (), 0, len(existing_connections))
        source, target = existing_connections[int(conn_idx)]
        
        # Generate weight change
        key, subkey = random.split(key)
        old_weight = topology.connection_weights[source, target]
        weight_change = random.normal(subkey, ()) * self.params.weight_mutation_std
        
        # Preserve sign for excitatory/inhibitory constraint
        source_type = topology.neuron_parameters[source].connection_type
        if source_type == ConnectionType.EXCITATORY:
            new_weight = max(0.001, abs(old_weight + weight_change))  # Keep positive
        else:
            new_weight = min(-0.001, -abs(old_weight + weight_change))  # Keep negative
        
        return TopologyMutation(
            MutationType.MODIFY_WEIGHT,
            {
                'source': source,
                'target': target,
                'old_weight': float(old_weight),
                'new_weight': float(new_weight)
            }
        )
    
    def _generate_add_neuron_mutation(
        self,
        topology: NetworkTopology,
        key: jax.random.PRNGKey
    ) -> Optional[TopologyMutation]:
        """Generate mutation to add a new neuron."""
        
        if topology.n_neurons >= self.params.max_total_neurons:
            return None
        
        # Determine neuron type to maintain excitatory/inhibitory balance
        if self.params.maintain_excitatory_ratio:
            current_excitatory = sum(1 for params in topology.neuron_parameters.values()
                                   if params.connection_type == ConnectionType.EXCITATORY)
            current_ratio = current_excitatory / topology.n_neurons
            
            if current_ratio < self.params.target_excitatory_ratio:
                neuron_type = ConnectionType.EXCITATORY
            else:
                neuron_type = ConnectionType.INHIBITORY
        else:
            # Random choice
            key, subkey = random.split(key)
            neuron_type = ConnectionType.EXCITATORY if random.bernoulli(subkey, 0.8) else ConnectionType.INHIBITORY
        
        return TopologyMutation(
            MutationType.ADD_NEURON,
            {
                'neuron_type': neuron_type,
                'position': topology.n_neurons  # Add at end
            }
        )
    
    def _generate_remove_neuron_mutation(
        self,
        topology: NetworkTopology,
        key: jax.random.PRNGKey
    ) -> Optional[TopologyMutation]:
        """Generate mutation to remove a neuron."""
        
        if topology.n_neurons <= self.params.min_total_neurons:
            return None
        
        # Choose neuron to remove (avoid highly connected neurons)
        neuron_degrees = jnp.sum(topology.adjacency_matrix, axis=0) + jnp.sum(topology.adjacency_matrix, axis=1)
        
        # Prefer removing neurons with lower connectivity
        removal_probs = 1.0 / (neuron_degrees + 1.0)  # Add 1 to avoid division by zero
        removal_probs = removal_probs / jnp.sum(removal_probs)
        
        key, subkey = random.split(key)
        neuron_to_remove = random.categorical(subkey, jnp.log(removal_probs))
        
        return TopologyMutation(
            MutationType.REMOVE_NEURON,
            {
                'neuron_id': int(neuron_to_remove)
            }
        )
    
    def _generate_rewire_connection_mutation(
        self,
        topology: NetworkTopology,
        key: jax.random.PRNGKey
    ) -> Optional[TopologyMutation]:
        """Generate mutation to rewire an existing connection."""
        
        # Find existing connections
        existing_connections = []
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if topology.adjacency_matrix[i, j]:
                    existing_connections.append((i, j))
        
        if not existing_connections:
            return None
        
        # Choose connection to rewire
        key, subkey = random.split(key)
        conn_idx = random.randint(subkey, (), 0, len(existing_connections))
        old_source, old_target = existing_connections[int(conn_idx)]
        
        # Choose new target (keep same source)
        possible_targets = []
        for j in range(topology.n_neurons):
            if j != old_source and not topology.adjacency_matrix[old_source, j]:
                possible_targets.append(j)
        
        if not possible_targets:
            return None
        
        key, subkey = random.split(key)
        new_target_idx = random.randint(subkey, (), 0, len(possible_targets))
        new_target = possible_targets[int(new_target_idx)]
        
        return TopologyMutation(
            MutationType.REWIRE_CONNECTION,
            {
                'source': old_source,
                'old_target': old_target,
                'new_target': new_target,
                'weight': float(topology.connection_weights[old_source, old_target]),
                'delay': float(topology.connection_delays[old_source, old_target])
            }
        )
    
    def _generate_modify_delay_mutation(
        self,
        topology: NetworkTopology,
        key: jax.random.PRNGKey
    ) -> Optional[TopologyMutation]:
        """Generate mutation to modify connection delay."""
        
        # Find existing connections
        existing_connections = []
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if topology.adjacency_matrix[i, j]:
                    existing_connections.append((i, j))
        
        if not existing_connections:
            return None
        
        # Choose random connection
        key, subkey = random.split(key)
        conn_idx = random.randint(subkey, (), 0, len(existing_connections))
        source, target = existing_connections[int(conn_idx)]
        
        # Generate delay change
        key, subkey = random.split(key)
        old_delay = topology.connection_delays[source, target]
        delay_change = random.normal(subkey, ()) * self.params.delay_mutation_std
        new_delay = jnp.clip(old_delay + delay_change, 1e-3, 10e-3)  # 1-10ms range
        
        return TopologyMutation(
            MutationType.MODIFY_DELAY,
            {
                'source': source,
                'target': target,
                'old_delay': float(old_delay),
                'new_delay': float(new_delay)
            }
        )
    
    def _generate_prune_weak_mutation(
        self,
        topology: NetworkTopology,
        key: jax.random.PRNGKey
    ) -> Optional[TopologyMutation]:
        """Generate mutation to prune weak connections."""
        
        # Find weak connections
        weak_connections = []
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if (topology.adjacency_matrix[i, j] and 
                    abs(topology.connection_weights[i, j]) < self.params.weak_connection_threshold):
                    weak_connections.append((i, j))
        
        if not weak_connections:
            return None
        
        return TopologyMutation(
            MutationType.PRUNE_WEAK_CONNECTIONS,
            {
                'connections_to_prune': weak_connections
            }
        )
    
    def _generate_duplicate_neuron_mutation(
        self,
        topology: NetworkTopology,
        key: jax.random.PRNGKey
    ) -> Optional[TopologyMutation]:
        """Generate mutation to duplicate a neuron."""
        
        if topology.n_neurons >= self.params.max_total_neurons:
            return None
        
        # Choose neuron to duplicate (prefer highly connected ones)
        neuron_degrees = jnp.sum(topology.adjacency_matrix, axis=0) + jnp.sum(topology.adjacency_matrix, axis=1)
        
        if jnp.sum(neuron_degrees) == 0:
            return None
        
        duplication_probs = neuron_degrees / jnp.sum(neuron_degrees)
        
        key, subkey = random.split(key)
        neuron_to_duplicate = random.categorical(subkey, jnp.log(duplication_probs + 1e-10))
        
        return TopologyMutation(
            MutationType.DUPLICATE_NEURON,
            {
                'original_neuron': int(neuron_to_duplicate),
                'new_position': topology.n_neurons
            }
        )
    
    def _generate_merge_neurons_mutation(
        self,
        topology: NetworkTopology,
        key: jax.random.PRNGKey
    ) -> Optional[TopologyMutation]:
        """Generate mutation to merge two neurons."""
        
        if topology.n_neurons <= self.params.min_total_neurons:
            return None
        
        # Choose two neurons to merge
        key, subkey1, subkey2 = random.split(key, 3)
        neuron1 = random.randint(subkey1, (), 0, topology.n_neurons)
        neuron2 = random.randint(subkey2, (), 0, topology.n_neurons)
        
        if neuron1 == neuron2:
            return None
        
        return TopologyMutation(
            MutationType.MERGE_NEURONS,
            {
                'neuron1': int(neuron1),
                'neuron2': int(neuron2),
                'keep_neuron': int(neuron1)  # Keep the first neuron
            }
        )
    
    def apply_mutation(
        self,
        topology: NetworkTopology,
        mutation: TopologyMutation
    ) -> NetworkTopology:
        """
        Apply a mutation to the network topology.
        
        Args:
            topology: Current topology
            mutation: Mutation to apply
            
        Returns:
            Modified topology
        """
        if mutation.mutation_type == MutationType.ADD_CONNECTION:
            return self._apply_add_connection(topology, mutation)
        elif mutation.mutation_type == MutationType.REMOVE_CONNECTION:
            return self._apply_remove_connection(topology, mutation)
        elif mutation.mutation_type == MutationType.MODIFY_WEIGHT:
            return self._apply_modify_weight(topology, mutation)
        elif mutation.mutation_type == MutationType.ADD_NEURON:
            return self._apply_add_neuron(topology, mutation)
        elif mutation.mutation_type == MutationType.REMOVE_NEURON:
            return self._apply_remove_neuron(topology, mutation)
        elif mutation.mutation_type == MutationType.REWIRE_CONNECTION:
            return self._apply_rewire_connection(topology, mutation)
        elif mutation.mutation_type == MutationType.MODIFY_DELAY:
            return self._apply_modify_delay(topology, mutation)
        elif mutation.mutation_type == MutationType.PRUNE_WEAK_CONNECTIONS:
            return self._apply_prune_weak(topology, mutation)
        elif mutation.mutation_type == MutationType.DUPLICATE_NEURON:
            return self._apply_duplicate_neuron(topology, mutation)
        elif mutation.mutation_type == MutationType.MERGE_NEURONS:
            return self._apply_merge_neurons(topology, mutation)
        
        return topology
    
    def _apply_add_connection(self, topology: NetworkTopology, mutation: TopologyMutation) -> NetworkTopology:
        """Apply add connection mutation."""
        params = mutation.parameters
        source, target = params['source'], params['target']
        weight, delay = params['weight'], params['delay']
        
        # Update adjacency matrix
        new_adjacency = topology.adjacency_matrix.at[source, target].set(True)
        
        # Update weights and delays
        new_weights = topology.connection_weights.at[source, target].set(weight)
        new_delays = topology.connection_delays.at[source, target].set(delay)
        
        # Update connection parameters
        new_conn_params = topology.connection_parameters.copy()
        source_type = topology.neuron_parameters[source].connection_type
        new_conn_params[(source, target)] = ConnectionParams(
            weight=weight,
            delay=delay,
            connection_type=source_type
        )
        
        # Update metrics
        new_n_connections = topology.n_connections + 1
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
    
    def _apply_remove_connection(self, topology: NetworkTopology, mutation: TopologyMutation) -> NetworkTopology:
        """Apply remove connection mutation."""
        params = mutation.parameters
        source, target = params['source'], params['target']
        
        # Update adjacency matrix
        new_adjacency = topology.adjacency_matrix.at[source, target].set(False)
        
        # Update weights and delays
        new_weights = topology.connection_weights.at[source, target].set(0.0)
        new_delays = topology.connection_delays.at[source, target].set(0.0)
        
        # Update connection parameters
        new_conn_params = topology.connection_parameters.copy()
        if (source, target) in new_conn_params:
            del new_conn_params[(source, target)]
        
        # Update metrics
        new_n_connections = topology.n_connections - 1
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
    
    def _apply_modify_weight(self, topology: NetworkTopology, mutation: TopologyMutation) -> NetworkTopology:
        """Apply modify weight mutation."""
        params = mutation.parameters
        source, target = params['source'], params['target']
        new_weight = params['new_weight']
        
        # Update weights
        new_weights = topology.connection_weights.at[source, target].set(new_weight)
        
        # Update connection parameters
        new_conn_params = topology.connection_parameters.copy()
        if (source, target) in new_conn_params:
            old_params = new_conn_params[(source, target)]
            new_conn_params[(source, target)] = ConnectionParams(
                weight=new_weight,
                delay=old_params.delay,
                connection_type=old_params.connection_type
            )
        
        # Update spectral radius
        new_spectral_radius = self.topology_manager._compute_spectral_radius(new_weights)
        
        return topology._replace(
            connection_weights=new_weights,
            connection_parameters=new_conn_params,
            spectral_radius=new_spectral_radius
        )
    
    def _apply_modify_delay(self, topology: NetworkTopology, mutation: TopologyMutation) -> NetworkTopology:
        """Apply modify delay mutation."""
        params = mutation.parameters
        source, target = params['source'], params['target']
        new_delay = params['new_delay']
        
        # Update delays
        new_delays = topology.connection_delays.at[source, target].set(new_delay)
        
        # Update connection parameters
        new_conn_params = topology.connection_parameters.copy()
        if (source, target) in new_conn_params:
            old_params = new_conn_params[(source, target)]
            new_conn_params[(source, target)] = ConnectionParams(
                weight=old_params.weight,
                delay=new_delay,
                connection_type=old_params.connection_type
            )
        
        return topology._replace(
            connection_delays=new_delays,
            connection_parameters=new_conn_params
        )
    
    def _apply_add_neuron(self, topology: NetworkTopology, mutation: TopologyMutation) -> NetworkTopology:
        """Apply add neuron mutation (simplified - would need full matrix expansion)."""
        # This is a simplified version - full implementation would require
        # expanding all matrices and updating indices
        params = mutation.parameters
        neuron_type = params['neuron_type']
        
        # For now, return original topology
        # Full implementation would expand matrices and add neuron
        return topology
    
    def _apply_remove_neuron(self, topology: NetworkTopology, mutation: TopologyMutation) -> NetworkTopology:
        """Apply remove neuron mutation (simplified)."""
        # This is a simplified version - full implementation would require
        # matrix contraction and index remapping
        return topology
    
    def _apply_rewire_connection(self, topology: NetworkTopology, mutation: TopologyMutation) -> NetworkTopology:
        """Apply rewire connection mutation."""
        params = mutation.parameters
        source = params['source']
        old_target = params['old_target']
        new_target = params['new_target']
        weight = params['weight']
        delay = params['delay']
        
        # Remove old connection
        new_adjacency = topology.adjacency_matrix.at[source, old_target].set(False)
        new_weights = topology.connection_weights.at[source, old_target].set(0.0)
        new_delays = topology.connection_delays.at[source, old_target].set(0.0)
        
        # Add new connection
        new_adjacency = new_adjacency.at[source, new_target].set(True)
        new_weights = new_weights.at[source, new_target].set(weight)
        new_delays = new_delays.at[source, new_target].set(delay)
        
        # Update connection parameters
        new_conn_params = topology.connection_parameters.copy()
        if (source, old_target) in new_conn_params:
            del new_conn_params[(source, old_target)]
        
        source_type = topology.neuron_parameters[source].connection_type
        new_conn_params[(source, new_target)] = ConnectionParams(
            weight=weight,
            delay=delay,
            connection_type=source_type
        )
        
        # Update metrics
        new_spectral_radius = self.topology_manager._compute_spectral_radius(new_weights)
        new_modularity = self.topology_manager._compute_modularity(new_adjacency)
        
        return topology._replace(
            adjacency_matrix=new_adjacency,
            connection_weights=new_weights,
            connection_delays=new_delays,
            connection_parameters=new_conn_params,
            spectral_radius=new_spectral_radius,
            modularity=new_modularity
        )
    
    def _apply_prune_weak(self, topology: NetworkTopology, mutation: TopologyMutation) -> NetworkTopology:
        """Apply prune weak connections mutation."""
        params = mutation.parameters
        connections_to_prune = params['connections_to_prune']
        
        new_adjacency = topology.adjacency_matrix
        new_weights = topology.connection_weights
        new_delays = topology.connection_delays
        new_conn_params = topology.connection_parameters.copy()
        
        # Remove each weak connection
        for source, target in connections_to_prune:
            new_adjacency = new_adjacency.at[source, target].set(False)
            new_weights = new_weights.at[source, target].set(0.0)
            new_delays = new_delays.at[source, target].set(0.0)
            
            if (source, target) in new_conn_params:
                del new_conn_params[(source, target)]
        
        # Update metrics
        new_n_connections = topology.n_connections - len(connections_to_prune)
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
    
    def _apply_duplicate_neuron(self, topology: NetworkTopology, mutation: TopologyMutation) -> NetworkTopology:
        """Apply duplicate neuron mutation (simplified)."""
        # Simplified version - full implementation would expand matrices
        return topology
    
    def _apply_merge_neurons(self, topology: NetworkTopology, mutation: TopologyMutation) -> NetworkTopology:
        """Apply merge neurons mutation (simplified)."""
        # Simplified version - full implementation would contract matrices
        return topology
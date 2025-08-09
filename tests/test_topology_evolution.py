"""
Tests for Topology Evolution System

This module contains comprehensive tests for the topology evolution components
including network topology representation, mutation operators, evolution engine,
performance selection, and synaptic pruning/growth mechanisms.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import List, Dict, Any

from src.core.topology.network_topology import (
    NetworkTopology, TopologyManager, NeuronParams, ConnectionParams, ConnectionType
)
from src.core.topology.mutation_operators import (
    MutationOperator, TopologyMutation, MutationParams, MutationType
)
from src.training.evolution.topology_evolution import (
    TopologyEvolution, EvolutionParams, EvolutionState
)
from src.training.evolution.performance_selection import (
    PerformanceSelector, SelectionParams, SelectionMethod, PerformanceEvaluator, PerformanceMetrics
)
from src.training.evolution.synaptic_pruning import (
    SynapticPruner, SynapticGrower, SynapticPlasticityManager,
    PruningParams, GrowthParams, SynapticState, PruningStrategy, GrowthStrategy
)


class TestNetworkTopology:
    """Test network topology representation and management."""
    
    def test_topology_manager_initialization(self):
        """Test TopologyManager initialization."""
        manager = TopologyManager()
        assert manager.max_neurons == 10000
        assert manager.max_connections_per_neuron == 1000
    
    def test_create_random_topology(self):
        """Test random topology creation."""
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        topology = manager.create_random_topology(
            n_neurons=100,
            connectivity=0.1,
            excitatory_ratio=0.8,
            key=key
        )
        
        # Check basic properties
        assert topology.n_neurons == 100
        assert topology.adjacency_matrix.shape == (100, 100)
        assert topology.connection_weights.shape == (100, 100)
        assert topology.connection_delays.shape == (100, 100)
        
        # Check no self-connections
        assert not jnp.any(jnp.diag(topology.adjacency_matrix))
        
        # Check excitatory/inhibitory structure
        n_excitatory = int(0.8 * 100)
        excitatory_weights = topology.connection_weights[:n_excitatory, :]
        inhibitory_weights = topology.connection_weights[n_excitatory:, :]
        
        # Excitatory neurons should have non-negative outgoing weights
        excitatory_nonzero = excitatory_weights[topology.adjacency_matrix[:n_excitatory, :]]
        if len(excitatory_nonzero) > 0:
            assert jnp.all(excitatory_nonzero >= 0)
        
        # Inhibitory neurons should have non-positive outgoing weights
        inhibitory_nonzero = inhibitory_weights[topology.adjacency_matrix[n_excitatory:, :]]
        if len(inhibitory_nonzero) > 0:
            assert jnp.all(inhibitory_nonzero <= 0)
        
        # Check connectivity is approximately correct
        actual_connectivity = topology.n_connections / (100 * 99)
        assert abs(actual_connectivity - 0.1) < 0.05  # Allow some variance
    
    def test_create_small_world_topology(self):
        """Test small-world topology creation."""
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        topology = manager.create_small_world_topology(
            n_neurons=50,
            k_neighbors=6,
            rewiring_prob=0.1,
            key=key
        )
        
        assert topology.n_neurons == 50
        assert topology.adjacency_matrix.shape == (50, 50)
        assert topology.n_connections > 0
        assert not jnp.any(jnp.diag(topology.adjacency_matrix))
    
    def test_topology_validation(self):
        """Test topology validation."""
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        # Create valid topology
        valid_topology = manager.create_random_topology(n_neurons=50, key=key)
        assert manager.validate_topology(valid_topology)
        
        # Test invalid topology (self-connections)
        invalid_adjacency = valid_topology.adjacency_matrix.at[0, 0].set(True)
        invalid_topology = valid_topology._replace(adjacency_matrix=invalid_adjacency)
        assert not manager.validate_topology(invalid_topology)
    
    def test_topology_analysis(self):
        """Test topology analysis."""
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        topology = manager.create_random_topology(n_neurons=50, key=key)
        analysis = manager.analyze_topology(topology)
        
        # Check required fields
        required_fields = [
            'n_neurons', 'n_connections', 'connectivity', 'spectral_radius',
            'modularity', 'mean_weight', 'mean_in_degree', 'mean_out_degree',
            'excitatory_ratio', 'current_fitness'
        ]
        
        for field in required_fields:
            assert field in analysis
            assert isinstance(analysis[field], (int, float))
    
    def test_performance_update(self):
        """Test performance tracking update."""
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        topology = manager.create_random_topology(n_neurons=50, key=key)
        
        # Update performance
        updated_topology = manager.update_performance(topology, 0.8)
        
        assert updated_topology.fitness_score > topology.fitness_score
        assert jnp.any(updated_topology.performance_history != topology.performance_history)


class TestMutationOperators:
    """Test mutation operators for topology evolution."""
    
    def test_mutation_operator_initialization(self):
        """Test MutationOperator initialization."""
        params = MutationParams()
        operator = MutationOperator(params)
        
        assert operator.params == params
        assert operator.topology_manager is not None
    
    def test_generate_mutations(self):
        """Test mutation generation."""
        manager = TopologyManager()
        operator = MutationOperator()
        key = random.PRNGKey(42)
        
        topology = manager.create_random_topology(n_neurons=50, key=key)
        
        mutations = operator.generate_mutations(topology, n_mutations=5, key=key)
        
        assert len(mutations) <= 5  # Some mutations might fail to generate
        for mutation in mutations:
            assert isinstance(mutation, TopologyMutation)
            assert mutation.mutation_type in MutationType
            assert isinstance(mutation.parameters, dict)
    
    def test_add_connection_mutation(self):
        """Test add connection mutation."""
        manager = TopologyManager()
        operator = MutationOperator()
        key = random.PRNGKey(42)
        
        # Create sparse topology to ensure space for new connections
        topology = manager.create_random_topology(n_neurons=20, connectivity=0.05, key=key)
        
        mutation = operator._generate_add_connection_mutation(topology, key)
        
        if mutation is not None:  # Mutation might not be possible
            assert mutation.mutation_type == MutationType.ADD_CONNECTION
            assert 'source' in mutation.parameters
            assert 'target' in mutation.parameters
            assert 'weight' in mutation.parameters
            assert 'delay' in mutation.parameters
            
            # Apply mutation
            new_topology = operator.apply_mutation(topology, mutation)
            assert new_topology.n_connections >= topology.n_connections
    
    def test_remove_connection_mutation(self):
        """Test remove connection mutation."""
        manager = TopologyManager()
        operator = MutationOperator()
        key = random.PRNGKey(42)
        
        topology = manager.create_random_topology(n_neurons=50, connectivity=0.2, key=key)
        
        mutation = operator._generate_remove_connection_mutation(topology, key)
        
        if mutation is not None:
            assert mutation.mutation_type == MutationType.REMOVE_CONNECTION
            assert 'source' in mutation.parameters
            assert 'target' in mutation.parameters
            
            # Apply mutation
            new_topology = operator.apply_mutation(topology, mutation)
            assert new_topology.n_connections <= topology.n_connections
    
    def test_modify_weight_mutation(self):
        """Test weight modification mutation."""
        manager = TopologyManager()
        operator = MutationOperator()
        key = random.PRNGKey(42)
        
        topology = manager.create_random_topology(n_neurons=50, key=key)
        
        mutation = operator._generate_modify_weight_mutation(topology, key)
        
        if mutation is not None:
            assert mutation.mutation_type == MutationType.MODIFY_WEIGHT
            assert 'source' in mutation.parameters
            assert 'target' in mutation.parameters
            assert 'old_weight' in mutation.parameters
            assert 'new_weight' in mutation.parameters
            
            # Apply mutation
            new_topology = operator.apply_mutation(topology, mutation)
            source, target = mutation.parameters['source'], mutation.parameters['target']
            
            # Check weight was changed
            assert new_topology.connection_weights[source, target] != topology.connection_weights[source, target]
    
    def test_rewire_connection_mutation(self):
        """Test connection rewiring mutation."""
        manager = TopologyManager()
        operator = MutationOperator()
        key = random.PRNGKey(42)
        
        topology = manager.create_random_topology(n_neurons=50, connectivity=0.1, key=key)
        
        mutation = operator._generate_rewire_connection_mutation(topology, key)
        
        if mutation is not None:
            assert mutation.mutation_type == MutationType.REWIRE_CONNECTION
            assert 'source' in mutation.parameters
            assert 'old_target' in mutation.parameters
            assert 'new_target' in mutation.parameters
            
            # Apply mutation
            new_topology = operator.apply_mutation(topology, mutation)
            
            # Check connection count unchanged
            assert new_topology.n_connections == topology.n_connections


class TestTopologyEvolution:
    """Test topology evolution engine."""
    
    def test_evolution_initialization(self):
        """Test TopologyEvolution initialization."""
        evolution = TopologyEvolution()
        
        assert evolution.evolution_params is not None
        assert evolution.mutation_params is not None
        assert evolution.fitness_function is not None
        assert evolution.state.generation == 0
    
    def test_population_initialization(self):
        """Test population initialization."""
        evolution = TopologyEvolution()
        key = random.PRNGKey(42)
        
        evolution.initialize_population(key=key)
        
        assert len(evolution.state.population) == evolution.evolution_params.population_size
        assert evolution.state.best_topology is not None
        assert evolution.state.best_fitness > -jnp.inf
        
        # Check all topologies are valid
        for topology in evolution.state.population:
            assert evolution.topology_manager.validate_topology(topology)
    
    def test_single_generation_evolution(self):
        """Test single generation evolution."""
        # Use smaller parameters for faster testing
        evolution_params = EvolutionParams(population_size=10, elite_size=2)
        evolution = TopologyEvolution(evolution_params)
        key = random.PRNGKey(42)
        
        evolution.initialize_population(key=key)
        initial_generation = evolution.state.generation
        initial_best_fitness = evolution.state.best_fitness
        
        stats = evolution.evolve_generation(key)
        
        assert evolution.state.generation == initial_generation + 1
        assert 'generation' in stats
        assert 'best_fitness' in stats
        assert 'mean_fitness' in stats
        assert len(evolution.state.population) == evolution_params.population_size
    
    def test_multi_generation_evolution(self):
        """Test multi-generation evolution."""
        evolution_params = EvolutionParams(
            population_size=8, 
            elite_size=2, 
            max_generations=5
        )
        evolution = TopologyEvolution(evolution_params)
        key = random.PRNGKey(42)
        
        final_stats = evolution.evolve(max_generations=5, key=key)
        
        assert evolution.state.generation >= 5
        assert 'total_evolution_time' in final_stats
        assert 'total_generations' in final_stats
        assert evolution.state.best_topology is not None
    
    def test_fitness_function(self):
        """Test default fitness function."""
        evolution = TopologyEvolution()
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        topology = manager.create_random_topology(n_neurons=100, key=key)
        fitness = evolution.fitness_function(topology)
        
        assert isinstance(fitness, float)
        assert not jnp.isnan(fitness)
        assert not jnp.isinf(fitness)
    
    def test_custom_fitness_function(self):
        """Test custom fitness function."""
        def custom_fitness(topology: NetworkTopology) -> float:
            return float(topology.n_connections / topology.n_neurons)
        
        evolution = TopologyEvolution(fitness_function=custom_fitness)
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        topology = manager.create_random_topology(n_neurons=50, key=key)
        fitness = evolution.fitness_function(topology)
        
        expected_fitness = topology.n_connections / topology.n_neurons
        assert abs(fitness - expected_fitness) < 1e-6


class TestPerformanceSelection:
    """Test performance-based selection mechanisms."""
    
    def test_performance_evaluator(self):
        """Test performance evaluation."""
        evaluator = PerformanceEvaluator()
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        topology = manager.create_random_topology(n_neurons=50, key=key)
        metrics = evaluator.evaluate_topology(topology)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert isinstance(metrics.fitness, float)
        assert isinstance(metrics.connectivity, float)
        assert isinstance(metrics.spectral_radius, float)
        assert isinstance(metrics.modularity, float)
        assert isinstance(metrics.efficiency, float)
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        params = SelectionParams(method=SelectionMethod.TOURNAMENT, tournament_size=3)
        selector = PerformanceSelector(params)
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        # Create population
        population = []
        for i in range(10):
            key, subkey = random.split(key)
            topology = manager.create_random_topology(n_neurons=30, key=subkey)
            population.append(topology)
        
        # Select individuals
        selected = selector.select(population, n_select=5, key=key)
        
        assert len(selected) == 5
        assert all(topology in population for topology in selected)
    
    def test_roulette_wheel_selection(self):
        """Test roulette wheel selection."""
        params = SelectionParams(method=SelectionMethod.ROULETTE_WHEEL)
        selector = PerformanceSelector(params)
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        # Create population with varying fitness
        population = []
        for i in range(8):
            key, subkey = random.split(key)
            topology = manager.create_random_topology(n_neurons=25, key=subkey)
            # Manually set different fitness scores
            topology = manager.update_performance(topology, i * 0.1)
            population.append(topology)
        
        selected = selector.select(population, n_select=4, key=key)
        
        assert len(selected) == 4
        assert all(topology in population for topology in selected)
    
    def test_pareto_front_selection(self):
        """Test Pareto front selection."""
        params = SelectionParams(
            method=SelectionMethod.PARETO_FRONT,
            objectives=['fitness', 'connectivity', 'modularity']
        )
        selector = PerformanceSelector(params)
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        # Create population
        population = []
        for i in range(6):
            key, subkey = random.split(key)
            topology = manager.create_random_topology(n_neurons=20, key=subkey)
            population.append(topology)
        
        selected = selector.select(population, n_select=3, key=key)
        
        assert len(selected) == 3
        assert all(topology in population for topology in selected)


class TestSynapticPruning:
    """Test synaptic pruning and growth mechanisms."""
    
    def test_synaptic_pruner_initialization(self):
        """Test SynapticPruner initialization."""
        params = PruningParams(strategy=PruningStrategy.MAGNITUDE_BASED)
        pruner = SynapticPruner(params)
        
        assert pruner.params.strategy == PruningStrategy.MAGNITUDE_BASED
        assert pruner.topology_manager is not None
    
    def test_magnitude_based_pruning(self):
        """Test magnitude-based pruning."""
        params = PruningParams(
            strategy=PruningStrategy.MAGNITUDE_BASED,
            magnitude_threshold=0.05,
            min_connections_per_neuron=1
        )
        pruner = SynapticPruner(params)
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        # Create topology with some weak connections
        topology = manager.create_random_topology(n_neurons=30, connectivity=0.2, key=key)
        synaptic_state = SynapticState()
        
        # Apply pruning
        pruned_topology, updated_state = pruner.prune_connections(
            topology, synaptic_state, key=key
        )
        
        # Should have fewer or equal connections
        assert pruned_topology.n_connections <= topology.n_connections
        assert manager.validate_topology(pruned_topology)
    
    def test_activity_based_pruning(self):
        """Test activity-based pruning."""
        params = PruningParams(
            strategy=PruningStrategy.ACTIVITY_BASED,
            activity_threshold=0.01,
            activity_window=100
        )
        pruner = SynapticPruner(params)
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        topology = manager.create_random_topology(n_neurons=25, key=key)
        synaptic_state = SynapticState()
        
        # Add some activity history
        for i in range(topology.n_neurons):
            for j in range(topology.n_neurons):
                if topology.adjacency_matrix[i, j]:
                    connection = (i, j)
                    # Some connections have low activity
                    activity_level = 0.005 if (i + j) % 3 == 0 else 0.02
                    synaptic_state.activity_history[connection] = [activity_level] * 100
        
        pruned_topology, updated_state = pruner.prune_connections(
            topology, synaptic_state, current_time=1000.0, key=key
        )
        
        assert pruned_topology.n_connections <= topology.n_connections
        assert manager.validate_topology(pruned_topology)
    
    def test_synaptic_grower_initialization(self):
        """Test SynapticGrower initialization."""
        params = GrowthParams(strategy=GrowthStrategy.RANDOM_GROWTH)
        grower = SynapticGrower(params)
        
        assert grower.params.strategy == GrowthStrategy.RANDOM_GROWTH
        assert grower.topology_manager is not None
    
    def test_random_growth(self):
        """Test random synaptic growth."""
        params = GrowthParams(
            strategy=GrowthStrategy.RANDOM_GROWTH,
            growth_rate=0.1
        )
        grower = SynapticGrower(params)
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        # Create sparse topology with room for growth
        topology = manager.create_random_topology(n_neurons=20, connectivity=0.05, key=key)
        synaptic_state = SynapticState()
        
        grown_topology, updated_state = grower.grow_connections(
            topology, synaptic_state, current_time=0.0, key=key
        )
        
        # Should have more or equal connections
        assert grown_topology.n_connections >= topology.n_connections
        assert manager.validate_topology(grown_topology)
    
    def test_activity_dependent_growth(self):
        """Test activity-dependent growth."""
        params = GrowthParams(
            strategy=GrowthStrategy.ACTIVITY_DEPENDENT,
            activity_threshold=0.1,
            growth_rate=0.05
        )
        grower = SynapticGrower(params)
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        topology = manager.create_random_topology(n_neurons=15, connectivity=0.1, key=key)
        synaptic_state = SynapticState()
        
        # Set some neurons as active
        firing_rates = jnp.zeros(topology.n_neurons)
        firing_rates = firing_rates.at[:5].set(0.15)  # First 5 neurons are active
        synaptic_state.neuron_firing_rates = firing_rates
        
        grown_topology, updated_state = grower.grow_connections(
            topology, synaptic_state, current_time=0.0, key=key
        )
        
        assert grown_topology.n_connections >= topology.n_connections
        assert manager.validate_topology(grown_topology)
    
    def test_synaptic_plasticity_manager(self):
        """Test integrated synaptic plasticity management."""
        pruning_params = PruningParams(strategy=PruningStrategy.MAGNITUDE_BASED)
        growth_params = GrowthParams(strategy=GrowthStrategy.RANDOM_GROWTH)
        
        manager = SynapticPlasticityManager(pruning_params, growth_params)
        topology_manager = TopologyManager()
        key = random.PRNGKey(42)
        
        topology = topology_manager.create_random_topology(n_neurons=30, key=key)
        initial_connections = topology.n_connections
        
        # Update synaptic connections
        updated_topology = manager.update_synaptic_connections(
            topology, current_time=100.0, key=key
        )
        
        assert topology_manager.validate_topology(updated_topology)
        # Connections might increase, decrease, or stay the same
        assert updated_topology.n_connections >= 0
    
    def test_plasticity_statistics(self):
        """Test plasticity statistics collection."""
        manager = SynapticPlasticityManager()
        
        # Add some dummy data
        manager.synaptic_state.activity_history[(0, 1)] = [0.1, 0.2, 0.05]
        manager.synaptic_state.activity_history[(1, 2)] = [0.01, 0.001, 0.0]
        manager.synaptic_state.connection_ages[(0, 1)] = 100
        manager.synaptic_state.connection_ages[(1, 2)] = 50
        
        stats = manager.get_plasticity_statistics()
        
        assert 'total_tracked_connections' in stats
        assert 'active_connections' in stats
        assert 'average_connection_age' in stats
        assert 'pruning_strategy' in stats
        assert 'growth_strategy' in stats
        
        assert stats['total_tracked_connections'] == 2
        assert stats['average_connection_age'] == 75.0


class TestIntegration:
    """Integration tests for the complete topology evolution system."""
    
    def test_complete_evolution_pipeline(self):
        """Test complete evolution pipeline with all components."""
        # Setup evolution with small parameters for testing
        evolution_params = EvolutionParams(
            population_size=6,
            elite_size=2,
            max_generations=3,
            mutation_rate=0.2
        )
        
        mutation_params = MutationParams(
            add_connection_prob=0.3,
            remove_connection_prob=0.2,
            modify_weight_prob=0.4
        )
        
        def test_fitness_function(topology: NetworkTopology) -> float:
            # Simple fitness based on connectivity and spectral radius
            target_connectivity = 0.1
            actual_connectivity = topology.n_connections / (topology.n_neurons * (topology.n_neurons - 1))
            connectivity_score = 1.0 - abs(actual_connectivity - target_connectivity) * 10
            
            spectral_score = 1.0 - abs(topology.spectral_radius - 0.95)
            
            return max(0.0, 0.7 * connectivity_score + 0.3 * spectral_score)
        
        evolution = TopologyEvolution(
            evolution_params=evolution_params,
            mutation_params=mutation_params,
            fitness_function=test_fitness_function
        )
        
        key = random.PRNGKey(42)
        
        # Run evolution
        final_stats = evolution.evolve(max_generations=3, key=key)
        
        # Check results
        assert evolution.state.generation >= 3
        assert evolution.state.best_topology is not None
        assert evolution.state.best_fitness >= 0.0
        assert len(evolution.state.population) == evolution_params.population_size
        
        # Verify best topology is valid
        best_topology = evolution.get_best_topology()
        assert evolution.topology_manager.validate_topology(best_topology)
        
        # Check evolution made progress
        assert len(evolution.state.fitness_history) >= 3
    
    def test_evolution_with_plasticity(self):
        """Test evolution combined with synaptic plasticity."""
        # Create initial topology
        manager = TopologyManager()
        key = random.PRNGKey(42)
        
        initial_topology = manager.create_random_topology(n_neurons=40, key=key)
        
        # Setup plasticity manager
        plasticity_manager = SynapticPlasticityManager()
        
        # Setup evolution
        evolution_params = EvolutionParams(population_size=4, max_generations=2)
        evolution = TopologyEvolution(evolution_params)
        
        # Initialize with plasticity-modified topology
        key, subkey = random.split(key)
        modified_topology = plasticity_manager.update_synaptic_connections(
            initial_topology, key=subkey
        )
        
        evolution.initialize_population(base_topology=modified_topology, key=key)
        
        # Run short evolution
        final_stats = evolution.evolve(max_generations=2, key=key)
        
        assert evolution.state.best_topology is not None
        assert manager.validate_topology(evolution.state.best_topology)
    
    def test_performance_tracking(self):
        """Test performance tracking throughout evolution."""
        evolution_params = EvolutionParams(population_size=4, max_generations=3)
        evolution = TopologyEvolution(evolution_params)
        
        # Track generation statistics
        generation_stats = []
        
        def generation_callback(generation: int, stats: Dict[str, Any]):
            generation_stats.append(stats.copy())
        
        evolution.add_generation_callback(generation_callback)
        
        key = random.PRNGKey(42)
        evolution.evolve(max_generations=3, key=key)
        
        # Check statistics were collected
        assert len(generation_stats) >= 3
        
        for stats in generation_stats:
            assert 'generation' in stats
            assert 'best_fitness' in stats
            assert 'mean_fitness' in stats
            assert 'population_size' in stats
            assert stats['population_size'] == evolution_params.population_size


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
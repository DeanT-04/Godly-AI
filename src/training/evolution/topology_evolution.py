"""
Topology Evolution System

This module implements the core topology evolution engine for the Godly AI system.
It provides performance-based selection of architectural changes and manages
the evolutionary process for network optimization.
"""

from typing import List, Dict, Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
import numpy as np
from collections import deque
import time

from ...core.topology.network_topology import NetworkTopology, TopologyManager
from ...core.topology.mutation_operators import (
    MutationOperator, TopologyMutation, MutationParams, MutationType
)


@dataclass
class EvolutionParams:
    """Parameters for topology evolution."""
    
    # Population parameters
    population_size: int = 20           # Number of topology variants to maintain
    elite_size: int = 5                # Number of best topologies to preserve
    
    # Selection parameters
    tournament_size: int = 3            # Size of tournament selection
    selection_pressure: float = 1.5     # Strength of selection pressure
    
    # Mutation parameters
    mutation_rate: float = 0.1          # Probability of mutation per individual
    mutations_per_individual: int = 3   # Number of mutations to apply
    
    # Evolution control
    max_generations: int = 1000         # Maximum number of generations
    convergence_threshold: float = 1e-6 # Fitness improvement threshold for convergence
    stagnation_limit: int = 50          # Generations without improvement before restart
    
    # Performance evaluation
    evaluation_episodes: int = 10       # Number of episodes for fitness evaluation
    fitness_averaging_window: int = 5   # Window for fitness averaging
    
    # Diversity maintenance
    diversity_weight: float = 0.1       # Weight for diversity in fitness calculation
    min_diversity_threshold: float = 0.05  # Minimum diversity to maintain
    
    # Adaptive parameters
    adaptive_mutation_rate: bool = True  # Whether to adapt mutation rate
    mutation_rate_decay: float = 0.99   # Decay factor for mutation rate
    min_mutation_rate: float = 0.01     # Minimum mutation rate


@dataclass
class EvolutionState:
    """State of the evolution process."""
    
    generation: int = 0
    population: List[NetworkTopology] = None
    fitness_history: List[float] = None
    best_fitness: float = -jnp.inf
    best_topology: Optional[NetworkTopology] = None
    stagnation_counter: int = 0
    current_mutation_rate: float = 0.1
    diversity_scores: List[float] = None
    
    def __post_init__(self):
        if self.population is None:
            self.population = []
        if self.fitness_history is None:
            self.fitness_history = []
        if self.diversity_scores is None:
            self.diversity_scores = []


class TopologyEvolution:
    """
    Main topology evolution engine.
    
    This class implements a genetic algorithm for evolving network topologies
    based on performance feedback. It maintains a population of topologies,
    applies mutations, and selects the best performers for the next generation.
    """
    
    def __init__(
        self,
        evolution_params: Optional[EvolutionParams] = None,
        mutation_params: Optional[MutationParams] = None,
        fitness_function: Optional[Callable[[NetworkTopology], float]] = None
    ):
        """
        Initialize topology evolution system.
        
        Args:
            evolution_params: Parameters controlling evolution process
            mutation_params: Parameters controlling mutations
            fitness_function: Function to evaluate topology fitness
        """
        self.evolution_params = evolution_params or EvolutionParams()
        self.mutation_params = mutation_params or MutationParams()
        self.fitness_function = fitness_function or self._default_fitness_function
        
        # Initialize components
        self.topology_manager = TopologyManager()
        self.mutation_operator = MutationOperator(self.mutation_params)
        
        # Evolution state
        self.state = EvolutionState()
        self.state.current_mutation_rate = self.evolution_params.mutation_rate
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.generation_times = deque(maxlen=100)
        
        # Callbacks for monitoring
        self.generation_callbacks = []
        self.fitness_callbacks = []
    
    def initialize_population(
        self,
        base_topology: Optional[NetworkTopology] = None,
        key: Optional[jax.random.PRNGKey] = None
    ) -> None:
        """
        Initialize the evolution population.
        
        Args:
            base_topology: Base topology to start from (optional)
            key: Random key for initialization
        """
        if key is None:
            key = random.PRNGKey(42)
        
        self.state.population = []
        
        # Create initial population
        for i in range(self.evolution_params.population_size):
            key, subkey = random.split(key)
            
            if base_topology is not None and i == 0:
                # Use base topology as first individual
                topology = base_topology
            else:
                # Create random topology
                n_neurons = 100 + random.randint(subkey, (), 0, 400)  # 100-500 neurons
                key, subkey = random.split(key)
                connectivity = 0.05 + random.uniform(subkey, ()) * 0.15  # 5-20% connectivity
                
                key, subkey = random.split(key)
                topology = self.topology_manager.create_random_topology(
                    n_neurons=int(n_neurons),
                    connectivity=float(connectivity),
                    key=subkey
                )
            
            # Evaluate initial fitness
            fitness = self.fitness_function(topology)
            topology = self.topology_manager.update_performance(topology, fitness)
            
            self.state.population.append(topology)
        
        # Update best topology
        self._update_best_topology()
        
        # Initialize diversity tracking
        self._update_diversity_scores()
    
    def evolve_generation(self, key: Optional[jax.random.PRNGKey] = None) -> Dict[str, Any]:
        """
        Evolve one generation of the population.
        
        Args:
            key: Random key for evolution operations
            
        Returns:
            Dictionary with generation statistics
        """
        if key is None:
            key = random.PRNGKey(int(time.time() * 1000))
        
        start_time = time.time()
        
        # Selection
        key, subkey = random.split(key)
        selected_population = self._selection(subkey)
        
        # Mutation
        key, subkey = random.split(key)
        mutated_population = self._mutation(selected_population, subkey)
        
        # Evaluation
        evaluated_population = self._evaluation(mutated_population)
        
        # Replacement
        self.state.population = self._replacement(evaluated_population)
        
        # Update evolution state
        self.state.generation += 1
        self._update_best_topology()
        self._update_diversity_scores()
        self._update_mutation_rate()
        
        # Track performance
        generation_time = time.time() - start_time
        self.generation_times.append(generation_time)
        
        # Check for stagnation
        if len(self.state.fitness_history) > 1:
            improvement = self.state.fitness_history[-1] - self.state.fitness_history[-2]
            if improvement < self.evolution_params.convergence_threshold:
                self.state.stagnation_counter += 1
            else:
                self.state.stagnation_counter = 0
        
        # Prepare statistics
        stats = self._compute_generation_stats()
        stats['generation_time'] = generation_time
        
        # Call callbacks
        for callback in self.generation_callbacks:
            callback(self.state.generation, stats)
        
        return stats
    
    def _selection(self, key: jax.random.PRNGKey) -> List[NetworkTopology]:
        """Select individuals for reproduction using tournament selection."""
        selected = []
        
        for _ in range(self.evolution_params.population_size):
            key, subkey = random.split(key)
            
            # Tournament selection
            tournament_indices = random.choice(
                subkey,
                len(self.state.population),
                (self.evolution_params.tournament_size,),
                replace=False
            )
            
            # Select best from tournament
            tournament_fitness = [self.state.population[i].fitness_score for i in tournament_indices]
            winner_idx = tournament_indices[jnp.argmax(jnp.array(tournament_fitness))]
            
            selected.append(self.state.population[int(winner_idx)])
        
        return selected
    
    def _mutation(
        self,
        population: List[NetworkTopology],
        key: jax.random.PRNGKey
    ) -> List[NetworkTopology]:
        """Apply mutations to the population."""
        mutated = []
        
        for i, topology in enumerate(population):
            key, subkey = random.split(key)
            
            # Decide whether to mutate this individual
            if random.uniform(subkey, ()) < self.state.current_mutation_rate:
                # Generate mutations
                key, subkey = random.split(key)
                mutations = self.mutation_operator.generate_mutations(
                    topology,
                    self.evolution_params.mutations_per_individual,
                    subkey
                )
                
                # Apply mutations
                mutated_topology = topology
                for mutation in mutations:
                    mutated_topology = self.mutation_operator.apply_mutation(
                        mutated_topology, mutation
                    )
                
                # Validate mutated topology
                if self.topology_manager.validate_topology(mutated_topology):
                    mutated.append(mutated_topology)
                else:
                    # Keep original if mutation is invalid
                    mutated.append(topology)
            else:
                # No mutation
                mutated.append(topology)
        
        return mutated
    
    def _evaluation(self, population: List[NetworkTopology]) -> List[NetworkTopology]:
        """Evaluate fitness of all individuals in the population."""
        evaluated = []
        
        for topology in population:
            # Evaluate fitness
            fitness = self.fitness_function(topology)
            
            # Add diversity bonus
            diversity_bonus = self._compute_diversity_bonus(topology)
            adjusted_fitness = fitness + self.evolution_params.diversity_weight * diversity_bonus
            
            # Update topology with performance
            updated_topology = self.topology_manager.update_performance(topology, adjusted_fitness)
            evaluated.append(updated_topology)
        
        return evaluated
    
    def _replacement(self, population: List[NetworkTopology]) -> List[NetworkTopology]:
        """Select individuals for the next generation."""
        # Sort by fitness
        sorted_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
        
        # Elitism: keep best individuals
        elite = sorted_population[:self.evolution_params.elite_size]
        
        # Fill remaining slots with tournament selection from all individuals
        remaining_slots = self.evolution_params.population_size - self.evolution_params.elite_size
        remaining = sorted_population[:remaining_slots]
        
        return elite + remaining
    
    def _update_best_topology(self) -> None:
        """Update the best topology found so far."""
        current_best = max(self.state.population, key=lambda x: x.fitness_score)
        
        if current_best.fitness_score > self.state.best_fitness:
            self.state.best_fitness = current_best.fitness_score
            self.state.best_topology = current_best
        
        # Update fitness history
        self.state.fitness_history.append(current_best.fitness_score)
        
        # Keep history bounded
        if len(self.state.fitness_history) > 1000:
            self.state.fitness_history = self.state.fitness_history[-1000:]
    
    def _update_diversity_scores(self) -> None:
        """Update diversity scores for the population."""
        if len(self.state.population) < 2:
            self.state.diversity_scores = [0.0]
            return
        
        diversity_scores = []
        
        for i, topology in enumerate(self.state.population):
            # Compute diversity as average distance to other topologies
            distances = []
            for j, other_topology in enumerate(self.state.population):
                if i != j:
                    distance = self._compute_topology_distance(topology, other_topology)
                    distances.append(distance)
            
            avg_distance = jnp.mean(jnp.array(distances)) if distances else 0.0
            diversity_scores.append(float(avg_distance))
        
        self.state.diversity_scores = diversity_scores
    
    def _compute_topology_distance(
        self,
        topology1: NetworkTopology,
        topology2: NetworkTopology
    ) -> float:
        """Compute distance between two topologies."""
        # Simple distance based on connectivity differences
        if topology1.n_neurons != topology2.n_neurons:
            # Different sizes - use normalized comparison
            size_diff = abs(topology1.n_neurons - topology2.n_neurons) / max(topology1.n_neurons, topology2.n_neurons)
            return float(size_diff)
        
        # Same size - compare adjacency matrices
        adj_diff = jnp.sum(jnp.abs(topology1.adjacency_matrix.astype(float) - 
                                  topology2.adjacency_matrix.astype(float)))
        max_possible_diff = topology1.n_neurons * topology1.n_neurons
        
        return float(adj_diff / max_possible_diff)
    
    def _compute_diversity_bonus(self, topology: NetworkTopology) -> float:
        """Compute diversity bonus for a topology."""
        if not self.state.diversity_scores:
            return 0.0
        
        # Find the diversity score for this topology
        # (This is simplified - in practice would need better matching)
        avg_diversity = jnp.mean(jnp.array(self.state.diversity_scores))
        return float(avg_diversity)
    
    def _update_mutation_rate(self) -> None:
        """Update mutation rate based on evolution progress."""
        if not self.evolution_params.adaptive_mutation_rate:
            return
        
        # Decrease mutation rate if making good progress
        if self.state.stagnation_counter < 10:
            self.state.current_mutation_rate *= self.evolution_params.mutation_rate_decay
        else:
            # Increase mutation rate if stagnating
            self.state.current_mutation_rate *= 1.1
        
        # Clamp to reasonable bounds
        self.state.current_mutation_rate = jnp.clip(
            self.state.current_mutation_rate,
            self.evolution_params.min_mutation_rate,
            1.0
        )
    
    def _compute_generation_stats(self) -> Dict[str, Any]:
        """Compute statistics for the current generation."""
        fitness_scores = [topology.fitness_score for topology in self.state.population]
        
        stats = {
            'generation': self.state.generation,
            'best_fitness': self.state.best_fitness,
            'mean_fitness': float(jnp.mean(jnp.array(fitness_scores))),
            'std_fitness': float(jnp.std(jnp.array(fitness_scores))),
            'min_fitness': float(jnp.min(jnp.array(fitness_scores))),
            'max_fitness': float(jnp.max(jnp.array(fitness_scores))),
            'current_mutation_rate': float(self.state.current_mutation_rate),
            'stagnation_counter': self.state.stagnation_counter,
            'population_size': len(self.state.population),
            'mean_diversity': float(jnp.mean(jnp.array(self.state.diversity_scores))) if self.state.diversity_scores else 0.0
        }
        
        # Add topology statistics
        if self.state.best_topology:
            best_analysis = self.topology_manager.analyze_topology(self.state.best_topology)
            stats['best_topology'] = best_analysis
        
        return stats
    
    def _default_fitness_function(self, topology: NetworkTopology) -> float:
        """
        Default fitness function based on topology properties.
        
        This is a placeholder that should be replaced with actual
        performance evaluation on tasks.
        """
        # Combine multiple factors
        fitness = 0.0
        
        # Reward appropriate spectral radius (edge of chaos)
        target_spectral_radius = 0.95
        spectral_penalty = abs(topology.spectral_radius - target_spectral_radius)
        fitness += 1.0 - spectral_penalty
        
        # Reward moderate connectivity
        target_connectivity = 0.1
        actual_connectivity = topology.n_connections / (topology.n_neurons * (topology.n_neurons - 1))
        connectivity_penalty = abs(actual_connectivity - target_connectivity)
        fitness += 1.0 - connectivity_penalty * 10
        
        # Reward modularity
        fitness += topology.modularity
        
        # Penalize extreme sizes
        if topology.n_neurons < 50 or topology.n_neurons > 2000:
            fitness -= 0.5
        
        return float(fitness)
    
    def evolve(
        self,
        max_generations: Optional[int] = None,
        target_fitness: Optional[float] = None,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Dict[str, Any]:
        """
        Run the complete evolution process.
        
        Args:
            max_generations: Maximum generations to run (overrides params)
            target_fitness: Target fitness to reach (stops early if achieved)
            key: Random key for evolution
            
        Returns:
            Final evolution statistics
        """
        if key is None:
            key = random.PRNGKey(int(time.time() * 1000))
        
        max_gens = max_generations or self.evolution_params.max_generations
        
        # Initialize if not already done
        if not self.state.population:
            key, subkey = random.split(key)
            self.initialize_population(key=subkey)
        
        start_time = time.time()
        
        for generation in range(max_gens):
            key, subkey = random.split(key)
            stats = self.evolve_generation(subkey)
            
            # Check termination conditions
            if target_fitness and stats['best_fitness'] >= target_fitness:
                print(f"Target fitness {target_fitness} reached at generation {generation}")
                break
            
            if self.state.stagnation_counter >= self.evolution_params.stagnation_limit:
                print(f"Evolution stagnated at generation {generation}")
                break
            
            # Progress reporting
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {stats['best_fitness']:.4f}, "
                      f"Mean fitness = {stats['mean_fitness']:.4f}")
        
        total_time = time.time() - start_time
        
        # Final statistics
        final_stats = self._compute_generation_stats()
        final_stats['total_evolution_time'] = total_time
        final_stats['total_generations'] = self.state.generation
        final_stats['converged'] = self.state.stagnation_counter >= self.evolution_params.stagnation_limit
        
        return final_stats
    
    def get_best_topology(self) -> Optional[NetworkTopology]:
        """Get the best topology found during evolution."""
        return self.state.best_topology
    
    def add_generation_callback(self, callback: Callable[[int, Dict[str, Any]], None]) -> None:
        """Add callback to be called after each generation."""
        self.generation_callbacks.append(callback)
    
    def add_fitness_callback(self, callback: Callable[[NetworkTopology, float], None]) -> None:
        """Add callback to be called after fitness evaluation."""
        self.fitness_callbacks.append(callback)
    
    def save_evolution_state(self, filepath: str) -> None:
        """Save current evolution state to file."""
        # This would implement state serialization
        pass
    
    def load_evolution_state(self, filepath: str) -> None:
        """Load evolution state from file."""
        # This would implement state deserialization
        pass
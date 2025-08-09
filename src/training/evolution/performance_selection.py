"""
Performance-Based Selection System

This module implements sophisticated selection mechanisms for topology evolution
based on performance metrics, including multi-objective optimization and
adaptive selection strategies.
"""

from typing import List, Dict, Tuple, Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
import numpy as np
from enum import Enum
from collections import defaultdict

from ...core.topology.network_topology import NetworkTopology


class SelectionMethod(Enum):
    """Selection methods for evolution."""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_BASED = "rank_based"
    PARETO_FRONT = "pareto_front"
    NSGA_II = "nsga_ii"
    ADAPTIVE = "adaptive"


@dataclass
class SelectionParams:
    """Parameters for performance-based selection."""
    
    # Selection method
    method: SelectionMethod = SelectionMethod.TOURNAMENT
    
    # Tournament selection
    tournament_size: int = 3
    tournament_probability: float = 0.8  # Probability of selecting best in tournament
    
    # Roulette wheel selection
    fitness_scaling: float = 2.0  # Scaling factor for fitness values
    min_selection_probability: float = 0.01  # Minimum selection probability
    
    # Rank-based selection
    selection_pressure: float = 1.5  # Linear ranking selection pressure
    
    # Multi-objective selection
    objectives: List[str] = None  # List of objective names
    objective_weights: List[float] = None  # Weights for objectives
    
    # Pareto front selection
    crowding_distance_weight: float = 0.5  # Weight for crowding distance
    
    # Adaptive selection
    adaptation_window: int = 20  # Window for adaptation
    performance_threshold: float = 0.1  # Threshold for method switching
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ['fitness', 'diversity', 'efficiency']
        if self.objective_weights is None:
            self.objective_weights = [0.7, 0.2, 0.1]


@dataclass
class PerformanceMetrics:
    """Performance metrics for topology evaluation."""
    
    # Primary fitness
    fitness: float = 0.0
    
    # Task-specific performance
    task_accuracy: float = 0.0
    learning_speed: float = 0.0
    memory_efficiency: float = 0.0
    
    # Architectural properties
    connectivity: float = 0.0
    modularity: float = 0.0
    spectral_radius: float = 0.0
    
    # Computational efficiency
    inference_time: float = 0.0
    memory_usage: float = 0.0
    energy_consumption: float = 0.0
    
    # Robustness measures
    noise_tolerance: float = 0.0
    damage_resilience: float = 0.0
    
    # Diversity measures
    structural_diversity: float = 0.0
    functional_diversity: float = 0.0


class PerformanceEvaluator:
    """
    Evaluates topology performance across multiple metrics.
    
    This class provides comprehensive performance evaluation for network
    topologies, supporting both single-objective and multi-objective
    optimization scenarios.
    """
    
    def __init__(self, evaluation_functions: Optional[Dict[str, Callable]] = None):
        """
        Initialize performance evaluator.
        
        Args:
            evaluation_functions: Dictionary mapping metric names to evaluation functions
        """
        self.evaluation_functions = evaluation_functions or {}
        self._setup_default_evaluators()
    
    def _setup_default_evaluators(self):
        """Setup default evaluation functions."""
        
        def evaluate_connectivity(topology: NetworkTopology) -> float:
            """Evaluate connectivity appropriateness."""
            actual_connectivity = topology.n_connections / (topology.n_neurons * (topology.n_neurons - 1))
            target_connectivity = 0.1  # 10% connectivity target
            return 1.0 - abs(actual_connectivity - target_connectivity) * 10
        
        def evaluate_spectral_radius(topology: NetworkTopology) -> float:
            """Evaluate spectral radius for edge-of-chaos dynamics."""
            target_radius = 0.95
            return 1.0 - abs(topology.spectral_radius - target_radius)
        
        def evaluate_modularity(topology: NetworkTopology) -> float:
            """Evaluate network modularity."""
            return topology.modularity
        
        def evaluate_efficiency(topology: NetworkTopology) -> float:
            """Evaluate computational efficiency."""
            # Prefer smaller networks with good connectivity
            size_penalty = topology.n_neurons / 1000.0  # Normalize by 1000 neurons
            return max(0.0, 1.0 - size_penalty)
        
        # Set default evaluators
        if 'connectivity' not in self.evaluation_functions:
            self.evaluation_functions['connectivity'] = evaluate_connectivity
        if 'spectral_radius' not in self.evaluation_functions:
            self.evaluation_functions['spectral_radius'] = evaluate_spectral_radius
        if 'modularity' not in self.evaluation_functions:
            self.evaluation_functions['modularity'] = evaluate_modularity
        if 'efficiency' not in self.evaluation_functions:
            self.evaluation_functions['efficiency'] = evaluate_efficiency
    
    def evaluate_topology(self, topology: NetworkTopology) -> PerformanceMetrics:
        """
        Evaluate a topology across all performance metrics.
        
        Args:
            topology: Network topology to evaluate
            
        Returns:
            PerformanceMetrics with all evaluated metrics
        """
        metrics = PerformanceMetrics()
        
        # Evaluate each metric
        for metric_name, eval_func in self.evaluation_functions.items():
            try:
                value = eval_func(topology)
                setattr(metrics, metric_name, float(value))
            except Exception as e:
                print(f"Warning: Failed to evaluate {metric_name}: {e}")
                setattr(metrics, metric_name, 0.0)
        
        # Compute overall fitness as weighted combination
        metrics.fitness = (
            0.4 * metrics.connectivity +
            0.3 * metrics.spectral_radius +
            0.2 * metrics.modularity +
            0.1 * metrics.efficiency
        )
        
        return metrics
    
    def evaluate_population(self, population: List[NetworkTopology]) -> List[PerformanceMetrics]:
        """Evaluate performance metrics for entire population."""
        return [self.evaluate_topology(topology) for topology in population]


class PerformanceSelector:
    """
    Implements various selection strategies based on performance metrics.
    
    This class provides multiple selection algorithms for choosing individuals
    for reproduction based on their performance across multiple objectives.
    """
    
    def __init__(self, params: Optional[SelectionParams] = None):
        """Initialize performance selector."""
        self.params = params or SelectionParams()
        self.evaluator = PerformanceEvaluator()
        
        # Adaptive selection state
        self.method_performance_history = defaultdict(list)
        self.current_method = self.params.method
        self.adaptation_counter = 0
    
    def select(
        self,
        population: List[NetworkTopology],
        n_select: int,
        key: Optional[jax.random.PRNGKey] = None
    ) -> List[NetworkTopology]:
        """
        Select individuals from population based on performance.
        
        Args:
            population: Population to select from
            n_select: Number of individuals to select
            key: Random key for selection
            
        Returns:
            Selected individuals
        """
        if key is None:
            key = random.PRNGKey(42)
        
        # Evaluate population performance
        performance_metrics = self.evaluator.evaluate_population(population)
        
        # Choose selection method
        if self.params.method == SelectionMethod.ADAPTIVE:
            method = self._choose_adaptive_method(performance_metrics)
        else:
            method = self.params.method
        
        # Apply selection method
        if method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population, performance_metrics, n_select, key)
        elif method == SelectionMethod.ROULETTE_WHEEL:
            return self._roulette_wheel_selection(population, performance_metrics, n_select, key)
        elif method == SelectionMethod.RANK_BASED:
            return self._rank_based_selection(population, performance_metrics, n_select, key)
        elif method == SelectionMethod.PARETO_FRONT:
            return self._pareto_front_selection(population, performance_metrics, n_select, key)
        elif method == SelectionMethod.NSGA_II:
            return self._nsga_ii_selection(population, performance_metrics, n_select, key)
        else:
            # Default to tournament selection
            return self._tournament_selection(population, performance_metrics, n_select, key)
    
    def _tournament_selection(
        self,
        population: List[NetworkTopology],
        metrics: List[PerformanceMetrics],
        n_select: int,
        key: jax.random.PRNGKey
    ) -> List[NetworkTopology]:
        """Tournament selection based on fitness."""
        selected = []
        
        for _ in range(n_select):
            key, subkey = random.split(key)
            
            # Select tournament participants
            tournament_indices = random.choice(
                subkey,
                len(population),
                (self.params.tournament_size,),
                replace=False
            )
            
            # Find best in tournament
            tournament_fitness = [metrics[i].fitness for i in tournament_indices]
            
            # Probabilistic selection (not always best)
            key, subkey = random.split(key)
            if random.uniform(subkey, ()) < self.params.tournament_probability:
                # Select best
                winner_idx = tournament_indices[jnp.argmax(jnp.array(tournament_fitness))]
            else:
                # Select random from tournament
                key, subkey = random.split(key)
                winner_idx = random.choice(subkey, tournament_indices)
            
            selected.append(population[int(winner_idx)])
        
        return selected
    
    def _roulette_wheel_selection(
        self,
        population: List[NetworkTopology],
        metrics: List[PerformanceMetrics],
        n_select: int,
        key: jax.random.PRNGKey
    ) -> List[NetworkTopology]:
        """Roulette wheel selection based on fitness."""
        fitness_values = jnp.array([m.fitness for m in metrics])
        
        # Scale fitness values to ensure positive probabilities
        min_fitness = jnp.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values - min_fitness + 0.1
        
        # Apply fitness scaling
        fitness_values = jnp.power(fitness_values, self.params.fitness_scaling)
        
        # Ensure minimum selection probability
        min_prob = self.params.min_selection_probability
        fitness_values = jnp.maximum(fitness_values, min_prob)
        
        # Normalize to probabilities
        probabilities = fitness_values / jnp.sum(fitness_values)
        
        # Select individuals
        selected = []
        for _ in range(n_select):
            key, subkey = random.split(key)
            selected_idx = random.categorical(subkey, jnp.log(probabilities))
            selected.append(population[int(selected_idx)])
        
        return selected
    
    def _rank_based_selection(
        self,
        population: List[NetworkTopology],
        metrics: List[PerformanceMetrics],
        n_select: int,
        key: jax.random.PRNGKey
    ) -> List[NetworkTopology]:
        """Rank-based selection."""
        # Sort by fitness
        fitness_values = [m.fitness for m in metrics]
        sorted_indices = jnp.argsort(jnp.array(fitness_values))
        
        # Assign selection probabilities based on rank
        n_pop = len(population)
        ranks = jnp.arange(1, n_pop + 1)  # Ranks from 1 to n_pop
        
        # Linear ranking
        s = self.params.selection_pressure
        probabilities = (2 - s + 2 * (s - 1) * (ranks - 1) / (n_pop - 1)) / n_pop
        
        # Select individuals
        selected = []
        for _ in range(n_select):
            key, subkey = random.split(key)
            selected_rank_idx = random.categorical(subkey, jnp.log(probabilities))
            selected_idx = sorted_indices[selected_rank_idx]
            selected.append(population[int(selected_idx)])
        
        return selected
    
    def _pareto_front_selection(
        self,
        population: List[NetworkTopology],
        metrics: List[PerformanceMetrics],
        n_select: int,
        key: jax.random.PRNGKey
    ) -> List[NetworkTopology]:
        """Pareto front-based selection for multi-objective optimization."""
        # Extract objective values
        objectives = []
        for metric in metrics:
            obj_values = []
            for obj_name in self.params.objectives:
                if hasattr(metric, obj_name):
                    obj_values.append(getattr(metric, obj_name))
                else:
                    obj_values.append(0.0)
            objectives.append(obj_values)
        
        objectives = jnp.array(objectives)
        
        # Find Pareto front
        pareto_front_indices = self._find_pareto_front(objectives)
        
        # If Pareto front is smaller than needed, add more individuals
        if len(pareto_front_indices) >= n_select:
            # Use crowding distance to select from Pareto front
            selected_indices = self._crowding_distance_selection(
                objectives[pareto_front_indices], pareto_front_indices, n_select, key
            )
        else:
            # Include all Pareto front members and select additional individuals
            selected_indices = list(pareto_front_indices)
            remaining = n_select - len(selected_indices)
            
            # Select remaining individuals by fitness
            remaining_indices = [i for i in range(len(population)) if i not in pareto_front_indices]
            remaining_fitness = [metrics[i].fitness for i in remaining_indices]
            
            if remaining_indices:
                key, subkey = random.split(key)
                additional_indices = random.choice(
                    subkey, 
                    jnp.array(remaining_indices),
                    (remaining,),
                    p=jnp.array(remaining_fitness) / jnp.sum(jnp.array(remaining_fitness)),
                    replace=True
                )
                selected_indices.extend(additional_indices.tolist())
        
        return [population[i] for i in selected_indices[:n_select]]
    
    def _find_pareto_front(self, objectives: jnp.ndarray) -> List[int]:
        """Find Pareto front from objective values."""
        n_individuals = objectives.shape[0]
        pareto_front = []
        
        for i in range(n_individuals):
            is_dominated = False
            for j in range(n_individuals):
                if i != j:
                    # Check if j dominates i
                    if jnp.all(objectives[j] >= objectives[i]) and jnp.any(objectives[j] > objectives[i]):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append(i)
        
        return pareto_front
    
    def _crowding_distance_selection(
        self,
        objectives: jnp.ndarray,
        indices: List[int],
        n_select: int,
        key: jax.random.PRNGKey
    ) -> List[int]:
        """Select individuals based on crowding distance."""
        if len(indices) <= n_select:
            return indices
        
        # Compute crowding distances
        crowding_distances = self._compute_crowding_distance(objectives)
        
        # Sort by crowding distance (descending)
        sorted_indices = jnp.argsort(-crowding_distances)
        
        # Select top n_select individuals
        selected_local_indices = sorted_indices[:n_select]
        return [indices[i] for i in selected_local_indices]
    
    def _compute_crowding_distance(self, objectives: jnp.ndarray) -> jnp.ndarray:
        """Compute crowding distance for individuals."""
        n_individuals, n_objectives = objectives.shape
        distances = jnp.zeros(n_individuals)
        
        for obj_idx in range(n_objectives):
            # Sort by this objective
            sorted_indices = jnp.argsort(objectives[:, obj_idx])
            
            # Set boundary points to infinity
            distances = distances.at[sorted_indices[0]].set(jnp.inf)
            distances = distances.at[sorted_indices[-1]].set(jnp.inf)
            
            # Compute distances for intermediate points
            obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]
            if obj_range > 0:
                for i in range(1, n_individuals - 1):
                    distance_contribution = (
                        objectives[sorted_indices[i + 1], obj_idx] - 
                        objectives[sorted_indices[i - 1], obj_idx]
                    ) / obj_range
                    distances = distances.at[sorted_indices[i]].add(distance_contribution)
        
        return distances
    
    def _nsga_ii_selection(
        self,
        population: List[NetworkTopology],
        metrics: List[PerformanceMetrics],
        n_select: int,
        key: jax.random.PRNGKey
    ) -> List[NetworkTopology]:
        """NSGA-II selection algorithm."""
        # This is a simplified version of NSGA-II
        # Full implementation would include non-dominated sorting
        return self._pareto_front_selection(population, metrics, n_select, key)
    
    def _choose_adaptive_method(self, metrics: List[PerformanceMetrics]) -> SelectionMethod:
        """Choose selection method adaptively based on performance."""
        self.adaptation_counter += 1
        
        # Compute current population diversity
        fitness_values = [m.fitness for m in metrics]
        diversity = float(jnp.std(jnp.array(fitness_values)))
        
        # Track performance of current method
        self.method_performance_history[self.current_method].append(diversity)
        
        # Adapt method every adaptation_window generations
        if self.adaptation_counter % self.params.adaptation_window == 0:
            # Evaluate method performance
            method_scores = {}
            for method, history in self.method_performance_history.items():
                if history:
                    # Use recent performance trend
                    recent_performance = jnp.mean(jnp.array(history[-10:]))
                    method_scores[method] = recent_performance
            
            # Choose best performing method
            if method_scores:
                best_method = max(method_scores.keys(), key=lambda k: method_scores[k])
                
                # Switch if significantly better
                current_score = method_scores.get(self.current_method, 0)
                best_score = method_scores[best_method]
                
                if best_score > current_score + self.params.performance_threshold:
                    self.current_method = best_method
        
        return self.current_method
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about selection performance."""
        stats = {
            'current_method': self.current_method.value,
            'adaptation_counter': self.adaptation_counter,
            'method_performance': {}
        }
        
        for method, history in self.method_performance_history.items():
            if history:
                stats['method_performance'][method.value] = {
                    'mean_performance': float(jnp.mean(jnp.array(history))),
                    'recent_performance': float(jnp.mean(jnp.array(history[-10:]))),
                    'evaluations': len(history)
                }
        
        return stats
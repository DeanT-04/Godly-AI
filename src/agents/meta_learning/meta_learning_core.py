"""
Meta-Learning Core

This module implements the meta-learning capabilities for the Godly AI system,
including learning algorithm adaptation, hyperparameter optimization, and
domain adaptation for new task types.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import jax
import jax.numpy as jnp
from jax import random, grad, vmap
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
from collections import defaultdict, deque
import time

from ...core.topology.network_topology import NetworkTopology


class LearningAlgorithmType(Enum):
    """Types of learning algorithms."""
    GRADIENT_DESCENT = "gradient_descent"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    HEBBIAN = "hebbian"
    STDP = "stdp"
    CONTRASTIVE_DIVERGENCE = "contrastive_divergence"
    SELF_ORGANIZING = "self_organizing"


class OptimizationMethod(Enum):
    """Methods for hyperparameter optimization."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY_OPTIMIZATION = "evolutionary_optimization"
    GRADIENT_BASED = "gradient_based"
    POPULATION_BASED = "population_based"


@dataclass
class LearningAlgorithm:
    """Represents a learning algorithm with its parameters."""
    
    algorithm_type: LearningAlgorithmType
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[float] = field(default_factory=list)
    adaptation_count: int = 0
    success_rate: float = 0.0
    convergence_speed: float = 0.0
    stability_score: float = 0.0
    
    def __hash__(self):
        """Make LearningAlgorithm hashable for use as dictionary keys."""
        # Use algorithm type and a hash of hyperparameters for uniqueness
        hyperparams_str = str(sorted(self.hyperparameters.items()))
        return hash((self.algorithm_type, hyperparams_str, self.adaptation_count))
    
    def __eq__(self, other):
        """Define equality for LearningAlgorithm objects."""
        if not isinstance(other, LearningAlgorithm):
            return False
        return (self.algorithm_type == other.algorithm_type and 
                self.hyperparameters == other.hyperparameters and
                self.adaptation_count == other.adaptation_count)
    
    def update_performance(self, performance: float):
        """Update performance history and derived metrics."""
        self.performance_history.append(performance)
        
        # Keep history bounded
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        # Update derived metrics
        if len(self.performance_history) >= 10:
            recent_performance = self.performance_history[-10:]
            self.success_rate = sum(1 for p in recent_performance if p > 0.5) / len(recent_performance)
            
            # Convergence speed based on improvement rate
            if len(self.performance_history) >= 20:
                early_avg = jnp.mean(jnp.array(self.performance_history[-20:-10]))
                recent_avg = jnp.mean(jnp.array(recent_performance))
                self.convergence_speed = max(0.0, float(recent_avg - early_avg))
            elif len(self.performance_history) >= 10:
                # For smaller histories, compare first half to second half
                mid_point = len(self.performance_history) // 2
                early_avg = jnp.mean(jnp.array(self.performance_history[:mid_point]))
                recent_avg = jnp.mean(jnp.array(self.performance_history[mid_point:]))
                self.convergence_speed = max(0.0, float(recent_avg - early_avg))
            
            # Stability based on variance
            self.stability_score = 1.0 / (1.0 + float(jnp.var(jnp.array(recent_performance))))


@dataclass
class TaskDomain:
    """Represents a task domain with its characteristics."""
    
    domain_name: str
    task_type: str  # e.g., "classification", "regression", "control"
    input_dimensionality: int
    output_dimensionality: int
    temporal_structure: bool = False
    noise_level: float = 0.0
    complexity_score: float = 0.0
    
    # Domain-specific characteristics
    characteristics: Dict[str, Any] = field(default_factory=dict)
    
    # Performance requirements
    target_accuracy: float = 0.9
    max_training_time: float = 1000.0
    memory_constraints: int = 1000000  # bytes


@dataclass
class MetaLearningParams:
    """Parameters for meta-learning system."""
    
    # Algorithm adaptation
    algorithm_pool_size: int = 10
    adaptation_frequency: int = 100  # Steps between adaptations
    performance_window: int = 50     # Window for performance evaluation
    
    # Hyperparameter optimization
    optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN_OPTIMIZATION
    optimization_budget: int = 100   # Number of optimization steps
    exploration_rate: float = 0.1    # Exploration vs exploitation
    
    # Domain adaptation
    domain_similarity_threshold: float = 0.7
    transfer_learning_enabled: bool = True
    few_shot_adaptation_steps: int = 10
    
    # Meta-parameter adaptation
    meta_learning_rate: float = 0.01
    meta_adaptation_window: int = 1000
    
    # Performance tracking
    success_threshold: float = 0.8
    convergence_patience: int = 50
    
    # Resource constraints
    max_memory_usage: int = 1000000000  # 1GB
    max_computation_time: float = 3600.0  # 1 hour


class HyperparameterOptimizer:
    """
    Optimizes hyperparameters for learning algorithms.
    
    This class implements various optimization strategies for finding
    optimal hyperparameters based on performance feedback.
    """
    
    def __init__(self, method: OptimizationMethod = OptimizationMethod.BAYESIAN_OPTIMIZATION):
        """Initialize hyperparameter optimizer."""
        self.method = method
        self.optimization_history = []
        self.best_parameters = {}
        self.best_performance = -jnp.inf
        
        # Bayesian optimization state
        self.gaussian_process_data = []
        self.acquisition_function = self._expected_improvement
        
        # Evolutionary optimization state
        self.population = []
        self.generation = 0
    
    def optimize(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        objective_function: Callable[[Dict[str, Any]], float],
        n_iterations: int = 100,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using the configured method.
        
        Args:
            parameter_space: Dictionary mapping parameter names to (min, max) ranges
            objective_function: Function to evaluate parameter configurations
            n_iterations: Number of optimization iterations
            key: Random key for stochastic methods
            
        Returns:
            Best parameter configuration found
        """
        if key is None:
            key = random.PRNGKey(42)
        
        if self.method == OptimizationMethod.RANDOM_SEARCH:
            return self._random_search(parameter_space, objective_function, n_iterations, key)
        elif self.method == OptimizationMethod.GRID_SEARCH:
            return self._grid_search(parameter_space, objective_function, n_iterations)
        elif self.method == OptimizationMethod.BAYESIAN_OPTIMIZATION:
            return self._bayesian_optimization(parameter_space, objective_function, n_iterations, key)
        elif self.method == OptimizationMethod.EVOLUTIONARY_OPTIMIZATION:
            return self._evolutionary_optimization(parameter_space, objective_function, n_iterations, key)
        else:
            return self._random_search(parameter_space, objective_function, n_iterations, key)
    
    def _random_search(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        objective_function: Callable[[Dict[str, Any]], float],
        n_iterations: int,
        key: jax.random.PRNGKey
    ) -> Dict[str, Any]:
        """Random search optimization."""
        
        for i in range(n_iterations):
            key, subkey = random.split(key)
            
            # Sample random parameters
            params = {}
            for param_name, (min_val, max_val) in parameter_space.items():
                key, param_key = random.split(key)
                params[param_name] = float(random.uniform(param_key, (), minval=min_val, maxval=max_val))
            
            # Evaluate performance
            performance = objective_function(params)
            
            # Track best
            if performance > self.best_performance:
                self.best_performance = performance
                self.best_parameters = params.copy()
            
            # Record history
            self.optimization_history.append((params.copy(), performance))
        
        return self.best_parameters
    
    def _grid_search(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        objective_function: Callable[[Dict[str, Any]], float],
        n_iterations: int
    ) -> Dict[str, Any]:
        """Grid search optimization."""
        
        # Create grid points
        param_names = list(parameter_space.keys())
        n_params = len(param_names)
        
        if n_params == 0:
            return {}
        
        # Calculate grid resolution
        grid_resolution = max(2, int(n_iterations ** (1.0 / n_params)))
        
        # Generate grid points
        grid_points = []
        for param_name in param_names:
            min_val, max_val = parameter_space[param_name]
            points = jnp.linspace(min_val, max_val, grid_resolution)
            grid_points.append(points)
        
        # Evaluate all combinations
        best_params = {}
        best_performance = -jnp.inf
        
        def evaluate_recursive(param_idx: int, current_params: Dict[str, Any]):
            nonlocal best_params, best_performance
            
            if param_idx == len(param_names):
                # Evaluate this parameter combination
                performance = objective_function(current_params)
                
                if performance > best_performance:
                    best_performance = performance
                    best_params = current_params.copy()
                
                self.optimization_history.append((current_params.copy(), performance))
                return
            
            # Try all values for current parameter
            param_name = param_names[param_idx]
            for value in grid_points[param_idx]:
                current_params[param_name] = float(value)
                evaluate_recursive(param_idx + 1, current_params)
        
        evaluate_recursive(0, {})
        
        self.best_parameters = best_params
        self.best_performance = best_performance
        
        return self.best_parameters
    
    def _bayesian_optimization(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        objective_function: Callable[[Dict[str, Any]], float],
        n_iterations: int,
        key: jax.random.PRNGKey
    ) -> Dict[str, Any]:
        """Simplified Bayesian optimization."""
        
        # Start with random exploration
        n_initial = min(10, n_iterations // 2)
        
        for i in range(n_initial):
            key, subkey = random.split(key)
            
            # Random sample
            params = {}
            for param_name, (min_val, max_val) in parameter_space.items():
                key, param_key = random.split(key)
                params[param_name] = float(random.uniform(param_key, (), minval=min_val, maxval=max_val))
            
            performance = objective_function(params)
            
            if performance > self.best_performance:
                self.best_performance = performance
                self.best_parameters = params.copy()
            
            self.gaussian_process_data.append((params.copy(), performance))
            self.optimization_history.append((params.copy(), performance))
        
        # Continue with acquisition function-guided search
        for i in range(n_initial, n_iterations):
            key, subkey = random.split(key)
            
            # Find next point using acquisition function (simplified)
            best_acquisition = -jnp.inf
            best_candidate = None
            
            # Sample candidates and evaluate acquisition function
            for _ in range(20):  # Sample 20 candidates
                key, candidate_key = random.split(key)
                
                candidate = {}
                for param_name, (min_val, max_val) in parameter_space.items():
                    key, param_key = random.split(key)
                    candidate[param_name] = float(random.uniform(param_key, (), minval=min_val, maxval=max_val))
                
                acquisition_value = self._evaluate_acquisition(candidate)
                
                if acquisition_value > best_acquisition:
                    best_acquisition = acquisition_value
                    best_candidate = candidate
            
            if best_candidate is not None:
                performance = objective_function(best_candidate)
                
                if performance > self.best_performance:
                    self.best_performance = performance
                    self.best_parameters = best_candidate.copy()
                
                self.gaussian_process_data.append((best_candidate.copy(), performance))
                self.optimization_history.append((best_candidate.copy(), performance))
        
        return self.best_parameters
    
    def _evolutionary_optimization(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        objective_function: Callable[[Dict[str, Any]], float],
        n_iterations: int,
        key: jax.random.PRNGKey
    ) -> Dict[str, Any]:
        """Evolutionary optimization of hyperparameters."""
        
        population_size = 20
        mutation_rate = 0.1
        crossover_rate = 0.7
        
        # Initialize population
        if not self.population:
            for _ in range(population_size):
                key, subkey = random.split(key)
                
                individual = {}
                for param_name, (min_val, max_val) in parameter_space.items():
                    key, param_key = random.split(key)
                    individual[param_name] = float(random.uniform(param_key, (), minval=min_val, maxval=max_val))
                
                fitness = objective_function(individual)
                self.population.append((individual, fitness))
                
                if fitness > self.best_performance:
                    self.best_performance = fitness
                    self.best_parameters = individual.copy()
        
        # Evolution loop
        generations = n_iterations // population_size
        
        for gen in range(generations):
            # Selection, crossover, mutation
            new_population = []
            
            # Sort by fitness
            self.population.sort(key=lambda x: x[1], reverse=True)
            
            # Elitism - keep best individuals
            elite_size = population_size // 4
            for i in range(elite_size):
                new_population.append(self.population[i])
            
            # Generate offspring
            while len(new_population) < population_size:
                key, subkey = random.split(key)
                
                # Tournament selection
                parent1 = self._tournament_selection(key)
                key, subkey = random.split(key)
                parent2 = self._tournament_selection(subkey)
                
                # Crossover
                key, subkey = random.split(key)
                if random.uniform(subkey, ()) < crossover_rate:
                    child = self._crossover(parent1[0], parent2[0], key)
                else:
                    child = parent1[0].copy()
                
                # Mutation
                key, subkey = random.split(key)
                if random.uniform(subkey, ()) < mutation_rate:
                    child = self._mutate(child, parameter_space, key)
                
                # Evaluate child
                fitness = objective_function(child)
                new_population.append((child, fitness))
                
                if fitness > self.best_performance:
                    self.best_performance = fitness
                    self.best_parameters = child.copy()
                
                self.optimization_history.append((child.copy(), fitness))
            
            self.population = new_population
            self.generation += 1
        
        return self.best_parameters
    
    def _evaluate_acquisition(self, candidate: Dict[str, Any]) -> float:
        """Evaluate acquisition function for Bayesian optimization."""
        if not self.gaussian_process_data:
            return 0.0
        
        # Simplified acquisition function based on distance to existing points
        min_distance = jnp.inf
        
        for existing_params, existing_performance in self.gaussian_process_data:
            distance = 0.0
            for param_name, value in candidate.items():
                if param_name in existing_params:
                    distance += (value - existing_params[param_name]) ** 2
            
            distance = jnp.sqrt(distance)
            min_distance = min(min_distance, distance)
        
        # Encourage exploration of distant points
        return float(min_distance)
    
    def _tournament_selection(self, key: jax.random.PRNGKey) -> Tuple[Dict[str, Any], float]:
        """Tournament selection for evolutionary optimization."""
        tournament_size = 3
        
        # Select random individuals for tournament
        indices = random.choice(key, len(self.population), (tournament_size,), replace=False)
        
        # Find best in tournament
        best_fitness = -jnp.inf
        best_individual = None
        
        for idx in indices:
            individual, fitness = self.population[int(idx)]
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = (individual, fitness)
        
        return best_individual
    
    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        key: jax.random.PRNGKey
    ) -> Dict[str, Any]:
        """Crossover operation for evolutionary optimization."""
        child = {}
        
        for param_name in parent1.keys():
            key, subkey = random.split(key)
            
            if random.uniform(subkey, ()) < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        
        return child
    
    def _mutate(
        self,
        individual: Dict[str, Any],
        parameter_space: Dict[str, Tuple[float, float]],
        key: jax.random.PRNGKey
    ) -> Dict[str, Any]:
        """Mutation operation for evolutionary optimization."""
        mutated = individual.copy()
        
        for param_name, (min_val, max_val) in parameter_space.items():
            key, subkey = random.split(key)
            
            if random.uniform(subkey, ()) < 0.3:  # 30% chance to mutate each parameter
                # Gaussian mutation
                key, mutation_key = random.split(key)
                mutation_strength = (max_val - min_val) * 0.1  # 10% of range
                
                new_value = individual[param_name] + random.normal(mutation_key, ()) * mutation_strength
                new_value = jnp.clip(new_value, min_val, max_val)
                
                mutated[param_name] = float(new_value)
        
        return mutated
    
    def _expected_improvement(self, candidate: Dict[str, Any]) -> float:
        """Expected improvement acquisition function."""
        # Simplified version - would need full GP implementation
        return self._evaluate_acquisition(candidate)


class DomainAdapter:
    """
    Adapts learning algorithms to new task domains.
    
    This class implements domain adaptation strategies including
    transfer learning and few-shot adaptation.
    """
    
    def __init__(self):
        """Initialize domain adapter."""
        self.known_domains = {}
        self.domain_similarities = {}
        self.transfer_strategies = {}
        
        # Domain characterization
        self.domain_features = [
            'input_dimensionality',
            'output_dimensionality', 
            'temporal_structure',
            'noise_level',
            'complexity_score'
        ]
    
    def register_domain(self, domain: TaskDomain, performance_data: Dict[str, Any]):
        """Register a new domain with performance data."""
        self.known_domains[domain.domain_name] = {
            'domain': domain,
            'performance_data': performance_data,
            'successful_algorithms': [],
            'failed_algorithms': []
        }
        
        # Update similarity matrix
        self._update_domain_similarities(domain)
    
    def adapt_to_domain(
        self,
        target_domain: TaskDomain,
        available_algorithms: List[LearningAlgorithm],
        key: Optional[jax.random.PRNGKey] = None
    ) -> List[LearningAlgorithm]:
        """
        Adapt algorithms for a new domain.
        
        Args:
            target_domain: Domain to adapt to
            available_algorithms: Available learning algorithms
            key: Random key for stochastic operations
            
        Returns:
            Adapted algorithms ranked by expected performance
        """
        if key is None:
            key = random.PRNGKey(42)
        
        # Find most similar known domain
        most_similar_domain = self._find_most_similar_domain(target_domain)
        
        if most_similar_domain is None:
            # No similar domain found, use default ranking
            return self._rank_algorithms_by_default(available_algorithms, target_domain)
        
        # Transfer knowledge from similar domain
        adapted_algorithms = self._transfer_knowledge(
            target_domain, most_similar_domain, available_algorithms, key
        )
        
        return adapted_algorithms
    
    def few_shot_adaptation(
        self,
        algorithm: LearningAlgorithm,
        target_domain: TaskDomain,
        few_shot_data: List[Tuple[Any, Any]],
        n_adaptation_steps: int = 10,
        key: Optional[jax.random.PRNGKey] = None
    ) -> LearningAlgorithm:
        """
        Perform few-shot adaptation of an algorithm.
        
        Args:
            algorithm: Algorithm to adapt
            target_domain: Target domain
            few_shot_data: Small dataset for adaptation
            n_adaptation_steps: Number of adaptation steps
            key: Random key
            
        Returns:
            Adapted algorithm
        """
        if key is None:
            key = random.PRNGKey(42)
        
        adapted_algorithm = LearningAlgorithm(
            algorithm_type=algorithm.algorithm_type,
            hyperparameters=algorithm.hyperparameters.copy(),
            performance_history=algorithm.performance_history.copy(),
            adaptation_count=algorithm.adaptation_count + 1
        )
        
        # Adapt hyperparameters based on few-shot data
        if len(few_shot_data) > 0:
            # Simple adaptation: adjust learning rate based on data characteristics
            data_complexity = self._estimate_data_complexity(few_shot_data)
            
            if 'learning_rate' in adapted_algorithm.hyperparameters:
                # Adjust learning rate based on complexity
                base_lr = adapted_algorithm.hyperparameters['learning_rate']
                adapted_lr = base_lr * (1.0 / (1.0 + data_complexity))
                adapted_algorithm.hyperparameters['learning_rate'] = adapted_lr
            
            # Adjust regularization if present
            if 'regularization' in adapted_algorithm.hyperparameters:
                base_reg = adapted_algorithm.hyperparameters['regularization']
                adapted_reg = base_reg * (1.0 + data_complexity * 0.1)
                adapted_algorithm.hyperparameters['regularization'] = adapted_reg
        
        return adapted_algorithm
    
    def _update_domain_similarities(self, new_domain: TaskDomain):
        """Update domain similarity matrix with new domain."""
        for domain_name, domain_info in self.known_domains.items():
            if domain_name != new_domain.domain_name:
                existing_domain = domain_info['domain']
                similarity = self._compute_domain_similarity(new_domain, existing_domain)
                
                self.domain_similarities[(new_domain.domain_name, domain_name)] = similarity
                self.domain_similarities[(domain_name, new_domain.domain_name)] = similarity
    
    def _compute_domain_similarity(self, domain1: TaskDomain, domain2: TaskDomain) -> float:
        """Compute similarity between two domains."""
        similarity = 0.0
        n_features = 0
        
        # Compare numerical features
        for feature in self.domain_features:
            if hasattr(domain1, feature) and hasattr(domain2, feature):
                val1 = getattr(domain1, feature)
                val2 = getattr(domain2, feature)
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Normalized difference
                    max_val = max(abs(val1), abs(val2), 1.0)
                    feature_similarity = 1.0 - abs(val1 - val2) / max_val
                    similarity += feature_similarity
                    n_features += 1
                elif isinstance(val1, bool) and isinstance(val2, bool):
                    # Boolean match
                    similarity += 1.0 if val1 == val2 else 0.0
                    n_features += 1
        
        # Compare task type
        if domain1.task_type == domain2.task_type:
            similarity += 1.0
        n_features += 1
        
        return similarity / max(n_features, 1)
    
    def _find_most_similar_domain(self, target_domain: TaskDomain) -> Optional[str]:
        """Find the most similar known domain."""
        if not self.known_domains:
            return None
        
        best_similarity = -1.0
        best_domain = None
        
        for domain_name, domain_info in self.known_domains.items():
            existing_domain = domain_info['domain']
            similarity = self._compute_domain_similarity(target_domain, existing_domain)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_domain = domain_name
        
        return best_domain if best_similarity > 0.5 else None
    
    def _transfer_knowledge(
        self,
        target_domain: TaskDomain,
        source_domain_name: str,
        available_algorithms: List[LearningAlgorithm],
        key: jax.random.PRNGKey
    ) -> List[LearningAlgorithm]:
        """Transfer knowledge from source domain to target domain."""
        
        source_info = self.known_domains[source_domain_name]
        successful_algorithms = source_info.get('successful_algorithms', [])
        
        # Rank algorithms based on success in similar domain
        algorithm_scores = {}
        
        for algorithm in available_algorithms:
            score = 0.0
            
            # Check if algorithm was successful in source domain
            for successful_alg_type in successful_algorithms:
                if algorithm.algorithm_type == successful_alg_type:
                    score += 1.0
            
            # Add base score based on algorithm characteristics
            score += self._get_algorithm_domain_affinity(algorithm, target_domain)
            
            algorithm_scores[algorithm] = score
        
        # Sort by score
        ranked_algorithms = sorted(
            available_algorithms,
            key=lambda alg: algorithm_scores.get(alg, 0.0),
            reverse=True
        )
        
        return ranked_algorithms
    
    def _rank_algorithms_by_default(
        self,
        algorithms: List[LearningAlgorithm],
        domain: TaskDomain
    ) -> List[LearningAlgorithm]:
        """Default ranking when no similar domain is known."""
        
        algorithm_scores = {}
        
        for algorithm in algorithms:
            score = self._get_algorithm_domain_affinity(algorithm, domain)
            algorithm_scores[algorithm] = score
        
        return sorted(
            algorithms,
            key=lambda alg: algorithm_scores.get(alg, 0.0),
            reverse=True
        )
    
    def _get_algorithm_domain_affinity(
        self,
        algorithm: LearningAlgorithm,
        domain: TaskDomain
    ) -> float:
        """Get affinity score between algorithm and domain."""
        
        score = 0.0
        
        # Task type affinities
        if domain.task_type == "classification":
            if algorithm.algorithm_type in [LearningAlgorithmType.GRADIENT_DESCENT, 
                                          LearningAlgorithmType.EVOLUTIONARY]:
                score += 0.8
        elif domain.task_type == "regression":
            if algorithm.algorithm_type in [LearningAlgorithmType.GRADIENT_DESCENT]:
                score += 0.9
        elif domain.task_type == "control":
            if algorithm.algorithm_type in [LearningAlgorithmType.REINFORCEMENT_LEARNING,
                                          LearningAlgorithmType.EVOLUTIONARY]:
                score += 0.9
        
        # Temporal structure affinity
        if domain.temporal_structure:
            if algorithm.algorithm_type in [LearningAlgorithmType.STDP,
                                          LearningAlgorithmType.HEBBIAN]:
                score += 0.5
        
        # Noise tolerance
        if domain.noise_level > 0.1:
            if algorithm.algorithm_type in [LearningAlgorithmType.EVOLUTIONARY,
                                          LearningAlgorithmType.SELF_ORGANIZING]:
                score += 0.3
        
        return score
    
    def _estimate_data_complexity(self, data: List[Tuple[Any, Any]]) -> float:
        """Estimate complexity of few-shot data."""
        if not data:
            return 0.0
        
        # Simple complexity estimation based on data variance
        try:
            inputs = [x for x, y in data]
            if isinstance(inputs[0], (list, tuple, np.ndarray)):
                # Multi-dimensional input
                input_array = jnp.array(inputs)
                complexity = float(jnp.mean(jnp.var(input_array, axis=0)))
            else:
                # Scalar input
                complexity = float(jnp.var(jnp.array(inputs)))
            
            return jnp.clip(complexity, 0.0, 10.0)
        except:
            return 1.0  # Default complexity


class MetaLearningCore:
    """
    Core meta-learning system for the Godly AI architecture.
    
    This class coordinates learning algorithm adaptation, hyperparameter
    optimization, and domain adaptation to enable rapid learning on new tasks.
    """
    
    def __init__(self, params: Optional[MetaLearningParams] = None):
        """Initialize meta-learning core."""
        self.params = params or MetaLearningParams()
        
        # Components
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            self.params.optimization_method
        )
        self.domain_adapter = DomainAdapter()
        
        # Algorithm pool
        self.algorithm_pool = self._initialize_algorithm_pool()
        
        # Meta-learning state
        self.meta_parameters = {
            'global_learning_rate': 0.01,
            'adaptation_strength': 1.0,
            'exploration_bonus': 0.1,
            'transfer_threshold': 0.7
        }
        
        # Performance tracking
        self.task_history = []
        self.adaptation_history = []
        self.performance_trends = defaultdict(list)
        
        # Resource monitoring
        self.memory_usage = 0
        self.computation_time = 0.0
    
    def _initialize_algorithm_pool(self) -> List[LearningAlgorithm]:
        """Initialize pool of available learning algorithms."""
        
        algorithms = []
        
        # Gradient descent variants
        algorithms.append(LearningAlgorithm(
            algorithm_type=LearningAlgorithmType.GRADIENT_DESCENT,
            hyperparameters={
                'learning_rate': 0.01,
                'momentum': 0.9,
                'weight_decay': 1e-4
            }
        ))
        
        # Evolutionary algorithm
        algorithms.append(LearningAlgorithm(
            algorithm_type=LearningAlgorithmType.EVOLUTIONARY,
            hyperparameters={
                'population_size': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'selection_pressure': 1.5
            }
        ))
        
        # Reinforcement learning
        algorithms.append(LearningAlgorithm(
            algorithm_type=LearningAlgorithmType.REINFORCEMENT_LEARNING,
            hyperparameters={
                'learning_rate': 0.001,
                'discount_factor': 0.99,
                'exploration_rate': 0.1,
                'target_update_frequency': 100
            }
        ))
        
        # Hebbian learning
        algorithms.append(LearningAlgorithm(
            algorithm_type=LearningAlgorithmType.HEBBIAN,
            hyperparameters={
                'learning_rate': 0.01,
                'decay_rate': 0.001,
                'normalization': True
            }
        ))
        
        # STDP
        algorithms.append(LearningAlgorithm(
            algorithm_type=LearningAlgorithmType.STDP,
            hyperparameters={
                'a_plus': 0.005,
                'a_minus': 0.005,
                'tau_plus': 20e-3,
                'tau_minus': 20e-3
            }
        ))
        
        return algorithms
    
    def learn_learning_algorithm(
        self,
        task_distribution: List[TaskDomain],
        performance_data: Dict[str, List[float]],
        key: Optional[jax.random.PRNGKey] = None
    ) -> LearningAlgorithm:
        """
        Learn an optimal learning algorithm for a task distribution.
        
        Args:
            task_distribution: Distribution of tasks to optimize for
            performance_data: Historical performance data
            key: Random key for stochastic operations
            
        Returns:
            Optimized learning algorithm
        """
        if key is None:
            key = random.PRNGKey(42)
        
        # Analyze task distribution characteristics
        distribution_characteristics = self._analyze_task_distribution(task_distribution)
        
        # Select base algorithm type
        best_algorithm_type = self._select_algorithm_type(
            distribution_characteristics, performance_data
        )
        
        # Optimize hyperparameters for selected algorithm type
        base_algorithm = next(
            (alg for alg in self.algorithm_pool if alg.algorithm_type == best_algorithm_type),
            self.algorithm_pool[0]
        )
        
        # Define parameter space for optimization
        parameter_space = self._get_parameter_space(base_algorithm.algorithm_type)
        
        # Define objective function
        def objective_function(params: Dict[str, Any]) -> float:
            # Simulate performance with these parameters
            return self._evaluate_algorithm_parameters(
                best_algorithm_type, params, task_distribution, performance_data
            )
        
        # Optimize hyperparameters
        optimal_params = self.hyperparameter_optimizer.optimize(
            parameter_space, objective_function, 
            self.params.optimization_budget, key
        )
        
        # Create optimized algorithm
        optimized_algorithm = LearningAlgorithm(
            algorithm_type=best_algorithm_type,
            hyperparameters=optimal_params,
            adaptation_count=1
        )
        
        return optimized_algorithm
    
    def adapt_to_new_domain(
        self,
        target_domain: TaskDomain,
        few_shot_data: Optional[List[Tuple[Any, Any]]] = None,
        key: Optional[jax.random.PRNGKey] = None
    ) -> List[LearningAlgorithm]:
        """
        Adapt algorithms for a new domain.
        
        Args:
            target_domain: Target domain to adapt to
            few_shot_data: Optional few-shot data for adaptation
            key: Random key
            
        Returns:
            List of adapted algorithms ranked by expected performance
        """
        if key is None:
            key = random.PRNGKey(42)
        
        # Get domain-adapted algorithm ranking
        adapted_algorithms = self.domain_adapter.adapt_to_domain(
            target_domain, self.algorithm_pool, key
        )
        
        # Apply few-shot adaptation if data is available
        if few_shot_data is not None and len(few_shot_data) > 0:
            few_shot_adapted = []
            
            for algorithm in adapted_algorithms[:3]:  # Adapt top 3 algorithms
                key, subkey = random.split(key)
                adapted_alg = self.domain_adapter.few_shot_adaptation(
                    algorithm, target_domain, few_shot_data,
                    self.params.few_shot_adaptation_steps, subkey
                )
                few_shot_adapted.append(adapted_alg)
            
            # Add remaining algorithms without few-shot adaptation
            few_shot_adapted.extend(adapted_algorithms[3:])
            adapted_algorithms = few_shot_adapted
        
        return adapted_algorithms
    
    def optimize_hyperparameters(
        self,
        algorithm: LearningAlgorithm,
        performance_feedback: List[float],
        key: Optional[jax.random.PRNGKey] = None
    ) -> LearningAlgorithm:
        """
        Optimize hyperparameters based on performance feedback.
        
        Args:
            algorithm: Algorithm to optimize
            performance_feedback: Recent performance measurements
            key: Random key
            
        Returns:
            Algorithm with optimized hyperparameters
        """
        if key is None:
            key = random.PRNGKey(42)
        
        if len(performance_feedback) < 5:
            # Not enough data for optimization
            return algorithm
        
        # Define parameter space
        parameter_space = self._get_parameter_space(algorithm.algorithm_type)
        
        # Define objective function based on recent performance
        def objective_function(params: Dict[str, Any]) -> float:
            # Estimate performance with these parameters
            # This is a simplified version - would need actual evaluation
            
            # Penalize large deviations from current parameters
            deviation_penalty = 0.0
            for param_name, param_value in params.items():
                if param_name in algorithm.hyperparameters:
                    current_value = algorithm.hyperparameters[param_name]
                    relative_deviation = abs(param_value - current_value) / max(abs(current_value), 1e-6)
                    deviation_penalty += relative_deviation * 0.1
            
            # Use recent performance trend as baseline
            recent_performance = jnp.mean(jnp.array(performance_feedback[-5:]))
            
            # Simple heuristic: assume performance improves with optimization
            estimated_performance = recent_performance * (1.0 + 0.1) - deviation_penalty
            
            return float(estimated_performance)
        
        # Optimize with smaller budget for online optimization
        optimal_params = self.hyperparameter_optimizer.optimize(
            parameter_space, objective_function, 
            min(20, self.params.optimization_budget), key
        )
        
        # Create optimized algorithm
        optimized_algorithm = LearningAlgorithm(
            algorithm_type=algorithm.algorithm_type,
            hyperparameters=optimal_params,
            performance_history=algorithm.performance_history.copy(),
            adaptation_count=algorithm.adaptation_count + 1
        )
        
        # Track this adaptation
        self.adaptation_history.append({
            'type': 'hyperparameter_optimization',
            'algorithm_type': algorithm.algorithm_type,
            'timestamp': time.time()
        })
        
        return optimized_algorithm
    
    def update_meta_parameters(self, performance_feedback: float):
        """Update meta-parameters based on performance feedback."""
        
        # Simple adaptive update of meta-parameters
        if len(self.performance_trends['overall']) > 0:
            recent_trend = performance_feedback - self.performance_trends['overall'][-1]
            
            # Adapt global learning rate
            if recent_trend > 0:
                # Performance improving, maintain or slightly increase learning rate
                self.meta_parameters['global_learning_rate'] *= 1.01
            else:
                # Performance declining, reduce learning rate
                self.meta_parameters['global_learning_rate'] *= 0.99
            
            # Clamp learning rate
            self.meta_parameters['global_learning_rate'] = jnp.clip(
                self.meta_parameters['global_learning_rate'], 1e-5, 0.1
            )
            
            # Adapt exploration bonus
            if recent_trend < -0.1:  # Significant performance drop
                self.meta_parameters['exploration_bonus'] *= 1.1  # Increase exploration
            elif recent_trend > 0.1:  # Significant improvement
                self.meta_parameters['exploration_bonus'] *= 0.95  # Reduce exploration
            
            # Clamp exploration bonus
            self.meta_parameters['exploration_bonus'] = jnp.clip(
                self.meta_parameters['exploration_bonus'], 0.01, 0.5
            )
        
        # Update performance trends
        self.performance_trends['overall'].append(performance_feedback)
        
        # Keep trends bounded
        if len(self.performance_trends['overall']) > 1000:
            self.performance_trends['overall'] = self.performance_trends['overall'][-1000:]
    
    def _analyze_task_distribution(self, task_distribution: List[TaskDomain]) -> Dict[str, Any]:
        """Analyze characteristics of task distribution."""
        
        if not task_distribution:
            return {}
        
        characteristics = {
            'n_tasks': len(task_distribution),
            'task_types': set(domain.task_type for domain in task_distribution),
            'avg_input_dim': float(jnp.mean(jnp.array([domain.input_dimensionality for domain in task_distribution]))),
            'avg_output_dim': float(jnp.mean(jnp.array([domain.output_dimensionality for domain in task_distribution]))),
            'temporal_fraction': sum(1 for domain in task_distribution if domain.temporal_structure) / len(task_distribution),
            'avg_noise_level': float(jnp.mean(jnp.array([domain.noise_level for domain in task_distribution]))),
            'avg_complexity': float(jnp.mean(jnp.array([domain.complexity_score for domain in task_distribution])))
        }
        
        return characteristics
    
    def _select_algorithm_type(
        self,
        distribution_characteristics: Dict[str, Any],
        performance_data: Dict[str, List[float]]
    ) -> LearningAlgorithmType:
        """Select best algorithm type for task distribution."""
        
        algorithm_scores = {}
        
        for algorithm in self.algorithm_pool:
            score = 0.0
            
            # Score based on historical performance
            if algorithm.algorithm_type.value in performance_data:
                recent_performance = performance_data[algorithm.algorithm_type.value]
                if recent_performance:
                    score += jnp.mean(jnp.array(recent_performance[-10:]))
            
            # Score based on distribution characteristics
            if 'task_types' in distribution_characteristics:
                task_types = distribution_characteristics['task_types']
                
                if 'classification' in task_types:
                    if algorithm.algorithm_type in [LearningAlgorithmType.GRADIENT_DESCENT,
                                                  LearningAlgorithmType.EVOLUTIONARY]:
                        score += 0.5
                
                if 'control' in task_types:
                    if algorithm.algorithm_type == LearningAlgorithmType.REINFORCEMENT_LEARNING:
                        score += 0.8
            
            # Score based on temporal structure
            if distribution_characteristics.get('temporal_fraction', 0) > 0.5:
                if algorithm.algorithm_type in [LearningAlgorithmType.STDP,
                                              LearningAlgorithmType.HEBBIAN]:
                    score += 0.3
            
            algorithm_scores[algorithm.algorithm_type] = score
        
        # Return algorithm type with highest score
        return max(algorithm_scores.keys(), key=lambda k: algorithm_scores[k])
    
    def _get_parameter_space(self, algorithm_type: LearningAlgorithmType) -> Dict[str, Tuple[float, float]]:
        """Get parameter space for hyperparameter optimization."""
        
        if algorithm_type == LearningAlgorithmType.GRADIENT_DESCENT:
            return {
                'learning_rate': (1e-5, 0.1),
                'momentum': (0.0, 0.99),
                'weight_decay': (1e-6, 1e-2)
            }
        elif algorithm_type == LearningAlgorithmType.EVOLUTIONARY:
            return {
                'population_size': (10, 200),
                'mutation_rate': (0.01, 0.5),
                'crossover_rate': (0.1, 0.9),
                'selection_pressure': (1.0, 3.0)
            }
        elif algorithm_type == LearningAlgorithmType.REINFORCEMENT_LEARNING:
            return {
                'learning_rate': (1e-5, 0.01),
                'discount_factor': (0.9, 0.999),
                'exploration_rate': (0.01, 0.5)
            }
        elif algorithm_type == LearningAlgorithmType.STDP:
            return {
                'a_plus': (0.001, 0.01),
                'a_minus': (0.001, 0.01),
                'tau_plus': (10e-3, 50e-3),
                'tau_minus': (10e-3, 50e-3)
            }
        else:
            # Default parameter space
            return {
                'learning_rate': (1e-5, 0.1)
            }
    
    def _evaluate_algorithm_parameters(
        self,
        algorithm_type: LearningAlgorithmType,
        parameters: Dict[str, Any],
        task_distribution: List[TaskDomain],
        performance_data: Dict[str, List[float]]
    ) -> float:
        """Evaluate algorithm parameters on task distribution."""
        
        # Simplified evaluation - would need actual implementation
        base_score = 0.5
        
        # Bonus for reasonable parameter values
        if algorithm_type == LearningAlgorithmType.GRADIENT_DESCENT:
            lr = parameters.get('learning_rate', 0.01)
            if 0.001 <= lr <= 0.1:
                base_score += 0.2
            
            momentum = parameters.get('momentum', 0.9)
            if 0.8 <= momentum <= 0.95:
                base_score += 0.1
        
        # Add noise for realistic evaluation
        noise = np.random.normal(0, 0.1)
        
        return float(jnp.clip(base_score + noise, 0.0, 1.0))
    
    def get_meta_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about meta-learning performance."""
        
        # Track total adaptations across all operations
        total_adaptations = sum(alg.adaptation_count for alg in self.algorithm_pool)
        # Add adaptations from adaptation history
        total_adaptations += len(self.adaptation_history)
        
        stats = {
            'algorithm_pool_size': len(self.algorithm_pool),
            'meta_parameters': self.meta_parameters.copy(),
            'total_adaptations': total_adaptations,
            'known_domains': len(self.domain_adapter.known_domains),
            'optimization_history_length': len(self.hyperparameter_optimizer.optimization_history),
            'memory_usage': self.memory_usage,
            'computation_time': self.computation_time
        }
        
        # Algorithm performance statistics
        for algorithm in self.algorithm_pool:
            alg_name = algorithm.algorithm_type.value
            stats[f'{alg_name}_success_rate'] = algorithm.success_rate
            stats[f'{alg_name}_convergence_speed'] = algorithm.convergence_speed
            stats[f'{alg_name}_stability_score'] = algorithm.stability_score
        
        # Performance trends
        if self.performance_trends['overall']:
            recent_performance = self.performance_trends['overall'][-10:]
            stats['recent_mean_performance'] = float(jnp.mean(jnp.array(recent_performance)))
            stats['performance_trend'] = float(jnp.mean(jnp.array(recent_performance[-5:])) - 
                                             jnp.mean(jnp.array(recent_performance[:5])))
        
        return stats


# Add the missing method to DomainAdapter class
def _get_algorithm_domain_affinity_method(self, algorithm: LearningAlgorithm, domain: TaskDomain) -> float:
    """Get affinity score between algorithm and domain."""
    
    score = 0.0
    
    # Task type affinities
    if domain.task_type == "classification":
        if algorithm.algorithm_type == LearningAlgorithmType.GRADIENT_DESCENT:
            score += 1.0  # Strong preference for gradient descent in classification
        elif algorithm.algorithm_type == LearningAlgorithmType.EVOLUTIONARY:
            score += 0.6
        else:
            score += 0.3
    elif domain.task_type == "regression":
        if algorithm.algorithm_type == LearningAlgorithmType.GRADIENT_DESCENT:
            score += 0.9
        else:
            score += 0.3
    elif domain.task_type == "control":
        if algorithm.algorithm_type == LearningAlgorithmType.REINFORCEMENT_LEARNING:
            score += 0.9
        elif algorithm.algorithm_type == LearningAlgorithmType.EVOLUTIONARY:
            score += 0.7
        else:
            score += 0.3
    
    # Temporal structure affinity
    if domain.temporal_structure:
        if algorithm.algorithm_type in [LearningAlgorithmType.STDP,
                                      LearningAlgorithmType.HEBBIAN]:
            score += 0.5
    
    return score

# Monkey patch the method to DomainAdapter
DomainAdapter._get_algorithm_domain_affinity = _get_algorithm_domain_affinity_method
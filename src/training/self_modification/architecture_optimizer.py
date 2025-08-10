"""
Architecture Optimizer

This module implements architecture optimization based on performance metrics,
including automated hyperparameter tuning and structural modifications.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Tuple, Optional, Callable, Any, NamedTuple
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import time


class OptimizationStrategy(Enum):
    """Optimization strategies for architecture modification."""
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"


@dataclass
class OptimizationConfig:
    """Configuration for architecture optimization."""
    
    # Strategy selection
    primary_strategy: OptimizationStrategy = OptimizationStrategy.GRADIENT_BASED
    fallback_strategy: OptimizationStrategy = OptimizationStrategy.RANDOM_SEARCH
    
    # Optimization parameters
    max_optimization_steps: int = 100
    convergence_threshold: float = 1e-6
    improvement_patience: int = 10
    
    # Search space parameters
    parameter_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    discrete_parameters: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Performance evaluation
    evaluation_episodes: int = 5
    evaluation_frequency: int = 10
    performance_aggregation: str = "mean"  # "mean", "median", "max"
    
    # Optimization constraints
    max_parameter_change: float = 0.5
    stability_weight: float = 0.2
    complexity_penalty: float = 0.1
    
    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: ["performance", "stability", "efficiency"])
    objective_weights: List[float] = field(default_factory=lambda: [0.7, 0.2, 0.1])


class PerformanceTracker:
    """Tracks performance metrics for optimization."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.parameter_history = deque(maxlen=window_size)
        
    def record_performance(self, 
                         parameters: Dict[str, Any], 
                         metrics: Dict[str, float]) -> None:
        """Record performance for given parameters."""
        self.parameter_history.append(parameters.copy())
        self.metrics_history.append(metrics.copy())
    
    def get_best_parameters(self, metric: str = "overall_performance") -> Dict[str, Any]:
        """Get parameters that achieved best performance."""
        if not self.metrics_history:
            return {}
        
        best_idx = max(range(len(self.metrics_history)), 
                      key=lambda i: self.metrics_history[i].get(metric, 0.0))
        return self.parameter_history[best_idx]
    
    def get_performance_trend(self, metric: str = "overall_performance") -> float:
        """Get performance trend (slope of recent performance)."""
        if len(self.metrics_history) < 5:
            return 0.0
        
        recent_values = [m.get(metric, 0.0) for m in list(self.metrics_history)[-10:]]
        x = np.arange(len(recent_values))
        trend = np.polyfit(x, recent_values, 1)[0]
        return float(trend)
    
    def get_performance_statistics(self, metric: str = "overall_performance") -> Dict[str, float]:
        """Get performance statistics."""
        if not self.metrics_history:
            return {}
        
        values = [m.get(metric, 0.0) for m in self.metrics_history]
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'current': values[-1] if values else 0.0
        }


class GradientBasedOptimizer:
    """Gradient-based optimization for continuous parameters."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def optimize_step(self, 
                     parameters: Dict[str, float],
                     gradients: Dict[str, float],
                     bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Perform one optimization step."""
        new_parameters = {}
        
        for param_name, current_value in parameters.items():
            if param_name in gradients:
                # Update velocity with momentum
                if param_name not in self.velocity:
                    self.velocity[param_name] = 0.0
                
                self.velocity[param_name] = (self.momentum * self.velocity[param_name] + 
                                           self.learning_rate * gradients[param_name])
                
                # Update parameter
                new_value = current_value + self.velocity[param_name]
                
                # Apply bounds
                if param_name in bounds:
                    min_val, max_val = bounds[param_name]
                    new_value = max(min_val, min(max_val, new_value))
                
                new_parameters[param_name] = new_value
            else:
                new_parameters[param_name] = current_value
        
        return new_parameters


class BayesianOptimizer:
    """Bayesian optimization for parameter search."""
    
    def __init__(self, acquisition_function: str = "expected_improvement"):
        self.acquisition_function = acquisition_function
        self.observations = []
        self.parameters_tried = []
    
    def suggest_parameters(self, 
                         bounds: Dict[str, Tuple[float, float]],
                         n_suggestions: int = 1) -> List[Dict[str, float]]:
        """Suggest next parameters to try."""
        suggestions = []
        
        for _ in range(n_suggestions):
            # Simple random suggestion for now (would use GP in full implementation)
            suggestion = {}
            for param_name, (min_val, max_val) in bounds.items():
                suggestion[param_name] = np.random.uniform(min_val, max_val)
            suggestions.append(suggestion)
        
        return suggestions
    
    def update_observations(self, 
                          parameters: Dict[str, float], 
                          performance: float) -> None:
        """Update observations with new data point."""
        self.parameters_tried.append(parameters.copy())
        self.observations.append(performance)


class EvolutionaryOptimizer:
    """Evolutionary optimization for parameter search."""
    
    def __init__(self, 
                 population_size: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.fitness_scores = []
    
    def initialize_population(self, bounds: Dict[str, Tuple[float, float]]) -> None:
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            individual = {}
            for param_name, (min_val, max_val) in bounds.items():
                individual[param_name] = np.random.uniform(min_val, max_val)
            self.population.append(individual)
    
    def evolve_population(self, 
                         fitness_scores: List[float],
                         bounds: Dict[str, Tuple[float, float]]) -> List[Dict[str, float]]:
        """Evolve population based on fitness scores."""
        self.fitness_scores = fitness_scores
        
        # Selection (tournament selection)
        new_population = []
        for _ in range(self.population_size):
            # Tournament selection
            tournament_size = 3
            tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
            winner_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
            new_population.append(self.population[winner_idx].copy())
        
        # Crossover and mutation
        for i in range(0, len(new_population), 2):
            if i + 1 < len(new_population) and np.random.random() < self.crossover_rate:
                # Crossover
                parent1, parent2 = new_population[i], new_population[i + 1]
                for param_name in parent1.keys():
                    if np.random.random() < 0.5:
                        parent1[param_name], parent2[param_name] = parent2[param_name], parent1[param_name]
            
            # Mutation
            for individual in new_population[i:i+2]:
                if i + 1 < len(new_population):
                    for param_name, value in individual.items():
                        if np.random.random() < self.mutation_rate:
                            min_val, max_val = bounds[param_name]
                            mutation_strength = (max_val - min_val) * 0.1
                            individual[param_name] = np.clip(
                                value + np.random.normal(0, mutation_strength),
                                min_val, max_val
                            )
        
        self.population = new_population
        return new_population


class ArchitectureOptimizer:
    """
    Architecture Optimizer
    
    Optimizes system architecture based on performance metrics using
    various optimization strategies.
    """
    
    def __init__(self, 
                 config: OptimizationConfig,
                 performance_evaluator: Callable[[Dict], Dict[str, float]],
                 key: jax.random.PRNGKey):
        self.config = config
        self.performance_evaluator = performance_evaluator
        self.key = key
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()
        
        # Initialize optimizers
        self.gradient_optimizer = GradientBasedOptimizer()
        self.bayesian_optimizer = BayesianOptimizer()
        self.evolutionary_optimizer = EvolutionaryOptimizer()
        
        # Optimization state
        self.current_parameters = {}
        self.best_parameters = {}
        self.best_performance = -float('inf')
        self.optimization_step = 0
        self.stagnation_count = 0
        
    def set_parameter_bounds(self, bounds: Dict[str, Tuple[float, float]]) -> None:
        """Set bounds for continuous parameters."""
        self.config.parameter_bounds.update(bounds)
    
    def set_discrete_parameters(self, discrete_params: Dict[str, List[Any]]) -> None:
        """Set options for discrete parameters."""
        self.config.discrete_parameters.update(discrete_params)
    
    def evaluate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate performance for given parameters."""
        # Apply parameters to system (this would be implemented by the caller)
        # For now, we'll use the performance evaluator directly
        metrics = self.performance_evaluator(parameters)
        
        # Record performance
        self.performance_tracker.record_performance(parameters, metrics)
        
        return metrics
    
    def compute_multi_objective_score(self, metrics: Dict[str, float]) -> float:
        """Compute multi-objective optimization score."""
        score = 0.0
        
        for i, objective in enumerate(self.config.objectives):
            if objective in metrics:
                weight = self.config.objective_weights[i] if i < len(self.config.objective_weights) else 1.0
                score += weight * metrics[objective]
        
        return score
    
    def estimate_gradients(self, 
                         parameters: Dict[str, float],
                         epsilon: float = 1e-6) -> Dict[str, float]:
        """Estimate gradients using finite differences."""
        gradients = {}
        base_metrics = self.evaluate_parameters(parameters)
        base_score = self.compute_multi_objective_score(base_metrics)
        
        for param_name, param_value in parameters.items():
            if param_name in self.config.parameter_bounds:
                # Perturb parameter
                perturbed_params = parameters.copy()
                perturbed_params[param_name] = param_value + epsilon
                
                # Evaluate perturbed parameters
                perturbed_metrics = self.evaluate_parameters(perturbed_params)
                perturbed_score = self.compute_multi_objective_score(perturbed_metrics)
                
                # Compute gradient
                gradient = (perturbed_score - base_score) / epsilon
                gradients[param_name] = gradient
        
        return gradients
    
    def optimize_with_gradient_based(self, 
                                   initial_parameters: Dict[str, float]) -> Dict[str, float]:
        """Optimize using gradient-based method."""
        current_params = initial_parameters.copy()
        
        for step in range(self.config.max_optimization_steps):
            # Estimate gradients
            gradients = self.estimate_gradients(current_params)
            
            # Optimization step
            current_params = self.gradient_optimizer.optimize_step(
                current_params, gradients, self.config.parameter_bounds
            )
            
            # Evaluate new parameters
            metrics = self.evaluate_parameters(current_params)
            score = self.compute_multi_objective_score(metrics)
            
            # Check for improvement
            if score > self.best_performance:
                self.best_performance = score
                self.best_parameters = current_params.copy()
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            
            # Check convergence
            if self.stagnation_count >= self.config.improvement_patience:
                print(f"Gradient-based optimization converged after {step + 1} steps")
                break
        
        return self.best_parameters
    
    def optimize_with_evolutionary(self, 
                                 initial_parameters: Dict[str, float]) -> Dict[str, float]:
        """Optimize using evolutionary algorithm."""
        # Initialize population
        self.evolutionary_optimizer.initialize_population(self.config.parameter_bounds)
        
        # Add initial parameters to population
        if initial_parameters:
            self.evolutionary_optimizer.population[0] = initial_parameters.copy()
        
        for generation in range(self.config.max_optimization_steps // self.evolutionary_optimizer.population_size):
            # Evaluate population
            fitness_scores = []
            for individual in self.evolutionary_optimizer.population:
                metrics = self.evaluate_parameters(individual)
                score = self.compute_multi_objective_score(metrics)
                fitness_scores.append(score)
                
                # Update best
                if score > self.best_performance:
                    self.best_performance = score
                    self.best_parameters = individual.copy()
            
            # Evolve population
            self.evolutionary_optimizer.evolve_population(fitness_scores, self.config.parameter_bounds)
            
            print(f"Generation {generation + 1}: Best score = {max(fitness_scores):.4f}")
        
        return self.best_parameters
    
    def optimize_with_bayesian(self, 
                             initial_parameters: Dict[str, float]) -> Dict[str, float]:
        """Optimize using Bayesian optimization."""
        # Evaluate initial parameters
        if initial_parameters:
            metrics = self.evaluate_parameters(initial_parameters)
            score = self.compute_multi_objective_score(metrics)
            self.bayesian_optimizer.update_observations(initial_parameters, score)
            
            if score > self.best_performance:
                self.best_performance = score
                self.best_parameters = initial_parameters.copy()
        
        for step in range(self.config.max_optimization_steps):
            # Get suggestions
            suggestions = self.bayesian_optimizer.suggest_parameters(
                self.config.parameter_bounds, n_suggestions=1
            )
            
            for params in suggestions:
                # Evaluate suggested parameters
                metrics = self.evaluate_parameters(params)
                score = self.compute_multi_objective_score(metrics)
                
                # Update Bayesian optimizer
                self.bayesian_optimizer.update_observations(params, score)
                
                # Update best
                if score > self.best_performance:
                    self.best_performance = score
                    self.best_parameters = params.copy()
                    self.stagnation_count = 0
                else:
                    self.stagnation_count += 1
            
            # Check convergence
            if self.stagnation_count >= self.config.improvement_patience:
                print(f"Bayesian optimization converged after {step + 1} steps")
                break
        
        return self.best_parameters
    
    def optimize_with_random_search(self, 
                                  initial_parameters: Dict[str, float]) -> Dict[str, float]:
        """Optimize using random search."""
        # Evaluate initial parameters
        if initial_parameters:
            metrics = self.evaluate_parameters(initial_parameters)
            score = self.compute_multi_objective_score(metrics)
            
            if score > self.best_performance:
                self.best_performance = score
                self.best_parameters = initial_parameters.copy()
        
        for step in range(self.config.max_optimization_steps):
            # Generate random parameters
            random_params = {}
            for param_name, (min_val, max_val) in self.config.parameter_bounds.items():
                random_params[param_name] = np.random.uniform(min_val, max_val)
            
            # Evaluate random parameters
            metrics = self.evaluate_parameters(random_params)
            score = self.compute_multi_objective_score(metrics)
            
            # Update best
            if score > self.best_performance:
                self.best_performance = score
                self.best_parameters = random_params.copy()
                print(f"Step {step + 1}: New best score = {score:.4f}")
        
        return self.best_parameters
    
    def optimize_architecture(self, 
                            initial_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize architecture using configured strategy."""
        print(f"Starting architecture optimization with {self.config.primary_strategy.value} strategy")
        
        # Separate continuous and discrete parameters
        continuous_params = {k: v for k, v in initial_parameters.items() 
                           if k in self.config.parameter_bounds}
        discrete_params = {k: v for k, v in initial_parameters.items() 
                         if k in self.config.discrete_parameters}
        
        # Optimize continuous parameters
        if continuous_params:
            if self.config.primary_strategy == OptimizationStrategy.GRADIENT_BASED:
                optimized_continuous = self.optimize_with_gradient_based(continuous_params)
            elif self.config.primary_strategy == OptimizationStrategy.EVOLUTIONARY:
                optimized_continuous = self.optimize_with_evolutionary(continuous_params)
            elif self.config.primary_strategy == OptimizationStrategy.BAYESIAN:
                optimized_continuous = self.optimize_with_bayesian(continuous_params)
            elif self.config.primary_strategy == OptimizationStrategy.RANDOM_SEARCH:
                optimized_continuous = self.optimize_with_random_search(continuous_params)
            else:
                optimized_continuous = continuous_params
        else:
            optimized_continuous = {}
        
        # Optimize discrete parameters (simple grid search for now)
        optimized_discrete = self.optimize_discrete_parameters(discrete_params)
        
        # Combine results
        optimized_parameters = {**optimized_continuous, **optimized_discrete}
        
        print(f"Architecture optimization completed. Best score: {self.best_performance:.4f}")
        return optimized_parameters
    
    def optimize_discrete_parameters(self, 
                                   discrete_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize discrete parameters using grid search."""
        if not discrete_params:
            return {}
        
        best_discrete = discrete_params.copy()
        best_score = -float('inf')
        
        # Simple approach: try each option for each parameter
        for param_name, current_value in discrete_params.items():
            if param_name in self.config.discrete_parameters:
                for option in self.config.discrete_parameters[param_name]:
                    test_params = discrete_params.copy()
                    test_params[param_name] = option
                    
                    # Combine with current best continuous parameters
                    full_params = {**self.best_parameters, **test_params}
                    
                    # Evaluate
                    metrics = self.evaluate_parameters(full_params)
                    score = self.compute_multi_objective_score(metrics)
                    
                    if score > best_score:
                        best_score = score
                        best_discrete[param_name] = option
        
        return best_discrete
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization process."""
        stats = self.performance_tracker.get_performance_statistics()
        
        return {
            'optimization_steps': self.optimization_step,
            'best_performance': self.best_performance,
            'best_parameters': self.best_parameters,
            'performance_statistics': stats,
            'performance_trend': self.performance_tracker.get_performance_trend(),
            'stagnation_count': self.stagnation_count,
            'strategy_used': self.config.primary_strategy.value
        }
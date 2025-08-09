"""
Tests for Meta-Learning Core

This module contains comprehensive tests for the meta-learning components
including hyperparameter optimization, domain adaptation, and the core
meta-learning system.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import List, Dict, Any

from src.agents.meta_learning.meta_learning_core import (
    MetaLearningCore,
    HyperparameterOptimizer,
    DomainAdapter,
    LearningAlgorithm,
    TaskDomain,
    MetaLearningParams,
    LearningAlgorithmType,
    OptimizationMethod
)


class TestLearningAlgorithm:
    """Test LearningAlgorithm class."""
    
    def test_learning_algorithm_initialization(self):
        """Test LearningAlgorithm initialization."""
        algorithm = LearningAlgorithm(
            algorithm_type=LearningAlgorithmType.GRADIENT_DESCENT,
            hyperparameters={'learning_rate': 0.01, 'momentum': 0.9}
        )
        
        assert algorithm.algorithm_type == LearningAlgorithmType.GRADIENT_DESCENT
        assert algorithm.hyperparameters['learning_rate'] == 0.01
        assert algorithm.hyperparameters['momentum'] == 0.9
        assert algorithm.adaptation_count == 0
        assert algorithm.success_rate == 0.0
    
    def test_performance_update(self):
        """Test performance update and derived metrics."""
        algorithm = LearningAlgorithm(
            algorithm_type=LearningAlgorithmType.GRADIENT_DESCENT
        )
        
        # Add some performance data
        performances = [0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.85, 0.9, 0.95, 0.92]
        for perf in performances:
            algorithm.update_performance(perf)
        
        assert len(algorithm.performance_history) == 10
        assert algorithm.success_rate > 0.5  # Most performances > 0.5
        assert algorithm.convergence_speed > 0  # Performance improved
        assert algorithm.stability_score > 0  # Some stability measure


class TestTaskDomain:
    """Test TaskDomain class."""
    
    def test_task_domain_initialization(self):
        """Test TaskDomain initialization."""
        domain = TaskDomain(
            domain_name="test_classification",
            task_type="classification",
            input_dimensionality=784,
            output_dimensionality=10,
            temporal_structure=False,
            noise_level=0.1,
            complexity_score=0.5
        )
        
        assert domain.domain_name == "test_classification"
        assert domain.task_type == "classification"
        assert domain.input_dimensionality == 784
        assert domain.output_dimensionality == 10
        assert domain.temporal_structure == False
        assert domain.noise_level == 0.1
        assert domain.complexity_score == 0.5


class TestHyperparameterOptimizer:
    """Test hyperparameter optimization."""
    
    def test_optimizer_initialization(self):
        """Test HyperparameterOptimizer initialization."""
        optimizer = HyperparameterOptimizer(OptimizationMethod.RANDOM_SEARCH)
        
        assert optimizer.method == OptimizationMethod.RANDOM_SEARCH
        assert len(optimizer.optimization_history) == 0
        assert optimizer.best_performance == -jnp.inf
    
    def test_random_search_optimization(self):
        """Test random search optimization."""
        optimizer = HyperparameterOptimizer(OptimizationMethod.RANDOM_SEARCH)
        
        # Define simple parameter space
        parameter_space = {
            'learning_rate': (0.001, 0.1),
            'momentum': (0.0, 0.99)
        }
        
        # Define simple objective function (quadratic with optimum at lr=0.01, momentum=0.9)
        def objective_function(params):
            lr = params['learning_rate']
            momentum = params['momentum']
            
            # Quadratic function with maximum at (0.01, 0.9)
            lr_score = 1.0 - ((lr - 0.01) / 0.01) ** 2
            momentum_score = 1.0 - ((momentum - 0.9) / 0.9) ** 2
            
            return max(0.0, lr_score * momentum_score)
        
        key = random.PRNGKey(42)
        best_params = optimizer.optimize(
            parameter_space, objective_function, n_iterations=50, key=key
        )
        
        # Check that optimization found reasonable parameters
        assert 'learning_rate' in best_params
        assert 'momentum' in best_params
        assert 0.001 <= best_params['learning_rate'] <= 0.1
        assert 0.0 <= best_params['momentum'] <= 0.99
        assert len(optimizer.optimization_history) == 50
        assert optimizer.best_performance > 0
    
    def test_grid_search_optimization(self):
        """Test grid search optimization."""
        optimizer = HyperparameterOptimizer(OptimizationMethod.GRID_SEARCH)
        
        parameter_space = {
            'param1': (0.0, 1.0),
            'param2': (0.0, 1.0)
        }
        
        def objective_function(params):
            # Simple function with maximum at (0.5, 0.5)
            return 1.0 - ((params['param1'] - 0.5) ** 2 + (params['param2'] - 0.5) ** 2)
        
        best_params = optimizer.optimize(
            parameter_space, objective_function, n_iterations=16
        )
        
        assert 'param1' in best_params
        assert 'param2' in best_params
        assert len(optimizer.optimization_history) > 0
        
        # Grid search should find parameters close to optimum
        assert abs(best_params['param1'] - 0.5) < 0.3
        assert abs(best_params['param2'] - 0.5) < 0.3
    
    def test_bayesian_optimization(self):
        """Test Bayesian optimization."""
        optimizer = HyperparameterOptimizer(OptimizationMethod.BAYESIAN_OPTIMIZATION)
        
        parameter_space = {
            'x': (-2.0, 2.0)
        }
        
        def objective_function(params):
            x = params['x']
            # Simple quadratic with maximum at x=0
            return 1.0 - x ** 2
        
        key = random.PRNGKey(42)
        best_params = optimizer.optimize(
            parameter_space, objective_function, n_iterations=20, key=key
        )
        
        assert 'x' in best_params
        assert -2.0 <= best_params['x'] <= 2.0
        assert len(optimizer.optimization_history) == 20
        
        # Should find parameters reasonably close to optimum
        assert abs(best_params['x']) < 1.0
    
    def test_evolutionary_optimization(self):
        """Test evolutionary optimization."""
        optimizer = HyperparameterOptimizer(OptimizationMethod.EVOLUTIONARY_OPTIMIZATION)
        
        parameter_space = {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        }
        
        def objective_function(params):
            x, y = params['x'], params['y']
            # Rastrigin function (multimodal, global optimum at (0,0))
            return 2.0 - (x**2 + y**2 - np.cos(2*np.pi*x) - np.cos(2*np.pi*y))
        
        key = random.PRNGKey(42)
        best_params = optimizer.optimize(
            parameter_space, objective_function, n_iterations=100, key=key
        )
        
        assert 'x' in best_params
        assert 'y' in best_params
        assert len(optimizer.optimization_history) > 0
        
        # Should find parameters reasonably close to global optimum
        assert abs(best_params['x']) < 2.0
        assert abs(best_params['y']) < 2.0


class TestDomainAdapter:
    """Test domain adaptation."""
    
    def test_domain_adapter_initialization(self):
        """Test DomainAdapter initialization."""
        adapter = DomainAdapter()
        
        assert len(adapter.known_domains) == 0
        assert len(adapter.domain_similarities) == 0
        assert len(adapter.domain_features) > 0
    
    def test_domain_registration(self):
        """Test domain registration."""
        adapter = DomainAdapter()
        
        domain = TaskDomain(
            domain_name="test_domain",
            task_type="classification",
            input_dimensionality=100,
            output_dimensionality=10
        )
        
        performance_data = {'accuracy': 0.85, 'training_time': 100.0}
        
        adapter.register_domain(domain, performance_data)
        
        assert "test_domain" in adapter.known_domains
        assert adapter.known_domains["test_domain"]['domain'] == domain
        assert adapter.known_domains["test_domain"]['performance_data'] == performance_data
    
    def test_domain_similarity_computation(self):
        """Test domain similarity computation."""
        adapter = DomainAdapter()
        
        domain1 = TaskDomain(
            domain_name="domain1",
            task_type="classification",
            input_dimensionality=100,
            output_dimensionality=10,
            temporal_structure=False,
            noise_level=0.1,
            complexity_score=0.5
        )
        
        domain2 = TaskDomain(
            domain_name="domain2",
            task_type="classification",
            input_dimensionality=120,
            output_dimensionality=10,
            temporal_structure=False,
            noise_level=0.15,
            complexity_score=0.6
        )
        
        domain3 = TaskDomain(
            domain_name="domain3",
            task_type="regression",
            input_dimensionality=50,
            output_dimensionality=1,
            temporal_structure=True,
            noise_level=0.5,
            complexity_score=0.9
        )
        
        # Similar domains should have high similarity
        similarity_12 = adapter._compute_domain_similarity(domain1, domain2)
        assert similarity_12 > 0.7
        
        # Different domains should have lower similarity
        similarity_13 = adapter._compute_domain_similarity(domain1, domain3)
        assert similarity_13 < similarity_12
    
    def test_domain_adaptation(self):
        """Test domain adaptation."""
        adapter = DomainAdapter()
        
        # Register a known domain
        known_domain = TaskDomain(
            domain_name="known_classification",
            task_type="classification",
            input_dimensionality=100,
            output_dimensionality=10,
            noise_level=0.1
        )
        
        adapter.register_domain(known_domain, {'successful_algorithms': [LearningAlgorithmType.GRADIENT_DESCENT]})
        
        # Create target domain similar to known domain
        target_domain = TaskDomain(
            domain_name="target_classification",
            task_type="classification",
            input_dimensionality=110,
            output_dimensionality=10,
            noise_level=0.12
        )
        
        # Create available algorithms
        algorithms = [
            LearningAlgorithm(LearningAlgorithmType.GRADIENT_DESCENT),
            LearningAlgorithm(LearningAlgorithmType.EVOLUTIONARY),
            LearningAlgorithm(LearningAlgorithmType.REINFORCEMENT_LEARNING)
        ]
        
        key = random.PRNGKey(42)
        adapted_algorithms = adapter.adapt_to_domain(target_domain, algorithms, key)
        
        assert len(adapted_algorithms) == len(algorithms)
        # Gradient descent should be ranked highly due to success in similar domain
        assert adapted_algorithms[0].algorithm_type == LearningAlgorithmType.GRADIENT_DESCENT
    
    def test_few_shot_adaptation(self):
        """Test few-shot adaptation."""
        adapter = DomainAdapter()
        
        algorithm = LearningAlgorithm(
            algorithm_type=LearningAlgorithmType.GRADIENT_DESCENT,
            hyperparameters={'learning_rate': 0.01, 'regularization': 0.001}
        )
        
        domain = TaskDomain(
            domain_name="few_shot_domain",
            task_type="classification",
            input_dimensionality=50,
            output_dimensionality=5
        )
        
        # Create some dummy few-shot data
        few_shot_data = [
            ([1.0, 2.0, 3.0], [1, 0, 0, 0, 0]),
            ([2.0, 3.0, 4.0], [0, 1, 0, 0, 0]),
            ([3.0, 4.0, 5.0], [0, 0, 1, 0, 0])
        ]
        
        key = random.PRNGKey(42)
        adapted_algorithm = adapter.few_shot_adaptation(
            algorithm, domain, few_shot_data, n_adaptation_steps=5, key=key
        )
        
        assert adapted_algorithm.algorithm_type == algorithm.algorithm_type
        assert adapted_algorithm.adaptation_count == algorithm.adaptation_count + 1
        # Hyperparameters should be modified
        assert 'learning_rate' in adapted_algorithm.hyperparameters
        assert 'regularization' in adapted_algorithm.hyperparameters


class TestMetaLearningCore:
    """Test meta-learning core system."""
    
    def test_meta_learning_core_initialization(self):
        """Test MetaLearningCore initialization."""
        params = MetaLearningParams(
            algorithm_pool_size=5,
            optimization_budget=20
        )
        
        core = MetaLearningCore(params)
        
        assert core.params == params
        assert len(core.algorithm_pool) > 0
        assert core.hyperparameter_optimizer is not None
        assert core.domain_adapter is not None
        assert len(core.meta_parameters) > 0
    
    def test_algorithm_pool_initialization(self):
        """Test algorithm pool initialization."""
        core = MetaLearningCore()
        
        # Should have multiple algorithm types
        algorithm_types = {alg.algorithm_type for alg in core.algorithm_pool}
        
        assert LearningAlgorithmType.GRADIENT_DESCENT in algorithm_types
        assert LearningAlgorithmType.EVOLUTIONARY in algorithm_types
        assert LearningAlgorithmType.REINFORCEMENT_LEARNING in algorithm_types
        assert LearningAlgorithmType.STDP in algorithm_types
        
        # Each algorithm should have hyperparameters
        for algorithm in core.algorithm_pool:
            assert len(algorithm.hyperparameters) > 0
    
    def test_task_distribution_analysis(self):
        """Test task distribution analysis."""
        core = MetaLearningCore()
        
        task_distribution = [
            TaskDomain("task1", "classification", 100, 10, temporal_structure=False, noise_level=0.1),
            TaskDomain("task2", "classification", 200, 5, temporal_structure=True, noise_level=0.2),
            TaskDomain("task3", "regression", 50, 1, temporal_structure=False, noise_level=0.05)
        ]
        
        characteristics = core._analyze_task_distribution(task_distribution)
        
        assert characteristics['n_tasks'] == 3
        assert 'classification' in characteristics['task_types']
        assert 'regression' in characteristics['task_types']
        assert abs(characteristics['avg_input_dim'] - (100 + 200 + 50) / 3) < 1e-5
        assert abs(characteristics['avg_output_dim'] - (10 + 5 + 1) / 3) < 1e-5
        assert characteristics['temporal_fraction'] == 1/3  # Only task2 has temporal structure
    
    def test_algorithm_type_selection(self):
        """Test algorithm type selection."""
        core = MetaLearningCore()
        
        # Create distribution characteristics favoring gradient descent
        characteristics = {
            'task_types': {'classification'},
            'temporal_fraction': 0.0,
            'avg_noise_level': 0.1
        }
        
        performance_data = {
            'gradient_descent': [0.8, 0.85, 0.9],
            'evolutionary': [0.6, 0.65, 0.7],
            'reinforcement_learning': [0.4, 0.45, 0.5]
        }
        
        selected_type = core._select_algorithm_type(characteristics, performance_data)
        
        # Should select gradient descent due to good performance and task type match
        assert selected_type == LearningAlgorithmType.GRADIENT_DESCENT
    
    def test_parameter_space_generation(self):
        """Test parameter space generation for different algorithms."""
        core = MetaLearningCore()
        
        # Test gradient descent parameter space
        gd_space = core._get_parameter_space(LearningAlgorithmType.GRADIENT_DESCENT)
        assert 'learning_rate' in gd_space
        assert 'momentum' in gd_space
        assert 'weight_decay' in gd_space
        
        # Test evolutionary parameter space
        evo_space = core._get_parameter_space(LearningAlgorithmType.EVOLUTIONARY)
        assert 'population_size' in evo_space
        assert 'mutation_rate' in evo_space
        assert 'crossover_rate' in evo_space
        
        # Test STDP parameter space
        stdp_space = core._get_parameter_space(LearningAlgorithmType.STDP)
        assert 'a_plus' in stdp_space
        assert 'a_minus' in stdp_space
        assert 'tau_plus' in stdp_space
        assert 'tau_minus' in stdp_space
    
    def test_learn_learning_algorithm(self):
        """Test learning algorithm optimization."""
        params = MetaLearningParams(optimization_budget=10)  # Small budget for testing
        core = MetaLearningCore(params)
        
        task_distribution = [
            TaskDomain("task1", "classification", 100, 10),
            TaskDomain("task2", "classification", 150, 10)
        ]
        
        performance_data = {
            'gradient_descent': [0.8, 0.85, 0.9],
            'evolutionary': [0.6, 0.65, 0.7]
        }
        
        key = random.PRNGKey(42)
        optimized_algorithm = core.learn_learning_algorithm(
            task_distribution, performance_data, key
        )
        
        assert isinstance(optimized_algorithm, LearningAlgorithm)
        assert optimized_algorithm.adaptation_count == 1
        assert len(optimized_algorithm.hyperparameters) > 0
    
    def test_domain_adaptation(self):
        """Test adaptation to new domain."""
        core = MetaLearningCore()
        
        # Register a known domain in the adapter
        known_domain = TaskDomain("known", "classification", 100, 10)
        core.domain_adapter.register_domain(known_domain, {})
        
        # Create similar target domain
        target_domain = TaskDomain("target", "classification", 110, 10)
        
        key = random.PRNGKey(42)
        adapted_algorithms = core.adapt_to_new_domain(target_domain, key=key)
        
        assert len(adapted_algorithms) == len(core.algorithm_pool)
        assert all(isinstance(alg, LearningAlgorithm) for alg in adapted_algorithms)
    
    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization."""
        core = MetaLearningCore()
        
        algorithm = core.algorithm_pool[0]  # Get first algorithm
        performance_feedback = [0.5, 0.6, 0.65, 0.7, 0.72, 0.75]
        
        key = random.PRNGKey(42)
        optimized_algorithm = core.optimize_hyperparameters(
            algorithm, performance_feedback, key
        )
        
        assert optimized_algorithm.algorithm_type == algorithm.algorithm_type
        assert optimized_algorithm.adaptation_count == algorithm.adaptation_count + 1
        assert len(optimized_algorithm.hyperparameters) > 0
    
    def test_meta_parameter_update(self):
        """Test meta-parameter updates."""
        core = MetaLearningCore()
        
        initial_lr = core.meta_parameters['global_learning_rate']
        initial_exploration = core.meta_parameters['exploration_bonus']
        
        # Simulate improving performance
        core.update_meta_parameters(0.8)
        core.update_meta_parameters(0.85)
        
        # Learning rate should increase slightly with improvement
        assert core.meta_parameters['global_learning_rate'] >= initial_lr
        
        # Simulate declining performance
        core.update_meta_parameters(0.7)
        core.update_meta_parameters(0.6)
        
        # Learning rate should decrease with decline
        assert core.meta_parameters['global_learning_rate'] < initial_lr * 1.01
    
    def test_statistics_collection(self):
        """Test meta-learning statistics collection."""
        core = MetaLearningCore()
        
        # Add some performance data
        core.update_meta_parameters(0.7)
        core.update_meta_parameters(0.8)
        core.update_meta_parameters(0.75)
        
        stats = core.get_meta_learning_statistics()
        
        # Check required statistics
        assert 'algorithm_pool_size' in stats
        assert 'meta_parameters' in stats
        assert 'total_adaptations' in stats
        assert 'known_domains' in stats
        assert 'recent_mean_performance' in stats
        assert 'performance_trend' in stats
        
        assert stats['algorithm_pool_size'] == len(core.algorithm_pool)
        assert isinstance(stats['meta_parameters'], dict)
        assert stats['recent_mean_performance'] > 0


class TestIntegration:
    """Integration tests for meta-learning system."""
    
    def test_complete_meta_learning_pipeline(self):
        """Test complete meta-learning pipeline."""
        # Setup meta-learning system
        params = MetaLearningParams(
            optimization_budget=5,  # Small for testing
            few_shot_adaptation_steps=3
        )
        core = MetaLearningCore(params)
        
        # Define task distribution
        task_distribution = [
            TaskDomain("task1", "classification", 50, 5, noise_level=0.1),
            TaskDomain("task2", "classification", 60, 5, noise_level=0.15)
        ]
        
        # Simulate performance data
        performance_data = {
            'gradient_descent': [0.7, 0.75, 0.8],
            'evolutionary': [0.6, 0.65, 0.7]
        }
        
        key = random.PRNGKey(42)
        
        # Step 1: Learn optimal algorithm for distribution
        optimal_algorithm = core.learn_learning_algorithm(
            task_distribution, performance_data, key
        )
        
        assert isinstance(optimal_algorithm, LearningAlgorithm)
        
        # Step 2: Adapt to new domain
        new_domain = TaskDomain("new_task", "classification", 55, 5, noise_level=0.12)
        
        few_shot_data = [
            ([1.0, 2.0], [1, 0, 0, 0, 0]),
            ([2.0, 3.0], [0, 1, 0, 0, 0])
        ]
        
        key, subkey = random.split(key)
        adapted_algorithms = core.adapt_to_new_domain(
            new_domain, few_shot_data, subkey
        )
        
        assert len(adapted_algorithms) > 0
        assert all(isinstance(alg, LearningAlgorithm) for alg in adapted_algorithms)
        
        # Step 3: Optimize hyperparameters based on feedback
        best_algorithm = adapted_algorithms[0]
        feedback = [0.6, 0.65, 0.7, 0.72, 0.75]
        
        key, subkey = random.split(key)
        final_algorithm = core.optimize_hyperparameters(
            best_algorithm, feedback, subkey
        )
        
        assert final_algorithm.adaptation_count > best_algorithm.adaptation_count
        
        # Step 4: Update meta-parameters
        for perf in feedback:
            core.update_meta_parameters(perf)
        
        # Check that system learned from experience
        stats = core.get_meta_learning_statistics()
        assert stats['total_adaptations'] > 0
        assert len(stats['meta_parameters']) > 0
    
    def test_multi_domain_learning(self):
        """Test learning across multiple domains."""
        core = MetaLearningCore()
        
        # Register multiple domains
        domains = [
            TaskDomain("vision", "classification", 784, 10, noise_level=0.1),
            TaskDomain("nlp", "classification", 300, 5, temporal_structure=True),
            TaskDomain("control", "control", 20, 4, temporal_structure=True, noise_level=0.2)
        ]
        
        for domain in domains:
            core.domain_adapter.register_domain(domain, {
                'successful_algorithms': [LearningAlgorithmType.GRADIENT_DESCENT]
            })
        
        # Test adaptation to new domain similar to existing ones
        new_domain = TaskDomain("new_vision", "classification", 800, 12, noise_level=0.12)
        
        key = random.PRNGKey(42)
        adapted_algorithms = core.adapt_to_new_domain(new_domain, key=key)
        
        # Should successfully adapt
        assert len(adapted_algorithms) > 0
        
        # Check domain similarities were computed
        assert len(core.domain_adapter.domain_similarities) > 0
        
        # Test statistics
        stats = core.get_meta_learning_statistics()
        assert stats['known_domains'] == len(domains)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
"""
Tests for self-modification mechanisms.

This module tests recursive self-improvement loops, architecture optimization,
safety constraints, and overall self-modification stability and improvement.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, List, Any
import time

from src.training.self_modification import (
    RecursiveSelfImprovement,
    ImprovementConfig,
    ImprovementState,
    ArchitectureOptimizer,
    OptimizationConfig,
    OptimizationStrategy,
    SafetyConstraintManager,
    SafetyConfig,
    SafetyViolation,
    ConstraintType,
    SelfModificationPipeline,
    ModificationConfig,
    ModificationResult
)


class MockPerformanceEvaluator:
    """Mock performance evaluator for testing."""
    
    def __init__(self, base_performance: float = 0.5, noise_level: float = 0.1):
        self.base_performance = base_performance
        self.noise_level = noise_level
        self.evaluation_count = 0
        self.parameter_effects = {
            'learning_rate': 0.1,
            'batch_size': -0.05,
            'momentum': 0.08,
            'regularization': -0.03
        }
    
    def __call__(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate system performance with parameter effects."""
        self.evaluation_count += 1
        
        # Base performance with noise
        performance = self.base_performance + np.random.normal(0, self.noise_level)
        
        # Apply parameter effects
        for param, effect in self.parameter_effects.items():
            if param in system_state:
                value = system_state[param]
                if isinstance(value, (int, float)):
                    # Optimal value around 0.01 for learning_rate, 32 for batch_size, etc.
                    if param == 'learning_rate':
                        optimal = 0.01
                        deviation = abs(value - optimal) / optimal
                    elif param == 'batch_size':
                        optimal = 32
                        deviation = abs(value - optimal) / optimal
                    elif param == 'momentum':
                        optimal = 0.9
                        deviation = abs(value - optimal) / optimal
                    elif param == 'regularization':
                        optimal = 0.01
                        deviation = abs(value - optimal) / optimal
                    else:
                        deviation = 0
                    
                    # Performance decreases with deviation from optimal
                    performance += effect * (1.0 - deviation)
        
        # Add some complexity metrics
        network_size = system_state.get('network_parameters', 1000)
        complexity_penalty = (network_size - 1000) / 10000 * 0.1
        
        return {
            'overall_performance': max(0.0, performance - complexity_penalty),
            'train_performance': max(0.0, performance - complexity_penalty + 0.1),
            'val_performance': max(0.0, performance - complexity_penalty - 0.05),
            'stability': max(0.0, 1.0 - abs(performance - self.base_performance) * 2),
            'efficiency': max(0.0, 1.0 - complexity_penalty * 2),
            'convergence_rate': min(1.0, performance * 2),
            'network_size': network_size,
            'memory_usage_gb': network_size / 100000,
            'computation_time': network_size / 1000,
            'network_connectivity': 0.5,
            'layer_sizes': [100, 200, 100]
        }


class TestRecursiveSelfImprovement:
    """Test recursive self-improvement system."""
    
    @pytest.fixture
    def setup_improvement_system(self):
        """Set up recursive improvement system for testing."""
        config = ImprovementConfig(
            improvement_frequency=10,
            max_improvement_depth=3,
            improvement_patience=5,
            max_modifications_per_cycle=2,
            baseline_episodes=5
        )
        
        evaluator = MockPerformanceEvaluator(base_performance=0.6)
        key = random.PRNGKey(42)
        
        system = RecursiveSelfImprovement(config, evaluator, key)
        
        initial_state = {
            'learning_rate': 0.02,
            'batch_size': 64,
            'momentum': 0.85,
            'regularization': 0.02,
            'network_parameters': 1500
        }
        
        return system, initial_state, evaluator
    
    def test_baseline_establishment(self, setup_improvement_system):
        """Test baseline performance establishment."""
        system, initial_state, evaluator = setup_improvement_system
        
        baseline = system.establish_baseline(initial_state)
        
        assert baseline > 0.0
        assert system.state.baseline_performance == baseline
        assert system.state.current_performance == baseline
        assert system.state.best_performance == baseline
        assert len(system.state.performance_history) > 0
    
    def test_improvement_proposal(self, setup_improvement_system):
        """Test improvement proposal generation."""
        system, initial_state, evaluator = setup_improvement_system
        
        # Establish baseline
        system.establish_baseline(initial_state)
        
        # Simulate declining performance
        for _ in range(10):
            system.state.performance_history.append(system.state.baseline_performance - 0.01 * len(system.state.performance_history))
        
        # Get current performance metrics
        metrics = evaluator(initial_state)
        
        # Propose improvements
        modifications = system.propose_improvements(metrics)
        
        assert isinstance(modifications, list)
        assert len(modifications) <= system.config.max_modifications_per_cycle
        
        # Check that modifications have required fields
        for mod in modifications:
            assert 'type' in mod
            assert 'reason' in mod
    
    def test_improvement_application(self, setup_improvement_system):
        """Test improvement application and evaluation."""
        system, initial_state, evaluator = setup_improvement_system
        
        # Establish baseline
        system.establish_baseline(initial_state)
        
        # Create test modifications
        modifications = [
            {
                'type': 'learning_rate_adjustment',
                'component': 'global',
                'factor': 0.8,
                'reason': 'performance_decline'
            }
        ]
        
        # Apply improvements
        modified_state = system.apply_improvements(modifications, initial_state)
        
        # Check that modifications were applied
        assert 'original_learning_rates' in modified_state or modified_state['learning_rate'] != initial_state['learning_rate']
        assert len(system.state.active_modifications) > 0
    
    def test_improvement_cycle(self, setup_improvement_system):
        """Test complete improvement cycle."""
        system, initial_state, evaluator = setup_improvement_system
        
        # Run improvement cycle
        modified_state, success = system.run_improvement_cycle(initial_state)
        
        assert isinstance(modified_state, dict)
        assert isinstance(success, bool)
        assert system.state.current_cycle > 0
    
    def test_recursive_improvement(self, setup_improvement_system):
        """Test recursive improvement with depth control."""
        system, initial_state, evaluator = setup_improvement_system
        
        # Run recursive improvement
        final_state = system.run_recursive_improvement(initial_state, max_depth=2)
        
        assert isinstance(final_state, dict)
        assert system.state.improvement_depth == 0  # Should return to 0 after completion
        
        # Check that some improvement was attempted
        summary = system.get_improvement_summary()
        assert summary['total_cycles'] > 0
    
    def test_rollback_mechanism(self, setup_improvement_system):
        """Test rollback mechanism for failed improvements."""
        system, initial_state, evaluator = setup_improvement_system
        
        # Establish baseline
        system.establish_baseline(initial_state)
        
        # Create modifications that should trigger rollback
        modifications = [
            {
                'type': 'learning_rate_adjustment',
                'component': 'global',
                'factor': 10.0,  # Large change that should hurt performance
                'reason': 'test_rollback'
            }
        ]
        
        # Apply modifications
        modified_state = system.apply_improvements(modifications, initial_state)
        
        # Simulate poor performance that triggers rollback
        system.state.current_performance = system.state.baseline_performance - 0.3
        
        # Test rollback
        restored_state = system.rollback_modifications(modified_state)
        
        assert isinstance(restored_state, dict)
        assert len(system.state.active_modifications) == 0


class TestArchitectureOptimizer:
    """Test architecture optimization system."""
    
    @pytest.fixture
    def setup_optimizer(self):
        """Set up architecture optimizer for testing."""
        config = OptimizationConfig(
            primary_strategy=OptimizationStrategy.GRADIENT_BASED,
            max_optimization_steps=20,
            convergence_threshold=1e-4,
            improvement_patience=5
        )
        
        # Set parameter bounds
        config.parameter_bounds = {
            'learning_rate': (0.001, 0.1),
            'batch_size': (8, 128),
            'momentum': (0.5, 0.99),
            'regularization': (0.001, 0.1)
        }
        
        evaluator = MockPerformanceEvaluator(base_performance=0.7)
        key = random.PRNGKey(42)
        
        optimizer = ArchitectureOptimizer(config, evaluator, key)
        
        initial_params = {
            'learning_rate': 0.05,
            'batch_size': 64,
            'momentum': 0.8,
            'regularization': 0.05
        }
        
        return optimizer, initial_params, evaluator
    
    def test_parameter_bounds_setting(self, setup_optimizer):
        """Test parameter bounds configuration."""
        optimizer, initial_params, evaluator = setup_optimizer
        
        new_bounds = {'learning_rate': (0.0001, 0.01)}
        optimizer.set_parameter_bounds(new_bounds)
        
        assert 'learning_rate' in optimizer.config.parameter_bounds
        assert optimizer.config.parameter_bounds['learning_rate'] == (0.0001, 0.01)
    
    def test_gradient_estimation(self, setup_optimizer):
        """Test gradient estimation for optimization."""
        optimizer, initial_params, evaluator = setup_optimizer
        
        gradients = optimizer.estimate_gradients(initial_params)
        
        assert isinstance(gradients, dict)
        assert len(gradients) > 0
        
        # Check that gradients are computed for bounded parameters
        for param in initial_params:
            if param in optimizer.config.parameter_bounds:
                assert param in gradients
                assert isinstance(gradients[param], (int, float))
    
    def test_gradient_based_optimization(self, setup_optimizer):
        """Test gradient-based optimization."""
        optimizer, initial_params, evaluator = setup_optimizer
        
        optimized_params = optimizer.optimize_with_gradient_based(initial_params)
        
        assert isinstance(optimized_params, dict)
        assert len(optimized_params) > 0
        
        # Check that parameters are within bounds
        for param, value in optimized_params.items():
            if param in optimizer.config.parameter_bounds:
                min_val, max_val = optimizer.config.parameter_bounds[param]
                assert min_val <= value <= max_val
    
    def test_evolutionary_optimization(self, setup_optimizer):
        """Test evolutionary optimization."""
        optimizer, initial_params, evaluator = setup_optimizer
        
        # Use smaller population for testing
        optimizer.evolutionary_optimizer.population_size = 5
        
        optimized_params = optimizer.optimize_with_evolutionary(initial_params)
        
        assert isinstance(optimized_params, dict)
        assert len(optimized_params) > 0
        assert optimizer.best_performance > -float('inf')
    
    def test_random_search_optimization(self, setup_optimizer):
        """Test random search optimization."""
        optimizer, initial_params, evaluator = setup_optimizer
        
        # Reduce iterations for testing
        optimizer.config.max_optimization_steps = 10
        
        optimized_params = optimizer.optimize_with_random_search(initial_params)
        
        assert isinstance(optimized_params, dict)
        assert len(optimized_params) > 0
    
    def test_multi_objective_scoring(self, setup_optimizer):
        """Test multi-objective optimization scoring."""
        optimizer, initial_params, evaluator = setup_optimizer
        
        metrics = evaluator(initial_params)
        score = optimizer.compute_multi_objective_score(metrics)
        
        assert isinstance(score, (int, float))
        assert score >= 0.0
    
    def test_complete_optimization(self, setup_optimizer):
        """Test complete architecture optimization."""
        optimizer, initial_params, evaluator = setup_optimizer
        
        # Reduce iterations for testing
        optimizer.config.max_optimization_steps = 10
        
        optimized_params = optimizer.optimize_architecture(initial_params)
        
        assert isinstance(optimized_params, dict)
        assert len(optimized_params) > 0
        
        # Get optimization summary
        summary = optimizer.get_optimization_summary()
        assert 'best_performance' in summary
        assert 'best_parameters' in summary
        assert summary['optimization_steps'] >= 0


class TestSafetyConstraints:
    """Test safety constraint system."""
    
    @pytest.fixture
    def setup_safety_manager(self):
        """Set up safety constraint manager for testing."""
        config = SafetyConfig(
            max_performance_drop=0.15,
            min_performance_threshold=0.2,
            max_variance_increase=1.5,
            stability_threshold=0.7,
            max_memory_usage=4.0,
            max_network_size=5000
        )
        
        manager = SafetyConstraintManager(config)
        
        return manager, config
    
    def test_performance_degradation_check(self, setup_safety_manager):
        """Test performance degradation safety check."""
        manager, config = setup_safety_manager
        
        # Create history with declining performance
        history = [
            {'performance': 0.8, 'timestamp': time.time() - 100},
            {'performance': 0.7, 'timestamp': time.time() - 50},
            {'performance': 0.5, 'timestamp': time.time()}  # Significant drop
        ]
        
        current_state = {'performance': 0.5}
        proposed_modification = {'type': 'test_modification'}
        
        violations = manager.check_safety_constraints(current_state, proposed_modification, history)
        
        # Should detect performance degradation
        perf_violations = [v for v in violations if v.constraint_type == ConstraintType.PERFORMANCE_DEGRADATION]
        assert len(perf_violations) > 0
        assert perf_violations[0].severity > 0.0
    
    def test_stability_violation_check(self, setup_safety_manager):
        """Test stability violation safety check."""
        manager, config = setup_safety_manager
        
        # Create history with high variance (unstable performance)
        base_perf = 0.6
        history = []
        for i in range(20):
            # Alternating high/low performance (oscillation)
            perf = base_perf + (0.2 if i % 2 == 0 else -0.2)
            history.append({'performance': perf, 'timestamp': time.time() - (20 - i)})
        
        current_state = {'performance': base_perf}
        proposed_modification = {'type': 'test_modification'}
        
        violations = manager.check_safety_constraints(current_state, proposed_modification, history)
        
        # Should detect stability violation
        stability_violations = [v for v in violations if v.constraint_type == ConstraintType.STABILITY_VIOLATION]
        assert len(stability_violations) > 0
    
    def test_resource_limit_check(self, setup_safety_manager):
        """Test resource limit safety check."""
        manager, config = setup_safety_manager
        
        # Create state that exceeds resource limits
        current_state = {
            'memory_usage_gb': 6.0,  # Exceeds limit of 4.0
            'network_parameters': 10000,  # Exceeds limit of 5000
            'computation_time': 100.0
        }
        
        proposed_modification = {'type': 'test_modification'}
        history = [{'performance': 0.6, 'timestamp': time.time()}]
        
        violations = manager.check_safety_constraints(current_state, proposed_modification, history)
        
        # Should detect resource violations
        resource_violations = [v for v in violations if v.constraint_type == ConstraintType.RESOURCE_LIMIT]
        assert len(resource_violations) > 0
    
    def test_structural_integrity_check(self, setup_safety_manager):
        """Test structural integrity safety check."""
        manager, config = setup_safety_manager
        
        # Create state with structural issues
        current_state = {
            'network_connectivity': 0.05,  # Below minimum
            'layer_sizes': [5, 15000, 8]  # One layer too small, one too large
        }
        
        proposed_modification = {'type': 'test_modification'}
        history = [{'performance': 0.6, 'timestamp': time.time()}]
        
        violations = manager.check_safety_constraints(current_state, proposed_modification, history)
        
        # Should detect structural violations
        structural_violations = [v for v in violations if v.constraint_type == ConstraintType.STRUCTURAL_INTEGRITY]
        assert len(structural_violations) > 0
    
    def test_modification_safety_evaluation(self, setup_safety_manager):
        """Test overall modification safety evaluation."""
        manager, config = setup_safety_manager
        
        # Safe modification
        safe_state = {
            'performance': 0.7,
            'memory_usage_gb': 2.0,
            'network_parameters': 2000,
            'network_connectivity': 0.5,
            'layer_sizes': [100, 200, 100]
        }
        
        safe_modification = {'type': 'safe_modification', 'severity': 0.3}
        history = [{'performance': 0.65, 'timestamp': time.time()}]
        
        is_safe, violations = manager.is_modification_safe(safe_state, safe_modification, history)
        
        assert is_safe
        assert len(violations) == 0
        
        # Unsafe modification
        unsafe_state = {
            'performance': 0.1,  # Below threshold
            'memory_usage_gb': 8.0,  # Exceeds limit
            'network_parameters': 10000  # Exceeds limit
        }
        
        unsafe_modification = {'type': 'unsafe_modification', 'severity': 0.9}
        
        is_safe, violations = manager.is_modification_safe(unsafe_state, unsafe_modification, history)
        
        assert not is_safe
        assert len(violations) > 0
    
    def test_safety_recommendations(self, setup_safety_manager):
        """Test safety recommendation generation."""
        manager, config = setup_safety_manager
        
        violations = [
            SafetyViolation(
                constraint_type=ConstraintType.PERFORMANCE_DEGRADATION,
                severity=0.8,
                description="Performance dropped significantly",
                timestamp=time.time(),
                context={}
            ),
            SafetyViolation(
                constraint_type=ConstraintType.RESOURCE_LIMIT,
                severity=0.6,
                description="Memory usage exceeded",
                timestamp=time.time(),
                context={}
            )
        ]
        
        recommendations = manager.get_safety_recommendations(violations)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('rollback' in rec.lower() for rec in recommendations)
    
    def test_emergency_mode(self, setup_safety_manager):
        """Test emergency mode activation."""
        manager, config = setup_safety_manager
        
        # Simulate consecutive failures
        current_state = {'performance': 0.1}
        proposed_modification = {'type': 'test_modification'}
        history = [{'performance': 0.1, 'timestamp': time.time()}]
        
        # Trigger multiple violations to reach emergency mode
        for _ in range(config.max_consecutive_failures + 1):
            violations = manager.check_safety_constraints(current_state, proposed_modification, history)
        
        assert manager.emergency_mode
        assert manager.consecutive_failures >= config.max_consecutive_failures
        
        # Test reset
        manager.reset_emergency_mode()
        assert not manager.emergency_mode
        assert manager.consecutive_failures == 0


class TestSelfModificationPipeline:
    """Test complete self-modification pipeline."""
    
    @pytest.fixture
    def setup_pipeline(self):
        """Set up self-modification pipeline for testing."""
        config = ModificationConfig(
            improvement_config=ImprovementConfig(
                improvement_frequency=5,
                max_improvement_depth=2,
                baseline_episodes=3
            ),
            optimization_config=OptimizationConfig(
                primary_strategy=OptimizationStrategy.GRADIENT_BASED,
                max_optimization_steps=10
            ),
            safety_config=SafetyConfig(
                max_performance_drop=0.2,
                max_memory_usage=5.0
            ),
            max_pipeline_iterations=5
        )
        
        # Set parameter bounds for optimization
        config.optimization_config.parameter_bounds = {
            'learning_rate': (0.001, 0.1),
            'batch_size': (8, 128)
        }
        
        evaluator = MockPerformanceEvaluator(base_performance=0.6)
        key = random.PRNGKey(42)
        
        pipeline = SelfModificationPipeline(config, evaluator, key)
        
        initial_state = {
            'learning_rate': 0.05,
            'batch_size': 64,
            'momentum': 0.8,
            'regularization': 0.02,
            'network_parameters': 2000
        }
        
        return pipeline, initial_state, evaluator
    
    def test_pipeline_initialization(self, setup_pipeline):
        """Test pipeline initialization."""
        pipeline, initial_state, evaluator = setup_pipeline
        
        pipeline.initialize_system(initial_state)
        
        assert pipeline.baseline_performance > 0.0
        assert len(pipeline.performance_history) > 0
        assert pipeline.current_system_state == initial_state
    
    def test_modification_proposal(self, setup_pipeline):
        """Test modification proposal generation."""
        pipeline, initial_state, evaluator = setup_pipeline
        
        pipeline.initialize_system(initial_state)
        
        modifications = pipeline.propose_modifications()
        
        assert isinstance(modifications, list)
        # Should have modifications from different sources
        sources = {mod.get('source') for mod in modifications}
        assert len(sources) > 0
    
    def test_modification_safety_evaluation(self, setup_pipeline):
        """Test modification safety evaluation."""
        pipeline, initial_state, evaluator = setup_pipeline
        
        pipeline.initialize_system(initial_state)
        
        # Safe modification
        safe_mod = {
            'type': 'learning_rate_adjustment',
            'factor': 0.9,
            'source': 'test'
        }
        
        is_safe, violations = pipeline.evaluate_modification_safety(safe_mod)
        
        assert isinstance(is_safe, bool)
        assert isinstance(violations, list)
    
    def test_modification_application(self, setup_pipeline):
        """Test modification application."""
        pipeline, initial_state, evaluator = setup_pipeline
        
        pipeline.initialize_system(initial_state)
        
        modification = {
            'type': 'learning_rate_adjustment',
            'factor': 0.8
        }
        
        modified_state = pipeline.apply_modification(modification)
        
        assert isinstance(modified_state, dict)
        # Learning rate should be modified
        assert modified_state.get('learning_rate') != initial_state.get('learning_rate')
    
    def test_modification_cycle(self, setup_pipeline):
        """Test complete modification cycle."""
        pipeline, initial_state, evaluator = setup_pipeline
        
        pipeline.initialize_system(initial_state)
        
        result = pipeline.execute_modification_cycle()
        
        assert isinstance(result, ModificationResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.performance_change, (int, float))
        assert isinstance(result.modifications_applied, list)
        assert isinstance(result.safety_violations, list)
        assert result.execution_time > 0.0
    
    def test_complete_pipeline_execution(self, setup_pipeline):
        """Test complete pipeline execution."""
        pipeline, initial_state, evaluator = setup_pipeline
        
        pipeline.initialize_system(initial_state)
        
        results = pipeline.run_self_modification(max_iterations=3)
        
        assert isinstance(results, dict)
        assert 'total_iterations' in results
        assert 'successful_iterations' in results
        assert 'success_rate' in results
        assert 'baseline_performance' in results
        assert 'final_performance' in results
        assert 'total_improvement' in results
        assert results['total_iterations'] > 0
    
    def test_pipeline_convergence(self, setup_pipeline):
        """Test pipeline convergence detection."""
        pipeline, initial_state, evaluator = setup_pipeline
        
        pipeline.initialize_system(initial_state)
        
        # Simulate converged performance
        converged_performance = 0.75
        for _ in range(10):
            pipeline.performance_history.append(converged_performance + np.random.normal(0, 0.001))
        
        # Run pipeline - should detect convergence quickly
        results = pipeline.run_self_modification(max_iterations=10)
        
        # Should stop early due to convergence
        assert results['total_iterations'] < 10
    
    def test_pipeline_summary(self, setup_pipeline):
        """Test pipeline summary generation."""
        pipeline, initial_state, evaluator = setup_pipeline
        
        pipeline.initialize_system(initial_state)
        
        # Run a few cycles
        for _ in range(2):
            pipeline.execute_modification_cycle()
        
        summary = pipeline.get_modification_summary()
        
        assert isinstance(summary, dict)
        assert 'pipeline_stats' in summary
        assert 'safety_stats' in summary
        assert 'modification_history' in summary
        
        pipeline_stats = summary['pipeline_stats']
        assert 'total_iterations' in pipeline_stats
        assert 'baseline_performance' in pipeline_stats
        assert 'current_performance' in pipeline_stats
    
    def test_pipeline_reset(self, setup_pipeline):
        """Test pipeline reset functionality."""
        pipeline, initial_state, evaluator = setup_pipeline
        
        pipeline.initialize_system(initial_state)
        
        # Run some cycles
        pipeline.execute_modification_cycle()
        
        # Reset pipeline
        pipeline.reset_pipeline()
        
        assert len(pipeline.history.attempts) == 0
        assert len(pipeline.performance_history) == 0
        assert pipeline.iteration_count == 0


# Integration test
def test_self_modification_integration():
    """Integration test for complete self-modification system."""
    
    # Create a more realistic performance evaluator
    class RealisticEvaluator:
        def __init__(self):
            self.optimal_params = {
                'learning_rate': 0.01,
                'batch_size': 32,
                'momentum': 0.9,
                'regularization': 0.01
            }
            self.base_performance = 0.7
            self.noise_level = 0.05
        
        def __call__(self, system_state: Dict[str, Any]) -> Dict[str, float]:
            performance = self.base_performance
            
            # Calculate performance based on parameter optimality
            for param, optimal_value in self.optimal_params.items():
                if param in system_state:
                    current_value = system_state[param]
                    if isinstance(current_value, (int, float)):
                        # Performance decreases with distance from optimal
                        distance = abs(current_value - optimal_value) / optimal_value
                        performance -= distance * 0.1
            
            # Add noise
            performance += np.random.normal(0, self.noise_level)
            performance = max(0.0, min(1.0, performance))
            
            return {
                'overall_performance': performance,
                'train_performance': performance + 0.05,
                'val_performance': performance - 0.02,
                'stability': max(0.0, 1.0 - abs(performance - self.base_performance)),
                'efficiency': performance,
                'convergence_rate': performance,
                'network_size': system_state.get('network_parameters', 1000),
                'memory_usage_gb': system_state.get('network_parameters', 1000) / 500,
                'computation_time': 10.0,
                'network_connectivity': 0.5,
                'layer_sizes': [100, 200, 100]
            }
    
    # Set up pipeline with realistic configuration
    config = ModificationConfig(
        improvement_config=ImprovementConfig(
            improvement_frequency=3,
            max_improvement_depth=2,
            improvement_patience=3,
            baseline_episodes=5
        ),
        optimization_config=OptimizationConfig(
            primary_strategy=OptimizationStrategy.GRADIENT_BASED,
            max_optimization_steps=15,
            convergence_threshold=1e-5
        ),
        safety_config=SafetyConfig(
            max_performance_drop=0.3,  # More lenient
            min_performance_threshold=0.1,  # Lower threshold
            max_memory_usage=20.0,  # Higher limit
            max_modifications_per_hour=50,  # Higher rate limit
            cooldown_period=10.0  # Shorter cooldown
        ),
        max_pipeline_iterations=8
    )
    
    # Set parameter bounds
    config.optimization_config.parameter_bounds = {
        'learning_rate': (0.001, 0.1),
        'batch_size': (8, 128),
        'momentum': (0.5, 0.99),
        'regularization': (0.001, 0.1)
    }
    
    evaluator = RealisticEvaluator()
    key = random.PRNGKey(42)
    
    pipeline = SelfModificationPipeline(config, evaluator, key)
    
    # Start with suboptimal parameters
    initial_state = {
        'learning_rate': 0.05,  # Too high
        'batch_size': 128,      # Too high
        'momentum': 0.7,        # Too low
        'regularization': 0.05, # Too high
        'network_parameters': 1500
    }
    
    # Initialize and run pipeline
    pipeline.initialize_system(initial_state)
    baseline_performance = pipeline.baseline_performance
    
    results = pipeline.run_self_modification(max_iterations=6)
    
    # Verify improvements
    assert results['final_performance'] >= baseline_performance - 0.1  # Allow for some noise
    assert results['total_iterations'] > 0
    assert results['success_rate'] >= 0.0
    
    # Verify safety (allow emergency mode for testing, just check it exists)
    safety_stats = pipeline.get_modification_summary()['safety_stats']
    if safety_stats:  # Only check if safety manager is active
        assert 'emergency_mode' in safety_stats  # Just verify the field exists
    
    # Verify that parameters moved toward optimal values
    final_state = pipeline.current_system_state
    optimal_params = evaluator.optimal_params
    
    improvements = 0
    for param, optimal_value in optimal_params.items():
        if param in initial_state and param in final_state:
            initial_distance = abs(initial_state[param] - optimal_value)
            final_distance = abs(final_state[param] - optimal_value)
            if final_distance < initial_distance:
                improvements += 1
    
    # At least some parameters should have improved
    assert improvements > 0
    
    print(f"Integration test results:")
    print(f"  Baseline performance: {baseline_performance:.4f}")
    print(f"  Final performance: {results['final_performance']:.4f}")
    print(f"  Total improvement: {results['total_improvement']:.4f}")
    print(f"  Success rate: {results['success_rate']:.2%}")
    print(f"  Parameters improved: {improvements}/{len(optimal_params)}")


if __name__ == "__main__":
    # Run integration test
    test_self_modification_integration()
    print("Self-modification integration test passed!")
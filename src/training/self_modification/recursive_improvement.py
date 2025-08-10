"""
Recursive Self-Improvement System

This module implements recursive self-improvement loops that allow the system
to continuously optimize its own architecture and learning algorithms based
on performance feedback.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Tuple, Optional, Callable, Any, NamedTuple
import numpy as np
from dataclasses import dataclass, field
import time
from collections import deque
from abc import ABC, abstractmethod


@dataclass
class ImprovementConfig:
    """Configuration for recursive self-improvement."""
    
    # Improvement cycle parameters
    improvement_frequency: int = 100        # Steps between improvement cycles
    max_improvement_depth: int = 5          # Maximum recursion depth
    improvement_patience: int = 10          # Cycles to wait for improvement
    
    # Performance tracking
    performance_window: int = 50            # Window for performance averaging
    improvement_threshold: float = 0.01     # Minimum improvement to continue
    baseline_episodes: int = 20             # Episodes to establish baseline
    
    # Modification parameters
    max_modifications_per_cycle: int = 3    # Maximum modifications per cycle
    modification_strength: float = 0.1      # Strength of modifications
    rollback_threshold: float = -0.05       # Performance drop threshold for rollback
    
    # Learning algorithm adaptation
    adapt_learning_rates: bool = True       # Whether to adapt learning rates
    adapt_network_structure: bool = True    # Whether to modify network structure
    adapt_hyperparameters: bool = True      # Whether to optimize hyperparameters
    
    # Safety and stability
    stability_check_frequency: int = 10     # Frequency of stability checks
    max_performance_variance: float = 0.2   # Maximum allowed performance variance
    emergency_rollback_enabled: bool = True # Enable emergency rollbacks


class ImprovementMetrics(NamedTuple):
    """Metrics for tracking improvement progress."""
    cycle: int
    performance_before: float
    performance_after: float
    improvement_ratio: float
    modifications_applied: int
    stability_score: float
    convergence_score: float


@dataclass
class ImprovementState:
    """State of the recursive improvement system."""
    
    current_cycle: int = 0
    improvement_depth: int = 0
    total_improvements: int = 0
    
    # Performance tracking
    performance_history: deque = field(default_factory=lambda: deque(maxlen=100))
    baseline_performance: float = 0.0
    current_performance: float = 0.0
    best_performance: float = 0.0
    
    # Modification tracking
    active_modifications: List[Dict] = field(default_factory=list)
    successful_modifications: List[Dict] = field(default_factory=list)
    failed_modifications: List[Dict] = field(default_factory=list)
    
    # State management
    last_improvement_cycle: int = 0
    consecutive_failures: int = 0
    stability_violations: int = 0
    
    # Checkpoints for rollback
    checkpoints: Dict[str, Any] = field(default_factory=dict)


class ModificationStrategy(ABC):
    """Abstract base class for modification strategies."""
    
    @abstractmethod
    def propose_modifications(self, 
                            current_state: ImprovementState,
                            performance_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Propose modifications based on current state and performance."""
        pass
    
    @abstractmethod
    def apply_modification(self, 
                         modification: Dict[str, Any],
                         system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific modification to the system."""
        pass
    
    @abstractmethod
    def rollback_modification(self, 
                            modification: Dict[str, Any],
                            system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback a modification."""
        pass


class LearningRateAdaptation(ModificationStrategy):
    """Strategy for adapting learning rates based on performance."""
    
    def __init__(self, adaptation_factor: float = 0.1):
        self.adaptation_factor = adaptation_factor
    
    def propose_modifications(self, 
                            current_state: ImprovementState,
                            performance_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Propose learning rate modifications."""
        modifications = []
        
        # Analyze performance trend
        if len(current_state.performance_history) >= 10:
            recent_performance = list(current_state.performance_history)[-10:]
            trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            
            if trend < 0:  # Performance declining
                # Reduce learning rate
                modifications.append({
                    'type': 'learning_rate_adjustment',
                    'component': 'global',
                    'factor': 1.0 - self.adaptation_factor,
                    'reason': 'performance_decline'
                })
            elif trend > 0.01:  # Performance improving rapidly
                # Increase learning rate slightly
                modifications.append({
                    'type': 'learning_rate_adjustment',
                    'component': 'global',
                    'factor': 1.0 + self.adaptation_factor * 0.5,
                    'reason': 'performance_improvement'
                })
        
        return modifications
    
    def apply_modification(self, 
                         modification: Dict[str, Any],
                         system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learning rate modification."""
        if modification['type'] == 'learning_rate_adjustment':
            component = modification['component']
            factor = modification['factor']
            
            # Store original value for rollback
            if 'original_learning_rates' not in system_state:
                system_state['original_learning_rates'] = {}
            
            if component == 'global':
                # Modify all learning rates
                for key in system_state.keys():
                    if 'learning_rate' in key:
                        system_state['original_learning_rates'][key] = system_state[key]
                        system_state[key] *= factor
        
        return system_state
    
    def rollback_modification(self, 
                            modification: Dict[str, Any],
                            system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback learning rate modification."""
        if modification['type'] == 'learning_rate_adjustment':
            if 'original_learning_rates' in system_state:
                for key, original_value in system_state['original_learning_rates'].items():
                    system_state[key] = original_value
                del system_state['original_learning_rates']
        
        return system_state


class NetworkStructureAdaptation(ModificationStrategy):
    """Strategy for adapting network structure."""
    
    def __init__(self, max_size_change: float = 0.2):
        self.max_size_change = max_size_change
    
    def propose_modifications(self, 
                            current_state: ImprovementState,
                            performance_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Propose network structure modifications."""
        modifications = []
        
        # Check if performance is stagnating
        if current_state.consecutive_failures > 3:
            # Try structural changes
            modifications.extend([
                {
                    'type': 'add_neurons',
                    'layer': 'hidden',
                    'count': int(performance_metrics.get('network_size', 100) * 0.1),
                    'reason': 'performance_stagnation'
                },
                {
                    'type': 'add_connections',
                    'density_increase': 0.05,
                    'reason': 'increase_capacity'
                }
            ])
        
        # Check for overfitting indicators
        train_performance = performance_metrics.get('train_performance', 0.0)
        val_performance = performance_metrics.get('val_performance', 0.0)
        
        if train_performance - val_performance > 0.1:  # Overfitting
            modifications.append({
                'type': 'prune_connections',
                'pruning_ratio': 0.1,
                'reason': 'reduce_overfitting'
            })
        
        return modifications
    
    def apply_modification(self, 
                         modification: Dict[str, Any],
                         system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply network structure modification."""
        mod_type = modification['type']
        
        if mod_type == 'add_neurons':
            # Store modification for potential rollback
            system_state['structure_modifications'] = system_state.get('structure_modifications', [])
            system_state['structure_modifications'].append(modification)
            
        elif mod_type == 'prune_connections':
            # Store original connections for rollback
            system_state['pruned_connections'] = modification
        
        return system_state
    
    def rollback_modification(self, 
                            modification: Dict[str, Any],
                            system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback network structure modification."""
        mod_type = modification['type']
        
        if mod_type in ['add_neurons', 'add_connections']:
            # Remove added components
            if 'structure_modifications' in system_state:
                system_state['structure_modifications'] = [
                    m for m in system_state['structure_modifications'] 
                    if m != modification
                ]
        
        elif mod_type == 'prune_connections':
            # Restore pruned connections
            if 'pruned_connections' in system_state:
                del system_state['pruned_connections']
        
        return system_state


class HyperparameterOptimization(ModificationStrategy):
    """Strategy for optimizing hyperparameters."""
    
    def __init__(self, optimization_strength: float = 0.1):
        self.optimization_strength = optimization_strength
    
    def propose_modifications(self, 
                            current_state: ImprovementState,
                            performance_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Propose hyperparameter modifications."""
        modifications = []
        
        # Analyze performance metrics to suggest hyperparameter changes
        convergence_rate = performance_metrics.get('convergence_rate', 0.0)
        stability = performance_metrics.get('stability', 1.0)
        
        if convergence_rate < 0.01:  # Slow convergence
            modifications.extend([
                {
                    'type': 'adjust_batch_size',
                    'factor': 1.2,
                    'reason': 'improve_convergence'
                },
                {
                    'type': 'adjust_momentum',
                    'factor': 1.1,
                    'reason': 'accelerate_learning'
                }
            ])
        
        if stability < 0.8:  # Unstable training
            modifications.extend([
                {
                    'type': 'adjust_regularization',
                    'factor': 1.2,
                    'reason': 'improve_stability'
                },
                {
                    'type': 'adjust_batch_size',
                    'factor': 0.8,
                    'reason': 'reduce_noise'
                }
            ])
        
        return modifications
    
    def apply_modification(self, 
                         modification: Dict[str, Any],
                         system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hyperparameter modification."""
        mod_type = modification['type']
        factor = modification['factor']
        
        # Store original values for rollback
        if 'original_hyperparameters' not in system_state:
            system_state['original_hyperparameters'] = {}
        
        if mod_type == 'adjust_batch_size':
            original_batch_size = system_state.get('batch_size', 32)
            system_state['original_hyperparameters']['batch_size'] = original_batch_size
            system_state['batch_size'] = int(original_batch_size * factor)
            
        elif mod_type == 'adjust_momentum':
            original_momentum = system_state.get('momentum', 0.9)
            system_state['original_hyperparameters']['momentum'] = original_momentum
            system_state['momentum'] = min(0.99, original_momentum * factor)
            
        elif mod_type == 'adjust_regularization':
            original_reg = system_state.get('regularization', 0.01)
            system_state['original_hyperparameters']['regularization'] = original_reg
            system_state['regularization'] = original_reg * factor
        
        return system_state
    
    def rollback_modification(self, 
                            modification: Dict[str, Any],
                            system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback hyperparameter modification."""
        if 'original_hyperparameters' in system_state:
            for param, value in system_state['original_hyperparameters'].items():
                system_state[param] = value
            del system_state['original_hyperparameters']
        
        return system_state


class RecursiveSelfImprovement:
    """
    Recursive Self-Improvement System
    
    Implements recursive loops for continuous system optimization based on
    performance feedback with safety constraints and rollback mechanisms.
    """
    
    def __init__(self, 
                 config: ImprovementConfig,
                 performance_evaluator: Callable[[Dict], Dict[str, float]],
                 key: jax.random.PRNGKey):
        self.config = config
        self.performance_evaluator = performance_evaluator
        self.key = key
        
        # Initialize state
        self.state = ImprovementState()
        
        # Initialize modification strategies
        self.strategies = {
            'learning_rate': LearningRateAdaptation(),
            'network_structure': NetworkStructureAdaptation(),
            'hyperparameters': HyperparameterOptimization()
        }
        
        # Performance tracking
        self.improvement_history = []
        self.rollback_count = 0
        
    def establish_baseline(self, system_state: Dict[str, Any]) -> float:
        """Establish baseline performance for comparison."""
        print("Establishing baseline performance...")
        
        performance_scores = []
        for episode in range(self.config.baseline_episodes):
            metrics = self.performance_evaluator(system_state)
            performance_scores.append(metrics.get('overall_performance', 0.0))
        
        baseline = float(np.mean(performance_scores))
        self.state.baseline_performance = baseline
        self.state.current_performance = baseline
        self.state.best_performance = baseline
        
        # Populate performance history with baseline scores
        for score in performance_scores:
            self.state.performance_history.append(score)
        
        print(f"Baseline performance established: {baseline:.4f}")
        return baseline
    
    def evaluate_current_performance(self, system_state: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate current system performance."""
        metrics = self.performance_evaluator(system_state)
        
        # Update performance tracking
        current_perf = metrics.get('overall_performance', 0.0)
        self.state.performance_history.append(current_perf)
        self.state.current_performance = current_perf
        
        if current_perf > self.state.best_performance:
            self.state.best_performance = current_perf
        
        return metrics
    
    def propose_improvements(self, 
                           performance_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Propose improvements based on current performance."""
        all_modifications = []
        
        # Get proposals from each strategy
        if self.config.adapt_learning_rates:
            lr_mods = self.strategies['learning_rate'].propose_modifications(
                self.state, performance_metrics
            )
            all_modifications.extend(lr_mods)
        
        if self.config.adapt_network_structure:
            struct_mods = self.strategies['network_structure'].propose_modifications(
                self.state, performance_metrics
            )
            all_modifications.extend(struct_mods)
        
        if self.config.adapt_hyperparameters:
            hyper_mods = self.strategies['hyperparameters'].propose_modifications(
                self.state, performance_metrics
            )
            all_modifications.extend(hyper_mods)
        
        # Limit number of modifications per cycle
        if len(all_modifications) > self.config.max_modifications_per_cycle:
            # Prioritize modifications (simple heuristic)
            priority_order = ['learning_rate_adjustment', 'adjust_regularization', 'prune_connections']
            prioritized = []
            
            for mod_type in priority_order:
                for mod in all_modifications:
                    if mod['type'] == mod_type and len(prioritized) < self.config.max_modifications_per_cycle:
                        prioritized.append(mod)
            
            # Add remaining modifications up to limit
            for mod in all_modifications:
                if mod not in prioritized and len(prioritized) < self.config.max_modifications_per_cycle:
                    prioritized.append(mod)
            
            all_modifications = prioritized
        
        return all_modifications
    
    def apply_improvements(self, 
                         modifications: List[Dict[str, Any]],
                         system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply proposed improvements to the system."""
        print(f"Applying {len(modifications)} modifications...")
        
        # Create checkpoint for rollback
        checkpoint_id = f"cycle_{self.state.current_cycle}"
        self.state.checkpoints[checkpoint_id] = {
            'system_state': system_state.copy(),
            'performance': self.state.current_performance,
            'modifications': modifications.copy()
        }
        
        # Apply modifications
        modified_state = system_state.copy()
        applied_modifications = []
        
        for modification in modifications:
            try:
                strategy_name = self._get_strategy_for_modification(modification)
                if strategy_name in self.strategies:
                    modified_state = self.strategies[strategy_name].apply_modification(
                        modification, modified_state
                    )
                    applied_modifications.append(modification)
                    print(f"Applied: {modification['type']} - {modification.get('reason', 'N/A')}")
            except Exception as e:
                print(f"Failed to apply modification {modification['type']}: {e}")
        
        self.state.active_modifications = applied_modifications
        return modified_state
    
    def evaluate_improvements(self, 
                            system_state: Dict[str, Any],
                            pre_modification_performance: float) -> bool:
        """Evaluate whether improvements were successful."""
        # Evaluate performance after modifications
        post_metrics = self.evaluate_current_performance(system_state)
        post_performance = post_metrics.get('overall_performance', 0.0)
        
        # Calculate improvement
        improvement = post_performance - pre_modification_performance
        improvement_ratio = improvement / max(abs(pre_modification_performance), 1e-8)
        
        # Create improvement metrics
        metrics = ImprovementMetrics(
            cycle=self.state.current_cycle,
            performance_before=pre_modification_performance,
            performance_after=post_performance,
            improvement_ratio=improvement_ratio,
            modifications_applied=len(self.state.active_modifications),
            stability_score=self._compute_stability_score(),
            convergence_score=self._compute_convergence_score()
        )
        
        self.improvement_history.append(metrics)
        
        # Determine if improvements were successful
        success = improvement_ratio > self.config.improvement_threshold
        
        if success:
            self.state.successful_modifications.extend(self.state.active_modifications)
            self.state.last_improvement_cycle = self.state.current_cycle
            self.state.consecutive_failures = 0
            self.state.total_improvements += 1
            print(f"Improvements successful! Performance: {pre_modification_performance:.4f} → {post_performance:.4f}")
        else:
            self.state.failed_modifications.extend(self.state.active_modifications)
            self.state.consecutive_failures += 1
            print(f"Improvements failed. Performance: {pre_modification_performance:.4f} → {post_performance:.4f}")
        
        return success
    
    def rollback_modifications(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback recent modifications if they were unsuccessful."""
        print("Rolling back modifications...")
        self.rollback_count += 1
        
        # Get latest checkpoint
        checkpoint_id = f"cycle_{self.state.current_cycle}"
        if checkpoint_id in self.state.checkpoints:
            checkpoint = self.state.checkpoints[checkpoint_id]
            
            # Restore system state
            restored_state = checkpoint['system_state'].copy()
            
            # Apply rollbacks for each modification
            for modification in reversed(self.state.active_modifications):
                try:
                    strategy_name = self._get_strategy_for_modification(modification)
                    if strategy_name in self.strategies:
                        restored_state = self.strategies[strategy_name].rollback_modification(
                            modification, restored_state
                        )
                except Exception as e:
                    print(f"Failed to rollback modification {modification['type']}: {e}")
            
            # Clean up
            del self.state.checkpoints[checkpoint_id]
            self.state.active_modifications = []
            
            print("Rollback completed")
            return restored_state
        
        return system_state
    
    def run_improvement_cycle(self, system_state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Run one complete improvement cycle."""
        print(f"\n=== Improvement Cycle {self.state.current_cycle} ===")
        
        # Evaluate current performance
        pre_metrics = self.evaluate_current_performance(system_state)
        pre_performance = pre_metrics.get('overall_performance', 0.0)
        
        # Check if improvement is needed
        if (self.state.current_cycle - self.state.last_improvement_cycle > 
            self.config.improvement_patience):
            print("No recent improvements, proposing modifications...")
            
            # Propose improvements
            modifications = self.propose_improvements(pre_metrics)
            
            if modifications:
                # Apply improvements
                modified_state = self.apply_improvements(modifications, system_state)
                
                # Evaluate improvements
                success = self.evaluate_improvements(modified_state, pre_performance)
                
                if success:
                    self.state.current_cycle += 1
                    return modified_state, True
                else:
                    # Rollback if performance degraded significantly
                    if (self.state.current_performance - pre_performance < 
                        self.config.rollback_threshold):
                        restored_state = self.rollback_modifications(system_state)
                        self.state.current_cycle += 1
                        return restored_state, False
                    else:
                        # Keep modifications even if no improvement
                        self.state.current_cycle += 1
                        return modified_state, False
            else:
                print("No modifications proposed")
        else:
            print("Recent improvements detected, skipping modification")
        
        self.state.current_cycle += 1
        return system_state, False
    
    def run_recursive_improvement(self, 
                                system_state: Dict[str, Any],
                                max_depth: Optional[int] = None) -> Dict[str, Any]:
        """Run recursive improvement with specified depth."""
        if max_depth is None:
            max_depth = self.config.max_improvement_depth
        
        if self.state.improvement_depth >= max_depth:
            print(f"Maximum improvement depth {max_depth} reached")
            return system_state
        
        print(f"Starting recursive improvement (depth {self.state.improvement_depth + 1}/{max_depth})")
        
        # Establish baseline if not done
        if self.state.baseline_performance == 0.0:
            self.establish_baseline(system_state)
        
        current_state = system_state
        improvement_made = False
        
        # Run improvement cycle
        self.state.improvement_depth += 1
        try:
            current_state, success = self.run_improvement_cycle(current_state)
            improvement_made = success
            
            # If improvement was made, recursively try to improve further
            if improvement_made and self.state.improvement_depth < max_depth:
                print("Improvement made, attempting recursive improvement...")
                current_state = self.run_recursive_improvement(current_state, max_depth)
        
        finally:
            self.state.improvement_depth -= 1
        
        return current_state
    
    def _get_strategy_for_modification(self, modification: Dict[str, Any]) -> str:
        """Get the strategy name for a given modification type."""
        mod_type = modification['type']
        
        if mod_type in ['learning_rate_adjustment']:
            return 'learning_rate'
        elif mod_type in ['add_neurons', 'add_connections', 'prune_connections']:
            return 'network_structure'
        elif mod_type in ['adjust_batch_size', 'adjust_momentum', 'adjust_regularization']:
            return 'hyperparameters'
        else:
            return 'unknown'
    
    def _compute_stability_score(self) -> float:
        """Compute stability score based on performance variance."""
        if len(self.state.performance_history) < 10:
            return 1.0
        
        recent_performance = list(self.state.performance_history)[-10:]
        variance = float(np.var(recent_performance))
        
        # Convert variance to stability score (lower variance = higher stability)
        stability = 1.0 / (1.0 + variance * 10)
        return stability
    
    def _compute_convergence_score(self) -> float:
        """Compute convergence score based on performance trend."""
        if len(self.state.performance_history) < 5:
            return 0.0
        
        recent_performance = list(self.state.performance_history)[-5:]
        
        # Compute trend (slope)
        x = np.arange(len(recent_performance))
        trend = np.polyfit(x, recent_performance, 1)[0]
        
        # Convert to convergence score (positive trend = good convergence)
        convergence = max(0.0, min(1.0, trend * 10 + 0.5))
        return float(convergence)
    
    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of improvement process."""
        return {
            'total_cycles': self.state.current_cycle,
            'total_improvements': self.state.total_improvements,
            'successful_modifications': len(self.state.successful_modifications),
            'failed_modifications': len(self.state.failed_modifications),
            'rollback_count': self.rollback_count,
            'baseline_performance': self.state.baseline_performance,
            'current_performance': self.state.current_performance,
            'best_performance': self.state.best_performance,
            'improvement_ratio': (self.state.current_performance - self.state.baseline_performance) / 
                               max(abs(self.state.baseline_performance), 1e-8),
            'stability_score': self._compute_stability_score(),
            'convergence_score': self._compute_convergence_score()
        }
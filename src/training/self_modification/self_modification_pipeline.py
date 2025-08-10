"""
Self-Modification Pipeline

This module integrates recursive self-improvement, architecture optimization,
and safety constraints into a unified self-modification system.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Tuple, Optional, Callable, Any, NamedTuple
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque

from .recursive_improvement import (
    RecursiveSelfImprovement, 
    ImprovementConfig, 
    ImprovementState,
    ImprovementMetrics
)
from .architecture_optimizer import (
    ArchitectureOptimizer, 
    OptimizationConfig, 
    OptimizationStrategy
)
from .safety_constraints import (
    SafetyConstraintManager, 
    SafetyConfig, 
    SafetyViolation,
    ConstraintType
)


class ModificationResult(NamedTuple):
    """Result of a self-modification attempt."""
    success: bool
    performance_change: float
    modifications_applied: List[Dict[str, Any]]
    safety_violations: List[SafetyViolation]
    execution_time: float
    rollback_performed: bool


@dataclass
class ModificationConfig:
    """Configuration for the self-modification pipeline."""
    
    # Component configurations
    improvement_config: ImprovementConfig = field(default_factory=ImprovementConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    safety_config: SafetyConfig = field(default_factory=SafetyConfig)
    
    # Pipeline settings
    enable_recursive_improvement: bool = True
    enable_architecture_optimization: bool = True
    enable_safety_constraints: bool = True
    
    # Execution settings
    max_pipeline_iterations: int = 10
    convergence_threshold: float = 1e-4
    stability_requirement: float = 0.8
    
    # Logging and monitoring
    log_all_attempts: bool = True
    save_checkpoints: bool = True
    checkpoint_frequency: int = 5


@dataclass
class ModificationHistory:
    """History of self-modification attempts."""
    
    attempts: List[ModificationResult] = field(default_factory=list)
    successful_modifications: List[Dict[str, Any]] = field(default_factory=list)
    failed_modifications: List[Dict[str, Any]] = field(default_factory=list)
    performance_timeline: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, performance)
    safety_violations: List[SafetyViolation] = field(default_factory=list)
    
    def add_result(self, result: ModificationResult, timestamp: float, performance: float) -> None:
        """Add a modification result to history."""
        self.attempts.append(result)
        self.performance_timeline.append((timestamp, performance))
        
        if result.success:
            self.successful_modifications.extend(result.modifications_applied)
        else:
            self.failed_modifications.extend(result.modifications_applied)
        
        self.safety_violations.extend(result.safety_violations)
    
    def get_success_rate(self, window: int = 10) -> float:
        """Get success rate over recent attempts."""
        if not self.attempts:
            return 0.0
        
        recent_attempts = self.attempts[-window:]
        successful = sum(1 for attempt in recent_attempts if attempt.success)
        return successful / len(recent_attempts)
    
    def get_performance_trend(self, window: int = 20) -> float:
        """Get performance trend over recent timeline."""
        if len(self.performance_timeline) < 2:
            return 0.0
        
        recent_timeline = self.performance_timeline[-window:]
        if len(recent_timeline) < 2:
            return 0.0
        
        times = [t for t, _ in recent_timeline]
        performances = [p for _, p in recent_timeline]
        
        # Simple linear trend
        x = np.arange(len(performances))
        trend = np.polyfit(x, performances, 1)[0]
        return float(trend)


class SelfModificationPipeline:
    """
    Self-Modification Pipeline
    
    Integrates recursive self-improvement, architecture optimization, and safety
    constraints to provide a comprehensive self-modification system.
    """
    
    def __init__(self, 
                 config: ModificationConfig,
                 performance_evaluator: Callable[[Dict], Dict[str, float]],
                 key: jax.random.PRNGKey):
        self.config = config
        self.performance_evaluator = performance_evaluator
        self.key = key
        
        # Initialize components
        self.recursive_improver = None
        self.architecture_optimizer = None
        self.safety_manager = None
        
        if config.enable_recursive_improvement:
            key, subkey = random.split(key)
            self.recursive_improver = RecursiveSelfImprovement(
                config.improvement_config, performance_evaluator, subkey
            )
        
        if config.enable_architecture_optimization:
            key, subkey = random.split(key)
            self.architecture_optimizer = ArchitectureOptimizer(
                config.optimization_config, performance_evaluator, subkey
            )
        
        if config.enable_safety_constraints:
            self.safety_manager = SafetyConstraintManager(config.safety_config)
        
        # Pipeline state
        self.history = ModificationHistory()
        self.current_system_state = {}
        self.baseline_performance = 0.0
        self.iteration_count = 0
        self.last_checkpoint = {}
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.modification_queue = deque()
        
    def initialize_system(self, initial_system_state: Dict[str, Any]) -> None:
        """Initialize the system with initial state."""
        self.current_system_state = initial_system_state.copy()
        
        # Establish baseline performance
        baseline_metrics = self.performance_evaluator(self.current_system_state)
        self.baseline_performance = baseline_metrics.get('overall_performance', 0.0)
        self.performance_history.append(self.baseline_performance)
        
        print(f"System initialized with baseline performance: {self.baseline_performance:.4f}")
        
        # Initialize recursive improver baseline
        if self.recursive_improver:
            self.recursive_improver.establish_baseline(self.current_system_state)
    
    def propose_modifications(self) -> List[Dict[str, Any]]:
        """Propose modifications using available strategies."""
        proposed_modifications = []
        
        # Get current performance metrics
        current_metrics = self.performance_evaluator(self.current_system_state)
        
        # Recursive improvement proposals
        if self.recursive_improver:
            improvement_proposals = self.recursive_improver.propose_improvements(current_metrics)
            for proposal in improvement_proposals:
                proposal['source'] = 'recursive_improvement'
                proposal['priority'] = 0.8
            proposed_modifications.extend(improvement_proposals)
        
        # Architecture optimization proposals
        if self.architecture_optimizer:
            # Extract optimizable parameters
            optimizable_params = self._extract_optimizable_parameters(self.current_system_state)
            if optimizable_params:
                # Run optimization to get suggestions
                optimized_params = self.architecture_optimizer.optimize_architecture(optimizable_params)
                
                # Convert to modification proposals
                for param_name, new_value in optimized_params.items():
                    if param_name in optimizable_params and optimizable_params[param_name] != new_value:
                        proposed_modifications.append({
                            'type': 'parameter_optimization',
                            'parameter': param_name,
                            'old_value': optimizable_params[param_name],
                            'new_value': new_value,
                            'source': 'architecture_optimizer',
                            'priority': 0.6
                        })
        
        # Sort by priority
        proposed_modifications.sort(key=lambda x: x.get('priority', 0.5), reverse=True)
        
        return proposed_modifications
    
    def evaluate_modification_safety(self, 
                                   proposed_modification: Dict[str, Any]) -> Tuple[bool, List[SafetyViolation]]:
        """Evaluate safety of proposed modification."""
        if not self.safety_manager:
            return True, []
        
        # Create history for safety evaluation
        history = []
        for timestamp, performance in self.history.performance_timeline[-20:]:  # Last 20 entries
            history.append({
                'performance': performance,
                'timestamp': timestamp
            })
        
        # Check safety
        is_safe, violations = self.safety_manager.is_modification_safe(
            self.current_system_state, proposed_modification, history
        )
        
        return is_safe, violations
    
    def apply_modification(self, modification: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a modification to the system state."""
        modified_state = self.current_system_state.copy()
        
        mod_type = modification['type']
        
        if mod_type == 'learning_rate_adjustment':
            # Apply learning rate modification
            factor = modification['factor']
            for key in modified_state.keys():
                if 'learning_rate' in key:
                    modified_state[key] = modified_state[key] * factor
        
        elif mod_type == 'parameter_optimization':
            # Apply parameter optimization
            param_name = modification['parameter']
            new_value = modification['new_value']
            modified_state[param_name] = new_value
        
        elif mod_type in ['add_neurons', 'prune_connections']:
            # Apply structural modifications
            if 'structural_modifications' not in modified_state:
                modified_state['structural_modifications'] = []
            modified_state['structural_modifications'].append(modification)
        
        elif mod_type in ['adjust_batch_size', 'adjust_momentum', 'adjust_regularization']:
            # Apply hyperparameter modifications
            param_map = {
                'adjust_batch_size': 'batch_size',
                'adjust_momentum': 'momentum',
                'adjust_regularization': 'regularization'
            }
            param_name = param_map[mod_type]
            factor = modification['factor']
            if param_name in modified_state:
                modified_state[param_name] = modified_state[param_name] * factor
        
        return modified_state
    
    def rollback_modification(self, modification: Dict[str, Any]) -> Dict[str, Any]:
        """Rollback a modification."""
        if self.last_checkpoint:
            return self.last_checkpoint.copy()
        else:
            return self.current_system_state
    
    def execute_modification_cycle(self) -> ModificationResult:
        """Execute one complete modification cycle."""
        start_time = time.time()
        
        # Create checkpoint
        self.last_checkpoint = self.current_system_state.copy()
        
        # Get baseline performance
        baseline_metrics = self.performance_evaluator(self.current_system_state)
        baseline_performance = baseline_metrics.get('overall_performance', 0.0)
        
        # Propose modifications
        proposed_modifications = self.propose_modifications()
        
        if not proposed_modifications:
            return ModificationResult(
                success=False,
                performance_change=0.0,
                modifications_applied=[],
                safety_violations=[],
                execution_time=time.time() - start_time,
                rollback_performed=False
            )
        
        # Filter safe modifications
        safe_modifications = []
        all_violations = []
        
        for modification in proposed_modifications:
            is_safe, violations = self.evaluate_modification_safety(modification)
            all_violations.extend(violations)
            
            if is_safe:
                safe_modifications.append(modification)
            else:
                print(f"Modification {modification['type']} rejected due to safety violations")
        
        if not safe_modifications:
            return ModificationResult(
                success=False,
                performance_change=0.0,
                modifications_applied=[],
                safety_violations=all_violations,
                execution_time=time.time() - start_time,
                rollback_performed=False
            )
        
        # Apply modifications (start with highest priority)
        applied_modifications = []
        current_state = self.current_system_state.copy()
        
        for modification in safe_modifications[:3]:  # Limit to top 3 modifications
            try:
                current_state = self.apply_modification(modification)
                applied_modifications.append(modification)
                print(f"Applied modification: {modification['type']}")
            except Exception as e:
                print(f"Failed to apply modification {modification['type']}: {e}")
        
        # Evaluate performance after modifications
        if applied_modifications:
            post_metrics = self.performance_evaluator(current_state)
            post_performance = post_metrics.get('overall_performance', 0.0)
            performance_change = post_performance - baseline_performance
            
            # Determine if modifications were successful
            success = performance_change > self.config.convergence_threshold
            rollback_performed = False
            
            if success:
                # Accept modifications
                self.current_system_state = current_state
                self.performance_history.append(post_performance)
                print(f"Modifications successful: {baseline_performance:.4f} → {post_performance:.4f}")
            else:
                # Check if rollback is needed
                if performance_change < -self.config.safety_config.max_performance_drop:
                    # Rollback
                    self.current_system_state = self.rollback_modification(applied_modifications[0])
                    rollback_performed = True
                    print(f"Performance degraded significantly, rolling back")
                else:
                    # Keep modifications even if no improvement
                    self.current_system_state = current_state
                    self.performance_history.append(post_performance)
                    print(f"Modifications kept despite no improvement")
            
            return ModificationResult(
                success=success,
                performance_change=performance_change,
                modifications_applied=applied_modifications,
                safety_violations=all_violations,
                execution_time=time.time() - start_time,
                rollback_performed=rollback_performed
            )
        
        return ModificationResult(
            success=False,
            performance_change=0.0,
            modifications_applied=[],
            safety_violations=all_violations,
            execution_time=time.time() - start_time,
            rollback_performed=False
        )
    
    def run_self_modification(self, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete self-modification pipeline."""
        if max_iterations is None:
            max_iterations = self.config.max_pipeline_iterations
        
        print(f"Starting self-modification pipeline (max {max_iterations} iterations)")
        
        pipeline_start_time = time.time()
        successful_iterations = 0
        consecutive_failures = 0
        
        for iteration in range(max_iterations):
            print(f"\n=== Self-Modification Iteration {iteration + 1} ===")
            
            # Execute modification cycle
            result = self.execute_modification_cycle()
            
            # Record result
            current_performance = self.performance_history[-1] if self.performance_history else 0.0
            self.history.add_result(result, time.time(), current_performance)
            
            if result.success:
                successful_iterations += 1
                consecutive_failures = 0
                print(f"Iteration {iteration + 1} successful")
            else:
                consecutive_failures += 1
                print(f"Iteration {iteration + 1} failed")
            
            # Check for early termination conditions
            if consecutive_failures >= 5:
                print("Too many consecutive failures, stopping pipeline")
                break
            
            # Check for convergence
            if len(self.performance_history) >= 5:
                recent_performance = list(self.performance_history)[-5:]
                performance_variance = float(np.var(recent_performance))
                if performance_variance < self.config.convergence_threshold:
                    print("Performance converged, stopping pipeline")
                    break
            
            # Save checkpoint periodically
            if (iteration + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint()
            
            self.iteration_count += 1
        
        # Final evaluation
        final_metrics = self.performance_evaluator(self.current_system_state)
        final_performance = final_metrics.get('overall_performance', 0.0)
        
        pipeline_results = {
            'total_iterations': self.iteration_count,
            'successful_iterations': successful_iterations,
            'success_rate': successful_iterations / max(self.iteration_count, 1),
            'baseline_performance': self.baseline_performance,
            'final_performance': final_performance,
            'total_improvement': final_performance - self.baseline_performance,
            'improvement_ratio': (final_performance - self.baseline_performance) / max(abs(self.baseline_performance), 1e-8),
            'execution_time': time.time() - pipeline_start_time,
            'safety_violations': len(self.history.safety_violations),
            'performance_trend': self.history.get_performance_trend(),
            'recent_success_rate': self.history.get_success_rate()
        }
        
        print(f"\nSelf-modification pipeline completed:")
        print(f"  Performance: {self.baseline_performance:.4f} → {final_performance:.4f}")
        print(f"  Improvement: {pipeline_results['improvement_ratio']:.2%}")
        print(f"  Success rate: {pipeline_results['success_rate']:.2%}")
        
        return pipeline_results
    
    def _extract_optimizable_parameters(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters that can be optimized."""
        optimizable = {}
        
        # Extract common optimizable parameters
        for key, value in system_state.items():
            if any(param in key.lower() for param in ['learning_rate', 'batch_size', 'momentum', 'regularization']):
                if isinstance(value, (int, float)):
                    optimizable[key] = value
        
        return optimizable
    
    def _save_checkpoint(self) -> None:
        """Save current state as checkpoint."""
        if self.config.save_checkpoints:
            checkpoint = {
                'system_state': self.current_system_state.copy(),
                'performance_history': list(self.performance_history),
                'iteration_count': self.iteration_count,
                'timestamp': time.time()
            }
            # In a real implementation, this would save to disk
            print(f"Checkpoint saved at iteration {self.iteration_count}")
    
    def get_modification_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of self-modification process."""
        safety_stats = self.safety_manager.get_safety_statistics() if self.safety_manager else {}
        improvement_summary = self.recursive_improver.get_improvement_summary() if self.recursive_improver else {}
        optimization_summary = self.architecture_optimizer.get_optimization_summary() if self.architecture_optimizer else {}
        
        return {
            'pipeline_stats': {
                'total_iterations': self.iteration_count,
                'baseline_performance': self.baseline_performance,
                'current_performance': self.performance_history[-1] if self.performance_history else 0.0,
                'performance_trend': self.history.get_performance_trend(),
                'success_rate': self.history.get_success_rate()
            },
            'safety_stats': safety_stats,
            'improvement_stats': improvement_summary,
            'optimization_stats': optimization_summary,
            'modification_history': {
                'total_attempts': len(self.history.attempts),
                'successful_modifications': len(self.history.successful_modifications),
                'failed_modifications': len(self.history.failed_modifications),
                'safety_violations': len(self.history.safety_violations)
            }
        }
    
    def reset_pipeline(self) -> None:
        """Reset the pipeline to initial state."""
        self.history = ModificationHistory()
        self.performance_history.clear()
        self.modification_queue.clear()
        self.iteration_count = 0
        
        if self.safety_manager:
            self.safety_manager.reset_emergency_mode()
        
        print("Self-modification pipeline reset")
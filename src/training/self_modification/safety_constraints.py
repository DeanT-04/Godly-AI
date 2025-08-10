"""
Safety Constraints Manager

This module implements safety constraints to prevent destructive modifications
during self-modification processes, ensuring system stability and reliability.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Tuple, Optional, Callable, Any, NamedTuple
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from abc import ABC, abstractmethod
import time
import warnings


class ConstraintType(Enum):
    """Types of safety constraints."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    STABILITY_VIOLATION = "stability_violation"
    RESOURCE_LIMIT = "resource_limit"
    STRUCTURAL_INTEGRITY = "structural_integrity"
    CONVERGENCE_FAILURE = "convergence_failure"
    PARAMETER_BOUNDS = "parameter_bounds"
    RATE_LIMIT = "rate_limit"


class SafetyViolation(NamedTuple):
    """Represents a safety constraint violation."""
    constraint_type: ConstraintType
    severity: float  # 0.0 to 1.0
    description: str
    timestamp: float
    context: Dict[str, Any]


@dataclass
class SafetyConfig:
    """Configuration for safety constraints."""
    
    # Performance constraints
    max_performance_drop: float = 0.2           # Maximum allowed performance drop
    min_performance_threshold: float = 0.1      # Minimum acceptable performance
    performance_window: int = 10                # Window for performance monitoring
    
    # Stability constraints
    max_variance_increase: float = 2.0          # Maximum variance increase factor
    stability_threshold: float = 0.8            # Minimum stability score
    oscillation_detection_window: int = 20      # Window for oscillation detection
    
    # Resource constraints
    max_memory_usage: float = 8.0               # Maximum memory usage (GB)
    max_computation_time: float = 3600.0        # Maximum computation time (seconds)
    max_network_size: int = 1000000             # Maximum network parameters
    
    # Structural constraints
    min_connectivity: float = 0.1               # Minimum network connectivity
    max_connectivity: float = 0.9               # Maximum network connectivity
    min_layer_size: int = 10                    # Minimum layer size
    max_layer_size: int = 10000                 # Maximum layer size
    
    # Modification rate limits
    max_modifications_per_hour: int = 10        # Maximum modifications per hour
    cooldown_period: float = 300.0              # Cooldown between major modifications
    
    # Emergency constraints
    emergency_rollback_threshold: float = 0.5   # Threshold for emergency rollback
    max_consecutive_failures: int = 5           # Maximum consecutive failures
    
    # Constraint weights (for prioritization)
    constraint_weights: Dict[ConstraintType, float] = field(default_factory=lambda: {
        ConstraintType.PERFORMANCE_DEGRADATION: 1.0,
        ConstraintType.STABILITY_VIOLATION: 0.8,
        ConstraintType.RESOURCE_LIMIT: 0.6,
        ConstraintType.STRUCTURAL_INTEGRITY: 0.7,
        ConstraintType.CONVERGENCE_FAILURE: 0.5,
        ConstraintType.PARAMETER_BOUNDS: 0.4,
        ConstraintType.RATE_LIMIT: 0.3
    })


class SafetyCheck(ABC):
    """Abstract base class for safety checks."""
    
    @abstractmethod
    def check(self, 
              current_state: Dict[str, Any],
              proposed_modification: Dict[str, Any],
              history: List[Dict[str, Any]]) -> Optional[SafetyViolation]:
        """Check if proposed modification violates safety constraints."""
        pass


class PerformanceDegradationCheck(SafetyCheck):
    """Check for excessive performance degradation."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
    
    def check(self, 
              current_state: Dict[str, Any],
              proposed_modification: Dict[str, Any],
              history: List[Dict[str, Any]]) -> Optional[SafetyViolation]:
        """Check for performance degradation."""
        
        if len(history) < 2:
            return None
        
        # Get recent performance history
        recent_performance = [h.get('performance', 0.0) for h in history[-self.config.performance_window:]]
        
        if len(recent_performance) < 2:
            return None
        
        # Calculate performance trend
        current_perf = recent_performance[-1]
        baseline_perf = max(recent_performance[:-1])  # Best recent performance
        
        if baseline_perf > 0:
            performance_drop = (baseline_perf - current_perf) / baseline_perf
            
            # Check for excessive degradation
            if performance_drop > self.config.max_performance_drop:
                return SafetyViolation(
                    constraint_type=ConstraintType.PERFORMANCE_DEGRADATION,
                    severity=min(1.0, performance_drop / self.config.max_performance_drop),
                    description=f"Performance dropped by {performance_drop:.2%} (limit: {self.config.max_performance_drop:.2%})",
                    timestamp=time.time(),
                    context={
                        'current_performance': current_perf,
                        'baseline_performance': baseline_perf,
                        'performance_drop': performance_drop
                    }
                )
            
            # Check absolute minimum threshold
            if current_perf < self.config.min_performance_threshold:
                return SafetyViolation(
                    constraint_type=ConstraintType.PERFORMANCE_DEGRADATION,
                    severity=1.0 - (current_perf / self.config.min_performance_threshold),
                    description=f"Performance {current_perf:.3f} below minimum threshold {self.config.min_performance_threshold:.3f}",
                    timestamp=time.time(),
                    context={
                        'current_performance': current_perf,
                        'minimum_threshold': self.config.min_performance_threshold
                    }
                )
        
        return None


class StabilityViolationCheck(SafetyCheck):
    """Check for stability violations."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
    
    def check(self, 
              current_state: Dict[str, Any],
              proposed_modification: Dict[str, Any],
              history: List[Dict[str, Any]]) -> Optional[SafetyViolation]:
        """Check for stability violations."""
        
        if len(history) < self.config.oscillation_detection_window:
            return None
        
        # Get recent performance values
        recent_performance = [h.get('performance', 0.0) for h in history[-self.config.oscillation_detection_window:]]
        
        # Calculate variance
        current_variance = float(np.var(recent_performance))
        
        # Get baseline variance (from earlier history)
        if len(history) >= 2 * self.config.oscillation_detection_window:
            baseline_performance = [h.get('performance', 0.0) for h in 
                                  history[-2*self.config.oscillation_detection_window:-self.config.oscillation_detection_window]]
            baseline_variance = float(np.var(baseline_performance))
            
            # Check for variance increase
            if baseline_variance > 0 and current_variance / baseline_variance > self.config.max_variance_increase:
                return SafetyViolation(
                    constraint_type=ConstraintType.STABILITY_VIOLATION,
                    severity=min(1.0, (current_variance / baseline_variance) / self.config.max_variance_increase),
                    description=f"Performance variance increased by {current_variance/baseline_variance:.1f}x (limit: {self.config.max_variance_increase:.1f}x)",
                    timestamp=time.time(),
                    context={
                        'current_variance': current_variance,
                        'baseline_variance': baseline_variance,
                        'variance_ratio': current_variance / baseline_variance
                    }
                )
        
        # Check for oscillations (alternating high/low performance)
        if len(recent_performance) >= 6:
            # Simple oscillation detection: check for alternating pattern
            diffs = np.diff(recent_performance)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            oscillation_ratio = sign_changes / len(diffs)
            
            if oscillation_ratio > 0.7:  # High frequency of sign changes
                return SafetyViolation(
                    constraint_type=ConstraintType.STABILITY_VIOLATION,
                    severity=oscillation_ratio,
                    description=f"Performance oscillation detected (ratio: {oscillation_ratio:.2f})",
                    timestamp=time.time(),
                    context={
                        'oscillation_ratio': oscillation_ratio,
                        'recent_performance': recent_performance[-6:]
                    }
                )
        
        return None


class ResourceLimitCheck(SafetyCheck):
    """Check for resource limit violations."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
    
    def check(self, 
              current_state: Dict[str, Any],
              proposed_modification: Dict[str, Any],
              history: List[Dict[str, Any]]) -> Optional[SafetyViolation]:
        """Check for resource limit violations."""
        
        # Check memory usage
        memory_usage = current_state.get('memory_usage_gb', 0.0)
        if memory_usage > self.config.max_memory_usage:
            return SafetyViolation(
                constraint_type=ConstraintType.RESOURCE_LIMIT,
                severity=min(1.0, memory_usage / self.config.max_memory_usage),
                description=f"Memory usage {memory_usage:.1f}GB exceeds limit {self.config.max_memory_usage:.1f}GB",
                timestamp=time.time(),
                context={'memory_usage': memory_usage, 'limit': self.config.max_memory_usage}
            )
        
        # Check computation time
        computation_time = current_state.get('computation_time', 0.0)
        if computation_time > self.config.max_computation_time:
            return SafetyViolation(
                constraint_type=ConstraintType.RESOURCE_LIMIT,
                severity=min(1.0, computation_time / self.config.max_computation_time),
                description=f"Computation time {computation_time:.1f}s exceeds limit {self.config.max_computation_time:.1f}s",
                timestamp=time.time(),
                context={'computation_time': computation_time, 'limit': self.config.max_computation_time}
            )
        
        # Check network size
        network_size = current_state.get('network_parameters', 0)
        if network_size > self.config.max_network_size:
            return SafetyViolation(
                constraint_type=ConstraintType.RESOURCE_LIMIT,
                severity=min(1.0, network_size / self.config.max_network_size),
                description=f"Network size {network_size} exceeds limit {self.config.max_network_size}",
                timestamp=time.time(),
                context={'network_size': network_size, 'limit': self.config.max_network_size}
            )
        
        return None


class StructuralIntegrityCheck(SafetyCheck):
    """Check for structural integrity violations."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
    
    def check(self, 
              current_state: Dict[str, Any],
              proposed_modification: Dict[str, Any],
              history: List[Dict[str, Any]]) -> Optional[SafetyViolation]:
        """Check for structural integrity violations."""
        
        # Check connectivity
        connectivity = current_state.get('network_connectivity', 0.5)
        if connectivity < self.config.min_connectivity:
            return SafetyViolation(
                constraint_type=ConstraintType.STRUCTURAL_INTEGRITY,
                severity=1.0 - (connectivity / self.config.min_connectivity),
                description=f"Network connectivity {connectivity:.3f} below minimum {self.config.min_connectivity:.3f}",
                timestamp=time.time(),
                context={'connectivity': connectivity, 'minimum': self.config.min_connectivity}
            )
        
        if connectivity > self.config.max_connectivity:
            return SafetyViolation(
                constraint_type=ConstraintType.STRUCTURAL_INTEGRITY,
                severity=(connectivity - self.config.max_connectivity) / (1.0 - self.config.max_connectivity),
                description=f"Network connectivity {connectivity:.3f} above maximum {self.config.max_connectivity:.3f}",
                timestamp=time.time(),
                context={'connectivity': connectivity, 'maximum': self.config.max_connectivity}
            )
        
        # Check layer sizes
        layer_sizes = current_state.get('layer_sizes', [])
        for i, size in enumerate(layer_sizes):
            if size < self.config.min_layer_size:
                return SafetyViolation(
                    constraint_type=ConstraintType.STRUCTURAL_INTEGRITY,
                    severity=1.0 - (size / self.config.min_layer_size),
                    description=f"Layer {i} size {size} below minimum {self.config.min_layer_size}",
                    timestamp=time.time(),
                    context={'layer_index': i, 'layer_size': size, 'minimum': self.config.min_layer_size}
                )
            
            if size > self.config.max_layer_size:
                return SafetyViolation(
                    constraint_type=ConstraintType.STRUCTURAL_INTEGRITY,
                    severity=min(1.0, size / self.config.max_layer_size),
                    description=f"Layer {i} size {size} above maximum {self.config.max_layer_size}",
                    timestamp=time.time(),
                    context={'layer_index': i, 'layer_size': size, 'maximum': self.config.max_layer_size}
                )
        
        return None


class RateLimitCheck(SafetyCheck):
    """Check for modification rate limit violations."""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.modification_timestamps = deque(maxlen=100)
        self.last_major_modification = 0.0
    
    def check(self, 
              current_state: Dict[str, Any],
              proposed_modification: Dict[str, Any],
              history: List[Dict[str, Any]]) -> Optional[SafetyViolation]:
        """Check for rate limit violations."""
        
        current_time = time.time()
        
        # Check cooldown period for major modifications
        modification_severity = proposed_modification.get('severity', 0.5)
        if modification_severity > 0.7:  # Major modification
            time_since_last = current_time - self.last_major_modification
            if time_since_last < self.config.cooldown_period:
                return SafetyViolation(
                    constraint_type=ConstraintType.RATE_LIMIT,
                    severity=(self.config.cooldown_period - time_since_last) / self.config.cooldown_period,
                    description=f"Major modification attempted {time_since_last:.1f}s after last (cooldown: {self.config.cooldown_period:.1f}s)",
                    timestamp=current_time,
                    context={
                        'time_since_last': time_since_last,
                        'cooldown_period': self.config.cooldown_period,
                        'modification_severity': modification_severity
                    }
                )
            self.last_major_modification = current_time
        
        # Check hourly rate limit
        self.modification_timestamps.append(current_time)
        recent_modifications = [t for t in self.modification_timestamps if current_time - t <= 3600.0]
        
        if len(recent_modifications) > self.config.max_modifications_per_hour:
            return SafetyViolation(
                constraint_type=ConstraintType.RATE_LIMIT,
                severity=len(recent_modifications) / self.config.max_modifications_per_hour,
                description=f"Too many modifications: {len(recent_modifications)} in last hour (limit: {self.config.max_modifications_per_hour})",
                timestamp=current_time,
                context={
                    'recent_modifications': len(recent_modifications),
                    'hourly_limit': self.config.max_modifications_per_hour
                }
            )
        
        return None


class SafetyConstraintManager:
    """
    Safety Constraint Manager
    
    Manages safety constraints to prevent destructive modifications during
    self-modification processes.
    """
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        
        # Initialize safety checks
        self.safety_checks = {
            ConstraintType.PERFORMANCE_DEGRADATION: PerformanceDegradationCheck(config),
            ConstraintType.STABILITY_VIOLATION: StabilityViolationCheck(config),
            ConstraintType.RESOURCE_LIMIT: ResourceLimitCheck(config),
            ConstraintType.STRUCTURAL_INTEGRITY: StructuralIntegrityCheck(config),
            ConstraintType.RATE_LIMIT: RateLimitCheck(config)
        }
        
        # Violation tracking
        self.violation_history = deque(maxlen=1000)
        self.consecutive_failures = 0
        self.emergency_mode = False
        
        # Statistics
        self.total_checks = 0
        self.total_violations = 0
        self.violations_by_type = {ct: 0 for ct in ConstraintType}
    
    def check_safety_constraints(self, 
                                current_state: Dict[str, Any],
                                proposed_modification: Dict[str, Any],
                                history: List[Dict[str, Any]]) -> List[SafetyViolation]:
        """Check all safety constraints for proposed modification."""
        
        self.total_checks += 1
        violations = []
        
        # Run all safety checks
        for constraint_type, safety_check in self.safety_checks.items():
            try:
                violation = safety_check.check(current_state, proposed_modification, history)
                if violation:
                    violations.append(violation)
                    self.violations_by_type[constraint_type] += 1
            except Exception as e:
                warnings.warn(f"Safety check {constraint_type} failed: {e}")
        
        # Record violations
        if violations:
            self.total_violations += len(violations)
            self.violation_history.extend(violations)
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        
        # Check for emergency mode
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            self.emergency_mode = True
            violations.append(SafetyViolation(
                constraint_type=ConstraintType.CONVERGENCE_FAILURE,
                severity=1.0,
                description=f"Emergency mode activated: {self.consecutive_failures} consecutive failures",
                timestamp=time.time(),
                context={'consecutive_failures': self.consecutive_failures}
            ))
        
        return violations
    
    def is_modification_safe(self, 
                           current_state: Dict[str, Any],
                           proposed_modification: Dict[str, Any],
                           history: List[Dict[str, Any]]) -> Tuple[bool, List[SafetyViolation]]:
        """Check if proposed modification is safe to apply."""
        
        violations = self.check_safety_constraints(current_state, proposed_modification, history)
        
        if not violations:
            return True, []
        
        # Calculate weighted severity
        total_weighted_severity = 0.0
        total_weight = 0.0
        
        for violation in violations:
            weight = self.config.constraint_weights.get(violation.constraint_type, 1.0)
            total_weighted_severity += violation.severity * weight
            total_weight += weight
        
        if total_weight > 0:
            average_severity = total_weighted_severity / total_weight
        else:
            average_severity = 1.0
        
        # Modification is safe if average severity is below emergency threshold
        is_safe = average_severity < self.config.emergency_rollback_threshold
        
        return is_safe, violations
    
    def get_safety_recommendations(self, 
                                 violations: List[SafetyViolation]) -> List[str]:
        """Get recommendations for addressing safety violations."""
        
        recommendations = []
        
        for violation in violations:
            if violation.constraint_type == ConstraintType.PERFORMANCE_DEGRADATION:
                recommendations.extend([
                    "Reduce modification strength",
                    "Increase evaluation episodes",
                    "Consider rollback to previous state"
                ])
            
            elif violation.constraint_type == ConstraintType.STABILITY_VIOLATION:
                recommendations.extend([
                    "Increase regularization",
                    "Reduce learning rate",
                    "Add stability constraints"
                ])
            
            elif violation.constraint_type == ConstraintType.RESOURCE_LIMIT:
                recommendations.extend([
                    "Reduce network size",
                    "Optimize memory usage",
                    "Implement model compression"
                ])
            
            elif violation.constraint_type == ConstraintType.STRUCTURAL_INTEGRITY:
                recommendations.extend([
                    "Maintain minimum connectivity",
                    "Avoid extreme structural changes",
                    "Validate network topology"
                ])
            
            elif violation.constraint_type == ConstraintType.RATE_LIMIT:
                recommendations.extend([
                    "Wait for cooldown period",
                    "Reduce modification frequency",
                    "Batch multiple small changes"
                ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def reset_emergency_mode(self) -> None:
        """Reset emergency mode (should be called after successful recovery)."""
        self.emergency_mode = False
        self.consecutive_failures = 0
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get safety constraint statistics."""
        
        recent_violations = [v for v in self.violation_history if time.time() - v.timestamp <= 3600.0]
        
        return {
            'total_checks': self.total_checks,
            'total_violations': self.total_violations,
            'violation_rate': self.total_violations / max(self.total_checks, 1),
            'violations_by_type': dict(self.violations_by_type),
            'recent_violations': len(recent_violations),
            'consecutive_failures': self.consecutive_failures,
            'emergency_mode': self.emergency_mode,
            'most_common_violation': max(self.violations_by_type.items(), key=lambda x: x[1])[0].value if self.violations_by_type else None
        }
    
    def update_config(self, new_config: SafetyConfig) -> None:
        """Update safety configuration."""
        self.config = new_config
        
        # Reinitialize safety checks with new config
        self.safety_checks = {
            ConstraintType.PERFORMANCE_DEGRADATION: PerformanceDegradationCheck(new_config),
            ConstraintType.STABILITY_VIOLATION: StabilityViolationCheck(new_config),
            ConstraintType.RESOURCE_LIMIT: ResourceLimitCheck(new_config),
            ConstraintType.STRUCTURAL_INTEGRITY: StructuralIntegrityCheck(new_config),
            ConstraintType.RATE_LIMIT: RateLimitCheck(new_config)
        }
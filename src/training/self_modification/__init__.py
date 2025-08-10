"""
Self-modification mechanisms for autonomous system improvement.

This module implements recursive self-improvement loops, architecture optimization
based on performance metrics, and safety constraints to prevent destructive modifications.
"""

from .recursive_improvement import (
    RecursiveSelfImprovement,
    ImprovementConfig,
    ImprovementState,
    ImprovementMetrics
)

from .architecture_optimizer import (
    ArchitectureOptimizer,
    OptimizationConfig,
    OptimizationStrategy,
    PerformanceTracker
)

from .safety_constraints import (
    SafetyConstraintManager,
    SafetyConfig,
    SafetyViolation,
    SafetyCheck,
    ConstraintType
)

from .self_modification_pipeline import (
    SelfModificationPipeline,
    ModificationConfig,
    ModificationResult,
    ModificationHistory
)

__all__ = [
    # Recursive Improvement
    'RecursiveSelfImprovement',
    'ImprovementConfig',
    'ImprovementState',
    'ImprovementMetrics',
    
    # Architecture Optimization
    'ArchitectureOptimizer',
    'OptimizationConfig',
    'OptimizationStrategy',
    'PerformanceTracker',
    
    # Safety Constraints
    'SafetyConstraintManager',
    'SafetyConfig',
    'SafetyViolation',
    'SafetyCheck',
    'ConstraintType',
    
    # Pipeline
    'SelfModificationPipeline',
    'ModificationConfig',
    'ModificationResult',
    'ModificationHistory'
]
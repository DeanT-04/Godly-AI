"""
Meta-learning and self-improvement for the Godly AI agents.

This module handles learning-to-learn and recursive self-improvement.
"""

from .meta_learning_core import (
    MetaLearningCore,
    HyperparameterOptimizer,
    DomainAdapter,
    LearningAlgorithm,
    TaskDomain,
    MetaLearningParams,
    LearningAlgorithmType,
    OptimizationMethod
)

__all__ = [
    'MetaLearningCore',
    'HyperparameterOptimizer',
    'DomainAdapter',
    'LearningAlgorithm',
    'TaskDomain',
    'MetaLearningParams',
    'LearningAlgorithmType',
    'OptimizationMethod'
]
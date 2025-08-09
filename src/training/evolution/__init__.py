"""
Architecture evolution and self-modification during training.

This module handles network topology evolution.
"""

from .topology_evolution import (
    TopologyEvolution,
    EvolutionParams,
    EvolutionState
)

from .performance_selection import (
    PerformanceSelector,
    SelectionParams,
    SelectionMethod,
    PerformanceEvaluator,
    PerformanceMetrics
)

from .synaptic_pruning import (
    SynapticPruner,
    SynapticGrower,
    SynapticPlasticityManager,
    PruningParams,
    GrowthParams,
    SynapticState,
    PruningStrategy,
    GrowthStrategy
)

__all__ = [
    'TopologyEvolution',
    'EvolutionParams',
    'EvolutionState',
    'PerformanceSelector',
    'SelectionParams',
    'SelectionMethod',
    'PerformanceEvaluator',
    'PerformanceMetrics',
    'SynapticPruner',
    'SynapticGrower',
    'SynapticPlasticityManager',
    'PruningParams',
    'GrowthParams',
    'SynapticState',
    'PruningStrategy',
    'GrowthStrategy'
]
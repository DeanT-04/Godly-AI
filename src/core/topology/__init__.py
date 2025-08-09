"""
Network topology and structure management for the Godly AI system.

This module handles network architecture and connectivity patterns.
"""

from .network_topology import (
    NetworkTopology,
    TopologyManager,
    NeuronParams,
    ConnectionParams,
    ConnectionType
)

from .mutation_operators import (
    MutationOperator,
    TopologyMutation,
    MutationParams,
    MutationType
)

__all__ = [
    'NetworkTopology',
    'TopologyManager', 
    'NeuronParams',
    'ConnectionParams',
    'ConnectionType',
    'MutationOperator',
    'TopologyMutation',
    'MutationParams',
    'MutationType'
]
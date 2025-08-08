"""
Plasticity and learning rules for the Godly AI neuromorphic system.

This module contains STDP and other plasticity mechanisms.
"""

from .stdp import (
    STDPLearningRule, STDPParams, STDPState,
    TripletsSTDP, create_stdp_rule, compute_stdp_window
)

__all__ = [
    "STDPLearningRule",
    "STDPParams",
    "STDPState", 
    "TripletsSTDP",
    "create_stdp_rule",
    "compute_stdp_window"
]
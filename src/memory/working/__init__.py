"""
Working Memory Module

This module implements working memory with dynamic spiking reservoirs
for pattern storage and retrieval with attention-based mechanisms.
"""

from .working_memory import (
    WorkingMemory,
    WorkingMemoryParams,
    WorkingMemoryState,
    MemoryPattern,
    create_working_memory
)

__all__ = [
    'WorkingMemory',
    'WorkingMemoryParams', 
    'WorkingMemoryState',
    'MemoryPattern',
    'create_working_memory'
]
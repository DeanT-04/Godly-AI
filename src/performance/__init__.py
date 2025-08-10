"""
Performance optimization layer for the Godly AI system.

This module provides CPU optimization, parallel processing, and JIT compilation
capabilities to maximize performance on consumer hardware.
"""

from .mkl_optimization import MKLOptimizer, enable_mkl_optimization
from .parallel_processing import ParallelProcessor, enable_parallel_processing
from .jit_compilation import JITCompiler, enable_jit_compilation

__all__ = [
    'MKLOptimizer',
    'enable_mkl_optimization',
    'ParallelProcessor', 
    'enable_parallel_processing',
    'JITCompiler',
    'enable_jit_compilation'
]
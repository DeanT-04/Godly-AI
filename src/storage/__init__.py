"""
Storage and persistence systems for the Godly AI System.

This module provides interfaces and implementations for various storage backends
including Redis for real-time memory, SQLite for persistent storage, and HDF5
for large-scale data.
"""

# Import storage classes with optional dependencies
__all__ = []

try:
    from .redis_storage import RedisStorage
    __all__.append('RedisStorage')
except ImportError:
    RedisStorage = None

try:
    from .sqlite_storage import SQLiteStorage
    __all__.append('SQLiteStorage')
except ImportError:
    SQLiteStorage = None

try:
    from .hdf5_storage import HDF5Storage
    __all__.append('HDF5Storage')
except ImportError:
    HDF5Storage = None
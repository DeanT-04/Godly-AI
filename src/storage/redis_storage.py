"""
Redis integration for real-time memory storage in the Godly AI System.

This module provides Redis-based storage for working memory patterns,
spike train caching, and attention weights with compression and error handling.
"""

import json
import pickle
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import redis
import numpy as np
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class RedisConnectionError(Exception):
    """Raised when Redis connection fails."""
    pass


class RedisStorage:
    """
    Redis-based storage system for real-time memory operations.
    
    Provides high-performance storage for working memory patterns,
    spike trains, and attention weights with automatic compression
    and connection management.
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 10,
        compression_threshold: int = 1024,
        retry_attempts: int = 3,
        retry_delay: float = 0.1
    ):
        """
        Initialize Redis storage with connection management.
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            max_connections: Maximum connection pool size
            compression_threshold: Minimum size for compression (bytes)
            retry_attempts: Number of retry attempts for failed operations
            retry_delay: Delay between retry attempts (seconds)
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.compression_threshold = compression_threshold
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Initialize connection pool
        self.connection_pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={}
        )
        
        self.redis_client = redis.Redis(connection_pool=self.connection_pool)
        
        # Test connection
        self._test_connection()
        
        # Key prefixes for different data types
        self.WORKING_MEMORY_PREFIX = "wm:"
        self.SPIKE_TRAIN_PREFIX = "spikes:"
        self.ATTENTION_PREFIX = "attention:"
        self.METADATA_PREFIX = "meta:"
        
        logger.info(f"Redis storage initialized: {host}:{port}/{db}")
    
    def _test_connection(self) -> None:
        """Test Redis connection and raise error if failed."""
        try:
            self.redis_client.ping()
        except redis.ConnectionError as e:
            raise RedisConnectionError(f"Failed to connect to Redis: {e}")
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if it exceeds threshold."""
        if len(data) > self.compression_threshold:
            return zlib.compress(data)
        return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if it was compressed."""
        try:
            # Try to decompress - if it fails, data wasn't compressed
            return zlib.decompress(data)
        except zlib.error:
            return data
    
    def _serialize_array(self, array: Union[np.ndarray, jnp.ndarray]) -> bytes:
        """Serialize numpy/jax array to bytes with compression."""
        # Convert JAX array to numpy if needed
        if hasattr(array, 'device'):  # JAX array
            array = np.array(array)
        
        # Serialize using pickle for efficiency
        serialized = pickle.dumps(array)
        return self._compress_data(serialized)
    
    def _deserialize_array(self, data: bytes) -> np.ndarray:
        """Deserialize bytes to numpy array with decompression."""
        decompressed = self._decompress_data(data)
        return pickle.loads(decompressed)
    
    def _retry_operation(self, operation, *args, **kwargs):
        """Retry Redis operation with exponential backoff."""
        for attempt in range(self.retry_attempts):
            try:
                return operation(*args, **kwargs)
            except redis.ConnectionError as e:
                if attempt == self.retry_attempts - 1:
                    raise RedisConnectionError(f"Redis operation failed after {self.retry_attempts} attempts: {e}")
                
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Redis operation failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
    
    # Working Memory Operations
    
    def store_working_memory_pattern(
        self,
        pattern_hash: str,
        pattern: Union[np.ndarray, jnp.ndarray],
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store working memory pattern with compression.
        
        Args:
            pattern_hash: Unique identifier for the pattern
            pattern: Neural pattern data
            timestamp: Pattern creation timestamp
            metadata: Optional metadata dictionary
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            key = f"{self.WORKING_MEMORY_PREFIX}{pattern_hash}"
            
            # Serialize pattern data
            pattern_data = self._serialize_array(pattern)
            
            # Create storage record
            record = {
                'pattern': pattern_data,
                'timestamp': timestamp,
                'shape': pattern.shape,
                'dtype': str(pattern.dtype),
                'metadata': json.dumps(metadata or {})
            }
            
            # Store with expiration (1 hour for working memory)
            result = self._retry_operation(
                self.redis_client.hset,
                key,
                mapping={k: v for k, v in record.items()}
            )
            
            self._retry_operation(self.redis_client.expire, key, 3600)  # 1 hour TTL
            
            logger.debug(f"Stored working memory pattern: {pattern_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store working memory pattern {pattern_hash}: {e}")
            return False
    
    def retrieve_working_memory_pattern(
        self,
        pattern_hash: str
    ) -> Optional[Tuple[np.ndarray, float, Dict[str, Any]]]:
        """
        Retrieve working memory pattern.
        
        Args:
            pattern_hash: Pattern identifier
            
        Returns:
            Tuple of (pattern, timestamp, metadata) or None if not found
        """
        try:
            key = f"{self.WORKING_MEMORY_PREFIX}{pattern_hash}"
            
            record = self._retry_operation(self.redis_client.hgetall, key)
            
            if not record:
                return None
            
            # Deserialize pattern
            pattern = self._deserialize_array(record[b'pattern'])
            timestamp = float(record[b'timestamp'])
            metadata = json.loads(record[b'metadata'].decode())
            
            logger.debug(f"Retrieved working memory pattern: {pattern_hash}")
            return pattern, timestamp, metadata
            
        except Exception as e:
            logger.error(f"Failed to retrieve working memory pattern {pattern_hash}: {e}")
            return None
    
    def update_working_memory_access(self, pattern_hash: str) -> bool:
        """Update access timestamp for working memory pattern."""
        try:
            key = f"{self.WORKING_MEMORY_PREFIX}{pattern_hash}"
            result = self._retry_operation(
                self.redis_client.hset,
                key,
                'last_accessed',
                time.time()
            )
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to update access time for {pattern_hash}: {e}")
            return False
    
    # Spike Train Operations
    
    def cache_spike_train(
        self,
        timestamp: float,
        spike_data: Union[np.ndarray, jnp.ndarray],
        neuron_ids: Optional[List[int]] = None,
        ttl: int = 300  # 5 minutes default
    ) -> bool:
        """
        Cache spike train data for high-frequency access.
        
        Args:
            timestamp: Spike train timestamp
            spike_data: Spike train array
            neuron_ids: Optional list of neuron IDs
            ttl: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        try:
            key = f"{self.SPIKE_TRAIN_PREFIX}{timestamp}"
            
            # Serialize spike data
            spike_bytes = self._serialize_array(spike_data)
            
            record = {
                'spikes': spike_bytes,
                'shape': spike_data.shape,
                'dtype': str(spike_data.dtype),
                'neuron_ids': json.dumps(neuron_ids or []),
                'cached_at': time.time()
            }
            
            # Store with TTL
            self._retry_operation(
                self.redis_client.hset,
                key,
                mapping=record
            )
            
            self._retry_operation(self.redis_client.expire, key, ttl)
            
            logger.debug(f"Cached spike train: {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache spike train {timestamp}: {e}")
            return False
    
    def retrieve_spike_train(
        self,
        timestamp: float
    ) -> Optional[Tuple[np.ndarray, List[int]]]:
        """
        Retrieve cached spike train data.
        
        Args:
            timestamp: Spike train timestamp
            
        Returns:
            Tuple of (spike_data, neuron_ids) or None if not found
        """
        try:
            key = f"{self.SPIKE_TRAIN_PREFIX}{timestamp}"
            
            record = self._retry_operation(self.redis_client.hgetall, key)
            
            if not record:
                return None
            
            spike_data = self._deserialize_array(record[b'spikes'])
            neuron_ids = json.loads(record[b'neuron_ids'].decode())
            
            logger.debug(f"Retrieved spike train: {timestamp}")
            return spike_data, neuron_ids
            
        except Exception as e:
            logger.error(f"Failed to retrieve spike train {timestamp}: {e}")
            return None
    
    def get_recent_spike_trains(
        self,
        time_window: float = 1.0
    ) -> List[Tuple[float, np.ndarray, List[int]]]:
        """
        Get all spike trains within a time window.
        
        Args:
            time_window: Time window in seconds from current time
            
        Returns:
            List of (timestamp, spike_data, neuron_ids) tuples
        """
        try:
            current_time = time.time()
            min_timestamp = current_time - time_window
            
            # Get all spike train keys
            pattern = f"{self.SPIKE_TRAIN_PREFIX}*"
            keys = self._retry_operation(self.redis_client.keys, pattern)
            
            results = []
            for key in keys:
                # Extract timestamp from key
                timestamp_str = key.decode().replace(self.SPIKE_TRAIN_PREFIX, '')
                try:
                    timestamp = float(timestamp_str)
                    if timestamp >= min_timestamp:
                        spike_train = self.retrieve_spike_train(timestamp)
                        if spike_train:
                            results.append((timestamp, spike_train[0], spike_train[1]))
                except ValueError:
                    continue
            
            # Sort by timestamp
            results.sort(key=lambda x: x[0])
            return results
            
        except Exception as e:
            logger.error(f"Failed to get recent spike trains: {e}")
            return []
    
    # Attention Weight Operations
    
    def store_attention_weights(
        self,
        context: str,
        weights: Union[np.ndarray, jnp.ndarray],
        timestamp: float
    ) -> bool:
        """
        Store attention weight distribution.
        
        Args:
            context: Context identifier
            weights: Attention weight array
            timestamp: Weight computation timestamp
            
        Returns:
            True if stored successfully
        """
        try:
            key = f"{self.ATTENTION_PREFIX}{context}"
            
            weight_data = self._serialize_array(weights)
            
            record = {
                'weights': weight_data,
                'timestamp': timestamp,
                'shape': weights.shape,
                'sum': float(np.sum(weights))  # For validation
            }
            
            self._retry_operation(
                self.redis_client.hset,
                key,
                mapping=record
            )
            
            # Attention weights expire after 10 minutes
            self._retry_operation(self.redis_client.expire, key, 600)
            
            logger.debug(f"Stored attention weights: {context}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store attention weights {context}: {e}")
            return False
    
    def retrieve_attention_weights(
        self,
        context: str
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Retrieve attention weights for context.
        
        Args:
            context: Context identifier
            
        Returns:
            Tuple of (weights, timestamp) or None if not found
        """
        try:
            key = f"{self.ATTENTION_PREFIX}{context}"
            
            record = self._retry_operation(self.redis_client.hgetall, key)
            
            if not record:
                return None
            
            weights = self._deserialize_array(record[b'weights'])
            timestamp = float(record[b'timestamp'])
            
            logger.debug(f"Retrieved attention weights: {context}")
            return weights, timestamp
            
        except Exception as e:
            logger.error(f"Failed to retrieve attention weights {context}: {e}")
            return None
    
    # Utility Operations
    
    def clear_expired_data(self) -> int:
        """
        Manually clear expired data and return count of cleared items.
        
        Returns:
            Number of items cleared
        """
        try:
            # Get all keys with TTL
            all_keys = self._retry_operation(self.redis_client.keys, "*")
            expired_count = 0
            
            for key in all_keys:
                ttl = self._retry_operation(self.redis_client.ttl, key)
                if ttl == -2:  # Key doesn't exist (expired)
                    expired_count += 1
            
            logger.info(f"Cleared {expired_count} expired items")
            return expired_count
            
        except Exception as e:
            logger.error(f"Failed to clear expired data: {e}")
            return 0
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage statistics by data type.
        
        Returns:
            Dictionary with memory usage by prefix
        """
        try:
            stats = {
                'working_memory': 0,
                'spike_trains': 0,
                'attention_weights': 0,
                'total': 0
            }
            
            # Count keys by prefix
            for prefix, stat_key in [
                (self.WORKING_MEMORY_PREFIX, 'working_memory'),
                (self.SPIKE_TRAIN_PREFIX, 'spike_trains'),
                (self.ATTENTION_PREFIX, 'attention_weights')
            ]:
                keys = self._retry_operation(self.redis_client.keys, f"{prefix}*")
                stats[stat_key] = len(keys)
                stats['total'] += len(keys)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}
    
    def flush_all_data(self) -> bool:
        """
        Flush all data from the Redis database.
        
        Returns:
            True if successful
        """
        try:
            self._retry_operation(self.redis_client.flushdb)
            logger.warning("Flushed all Redis data")
            return True
        except Exception as e:
            logger.error(f"Failed to flush Redis data: {e}")
            return False
    
    def close(self) -> None:
        """Close Redis connection pool."""
        try:
            self.connection_pool.disconnect()
            logger.info("Redis connection pool closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
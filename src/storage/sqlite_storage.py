"""
SQLite integration for persistent storage in the Godly AI System.

This module provides SQLite-based storage for episodes, concepts, learning events,
and other persistent data with ORM-like functionality and migration support.
"""

import json
import pickle
import sqlite3
import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import threading

import numpy as np
import jax.numpy as jnp

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    """Represents an episodic memory entry."""
    id: Optional[int] = None
    timestamp: float = 0.0
    experience_data: bytes = b''
    performance_score: float = 0.0
    context_hash: str = ''
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Concept:
    """Represents a semantic memory concept."""
    id: Optional[int] = None
    name: str = ''
    embedding: bytes = b''
    creation_time: float = 0.0
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LearningEvent:
    """Represents a learning event for meta-memory."""
    id: Optional[int] = None
    task_type: str = ''
    performance_delta: float = 0.0
    strategy_used: str = ''
    timestamp: float = 0.0
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class SQLiteStorageError(Exception):
    """Raised when SQLite storage operations fail."""
    pass


class SQLiteStorage:
    """
    SQLite-based persistent storage system.
    
    Provides persistent storage for episodes, concepts, learning events,
    and other long-term data with database migration and backup support.
    """
    
    # Database schema version
    SCHEMA_VERSION = 1
    
    def __init__(
        self,
        db_path: Union[str, Path] = "godly_ai.db",
        backup_interval: int = 3600,  # 1 hour
        max_backups: int = 10,
        enable_wal: bool = True,
        connection_timeout: float = 30.0
    ):
        """
        Initialize SQLite storage with database setup.
        
        Args:
            db_path: Path to SQLite database file
            backup_interval: Backup interval in seconds
            max_backups: Maximum number of backup files to keep
            enable_wal: Enable Write-Ahead Logging for better concurrency
            connection_timeout: Connection timeout in seconds
        """
        self.db_path = Path(db_path)
        self.backup_interval = backup_interval
        self.max_backups = max_backups
        self.connection_timeout = connection_timeout
        self.enable_wal = enable_wal
        
        # Thread-local storage for connections
        self._local = threading.local()
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()
        
        # Track last backup time
        self._last_backup = time.time()
        
        logger.info(f"SQLite storage initialized: {self.db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                timeout=self.connection_timeout,
                check_same_thread=False
            )
            
            # Configure connection
            conn = self._local.connection
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            
            if self.enable_wal:
                conn.execute("PRAGMA journal_mode=WAL")
            
            # Performance optimizations
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            
        return self._local.connection
    
    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database transaction failed: {e}")
            raise SQLiteStorageError(f"Transaction failed: {e}")
    
    def _initialize_database(self) -> None:
        """Initialize database schema and migrations."""
        with self._transaction() as conn:
            # Create schema version table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                )
            """)
            
            # Check current schema version
            cursor = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            current_version = cursor.fetchone()
            current_version = current_version[0] if current_version else 0
            
            # Apply migrations
            if current_version < self.SCHEMA_VERSION:
                self._apply_migrations(conn, current_version)
    
    def _apply_migrations(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Apply database migrations from current version to latest."""
        migrations = {
            1: self._create_initial_schema
        }
        
        for version in range(from_version + 1, self.SCHEMA_VERSION + 1):
            if version in migrations:
                logger.info(f"Applying migration to version {version}")
                migrations[version](conn)
                
                # Record migration
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (version, time.time())
                )
    
    def _create_initial_schema(self, conn: sqlite3.Connection) -> None:
        """Create initial database schema."""
        # Episodes table for episodic memory
        conn.execute("""
            CREATE TABLE episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                experience_data BLOB NOT NULL,
                performance_score REAL NOT NULL,
                context_hash TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        
        # Create indexes for episodes table
        conn.execute("CREATE INDEX idx_episodes_timestamp ON episodes(timestamp)")
        conn.execute("CREATE INDEX idx_episodes_context_hash ON episodes(context_hash)")
        conn.execute("CREATE INDEX idx_episodes_performance_score ON episodes(performance_score)")
        
        # Concepts table for semantic memory
        conn.execute("""
            CREATE TABLE concepts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                embedding BLOB NOT NULL,
                creation_time REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed REAL DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        """)
        
        # Create indexes for concepts table
        conn.execute("CREATE INDEX idx_concepts_name ON concepts(name)")
        conn.execute("CREATE INDEX idx_concepts_creation_time ON concepts(creation_time)")
        conn.execute("CREATE INDEX idx_concepts_access_count ON concepts(access_count)")
        
        # Learning events table for meta-memory
        conn.execute("""
            CREATE TABLE learning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                performance_delta REAL NOT NULL,
                strategy_used TEXT NOT NULL,
                timestamp REAL NOT NULL,
                parameters TEXT DEFAULT '{}'
            )
        """)
        
        # Create indexes for learning events table
        conn.execute("CREATE INDEX idx_learning_events_task_type ON learning_events(task_type)")
        conn.execute("CREATE INDEX idx_learning_events_timestamp ON learning_events(timestamp)")
        conn.execute("CREATE INDEX idx_learning_events_performance_delta ON learning_events(performance_delta)")
        
        # Concept relationships for knowledge graph
        conn.execute("""
            CREATE TABLE concept_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_concept_id INTEGER NOT NULL,
                target_concept_id INTEGER NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                created_at REAL DEFAULT (julianday('now')),
                FOREIGN KEY (source_concept_id) REFERENCES concepts (id),
                FOREIGN KEY (target_concept_id) REFERENCES concepts (id),
                UNIQUE(source_concept_id, target_concept_id, relationship_type)
            )
        """)
        
        # Create indexes for concept relationships table
        conn.execute("CREATE INDEX idx_concept_relationships_source ON concept_relationships(source_concept_id)")
        conn.execute("CREATE INDEX idx_concept_relationships_target ON concept_relationships(target_concept_id)")
        conn.execute("CREATE INDEX idx_concept_relationships_type ON concept_relationships(relationship_type)")
    
    def _serialize_array(self, array: Union[np.ndarray, jnp.ndarray]) -> bytes:
        """Serialize numpy/jax array to bytes."""
        if hasattr(array, 'device'):  # JAX array
            array = np.array(array)
        return pickle.dumps(array)
    
    def _deserialize_array(self, data: bytes) -> np.ndarray:
        """Deserialize bytes to numpy array."""
        return pickle.loads(data)
    
    # Episode Operations
    
    def store_episode(
        self,
        experience_data: Union[np.ndarray, jnp.ndarray, Dict[str, Any]],
        performance_score: float,
        context_hash: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store an episodic memory entry.
        
        Args:
            experience_data: Experience data (array or dict)
            performance_score: Performance score for this episode
            context_hash: Hash identifying the context
            timestamp: Episode timestamp (current time if None)
            metadata: Optional metadata dictionary
            
        Returns:
            Episode ID
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Serialize experience data
        if isinstance(experience_data, (np.ndarray, jnp.ndarray)):
            serialized_data = self._serialize_array(experience_data)
        else:
            serialized_data = pickle.dumps(experience_data)
        
        with self._transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO episodes (timestamp, experience_data, performance_score, context_hash, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp,
                serialized_data,
                performance_score,
                context_hash,
                json.dumps(metadata or {})
            ))
            
            episode_id = cursor.lastrowid
            logger.debug(f"Stored episode {episode_id} with score {performance_score}")
            return episode_id
    
    def retrieve_episode(self, episode_id: int) -> Optional[Episode]:
        """
        Retrieve an episode by ID.
        
        Args:
            episode_id: Episode identifier
            
        Returns:
            Episode object or None if not found
        """
        with self._transaction() as conn:
            cursor = conn.execute("""
                SELECT id, timestamp, experience_data, performance_score, context_hash, metadata
                FROM episodes WHERE id = ?
            """, (episode_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return Episode(
                id=row['id'],
                timestamp=row['timestamp'],
                experience_data=row['experience_data'],
                performance_score=row['performance_score'],
                context_hash=row['context_hash'],
                metadata=json.loads(row['metadata'])
            )
    
    def get_episodes_by_context(
        self,
        context_hash: str,
        limit: int = 100,
        min_score: Optional[float] = None
    ) -> List[Episode]:
        """
        Get episodes by context hash.
        
        Args:
            context_hash: Context identifier
            limit: Maximum number of episodes to return
            min_score: Minimum performance score filter
            
        Returns:
            List of Episode objects
        """
        query = """
            SELECT id, timestamp, experience_data, performance_score, context_hash, metadata
            FROM episodes WHERE context_hash = ?
        """
        params = [context_hash]
        
        if min_score is not None:
            query += " AND performance_score >= ?"
            params.append(min_score)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._transaction() as conn:
            cursor = conn.execute(query, params)
            
            episodes = []
            for row in cursor.fetchall():
                episodes.append(Episode(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    experience_data=row['experience_data'],
                    performance_score=row['performance_score'],
                    context_hash=row['context_hash'],
                    metadata=json.loads(row['metadata'])
                ))
            
            return episodes
    
    def get_recent_episodes(
        self,
        time_window: float = 3600.0,
        limit: int = 100
    ) -> List[Episode]:
        """
        Get recent episodes within time window.
        
        Args:
            time_window: Time window in seconds from current time
            limit: Maximum number of episodes
            
        Returns:
            List of recent Episode objects
        """
        min_timestamp = time.time() - time_window
        
        with self._transaction() as conn:
            cursor = conn.execute("""
                SELECT id, timestamp, experience_data, performance_score, context_hash, metadata
                FROM episodes 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (min_timestamp, limit))
            
            episodes = []
            for row in cursor.fetchall():
                episodes.append(Episode(
                    id=row['id'],
                    timestamp=row['timestamp'],
                    experience_data=row['experience_data'],
                    performance_score=row['performance_score'],
                    context_hash=row['context_hash'],
                    metadata=json.loads(row['metadata'])
                ))
            
            return episodes    
  
  # Concept Operations
    
    def store_concept(
        self,
        name: str,
        embedding: Union[np.ndarray, jnp.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
        creation_time: Optional[float] = None
    ) -> int:
        """
        Store a semantic memory concept.
        
        Args:
            name: Concept name (must be unique)
            embedding: Concept embedding vector
            metadata: Optional metadata dictionary
            creation_time: Creation timestamp (current time if None)
            
        Returns:
            Concept ID
        """
        if creation_time is None:
            creation_time = time.time()
        
        serialized_embedding = self._serialize_array(embedding)
        
        with self._transaction() as conn:
            try:
                cursor = conn.execute("""
                    INSERT INTO concepts (name, embedding, creation_time, metadata)
                    VALUES (?, ?, ?, ?)
                """, (
                    name,
                    serialized_embedding,
                    creation_time,
                    json.dumps(metadata or {})
                ))
                
                concept_id = cursor.lastrowid
                logger.debug(f"Stored concept '{name}' with ID {concept_id}")
                return concept_id
                
            except sqlite3.IntegrityError:
                # Concept name already exists, update instead
                cursor = conn.execute("""
                    UPDATE concepts 
                    SET embedding = ?, metadata = ?, last_accessed = ?
                    WHERE name = ?
                """, (
                    serialized_embedding,
                    json.dumps(metadata or {}),
                    time.time(),
                    name
                ))
                
                # Get the existing concept ID
                cursor = conn.execute("SELECT id FROM concepts WHERE name = ?", (name,))
                concept_id = cursor.fetchone()['id']
                logger.debug(f"Updated existing concept '{name}' with ID {concept_id}")
                return concept_id
    
    def retrieve_concept(self, concept_id: int) -> Optional[Concept]:
        """
        Retrieve a concept by ID.
        
        Args:
            concept_id: Concept identifier
            
        Returns:
            Concept object or None if not found
        """
        with self._transaction() as conn:
            cursor = conn.execute("""
                SELECT id, name, embedding, creation_time, access_count, last_accessed, metadata
                FROM concepts WHERE id = ?
            """, (concept_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return Concept(
                id=row['id'],
                name=row['name'],
                embedding=row['embedding'],
                creation_time=row['creation_time'],
                access_count=row['access_count'],
                last_accessed=row['last_accessed'],
                metadata=json.loads(row['metadata'])
            )
    
    def retrieve_concept_by_name(self, name: str) -> Optional[Concept]:
        """
        Retrieve a concept by name.
        
        Args:
            name: Concept name
            
        Returns:
            Concept object or None if not found
        """
        with self._transaction() as conn:
            cursor = conn.execute("""
                SELECT id, name, embedding, creation_time, access_count, last_accessed, metadata
                FROM concepts WHERE name = ?
            """, (name,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Update access count
            conn.execute("""
                UPDATE concepts 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """, (time.time(), row['id']))
            
            return Concept(
                id=row['id'],
                name=row['name'],
                embedding=row['embedding'],
                creation_time=row['creation_time'],
                access_count=row['access_count'] + 1,
                last_accessed=time.time(),
                metadata=json.loads(row['metadata'])
            )
    
    def search_concepts(
        self,
        query: str,
        limit: int = 10
    ) -> List[Concept]:
        """
        Search concepts by name pattern.
        
        Args:
            query: Search query (supports SQL LIKE patterns)
            limit: Maximum number of results
            
        Returns:
            List of matching Concept objects
        """
        with self._transaction() as conn:
            cursor = conn.execute("""
                SELECT id, name, embedding, creation_time, access_count, last_accessed, metadata
                FROM concepts 
                WHERE name LIKE ?
                ORDER BY access_count DESC, creation_time DESC
                LIMIT ?
            """, (f"%{query}%", limit))
            
            concepts = []
            for row in cursor.fetchall():
                concepts.append(Concept(
                    id=row['id'],
                    name=row['name'],
                    embedding=row['embedding'],
                    creation_time=row['creation_time'],
                    access_count=row['access_count'],
                    last_accessed=row['last_accessed'],
                    metadata=json.loads(row['metadata'])
                ))
            
            return concepts
    
    def get_concept_relationships(
        self,
        concept_id: int,
        relationship_type: Optional[str] = None
    ) -> List[Tuple[int, str, float]]:
        """
        Get relationships for a concept.
        
        Args:
            concept_id: Source concept ID
            relationship_type: Filter by relationship type (optional)
            
        Returns:
            List of (target_concept_id, relationship_type, strength) tuples
        """
        query = """
            SELECT target_concept_id, relationship_type, strength
            FROM concept_relationships 
            WHERE source_concept_id = ?
        """
        params = [concept_id]
        
        if relationship_type:
            query += " AND relationship_type = ?"
            params.append(relationship_type)
        
        query += " ORDER BY strength DESC"
        
        with self._transaction() as conn:
            cursor = conn.execute(query, params)
            return [(row['target_concept_id'], row['relationship_type'], row['strength']) 
                   for row in cursor.fetchall()]
    
    def add_concept_relationship(
        self,
        source_id: int,
        target_id: int,
        relationship_type: str,
        strength: float = 1.0
    ) -> bool:
        """
        Add or update a relationship between concepts.
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relationship_type: Type of relationship
            strength: Relationship strength (0.0 to 1.0)
            
        Returns:
            True if successful
        """
        with self._transaction() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO concept_relationships 
                    (source_concept_id, target_concept_id, relationship_type, strength)
                    VALUES (?, ?, ?, ?)
                """, (source_id, target_id, relationship_type, strength))
                
                logger.debug(f"Added relationship: {source_id} -> {target_id} ({relationship_type})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to add concept relationship: {e}")
                return False
    
    # Learning Event Operations
    
    def store_learning_event(
        self,
        task_type: str,
        performance_delta: float,
        strategy_used: str,
        parameters: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> int:
        """
        Store a learning event for meta-memory.
        
        Args:
            task_type: Type of learning task
            performance_delta: Change in performance
            strategy_used: Learning strategy identifier
            parameters: Optional strategy parameters
            timestamp: Event timestamp (current time if None)
            
        Returns:
            Learning event ID
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self._transaction() as conn:
            cursor = conn.execute("""
                INSERT INTO learning_events (task_type, performance_delta, strategy_used, timestamp, parameters)
                VALUES (?, ?, ?, ?, ?)
            """, (
                task_type,
                performance_delta,
                strategy_used,
                timestamp,
                json.dumps(parameters or {})
            ))
            
            event_id = cursor.lastrowid
            logger.debug(f"Stored learning event {event_id} for task '{task_type}'")
            return event_id
    
    def get_learning_events_by_task(
        self,
        task_type: str,
        limit: int = 100,
        min_performance_delta: Optional[float] = None
    ) -> List[LearningEvent]:
        """
        Get learning events by task type.
        
        Args:
            task_type: Task type to filter by
            limit: Maximum number of events
            min_performance_delta: Minimum performance improvement filter
            
        Returns:
            List of LearningEvent objects
        """
        query = """
            SELECT id, task_type, performance_delta, strategy_used, timestamp, parameters
            FROM learning_events WHERE task_type = ?
        """
        params = [task_type]
        
        if min_performance_delta is not None:
            query += " AND performance_delta >= ?"
            params.append(min_performance_delta)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with self._transaction() as conn:
            cursor = conn.execute(query, params)
            
            events = []
            for row in cursor.fetchall():
                events.append(LearningEvent(
                    id=row['id'],
                    task_type=row['task_type'],
                    performance_delta=row['performance_delta'],
                    strategy_used=row['strategy_used'],
                    timestamp=row['timestamp'],
                    parameters=json.loads(row['parameters'])
                ))
            
            return events
    
    def get_best_strategies(
        self,
        task_type: str,
        limit: int = 5
    ) -> List[Tuple[str, float, int]]:
        """
        Get best performing strategies for a task type.
        
        Args:
            task_type: Task type to analyze
            limit: Maximum number of strategies to return
            
        Returns:
            List of (strategy_name, avg_performance_delta, usage_count) tuples
        """
        with self._transaction() as conn:
            cursor = conn.execute("""
                SELECT strategy_used, 
                       AVG(performance_delta) as avg_delta,
                       COUNT(*) as usage_count
                FROM learning_events 
                WHERE task_type = ? AND performance_delta > 0
                GROUP BY strategy_used
                ORDER BY avg_delta DESC, usage_count DESC
                LIMIT ?
            """, (task_type, limit))
            
            return [(row['strategy_used'], row['avg_delta'], row['usage_count']) 
                   for row in cursor.fetchall()]
    
    # Database Management Operations
    
    def create_backup(self, backup_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Custom backup path (auto-generated if None)
            
        Returns:
            Path to backup file
        """
        if backup_path is None:
            timestamp = int(time.time())
            backup_path = self.db_path.parent / f"{self.db_path.stem}_backup_{timestamp}.db"
        else:
            backup_path = Path(backup_path)
        
        # Ensure backup directory exists
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._transaction() as conn:
            # Use SQLite backup API for consistent backup
            backup_conn = sqlite3.connect(str(backup_path))
            try:
                conn.backup(backup_conn)
                logger.info(f"Database backup created: {backup_path}")
            finally:
                backup_conn.close()
        
        # Clean up old backups
        self._cleanup_old_backups()
        
        return backup_path
    
    def _cleanup_old_backups(self) -> None:
        """Remove old backup files beyond max_backups limit."""
        backup_pattern = f"{self.db_path.stem}_backup_*.db"
        backup_files = list(self.db_path.parent.glob(backup_pattern))
        
        if len(backup_files) > self.max_backups:
            # Sort by modification time and remove oldest
            backup_files.sort(key=lambda p: p.stat().st_mtime)
            for old_backup in backup_files[:-self.max_backups]:
                try:
                    old_backup.unlink()
                    logger.debug(f"Removed old backup: {old_backup}")
                except Exception as e:
                    logger.warning(f"Failed to remove old backup {old_backup}: {e}")
    
    def vacuum_database(self) -> bool:
        """
        Vacuum the database to reclaim space and optimize performance.
        
        Returns:
            True if successful
        """
        try:
            with self._transaction() as conn:
                conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics and information.
        
        Returns:
            Dictionary with database statistics
        """
        with self._transaction() as conn:
            stats = {}
            
            # Table row counts
            for table in ['episodes', 'concepts', 'learning_events', 'concept_relationships']:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Database size
            stats['db_size_bytes'] = self.db_path.stat().st_size
            stats['db_size_mb'] = stats['db_size_bytes'] / (1024 * 1024)
            
            # Schema version
            cursor = conn.execute("SELECT MAX(version) FROM schema_version")
            stats['schema_version'] = cursor.fetchone()[0]
            
            # Performance stats
            cursor = conn.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor = conn.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            stats['total_pages'] = page_count
            stats['page_size'] = page_size
            
            return stats
    
    def auto_backup_if_needed(self) -> Optional[Path]:
        """
        Create backup if backup interval has elapsed.
        
        Returns:
            Path to backup file if created, None otherwise
        """
        if time.time() - self._last_backup >= self.backup_interval:
            backup_path = self.create_backup()
            self._last_backup = time.time()
            return backup_path
        return None
    
    def close(self) -> None:
        """Close database connections and perform cleanup."""
        try:
            # Close thread-local connections
            if hasattr(self._local, 'connection'):
                self._local.connection.close()
                delattr(self._local, 'connection')
            
            # Create final backup if needed
            self.auto_backup_if_needed()
            
            logger.info("SQLite storage closed")
        except Exception as e:
            logger.error(f"Error closing SQLite storage: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
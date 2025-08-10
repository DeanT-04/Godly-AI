"""
Tests for SQLite storage system.

Tests SQLite integration for episodes, concepts, learning events,
database migrations, and backup functionality.
"""

import json
import tempfile
import time
import pytest
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from unittest.mock import patch, Mock

from src.storage.sqlite_storage import (
    SQLiteStorage, SQLiteStorageError, Episode, Concept, LearningEvent
)


class TestSQLiteStorage:
    """Test suite for SQLite storage functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        yield db_path
        
        # Cleanup
        if db_path.exists():
            db_path.unlink()
        
        # Clean up backup files
        for backup in db_path.parent.glob(f"{db_path.stem}_backup_*.db"):
            backup.unlink()
    
    @pytest.fixture
    def sqlite_storage(self, temp_db_path):
        """SQLite storage instance with temporary database."""
        storage = SQLiteStorage(
            db_path=temp_db_path,
            backup_interval=3600,
            max_backups=3
        )
        yield storage
        storage.close()
    
    def test_initialization_and_schema_creation(self, temp_db_path):
        """Test SQLite storage initialization and schema creation."""
        storage = SQLiteStorage(db_path=temp_db_path)
        
        # Check that database file was created
        assert temp_db_path.exists()
        
        # Check schema version
        stats = storage.get_database_stats()
        assert stats['schema_version'] == 1
        assert 'episodes_count' in stats
        assert 'concepts_count' in stats
        assert 'learning_events_count' in stats
        
        storage.close()
    
    def test_episode_storage_and_retrieval(self, sqlite_storage):
        """Test episode storage and retrieval operations."""
        # Test storing episode with array data
        experience_array = np.random.rand(100, 50)
        episode_id = sqlite_storage.store_episode(
            experience_data=experience_array,
            performance_score=0.85,
            context_hash="test_context_123",
            metadata={"source": "test", "difficulty": "medium"}
        )
        
        assert isinstance(episode_id, int)
        assert episode_id > 0
        
        # Test retrieving episode
        retrieved_episode = sqlite_storage.retrieve_episode(episode_id)
        
        assert retrieved_episode is not None
        assert retrieved_episode.id == episode_id
        assert retrieved_episode.performance_score == 0.85
        assert retrieved_episode.context_hash == "test_context_123"
        assert retrieved_episode.metadata["source"] == "test"
        
        # Test storing episode with dict data
        experience_dict = {
            "observation": np.random.rand(10),
            "action": [1, 2, 3],
            "reward": 0.5
        }
        
        episode_id2 = sqlite_storage.store_episode(
            experience_data=experience_dict,
            performance_score=0.92,
            context_hash="test_context_456"
        )
        
        retrieved_episode2 = sqlite_storage.retrieve_episode(episode_id2)
        assert retrieved_episode2 is not None
        assert retrieved_episode2.performance_score == 0.92
    
    def test_episode_retrieval_by_context(self, sqlite_storage):
        """Test retrieving episodes by context hash."""
        context_hash = "shared_context"
        
        # Store multiple episodes with same context
        episode_ids = []
        for i in range(5):
            episode_id = sqlite_storage.store_episode(
                experience_data=np.random.rand(10, 5),
                performance_score=0.5 + i * 0.1,
                context_hash=context_hash,
                timestamp=time.time() + i  # Different timestamps
            )
            episode_ids.append(episode_id)
        
        # Retrieve episodes by context
        episodes = sqlite_storage.get_episodes_by_context(context_hash)
        
        assert len(episodes) == 5
        # Should be ordered by timestamp DESC
        for i in range(1, len(episodes)):
            assert episodes[i].timestamp <= episodes[i-1].timestamp
        
        # Test with minimum score filter
        high_score_episodes = sqlite_storage.get_episodes_by_context(
            context_hash, min_score=0.8
        )
        
        assert len(high_score_episodes) == 2  # Only scores 0.8 and 0.9
        for episode in high_score_episodes:
            assert episode.performance_score >= 0.8
    
    def test_recent_episodes_retrieval(self, sqlite_storage):
        """Test retrieving recent episodes within time window."""
        current_time = time.time()
        
        # Store episodes at different times
        old_episode = sqlite_storage.store_episode(
            experience_data=np.array([1, 2, 3]),
            performance_score=0.5,
            context_hash="old",
            timestamp=current_time - 7200  # 2 hours ago
        )
        
        recent_episode = sqlite_storage.store_episode(
            experience_data=np.array([4, 5, 6]),
            performance_score=0.8,
            context_hash="recent",
            timestamp=current_time - 1800  # 30 minutes ago
        )
        
        # Get recent episodes (1 hour window)
        recent_episodes = sqlite_storage.get_recent_episodes(time_window=3600)
        
        assert len(recent_episodes) == 1
        assert recent_episodes[0].id == recent_episode
        assert recent_episodes[0].context_hash == "recent"
    
    def test_concept_storage_and_retrieval(self, sqlite_storage):
        """Test concept storage and retrieval operations."""
        # Test storing new concept
        embedding = np.random.rand(256)
        concept_id = sqlite_storage.store_concept(
            name="test_concept",
            embedding=embedding,
            metadata={"category": "test", "confidence": 0.95}
        )
        
        assert isinstance(concept_id, int)
        assert concept_id > 0
        
        # Test retrieving concept by ID
        retrieved_concept = sqlite_storage.retrieve_concept(concept_id)
        
        assert retrieved_concept is not None
        assert retrieved_concept.id == concept_id
        assert retrieved_concept.name == "test_concept"
        assert retrieved_concept.metadata["category"] == "test"
        
        # Test retrieving concept by name (should update access count)
        retrieved_by_name = sqlite_storage.retrieve_concept_by_name("test_concept")
        
        assert retrieved_by_name is not None
        assert retrieved_by_name.access_count == 1
        assert retrieved_by_name.last_accessed > 0
        
        # Test updating existing concept
        new_embedding = np.random.rand(256)
        updated_id = sqlite_storage.store_concept(
            name="test_concept",  # Same name
            embedding=new_embedding,
            metadata={"category": "updated"}
        )
        
        assert updated_id == concept_id  # Should be same ID
        
        updated_concept = sqlite_storage.retrieve_concept(concept_id)
        assert updated_concept.metadata["category"] == "updated"
    
    def test_concept_search(self, sqlite_storage):
        """Test concept search functionality."""
        # Store multiple concepts
        concepts = [
            ("neural_network", np.random.rand(128)),
            ("neural_plasticity", np.random.rand(128)),
            ("machine_learning", np.random.rand(128)),
            ("deep_learning", np.random.rand(128))
        ]
        
        for name, embedding in concepts:
            sqlite_storage.store_concept(name, embedding)
        
        # Search for concepts containing "neural"
        neural_concepts = sqlite_storage.search_concepts("neural")
        
        assert len(neural_concepts) == 2
        concept_names = [c.name for c in neural_concepts]
        assert "neural_network" in concept_names
        assert "neural_plasticity" in concept_names
        
        # Search for concepts containing "learning"
        learning_concepts = sqlite_storage.search_concepts("learning")
        
        assert len(learning_concepts) == 2
        concept_names = [c.name for c in learning_concepts]
        assert "machine_learning" in concept_names
        assert "deep_learning" in concept_names
    
    def test_concept_relationships(self, sqlite_storage):
        """Test concept relationship management."""
        # Create concepts
        concept1_id = sqlite_storage.store_concept("concept1", np.random.rand(64))
        concept2_id = sqlite_storage.store_concept("concept2", np.random.rand(64))
        concept3_id = sqlite_storage.store_concept("concept3", np.random.rand(64))
        
        # Add relationships
        assert sqlite_storage.add_concept_relationship(
            concept1_id, concept2_id, "similar_to", 0.8
        )
        assert sqlite_storage.add_concept_relationship(
            concept1_id, concept3_id, "related_to", 0.6
        )
        assert sqlite_storage.add_concept_relationship(
            concept2_id, concept3_id, "opposite_of", 0.9
        )
        
        # Get relationships for concept1
        relationships = sqlite_storage.get_concept_relationships(concept1_id)
        
        assert len(relationships) == 2
        
        # Check specific relationship
        similar_rels = sqlite_storage.get_concept_relationships(
            concept1_id, "similar_to"
        )
        assert len(similar_rels) == 1
        assert similar_rels[0] == (concept2_id, "similar_to", 0.8)
        
        # Test updating relationship strength
        assert sqlite_storage.add_concept_relationship(
            concept1_id, concept2_id, "similar_to", 0.95  # Updated strength
        )
        
        updated_rels = sqlite_storage.get_concept_relationships(
            concept1_id, "similar_to"
        )
        assert updated_rels[0][2] == 0.95  # Check updated strength
    
    def test_learning_event_storage(self, sqlite_storage):
        """Test learning event storage and retrieval."""
        # Store learning events
        event_id1 = sqlite_storage.store_learning_event(
            task_type="pattern_recognition",
            performance_delta=0.15,
            strategy_used="gradient_descent",
            parameters={"learning_rate": 0.01, "batch_size": 32}
        )
        
        event_id2 = sqlite_storage.store_learning_event(
            task_type="pattern_recognition",
            performance_delta=0.08,
            strategy_used="adam_optimizer",
            parameters={"learning_rate": 0.001, "beta1": 0.9}
        )
        
        event_id3 = sqlite_storage.store_learning_event(
            task_type="sequence_learning",
            performance_delta=0.22,
            strategy_used="lstm_network",
            parameters={"hidden_size": 128, "num_layers": 2}
        )
        
        # Get events by task type
        pattern_events = sqlite_storage.get_learning_events_by_task("pattern_recognition")
        
        assert len(pattern_events) == 2
        assert all(event.task_type == "pattern_recognition" for event in pattern_events)
        
        # Test with minimum performance filter
        high_perf_events = sqlite_storage.get_learning_events_by_task(
            "pattern_recognition", min_performance_delta=0.1
        )
        
        assert len(high_perf_events) == 1
        assert high_perf_events[0].performance_delta == 0.15
    
    def test_best_strategies_analysis(self, sqlite_storage):
        """Test best strategies analysis functionality."""
        task_type = "optimization_task"
        
        # Store multiple learning events with different strategies
        strategies_data = [
            ("strategy_A", [0.1, 0.15, 0.12]),  # avg: 0.123
            ("strategy_B", [0.2, 0.18, 0.22]),  # avg: 0.2
            ("strategy_C", [0.05, 0.08]),       # avg: 0.065
        ]
        
        for strategy, deltas in strategies_data:
            for delta in deltas:
                sqlite_storage.store_learning_event(
                    task_type=task_type,
                    performance_delta=delta,
                    strategy_used=strategy
                )
        
        # Get best strategies
        best_strategies = sqlite_storage.get_best_strategies(task_type, limit=3)
        
        assert len(best_strategies) == 3
        
        # Should be ordered by average performance (descending)
        assert best_strategies[0][0] == "strategy_B"  # Highest avg
        assert best_strategies[1][0] == "strategy_A"  # Second highest
        assert best_strategies[2][0] == "strategy_C"  # Lowest
        
        # Check usage counts
        assert best_strategies[0][2] == 3  # strategy_B used 3 times
        assert best_strategies[1][2] == 3  # strategy_A used 3 times
        assert best_strategies[2][2] == 2  # strategy_C used 2 times
    
    def test_array_serialization_with_jax(self, sqlite_storage):
        """Test array serialization with JAX arrays."""
        # Test with JAX array
        jax_array = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        
        concept_id = sqlite_storage.store_concept(
            name="jax_concept",
            embedding=jax_array
        )
        
        retrieved_concept = sqlite_storage.retrieve_concept(concept_id)
        assert retrieved_concept is not None
        
        # Deserialize and check
        deserialized = sqlite_storage._deserialize_array(retrieved_concept.embedding)
        np.testing.assert_array_equal(np.array(jax_array), deserialized)
    
    def test_database_backup_and_restore(self, sqlite_storage, temp_db_path):
        """Test database backup functionality."""
        # Add some data
        sqlite_storage.store_episode(
            experience_data=np.random.rand(10, 5),
            performance_score=0.8,
            context_hash="backup_test"
        )
        
        sqlite_storage.store_concept("backup_concept", np.random.rand(64))
        
        # Create backup
        backup_path = sqlite_storage.create_backup()
        
        assert backup_path.exists()
        assert backup_path.name.startswith(temp_db_path.stem + "_backup_")
        
        # Verify backup contains data
        backup_storage = SQLiteStorage(db_path=backup_path)
        stats = backup_storage.get_database_stats()
        
        assert stats['episodes_count'] == 1
        assert stats['concepts_count'] == 1
        
        backup_storage.close()
    
    def test_database_vacuum(self, sqlite_storage):
        """Test database vacuum operation."""
        # Add and remove data to create fragmentation
        episode_ids = []
        for i in range(10):
            episode_id = sqlite_storage.store_episode(
                experience_data=np.random.rand(100, 50),
                performance_score=0.5,
                context_hash=f"vacuum_test_{i}"
            )
            episode_ids.append(episode_id)
        
        # Vacuum database
        result = sqlite_storage.vacuum_database()
        assert result is True
    
    def test_database_statistics(self, sqlite_storage):
        """Test database statistics collection."""
        # Add some data
        sqlite_storage.store_episode(
            experience_data=np.random.rand(10, 5),
            performance_score=0.8,
            context_hash="stats_test"
        )
        
        sqlite_storage.store_concept("stats_concept", np.random.rand(64))
        
        sqlite_storage.store_learning_event(
            task_type="stats_task",
            performance_delta=0.1,
            strategy_used="stats_strategy"
        )
        
        # Get statistics
        stats = sqlite_storage.get_database_stats()
        
        assert stats['episodes_count'] == 1
        assert stats['concepts_count'] == 1
        assert stats['learning_events_count'] == 1
        assert stats['concept_relationships_count'] == 0
        assert 'db_size_bytes' in stats
        assert 'db_size_mb' in stats
        assert stats['schema_version'] == 1
    
    def test_auto_backup_functionality(self, sqlite_storage):
        """Test automatic backup functionality."""
        # Set short backup interval for testing
        sqlite_storage.backup_interval = 1  # 1 second
        sqlite_storage._last_backup = time.time() - 2  # 2 seconds ago
        
        # Should create backup
        backup_path = sqlite_storage.auto_backup_if_needed()
        assert backup_path is not None
        assert backup_path.exists()
        
        # Should not create backup immediately after
        backup_path2 = sqlite_storage.auto_backup_if_needed()
        assert backup_path2 is None
    
    def test_backup_cleanup(self, sqlite_storage, temp_db_path):
        """Test old backup cleanup functionality."""
        sqlite_storage.max_backups = 2
        
        # Create multiple backups
        backup_paths = []
        for i in range(4):
            time.sleep(0.1)  # Ensure different timestamps
            backup_path = sqlite_storage.create_backup()
            backup_paths.append(backup_path)
        
        # Check that only max_backups files remain
        backup_pattern = f"{temp_db_path.stem}_backup_*.db"
        remaining_backups = list(temp_db_path.parent.glob(backup_pattern))
        
        assert len(remaining_backups) == 2
        
        # Check that newest backups are kept
        remaining_names = [b.name for b in remaining_backups]
        assert backup_paths[-1].name in remaining_names  # Most recent
        assert backup_paths[-2].name in remaining_names  # Second most recent
    
    def test_context_manager_usage(self, temp_db_path):
        """Test SQLite storage as context manager."""
        with SQLiteStorage(db_path=temp_db_path) as storage:
            episode_id = storage.store_episode(
                experience_data=np.array([1, 2, 3]),
                performance_score=0.7,
                context_hash="context_manager_test"
            )
            
            assert episode_id > 0
        
        # Storage should be closed after context exit
        # Verify data persists
        with SQLiteStorage(db_path=temp_db_path) as storage2:
            episode = storage2.retrieve_episode(episode_id)
            assert episode is not None
            assert episode.context_hash == "context_manager_test"
    
    def test_error_handling(self, sqlite_storage):
        """Test error handling in various scenarios."""
        # Test retrieving non-existent episode
        non_existent = sqlite_storage.retrieve_episode(99999)
        assert non_existent is None
        
        # Test retrieving non-existent concept
        non_existent_concept = sqlite_storage.retrieve_concept_by_name("non_existent")
        assert non_existent_concept is None
        
        # Test empty search results
        empty_results = sqlite_storage.search_concepts("nonexistentpattern")
        assert len(empty_results) == 0
        
        # Test relationships for non-existent concept
        no_relationships = sqlite_storage.get_concept_relationships(99999)
        assert len(no_relationships) == 0
    
    def test_concurrent_access_simulation(self, sqlite_storage):
        """Test concurrent access patterns (simulated)."""
        import threading
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(5):
                    episode_id = sqlite_storage.store_episode(
                        experience_data=np.random.rand(10, 5),
                        performance_score=0.5 + i * 0.1,
                        context_hash=f"worker_{worker_id}_episode_{i}"
                    )
                    results.append(episode_id)
                    
                    # Retrieve episode
                    episode = sqlite_storage.retrieve_episode(episode_id)
                    assert episode is not None
                    
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 15  # 3 workers * 5 episodes each
        assert len(set(results)) == 15  # All IDs should be unique


class TestSQLiteStorageIntegration:
    """Integration tests for SQLite storage."""
    
    @pytest.mark.integration
    def test_large_data_handling(self):
        """Test handling of large datasets."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        try:
            with SQLiteStorage(db_path=db_path) as storage:
                # Store large episodes
                large_episodes = []
                for i in range(100):
                    large_data = np.random.rand(1000, 100)  # Large array
                    episode_id = storage.store_episode(
                        experience_data=large_data,
                        performance_score=np.random.rand(),
                        context_hash=f"large_context_{i % 10}"
                    )
                    large_episodes.append(episode_id)
                
                # Store many concepts
                for i in range(500):
                    embedding = np.random.rand(512)  # Large embedding
                    storage.store_concept(
                        name=f"concept_{i}",
                        embedding=embedding,
                        metadata={"batch": i // 50}
                    )
                
                # Verify data integrity
                stats = storage.get_database_stats()
                assert stats['episodes_count'] == 100
                assert stats['concepts_count'] == 500
                
                # Test retrieval performance
                start_time = time.time()
                for episode_id in large_episodes[:10]:
                    episode = storage.retrieve_episode(episode_id)
                    assert episode is not None
                retrieval_time = time.time() - start_time
                
                # Should retrieve 10 large episodes in reasonable time
                assert retrieval_time < 5.0
        
        finally:
            if db_path.exists():
                db_path.unlink()
    
    @pytest.mark.integration
    def test_performance_benchmarks(self):
        """Test SQLite storage performance benchmarks."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        try:
            with SQLiteStorage(db_path=db_path) as storage:
                # Benchmark episode storage
                episodes_data = [
                    (np.random.rand(100, 50), np.random.rand(), f"context_{i}")
                    for i in range(1000)
                ]
                
                start_time = time.time()
                episode_ids = []
                for data, score, context in episodes_data:
                    episode_id = storage.store_episode(data, score, context)
                    episode_ids.append(episode_id)
                storage_time = time.time() - start_time
                
                # Benchmark episode retrieval
                start_time = time.time()
                for episode_id in episode_ids[:100]:
                    episode = storage.retrieve_episode(episode_id)
                    assert episode is not None
                retrieval_time = time.time() - start_time
                
                # Performance assertions (adjust based on hardware)
                assert storage_time < 30.0  # Store 1000 episodes in < 30 seconds
                assert retrieval_time < 5.0  # Retrieve 100 episodes in < 5 seconds
                
                logger.info(f"Storage time: {storage_time:.2f}s, Retrieval time: {retrieval_time:.2f}s")
        
        finally:
            if db_path.exists():
                db_path.unlink()
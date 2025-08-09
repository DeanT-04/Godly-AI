"""
Tests for Episodic Memory System

This module contains comprehensive tests for the episodic memory implementation,
including experience storage, replay mechanisms, consolidation, and retrieval.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time
from typing import Dict, List, Any

from src.memory.episodic import (
    EpisodicMemory,
    EpisodicMemoryParams,
    EpisodicMemoryState,
    Episode,
    Experience,
    ConsolidationNode,
    create_episodic_memory
)


class TestEpisodicMemoryParams:
    """Test EpisodicMemoryParams configuration."""
    
    def test_default_params(self):
        """Test default parameter values."""
        params = EpisodicMemoryParams()
        
        assert params.max_episodes == 10000
        assert params.max_episode_length == 1000
        assert params.compression_threshold == 0.8
        assert params.compression_ratio == 0.5
        assert params.temporal_window == 10.0
        assert params.context_decay == 0.1
        assert params.replay_batch_size == 32
        assert params.replay_priority_alpha == 0.6
        assert params.replay_beta == 0.4
        assert params.consolidation_interval == 3600.0
        assert params.consolidation_threshold == 100
        assert params.similarity_threshold == 0.7
        assert params.max_retrieval_results == 10
        assert params.reservoir_size == 300
        assert params.encoding_timesteps == 20
    
    def test_custom_params(self):
        """Test custom parameter configuration."""
        params = EpisodicMemoryParams(
            max_episodes=5000,
            replay_batch_size=64,
            similarity_threshold=0.9
        )
        
        assert params.max_episodes == 5000
        assert params.replay_batch_size == 64
        assert params.similarity_threshold == 0.9
        # Other params should remain default
        assert params.max_episode_length == 1000
        assert params.compression_threshold == 0.8


class TestExperience:
    """Test Experience data structure."""
    
    def test_experience_creation(self):
        """Test creating Experience objects."""
        observation = jnp.array([1.0, 2.0, 3.0])
        action = jnp.array([0.5, -0.5])
        reward = 1.5
        next_observation = jnp.array([1.1, 2.1, 3.1])
        timestamp = time.time()
        context = {"task": "navigation", "difficulty": 0.7}
        
        experience = Experience(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            timestamp=timestamp,
            context=context
        )
        
        assert jnp.allclose(experience.observation, observation)
        assert jnp.allclose(experience.action, action)
        assert experience.reward == reward
        assert jnp.allclose(experience.next_observation, next_observation)
        assert experience.timestamp == timestamp
        assert experience.context == context
    
    def test_experience_with_empty_context(self):
        """Test Experience with empty context."""
        experience = Experience(
            observation=jnp.array([1.0]),
            action=jnp.array([0.0]),
            reward=0.0,
            next_observation=jnp.array([1.0]),
            timestamp=time.time(),
            context={}
        )
        
        assert experience.context == {}


class TestEpisodicMemory:
    """Test EpisodicMemory class functionality."""
    
    @pytest.fixture
    def memory(self):
        """Create EpisodicMemory instance for testing."""
        params = EpisodicMemoryParams(
            max_episodes=100,
            max_episode_length=50,
            reservoir_size=50,  # Smaller for faster tests
            encoding_timesteps=5,
            consolidation_threshold=5
        )
        return EpisodicMemory(params)
    
    @pytest.fixture
    def sample_experience(self):
        """Create sample experience for testing."""
        return Experience(
            observation=jnp.array([1.0, 2.0, 3.0]),
            action=jnp.array([0.5, -0.5]),
            reward=1.0,
            next_observation=jnp.array([1.1, 2.1, 3.1]),
            timestamp=time.time(),
            context={"task": "test", "step": 1}
        )
    
    def test_initialization(self, memory):
        """Test EpisodicMemory initialization."""
        assert isinstance(memory, EpisodicMemory)
        assert isinstance(memory.params, EpisodicMemoryParams)
        assert hasattr(memory, 'encoding_lsm')
    
    def test_init_state(self, memory):
        """Test state initialization."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        assert isinstance(state, EpisodicMemoryState)
        assert len(state.episodes) == 0
        assert len(state.consolidation_tree) == 0
        assert state.current_episode is None
        assert state.total_episodes == 0
        assert isinstance(state.global_time, float)
    
    def test_store_single_episode(self, memory, sample_experience):
        """Test storing a single episode."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        new_state, episode_id = memory.store_episode(state, sample_experience, key)
        
        assert len(new_state.episodes) == 1
        assert episode_id in new_state.episodes
        assert new_state.total_episodes == 1
        
        stored_episode = new_state.episodes[episode_id]
        assert isinstance(stored_episode, Episode)
        assert len(stored_episode.experiences) == 1
        assert stored_episode.experiences[0].reward == sample_experience.reward
        assert stored_episode.total_reward == sample_experience.reward
        assert stored_episode.episode_length == 1
    
    def test_store_multiple_episodes(self, memory):
        """Test storing multiple episodes."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Store multiple episodes
        for i in range(5):
            experience = Experience(
                observation=jnp.array([float(i), float(i+1)]),
                action=jnp.array([float(i)]),
                reward=float(i),
                next_observation=jnp.array([float(i+1), float(i+2)]),
                timestamp=time.time() + i,
                context={"episode": i}
            )
            
            key, subkey = random.split(key)
            state, episode_id = memory.store_episode(state, experience, subkey)
        
        assert len(state.episodes) == 5
        assert state.total_episodes == 5
        assert len(state.replay_priorities) == 5
    
    def test_replay_episode(self, memory, sample_experience):
        """Test episode replay functionality."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Store episode
        state, episode_id = memory.store_episode(state, sample_experience, key)
        
        # Replay episode
        new_state, replayed_experience = memory.replay_episode(state, episode_id)
        
        assert replayed_experience is not None
        assert jnp.allclose(replayed_experience.observation, sample_experience.observation)
        assert replayed_experience.reward == sample_experience.reward
        
        # Check that access count increased
        replayed_episode = new_state.episodes[episode_id]
        original_episode = state.episodes[episode_id]
        assert replayed_episode.access_count == original_episode.access_count + 1
        assert replayed_episode.importance_weight > original_episode.importance_weight
    
    def test_replay_nonexistent_episode(self, memory):
        """Test replaying non-existent episode."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        new_state, replayed_experience = memory.replay_episode(state, "nonexistent")
        
        assert replayed_experience is None
        assert new_state == state  # State should be unchanged
    
    def test_sample_replay_batch(self, memory):
        """Test sampling replay batches."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Store multiple episodes with different rewards (importance)
        experiences = []
        for i in range(10):
            experience = Experience(
                observation=jnp.array([float(i)]),
                action=jnp.array([0.0]),
                reward=float(i),  # Different rewards for different importance
                next_observation=jnp.array([float(i+1)]),
                timestamp=time.time() + i,
                context={"batch_test": i}
            )
            experiences.append(experience)
            
            key, subkey = random.split(key)
            state, _ = memory.store_episode(state, experience, subkey)
        
        # Sample replay batch
        key, subkey = random.split(key)
        new_state, batch = memory.sample_replay_batch(state, batch_size=5, key=subkey)
        
        assert len(batch) == 5
        assert all(isinstance(exp, Experience) for exp in batch)
        
        # Test with default batch size
        key, subkey = random.split(key)
        new_state, batch = memory.sample_replay_batch(new_state, key=subkey)
        assert len(batch) <= memory.params.replay_batch_size
    
    def test_sample_replay_batch_empty_memory(self, memory):
        """Test sampling from empty memory."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        new_state, batch = memory.sample_replay_batch(state, key=key)
        
        assert len(batch) == 0
        assert new_state == state
    
    def test_start_episode(self, memory):
        """Test starting new episodes."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Start first episode
        new_state = memory.start_episode(state, {"initial": "context"})
        
        assert new_state.current_episode == []
        assert isinstance(new_state.current_episode_start, float)
        
        # Add experience to current episode
        experience = Experience(
            observation=jnp.array([1.0]),
            action=jnp.array([0.0]),
            reward=1.0,
            next_observation=jnp.array([2.0]),
            timestamp=time.time(),
            context={"step": 1}
        )
        
        new_state = memory.store_experience(new_state, experience)
        assert len(new_state.current_episode) == 1
        
        # Start new episode (should finalize previous)
        final_state = memory.start_episode(new_state)
        assert final_state.current_episode == []
        assert len(final_state.episodes) == 1  # Previous episode stored
    
    def test_store_experience_sequence(self, memory):
        """Test storing sequence of experiences in episode."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Start episode
        state = memory.start_episode(state)
        
        # Add multiple experiences
        for i in range(5):
            experience = Experience(
                observation=jnp.array([float(i)]),
                action=jnp.array([float(i % 2)]),
                reward=1.0 if i % 2 == 0 else -1.0,
                next_observation=jnp.array([float(i+1)]),
                timestamp=time.time() + i,
                context={"step": i}
            )
            
            state = memory.store_experience(state, experience)
        
        assert len(state.current_episode) == 5
        
        # Finalize episode by starting new one
        final_state = memory.start_episode(state)
        assert len(final_state.episodes) == 1
        
        # Check stored episode
        episode = list(final_state.episodes.values())[0]
        assert episode.episode_length == 5
        assert len(episode.experiences) == 5
        assert episode.total_reward == 1.0  # 3 positive, 2 negative
    
    def test_episode_length_limit(self, memory):
        """Test automatic episode finalization at length limit."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Start episode
        state = memory.start_episode(state)
        
        # Add experiences up to limit
        max_length = memory.params.max_episode_length
        for i in range(max_length + 5):  # Go over limit
            experience = Experience(
                observation=jnp.array([float(i)]),
                action=jnp.array([0.0]),
                reward=1.0,
                next_observation=jnp.array([float(i+1)]),
                timestamp=time.time() + i,
                context={"step": i}
            )
            
            state = memory.store_experience(state, experience)
            
            # Should auto-finalize and start new episode at limit
            if i >= max_length - 1:
                assert len(state.current_episode) < max_length
        
        # Should have created at least one complete episode
        assert len(state.episodes) >= 1
    
    def test_query_episodes_by_context(self, memory):
        """Test querying episodes by context."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Store episodes with different contexts
        contexts = [
            {"task": "navigation", "difficulty": 0.5},
            {"task": "navigation", "difficulty": 0.8},
            {"task": "manipulation", "difficulty": 0.5},
            {"task": "communication", "difficulty": 0.3}
        ]
        
        for i, context in enumerate(contexts):
            experience = Experience(
                observation=jnp.array([float(i)]),
                action=jnp.array([0.0]),
                reward=1.0,
                next_observation=jnp.array([float(i+1)]),
                timestamp=time.time() + i,
                context=context
            )
            
            key, subkey = random.split(key)
            state, _ = memory.store_episode(state, experience, subkey)
        
        # Query for navigation tasks
        key, subkey = random.split(key)
        new_state, results = memory.query_episodes(
            state, 
            {"task": "navigation"}, 
            key=subkey
        )
        
        # Should find episodes with navigation task
        assert len(results) >= 1
        
        # Verify results contain navigation episodes
        for episode_id in results:
            episode = state.episodes[episode_id]
            assert episode.context_summary.get("task") == "navigation"
    
    def test_query_episodes_by_temporal_range(self, memory):
        """Test querying episodes by temporal range."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        base_time = time.time()
        
        # Store episodes at different times
        for i in range(5):
            experience = Experience(
                observation=jnp.array([float(i)]),
                action=jnp.array([0.0]),
                reward=1.0,
                next_observation=jnp.array([float(i+1)]),
                timestamp=base_time + i * 10,  # 10 second intervals
                context={"time_test": i}
            )
            
            key, subkey = random.split(key)
            state, _ = memory.store_episode(state, experience, subkey)
        
        # Query for episodes in middle time range
        temporal_range = (base_time + 15, base_time + 35)
        key, subkey = random.split(key)
        new_state, results = memory.query_episodes(
            state, 
            {}, 
            temporal_range=temporal_range,
            key=subkey
        )
        
        # Should find episodes in time range
        for episode_id in results:
            episode = state.episodes[episode_id]
            assert temporal_range[0] <= episode.start_time <= temporal_range[1]
    
    def test_consolidate_memories(self, memory):
        """Test memory consolidation functionality."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Store multiple similar episodes (old enough for consolidation)
        old_time = time.time() - 7200  # 2 hours ago
        
        for i in range(10):
            experience = Experience(
                observation=jnp.array([1.0, 2.0]),  # Similar observations
                action=jnp.array([0.5]),
                reward=1.0,
                next_observation=jnp.array([1.1, 2.1]),
                timestamp=old_time + i,
                context={"consolidation_test": True, "group": i % 2}
            )
            
            key, subkey = random.split(key)
            state, episode_id = memory.store_episode(state, experience, subkey)
            
            # Make episodes old by updating timestamps
            episode = state.episodes[episode_id]
            updated_episode = Episode(
                episode_id=episode.episode_id,
                experiences=episode.experiences,
                start_time=old_time + i,
                end_time=old_time + i,
                total_reward=episode.total_reward,
                episode_length=episode.episode_length,
                context_summary=episode.context_summary,
                encoding=episode.encoding,
                compression_level=episode.compression_level,
                access_count=episode.access_count,
                last_accessed=old_time + i,  # Make it old
                importance_weight=episode.importance_weight,
                consolidation_level=episode.consolidation_level
            )
            
            new_episodes = state.episodes.copy()
            new_episodes[episode_id] = updated_episode
            state = state._replace(episodes=new_episodes)
        
        # Perform consolidation
        key, subkey = random.split(key)
        consolidated_state = memory.consolidate_memories(state, subkey)
        
        # Should have created some consolidation nodes
        assert len(consolidated_state.consolidation_tree) >= 0
        
        # Episodes should have increased consolidation levels
        for episode in consolidated_state.episodes.values():
            assert episode.consolidation_level >= 0
    
    def test_memory_statistics(self, memory):
        """Test memory statistics computation."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Test empty memory statistics
        stats = memory.get_memory_statistics(state)
        assert stats['num_episodes'] == 0
        assert stats['total_experiences'] == 0
        assert stats['memory_utilization'] == 0.0
        
        # Add some episodes
        for i in range(5):
            experience = Experience(
                observation=jnp.array([float(i)]),
                action=jnp.array([0.0]),
                reward=float(i),
                next_observation=jnp.array([float(i+1)]),
                timestamp=time.time() + i,
                context={"stats_test": i}
            )
            
            key, subkey = random.split(key)
            state, _ = memory.store_episode(state, experience, subkey)
        
        # Test populated memory statistics
        stats = memory.get_memory_statistics(state)
        assert stats['num_episodes'] == 5
        assert stats['total_experiences'] == 5
        assert stats['memory_utilization'] == 5 / memory.params.max_episodes
        assert stats['average_episode_length'] == 1.0
        assert stats['total_reward_stored'] == 10.0  # 0+1+2+3+4
        assert isinstance(stats['oldest_episode_age'], float)
        assert stats['most_accessed_episode'] is not None
    
    def test_capacity_management(self, memory):
        """Test memory capacity management."""
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Fill memory to capacity
        max_episodes = memory.params.max_episodes
        
        for i in range(max_episodes + 10):  # Go over capacity
            experience = Experience(
                observation=jnp.array([float(i)]),
                action=jnp.array([0.0]),
                reward=float(i % 5),  # Varying importance
                next_observation=jnp.array([float(i+1)]),
                timestamp=time.time() + i,
                context={"capacity_test": i}
            )
            
            key, subkey = random.split(key)
            state, _ = memory.store_episode(state, experience, subkey)
        
        # Should not exceed capacity
        assert len(state.episodes) <= max_episodes
        assert state.total_episodes == max_episodes + 10  # Total stored count


class TestConvenienceFunctions:
    """Test convenience functions for creating episodic memory."""
    
    def test_create_standard_memory(self):
        """Test creating standard episodic memory."""
        memory = create_episodic_memory("standard")
        
        assert isinstance(memory, EpisodicMemory)
        assert memory.params.max_episodes == 10000
        assert memory.params.replay_batch_size == 32
    
    def test_create_small_memory(self):
        """Test creating small episodic memory."""
        memory = create_episodic_memory("small")
        
        assert isinstance(memory, EpisodicMemory)
        assert memory.params.max_episodes == 100
        assert memory.params.max_episode_length == 50
        assert memory.params.reservoir_size == 100
        assert memory.params.consolidation_threshold == 10
    
    def test_create_large_memory(self):
        """Test creating large episodic memory."""
        memory = create_episodic_memory("large")
        
        assert isinstance(memory, EpisodicMemory)
        assert memory.params.max_episodes == 50000
        assert memory.params.max_episode_length == 5000
        assert memory.params.reservoir_size == 500
        assert memory.params.consolidation_threshold == 1000
    
    def test_create_fast_consolidation_memory(self):
        """Test creating fast consolidation memory."""
        memory = create_episodic_memory("fast_consolidation")
        
        assert isinstance(memory, EpisodicMemory)
        assert memory.params.consolidation_interval == 600.0  # 10 minutes
        assert memory.params.consolidation_threshold == 20
        assert memory.params.compression_threshold == 0.6
    
    def test_create_slow_consolidation_memory(self):
        """Test creating slow consolidation memory."""
        memory = create_episodic_memory("slow_consolidation")
        
        assert isinstance(memory, EpisodicMemory)
        assert memory.params.consolidation_interval == 86400.0  # 24 hours
        assert memory.params.consolidation_threshold == 500
        assert memory.params.compression_threshold == 0.9
    
    def test_create_high_priority_memory(self):
        """Test creating high priority memory."""
        memory = create_episodic_memory("high_priority")
        
        assert isinstance(memory, EpisodicMemory)
        assert memory.params.replay_priority_alpha == 1.0
        assert memory.params.replay_beta == 0.8
        assert memory.params.replay_batch_size == 64
    
    def test_create_invalid_memory_type(self):
        """Test creating memory with invalid type."""
        with pytest.raises(ValueError, match="Unknown memory type"):
            create_episodic_memory("invalid_type")


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    def test_full_episode_lifecycle(self):
        """Test complete episode lifecycle."""
        memory = create_episodic_memory("small")
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Start episode
        state = memory.start_episode(state, {"task": "integration_test"})
        
        # Add experiences
        experiences = []
        for i in range(10):
            experience = Experience(
                observation=jnp.array([float(i), float(i+1)]),
                action=jnp.array([float(i % 3)]),
                reward=1.0 if i % 2 == 0 else -0.5,
                next_observation=jnp.array([float(i+1), float(i+2)]),
                timestamp=time.time() + i,
                context={"step": i, "phase": "training"}
            )
            experiences.append(experience)
            state = memory.store_experience(state, experience)
        
        # Finalize episode
        state = memory.start_episode(state)
        
        # Verify episode was stored
        assert len(state.episodes) == 1
        episode = list(state.episodes.values())[0]
        assert episode.episode_length == 10
        assert len(episode.experiences) == 10
        
        # Test replay
        episode_id = list(state.episodes.keys())[0]
        state, replayed_exp = memory.replay_episode(state, episode_id)
        assert replayed_exp is not None
        
        # Test batch sampling
        key, subkey = random.split(key)
        state, batch = memory.sample_replay_batch(state, batch_size=5, key=subkey)
        assert len(batch) == 1  # Only one episode available
        
        # Test querying
        key, subkey = random.split(key)
        state, results = memory.query_episodes(
            state, 
            {"task": "integration_test"}, 
            key=subkey
        )
        assert len(results) == 1
        assert results[0] == episode_id
    
    def test_memory_under_stress(self):
        """Test memory system under stress conditions."""
        memory = create_episodic_memory("small")
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Rapidly store many episodes
        for i in range(200):  # More than capacity
            experience = Experience(
                observation=jnp.array([float(i % 10)]),
                action=jnp.array([float(i % 3)]),
                reward=np.random.normal(0, 1),
                next_observation=jnp.array([float((i+1) % 10)]),
                timestamp=time.time() + i * 0.1,
                context={"stress_test": i, "batch": i // 10}
            )
            
            key, subkey = random.split(key)
            state, _ = memory.store_episode(state, experience, subkey)
        
        # Memory should handle capacity gracefully
        assert len(state.episodes) <= memory.params.max_episodes
        assert state.total_episodes == 200
        
        # Should still be able to sample and query
        key, subkey = random.split(key)
        state, batch = memory.sample_replay_batch(state, key=subkey)
        assert len(batch) > 0
        
        # Statistics should be reasonable
        stats = memory.get_memory_statistics(state)
        assert stats['num_episodes'] > 0
        assert stats['memory_utilization'] <= 1.0
    
    def test_temporal_consistency(self):
        """Test temporal consistency of stored episodes."""
        memory = create_episodic_memory("small")
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        base_time = time.time()
        timestamps = []
        
        # Store episodes with specific timestamps
        for i in range(10):
            timestamp = base_time + i * 60  # 1 minute intervals
            timestamps.append(timestamp)
            
            experience = Experience(
                observation=jnp.array([float(i)]),
                action=jnp.array([0.0]),
                reward=1.0,
                next_observation=jnp.array([float(i+1)]),
                timestamp=timestamp,
                context={"temporal_test": i}
            )
            
            key, subkey = random.split(key)
            state, _ = memory.store_episode(state, experience, subkey)
        
        # Query episodes in temporal ranges
        for i in range(5):
            start_time = base_time + i * 120
            end_time = start_time + 180
            
            key, subkey = random.split(key)
            state, results = memory.query_episodes(
                state, 
                {}, 
                temporal_range=(start_time, end_time),
                key=subkey
            )
            
            # Verify temporal consistency
            for episode_id in results:
                episode = state.episodes[episode_id]
                assert start_time <= episode.start_time <= end_time


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
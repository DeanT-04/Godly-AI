"""
Enhanced Unit Tests for Memory Systems

Comprehensive testing of working memory, episodic memory, semantic memory,
and meta-memory with property-based testing and extensive coverage.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock
import logging
from typing import Dict, Any, List, Tuple

from .test_framework import (
    TestDataGenerators,
    MockDependencies,
    CoverageAnalyzer,
    PerformanceProfiler,
    assert_memory_consistency,
    TestMetrics
)

logger = logging.getLogger(__name__)


class TestEnhancedWorkingMemory:
    """Enhanced unit tests for working memory system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.profiler = PerformanceProfiler()
        self.test_data_gen = TestDataGenerators()
        
    @pytest.mark.unit
    def test_working_memory_initialization_comprehensive(self):
        """Test comprehensive working memory initialization"""
        from src.memory.working.working_memory import WorkingMemory
        
        # Test default initialization
        wm = WorkingMemory()
        assert wm.capacity > 0
        assert wm.decay_rate > 0
        assert wm.decay_rate < 1.0
        
        # Test custom parameters
        custom_params = {
            'capacity': 50,
            'decay_rate': 0.02,
            'attention_strength': 2.0,
            'pattern_dim': 128
        }
        
        wm_custom = WorkingMemory(**custom_params)
        for param, value in custom_params.items():
            if hasattr(wm_custom, param):
                assert getattr(wm_custom, param) == value
    
    @pytest.mark.unit
    @given(
        capacity=st.integers(min_value=5, max_value=100),
        pattern_dim=st.integers(min_value=10, max_value=200),
        n_patterns=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=20, deadline=2000)
    def test_working_memory_capacity_properties(self, capacity, pattern_dim, n_patterns):
        """Property-based test for working memory capacity management"""
        from src.memory.working.working_memory import WorkingMemory
        
        assume(n_patterns <= capacity * 2)  # Allow some overflow testing
        
        wm = WorkingMemory(capacity=capacity, pattern_dim=pattern_dim)
        state = wm.init_state()
        
        # Generate test patterns
        key = jax.random.PRNGKey(42)
        patterns = []
        stored_ids = []
        
        for i in range(n_patterns):
            key, subkey = jax.random.split(key)
            pattern = jax.random.normal(subkey, (pattern_dim,))
            patterns.append(pattern)
            
            pattern_id = wm.store_pattern(state, pattern, timestamp=float(i))
            stored_ids.append(pattern_id)
        
        # Property: should not exceed capacity
        active_patterns = len([pid for pid in stored_ids if pid is not None])
        assert active_patterns <= capacity, f"Exceeded capacity: {active_patterns} > {capacity}"
        
        # Property: retrieval should work for stored patterns
        for i, pattern_id in enumerate(stored_ids):
            if pattern_id is not None:
                retrieved, confidence = wm.retrieve_pattern(state, patterns[i])
                assert retrieved is not None, f"Failed to retrieve pattern {i}"
                assert confidence >= 0.0, f"Invalid confidence: {confidence}"
    
    @pytest.mark.unit
    def test_working_memory_attention_mechanism(self):
        """Test attention mechanism in working memory"""
        from src.memory.working.working_memory import WorkingMemory
        
        wm = WorkingMemory(capacity=10, attention_strength=1.5)
        state = wm.init_state()
        
        # Store patterns with different attention weights
        key = jax.random.PRNGKey(42)
        patterns = []
        attention_weights = [0.1, 0.5, 1.0, 2.0, 0.3]
        
        for i, attention in enumerate(attention_weights):
            key, subkey = jax.random.split(key)
            pattern = jax.random.normal(subkey, (64,))
            patterns.append(pattern)
            
            wm.store_pattern(state, pattern, timestamp=float(i))
            wm.update_attention_weights(state, jnp.array([attention]))
        
        # Test retrieval with attention bias
        query = patterns[2]  # High attention pattern
        retrieved, confidence = wm.retrieve_pattern(state, query)
        
        assert retrieved is not None
        assert confidence > 0.5, f"High attention pattern should have high confidence: {confidence}"
        
        # Test competitive retrieval
        similar_query = patterns[2] + jax.random.normal(key, (64,)) * 0.1
        retrieved_similar, confidence_similar = wm.retrieve_pattern(state, similar_query)
        
        assert confidence >= confidence_similar, "Original should have higher confidence than similar"
    
    @pytest.mark.unit
    def test_working_memory_decay_dynamics(self):
        """Test memory decay dynamics"""
        from src.memory.working.working_memory import WorkingMemory
        
        wm = WorkingMemory(capacity=20, decay_rate=0.1)
        state = wm.init_state()
        
        # Store patterns at different times
        key = jax.random.PRNGKey(42)
        patterns = []
        timestamps = [0.0, 1.0, 2.0, 5.0, 10.0]
        
        for i, timestamp in enumerate(timestamps):
            key, subkey = jax.random.split(key)
            pattern = jax.random.normal(subkey, (64,))
            patterns.append(pattern)
            wm.store_pattern(state, pattern, timestamp=timestamp)
        
        # Simulate time passage and test retrieval
        current_time = 15.0
        retrieval_confidences = []
        
        for i, pattern in enumerate(patterns):
            retrieved, confidence = wm.retrieve_pattern(state, pattern, current_time=current_time)
            retrieval_confidences.append(confidence)
        
        # Property: older patterns should have lower confidence due to decay
        for i in range(1, len(retrieval_confidences)):
            age_diff = timestamps[i] - timestamps[i-1]
            if age_diff > 0:
                # Allow some noise, but general trend should be decreasing confidence with age
                assert retrieval_confidences[i-1] >= retrieval_confidences[i] - 0.2, \
                    f"Older patterns should decay more: {retrieval_confidences}"
    
    @pytest.mark.unit
    def test_working_memory_pattern_completion(self):
        """Test pattern completion capabilities"""
        from src.memory.working.working_memory import WorkingMemory
        
        wm = WorkingMemory(capacity=15, pattern_dim=100)
        state = wm.init_state()
        
        # Store complete patterns
        key = jax.random.PRNGKey(42)
        complete_patterns = []
        
        for i in range(5):
            key, subkey = jax.random.split(key)
            pattern = jax.random.normal(subkey, (100,))
            complete_patterns.append(pattern)
            wm.store_pattern(state, pattern, timestamp=float(i))
        
        # Test partial pattern retrieval
        for i, complete_pattern in enumerate(complete_patterns):
            # Create partial pattern (50% missing)
            key, subkey = jax.random.split(key)
            mask = jax.random.bernoulli(subkey, 0.5, shape=(100,))
            partial_pattern = complete_pattern * mask
            
            # Attempt retrieval
            retrieved, confidence = wm.retrieve_pattern(state, partial_pattern)
            
            if retrieved is not None:
                # Check similarity to original
                similarity = jnp.corrcoef(retrieved, complete_pattern)[0, 1]
                assert similarity > 0.3, f"Pattern completion failed: similarity = {similarity}"
    
    @pytest.mark.unit
    def test_working_memory_performance_scaling(self):
        """Test performance scaling with memory size"""
        from src.memory.working.working_memory import WorkingMemory
        
        capacities = [10, 50, 100, 200]
        performance_metrics = []
        
        for capacity in capacities:
            wm = WorkingMemory(capacity=capacity, pattern_dim=64)
            state = wm.init_state()
            
            # Fill memory to capacity
            key = jax.random.PRNGKey(42)
            patterns = []
            
            def fill_memory():
                nonlocal patterns
                for i in range(capacity):
                    key_local, subkey = jax.random.split(jax.random.PRNGKey(42 + i))
                    pattern = jax.random.normal(subkey, (64,))
                    patterns.append(pattern)
                    wm.store_pattern(state, pattern, timestamp=float(i))
            
            # Profile memory filling
            metrics = self.profiler.profile_test_execution(fill_memory)
            performance_metrics.append(metrics)
        
        # Property: performance should scale reasonably with capacity
        for i in range(1, len(performance_metrics)):
            capacity_ratio = capacities[i] / capacities[i-1]
            time_ratio = performance_metrics[i].execution_time / performance_metrics[i-1].execution_time
            
            # Time should not scale worse than quadratically
            assert time_ratio <= capacity_ratio ** 2 + 1.0, \
                f"Performance scaling too poor: {time_ratio} vs capacity ratio {capacity_ratio}"


class TestEnhancedEpisodicMemory:
    """Enhanced unit tests for episodic memory system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.profiler = PerformanceProfiler()
        
    @pytest.mark.unit
    def test_episodic_memory_initialization_comprehensive(self):
        """Test comprehensive episodic memory initialization"""
        from src.memory.episodic.episodic_memory import EpisodicMemory
        
        # Test default initialization
        em = EpisodicMemory()
        assert em.max_episodes > 0
        assert em.consolidation_threshold > 0
        
        # Test custom parameters
        custom_params = {
            'max_episodes': 500,
            'consolidation_threshold': 0.8,
            'compression_ratio': 0.3,
            'episode_length_limit': 200
        }
        
        em_custom = EpisodicMemory(**custom_params)
        for param, value in custom_params.items():
            if hasattr(em_custom, param):
                assert getattr(em_custom, param) == value
    
    @pytest.mark.unit
    @given(
        episode_length=st.integers(min_value=5, max_value=50),
        n_episodes=st.integers(min_value=1, max_value=20),
        observation_dim=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=15, deadline=3000)
    def test_episodic_memory_storage_properties(self, episode_length, n_episodes, observation_dim):
        """Property-based test for episodic memory storage"""
        from src.memory.episodic.episodic_memory import EpisodicMemory
        
        em = EpisodicMemory(max_episodes=n_episodes * 2)
        state = em.init_state()
        
        # Generate and store episodes
        key = jax.random.PRNGKey(42)
        stored_episodes = []
        
        for episode_id in range(n_episodes):
            episode_data = []
            
            for step in range(episode_length):
                key, subkey = jax.random.split(key)
                experience = {
                    'observation': jax.random.normal(subkey, (observation_dim,)),
                    'action': jax.random.randint(subkey, (), 0, 4),
                    'reward': jax.random.uniform(subkey, minval=-1.0, maxval=1.0),
                    'timestamp': float(episode_id * episode_length + step)
                }
                episode_data.append(experience)
            
            episode_id_stored = em.store_episode(state, episode_data)
            stored_episodes.append((episode_id_stored, episode_data))
        
        # Property: all episodes should be stored (within capacity)
        valid_episodes = [ep for ep in stored_episodes if ep[0] is not None]
        assert len(valid_episodes) <= em.max_episodes
        
        # Property: stored episodes should be retrievable
        for episode_id, original_data in valid_episodes:
            retrieved_data = em.replay_episode(state, episode_id)
            assert retrieved_data is not None, f"Failed to retrieve episode {episode_id}"
            assert len(retrieved_data) > 0, "Retrieved episode should not be empty"
    
    @pytest.mark.unit
    def test_episodic_memory_consolidation(self):
        """Test memory consolidation mechanism"""
        from src.memory.episodic.episodic_memory import EpisodicMemory
        
        em = EpisodicMemory(max_episodes=20, consolidation_threshold=0.7)
        state = em.init_state()
        
        # Store episodes with varying importance
        key = jax.random.PRNGKey(42)
        episode_rewards = [0.1, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4]  # Some high-reward episodes
        
        for i, reward_level in enumerate(episode_rewards):
            episode_data = []
            
            for step in range(10):
                key, subkey = jax.random.split(key)
                experience = {
                    'observation': jax.random.normal(subkey, (32,)),
                    'action': jax.random.randint(subkey, (), 0, 4),
                    'reward': reward_level + jax.random.normal(subkey) * 0.1,
                    'timestamp': float(i * 10 + step)
                }
                episode_data.append(experience)
            
            em.store_episode(state, episode_data)
        
        # Trigger consolidation
        em.consolidate_memories(state)
        
        # Check that high-reward episodes are preserved
        stats = em.get_memory_statistics(state)
        assert stats['total_episodes'] > 0
        
        # Test that consolidated episodes are still accessible
        for episode_id in range(min(len(episode_rewards), em.max_episodes)):
            retrieved = em.replay_episode(state, str(episode_id))
            # High-reward episodes should be more likely to be preserved
            if episode_rewards[episode_id] > 0.6:
                assert retrieved is not None, f"High-reward episode {episode_id} should be preserved"
    
    @pytest.mark.unit
    def test_episodic_memory_temporal_queries(self):
        """Test temporal query capabilities"""
        from src.memory.episodic.episodic_memory import EpisodicMemory
        
        em = EpisodicMemory()
        state = em.init_state()
        
        # Store episodes with specific timestamps
        key = jax.random.PRNGKey(42)
        time_ranges = [(0.0, 10.0), (15.0, 25.0), (30.0, 40.0), (50.0, 60.0)]
        
        for i, (start_time, end_time) in enumerate(time_ranges):
            episode_data = []
            
            for step in range(10):
                timestamp = start_time + (end_time - start_time) * step / 9
                key, subkey = jax.random.split(key)
                
                experience = {
                    'observation': jax.random.normal(subkey, (32,)),
                    'action': i,  # Use episode index as action for identification
                    'reward': 0.5,
                    'timestamp': timestamp
                }
                episode_data.append(experience)
            
            em.store_episode(state, episode_data)
        
        # Query episodes by temporal range
        query_ranges = [(5.0, 20.0), (25.0, 45.0), (35.0, 55.0)]
        
        for query_start, query_end in query_ranges:
            matching_episodes = em.query_episodes_by_temporal_range(
                state, query_start, query_end
            )
            
            # Verify that returned episodes overlap with query range
            for episode_id in matching_episodes:
                episode_data = em.replay_episode(state, episode_id)
                if episode_data:
                    episode_times = [exp['timestamp'] for exp in episode_data]
                    episode_start = min(episode_times)
                    episode_end = max(episode_times)
                    
                    # Check for temporal overlap
                    overlap = not (episode_end < query_start or episode_start > query_end)
                    assert overlap, f"Episode {episode_id} should overlap with query range [{query_start}, {query_end}]"
    
    @pytest.mark.unit
    def test_episodic_memory_replay_batch(self):
        """Test batch replay functionality"""
        from src.memory.episodic.episodic_memory import EpisodicMemory
        
        em = EpisodicMemory()
        state = em.init_state()
        
        # Store multiple episodes
        key = jax.random.PRNGKey(42)
        n_episodes = 10
        
        for i in range(n_episodes):
            episode_data = []
            
            for step in range(5):
                key, subkey = jax.random.split(key)
                experience = {
                    'observation': jax.random.normal(subkey, (16,)),
                    'action': i % 4,  # Cycle through actions
                    'reward': jax.random.uniform(subkey, minval=0.0, maxval=1.0),
                    'timestamp': float(i * 5 + step)
                }
                episode_data.append(experience)
            
            em.store_episode(state, episode_data)
        
        # Test batch replay
        batch_sizes = [1, 3, 5, 8]
        
        for batch_size in batch_sizes:
            batch = em.sample_replay_batch(state, batch_size)
            
            assert len(batch) <= batch_size, f"Batch size exceeded: {len(batch)} > {batch_size}"
            assert len(batch) > 0, "Batch should not be empty"
            
            # Verify batch contents
            for episode_data in batch:
                assert len(episode_data) > 0, "Episode in batch should not be empty"
                assert all('observation' in exp for exp in episode_data), "All experiences should have observations"


class TestEnhancedSemanticMemory:
    """Enhanced unit tests for semantic memory system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.profiler = PerformanceProfiler()
        
    @pytest.mark.unit
    def test_semantic_memory_initialization_comprehensive(self):
        """Test comprehensive semantic memory initialization"""
        from src.memory.semantic.semantic_memory import SemanticMemory
        
        # Test default initialization
        sm = SemanticMemory()
        assert sm.embedding_dim > 0
        assert sm.similarity_threshold > 0
        assert sm.similarity_threshold < 1.0
        
        # Test custom parameters
        custom_params = {
            'embedding_dim': 256,
            'similarity_threshold': 0.8,
            'max_concepts': 1000,
            'learning_rate': 0.02
        }
        
        sm_custom = SemanticMemory(**custom_params)
        for param, value in custom_params.items():
            if hasattr(sm_custom, param):
                assert getattr(sm_custom, param) == value
    
    @pytest.mark.unit
    @given(
        n_experiences=st.integers(min_value=5, max_value=30),
        embedding_dim=st.integers(min_value=32, max_value=128),
        concept_clusters=st.integers(min_value=2, max_value=8)
    )
    @settings(max_examples=10, deadline=4000)
    def test_semantic_memory_concept_formation_properties(self, n_experiences, embedding_dim, concept_clusters):
        """Property-based test for concept formation"""
        from src.memory.semantic.semantic_memory import SemanticMemory
        
        sm = SemanticMemory(embedding_dim=embedding_dim, similarity_threshold=0.6)
        state = sm.init_state()
        
        # Generate clustered experiences
        key = jax.random.PRNGKey(42)
        experiences = []
        
        # Create concept clusters
        cluster_centers = []
        for i in range(concept_clusters):
            key, subkey = jax.random.split(key)
            center = jax.random.normal(subkey, (embedding_dim,))
            cluster_centers.append(center)
        
        # Generate experiences around cluster centers
        for i in range(n_experiences):
            cluster_id = i % concept_clusters
            key, subkey = jax.random.split(key)
            
            # Add noise to cluster center
            noise = jax.random.normal(subkey, (embedding_dim,)) * 0.3
            observation = cluster_centers[cluster_id] + noise
            
            experience = {
                'observation': observation,
                'action': cluster_id,
                'reward': 0.5,
                'timestamp': float(i),
                'context': {'cluster': cluster_id}
            }
            experiences.append(experience)
        
        # Extract concepts
        concepts = sm.extract_concepts(state, experiences)
        
        # Property: should form reasonable number of concepts
        assert len(concepts) >= 1, "Should form at least one concept"
        assert len(concepts) <= concept_clusters * 2, f"Too many concepts formed: {len(concepts)} > {concept_clusters * 2}"
        
        # Property: concepts should have valid embeddings
        for concept in concepts:
            assert concept.embedding.shape == (embedding_dim,), f"Invalid concept embedding shape: {concept.embedding.shape}"
            assert not jnp.any(jnp.isnan(concept.embedding)), "Concept embedding contains NaN"
    
    @pytest.mark.unit
    def test_semantic_memory_knowledge_graph_construction(self):
        """Test knowledge graph construction"""
        from src.memory.semantic.semantic_memory import SemanticMemory
        
        sm = SemanticMemory(embedding_dim=64, similarity_threshold=0.7)
        state = sm.init_state()
        
        # Create related concepts
        key = jax.random.PRNGKey(42)
        concept_data = [
            {'name': 'animal', 'features': [1, 1, 0, 0, 1]},
            {'name': 'dog', 'features': [1, 1, 1, 0, 1]},
            {'name': 'cat', 'features': [1, 1, 1, 1, 1]},
            {'name': 'vehicle', 'features': [0, 0, 0, 1, 0]},
            {'name': 'car', 'features': [0, 0, 0, 1, 1]},
        ]
        
        # Generate experiences for each concept
        experiences = []
        for i, concept_info in enumerate(concept_data):
            for rep in range(3):  # Multiple instances per concept
                key, subkey = jax.random.split(key)
                
                # Create embedding based on features
                base_embedding = jnp.array(concept_info['features'] * 12 + [0] * 4, dtype=jnp.float32)
                noise = jax.random.normal(subkey, (64,)) * 0.1
                observation = base_embedding + noise
                
                experience = {
                    'observation': observation,
                    'action': i,
                    'reward': 0.5,
                    'timestamp': float(i * 3 + rep),
                    'context': {'concept_name': concept_info['name']}
                }
                experiences.append(experience)
        
        # Extract concepts and build knowledge graph
        concepts = sm.extract_concepts(state, experiences)
        knowledge_graph = sm.build_knowledge_graph(state, concepts)
        
        # Property: graph should have nodes and edges
        assert knowledge_graph.number_of_nodes() > 0, "Knowledge graph should have nodes"
        
        # Property: similar concepts should be connected
        # Check for hierarchical relationships (animal -> dog, cat)
        node_names = [node for node in knowledge_graph.nodes()]
        
        # Find concept pairs that should be related
        related_pairs = [('animal', 'dog'), ('animal', 'cat'), ('vehicle', 'car')]
        
        for concept1, concept2 in related_pairs:
            # Check if concepts exist in graph (by similarity to expected names)
            nodes1 = [n for n in node_names if concept1 in str(n).lower()]
            nodes2 = [n for n in node_names if concept2 in str(n).lower()]
            
            if nodes1 and nodes2:
                # Check if there's a path between related concepts
                try:
                    import networkx as nx
                    path_exists = nx.has_path(knowledge_graph, nodes1[0], nodes2[0])
                    # Related concepts should be connected (directly or indirectly)
                    assert path_exists or knowledge_graph.has_edge(nodes1[0], nodes2[0]), \
                        f"Related concepts {concept1} and {concept2} should be connected"
                except ImportError:
                    # If NetworkX not available, just check basic structure
                    pass
    
    @pytest.mark.unit
    def test_semantic_memory_similarity_computation(self):
        """Test semantic similarity computation"""
        from src.memory.semantic.semantic_memory import SemanticMemory
        
        sm = SemanticMemory(embedding_dim=32)
        state = sm.init_state()
        
        # Create test concepts with known relationships
        key = jax.random.PRNGKey(42)
        
        # Identical concepts
        concept1 = jax.random.normal(key, (32,))
        concept2 = concept1.copy()
        
        # Similar concepts
        key, subkey = jax.random.split(key)
        concept3 = concept1 + jax.random.normal(subkey, (32,)) * 0.1
        
        # Dissimilar concept
        key, subkey = jax.random.split(key)
        concept4 = jax.random.normal(subkey, (32,))
        
        # Test similarity computation
        sim_identical = sm.compute_semantic_similarity(state, concept1, concept2)
        sim_similar = sm.compute_semantic_similarity(state, concept1, concept3)
        sim_dissimilar = sm.compute_semantic_similarity(state, concept1, concept4)
        
        # Property: identical concepts should have highest similarity
        assert sim_identical > sim_similar, f"Identical should be more similar: {sim_identical} vs {sim_similar}"
        assert sim_similar > sim_dissimilar, f"Similar should be more similar than dissimilar: {sim_similar} vs {sim_dissimilar}"
        
        # Property: similarity should be symmetric
        sim_reverse = sm.compute_semantic_similarity(state, concept2, concept1)
        assert abs(sim_identical - sim_reverse) < 1e-6, "Similarity should be symmetric"
        
        # Property: similarity should be bounded
        assert 0.0 <= sim_identical <= 1.0, f"Similarity out of bounds: {sim_identical}"
        assert 0.0 <= sim_similar <= 1.0, f"Similarity out of bounds: {sim_similar}"
        assert 0.0 <= sim_dissimilar <= 1.0, f"Similarity out of bounds: {sim_dissimilar}"
    
    @pytest.mark.unit
    def test_semantic_memory_knowledge_consolidation(self):
        """Test knowledge consolidation process"""
        from src.memory.semantic.semantic_memory import SemanticMemory
        
        sm = SemanticMemory(embedding_dim=64, max_concepts=20)
        state = sm.init_state()
        
        # Generate many overlapping concepts
        key = jax.random.PRNGKey(42)
        experiences = []
        
        # Create overlapping concept clusters
        base_concepts = []
        for i in range(5):
            key, subkey = jax.random.split(key)
            base_concept = jax.random.normal(subkey, (64,))
            base_concepts.append(base_concept)
        
        # Generate experiences with overlap
        for i in range(50):
            base_idx = i % len(base_concepts)
            key, subkey = jax.random.split(key)
            
            # Add varying amounts of noise
            noise_level = 0.2 if i % 10 < 7 else 0.5  # Most similar, some different
            noise = jax.random.normal(subkey, (64,)) * noise_level
            observation = base_concepts[base_idx] + noise
            
            experience = {
                'observation': observation,
                'action': base_idx,
                'reward': 0.5,
                'timestamp': float(i),
                'context': {'base_concept': base_idx}
            }
            experiences.append(experience)
        
        # Extract initial concepts
        initial_concepts = sm.extract_concepts(state, experiences)
        initial_count = len(initial_concepts)
        
        # Perform consolidation
        sm.consolidate_knowledge(state)
        
        # Get final concept count
        stats = sm.get_knowledge_statistics(state)
        final_count = stats.get('total_concepts', initial_count)
        
        # Property: consolidation should reduce redundant concepts
        assert final_count <= initial_count, f"Consolidation should reduce concepts: {final_count} <= {initial_count}"
        
        # Property: should maintain core concepts
        assert final_count >= len(base_concepts) // 2, f"Should maintain core concepts: {final_count} >= {len(base_concepts) // 2}"


class TestEnhancedMetaMemory:
    """Enhanced unit tests for meta-memory system"""
    
    def setup_method(self):
        """Setup test environment"""
        self.profiler = PerformanceProfiler()
        
    @pytest.mark.unit
    def test_meta_memory_initialization_comprehensive(self):
        """Test comprehensive meta-memory initialization"""
        from src.memory.meta.meta_memory import MetaMemory
        
        # Test default initialization
        mm = MetaMemory()
        assert mm.learning_history_size > 0
        assert mm.adaptation_rate > 0
        assert mm.adaptation_rate < 1.0
        
        # Test custom parameters
        custom_params = {
            'learning_history_size': 200,
            'adaptation_rate': 0.05,
            'strategy_similarity_threshold': 0.8,
            'meta_learning_rate': 0.02
        }
        
        mm_custom = MetaMemory(**custom_params)
        for param, value in custom_params.items():
            if hasattr(mm_custom, param):
                assert getattr(mm_custom, param) == value
    
    @pytest.mark.unit
    @given(
        n_learning_experiences=st.integers(min_value=5, max_value=30),
        n_task_types=st.integers(min_value=2, max_value=8),
        performance_range=st.tuples(
            st.floats(min_value=0.1, max_value=0.5),
            st.floats(min_value=0.6, max_value=1.0)
        )
    )
    @settings(max_examples=10, deadline=3000)
    def test_meta_memory_learning_experience_properties(self, n_learning_experiences, n_task_types, performance_range):
        """Property-based test for learning experience storage"""
        from src.memory.meta.meta_memory import MetaMemory
        
        assume(performance_range[0] < performance_range[1])
        
        mm = MetaMemory(learning_history_size=n_learning_experiences * 2)
        state = mm.init_state()
        
        # Generate learning experiences
        key = jax.random.PRNGKey(42)
        task_types = [f"task_type_{i}" for i in range(n_task_types)]
        strategies = ["strategy_A", "strategy_B", "strategy_C"]
        
        stored_experiences = []
        
        for i in range(n_learning_experiences):
            task_type = task_types[i % n_task_types]
            strategy = strategies[i % len(strategies)]
            
            key, subkey = jax.random.split(key)
            performance = jax.random.uniform(
                subkey, 
                minval=performance_range[0], 
                maxval=performance_range[1]
            )
            
            learning_time = jax.random.uniform(subkey, minval=1.0, maxval=100.0)
            
            mm.store_learning_experience(
                state, 
                task_type=task_type,
                performance=float(performance),
                strategy=strategy,
                learning_time=float(learning_time)
            )
            
            stored_experiences.append({
                'task_type': task_type,
                'performance': performance,
                'strategy': strategy,
                'learning_time': learning_time
            })
        
        # Property: should be able to retrieve strategies for known task types
        for task_type in task_types:
            retrieved_strategy = mm.retrieve_learning_strategy(state, task_type)
            assert retrieved_strategy is not None, f"Should retrieve strategy for task type {task_type}"
        
        # Property: learning statistics should be reasonable
        stats = mm.get_learning_statistics(state)
        assert stats['total_experiences'] <= n_learning_experiences
        assert stats['total_experiences'] > 0
        
        if 'average_performance' in stats:
            avg_perf = stats['average_performance']
            assert performance_range[0] <= avg_perf <= performance_range[1], \
                f"Average performance out of range: {avg_perf}"
    
    @pytest.mark.unit
    def test_meta_memory_strategy_adaptation(self):
        """Test strategy adaptation based on performance feedback"""
        from src.memory.meta.meta_memory import MetaMemory
        
        mm = MetaMemory(adaptation_rate=0.1)
        state = mm.init_state()
        
        # Store initial learning experiences with different strategies
        task_type = "classification_task"
        strategies = ["gradient_descent", "adam_optimizer", "rmsprop"]
        
        # Simulate that adam_optimizer performs better
        strategy_performances = {
            "gradient_descent": [0.6, 0.65, 0.7],
            "adam_optimizer": [0.8, 0.85, 0.9],
            "rmsprop": [0.7, 0.72, 0.75]
        }
        
        # Store experiences
        for strategy, performances in strategy_performances.items():
            for perf in performances:
                mm.store_learning_experience(
                    state,
                    task_type=task_type,
                    performance=perf,
                    strategy=strategy,
                    learning_time=10.0
                )
        
        # Retrieve strategy (should prefer adam_optimizer)
        best_strategy = mm.retrieve_learning_strategy(state, task_type)
        
        # Test meta-parameter updates based on feedback
        positive_feedback = 0.1  # Good performance
        mm.update_meta_parameters(state, positive_feedback)
        
        negative_feedback = -0.2  # Poor performance
        mm.update_meta_parameters(state, negative_feedback)
        
        # Property: system should adapt to feedback
        stats_after = mm.get_learning_statistics(state)
        assert 'adaptation_history' in stats_after or 'meta_parameters' in stats_after
    
    @pytest.mark.unit
    def test_meta_memory_strategy_generalization(self):
        """Test strategy generalization across similar tasks"""
        from src.memory.meta.meta_memory import MetaMemory
        
        mm = MetaMemory(strategy_similarity_threshold=0.7)
        state = mm.init_state()
        
        # Store experiences for related tasks
        related_tasks = [
            "image_classification_cifar10",
            "image_classification_mnist", 
            "image_classification_imagenet",
            "text_classification_sentiment",
            "text_classification_topic"
        ]
        
        # Image tasks should use similar strategies
        image_strategy = "convolutional_network"
        text_strategy = "transformer_network"
        
        for task in related_tasks:
            if "image" in task:
                strategy = image_strategy
                performance = 0.85
            else:
                strategy = text_strategy
                performance = 0.80
            
            mm.store_learning_experience(
                state,
                task_type=task,
                performance=performance,
                strategy=strategy,
                learning_time=20.0
            )
        
        # Test generalization to new similar task
        new_image_task = "image_classification_custom"
        retrieved_strategy = mm.retrieve_learning_strategy(state, new_image_task)
        
        # Should generalize from similar image tasks
        # Note: This test depends on implementation details of similarity computation
        assert retrieved_strategy is not None, "Should retrieve strategy for similar task"
    
    @pytest.mark.unit
    def test_meta_memory_consolidation_trigger(self):
        """Test consolidation trigger mechanism"""
        from src.memory.meta.meta_memory import MetaMemory
        
        mm = MetaMemory(learning_history_size=10)  # Small size to trigger consolidation
        state = mm.init_state()
        
        # Fill memory beyond capacity
        task_type = "test_task"
        
        for i in range(15):  # Exceed capacity
            mm.store_learning_experience(
                state,
                task_type=task_type,
                performance=0.5 + i * 0.02,  # Gradually improving
                strategy=f"strategy_{i % 3}",
                learning_time=10.0
            )
        
        # Check that consolidation was triggered
        stats = mm.get_learning_statistics(state)
        
        # Property: should not exceed capacity significantly
        assert stats['total_experiences'] <= mm.learning_history_size * 1.5, \
            f"Memory size not managed: {stats['total_experiences']} > {mm.learning_history_size * 1.5}"
        
        # Property: should still be able to retrieve strategies
        retrieved_strategy = mm.retrieve_learning_strategy(state, task_type)
        assert retrieved_strategy is not None, "Should still retrieve strategies after consolidation"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
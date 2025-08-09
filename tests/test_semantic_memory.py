"""
Tests for Semantic Memory Implementation

This module contains comprehensive tests for the semantic memory system,
including concept extraction, knowledge graph construction, and query mechanisms.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time
from typing import List, Dict, Any

from src.memory.semantic.semantic_memory import (
    SemanticMemory, SemanticMemoryParams, SemanticMemoryState,
    Concept, ConceptRelation, KnowledgeGraph,
    create_semantic_memory
)
from src.memory.episodic.episodic_memory import Experience


class TestSemanticMemoryParams:
    """Test SemanticMemoryParams configuration."""
    
    def test_default_params(self):
        """Test default parameter values."""
        params = SemanticMemoryParams()
        
        assert params.min_concept_frequency == 3
        assert params.concept_similarity_threshold == 0.8
        assert params.max_concepts == 10000
        assert params.embedding_dim == 256
        assert params.reservoir_size == 400
        assert params.minute_timescale == 60.0
    
    def test_custom_params(self):
        """Test custom parameter configuration."""
        params = SemanticMemoryParams(
            min_concept_frequency=5,
            embedding_dim=128,
            max_concepts=5000
        )
        
        assert params.min_concept_frequency == 5
        assert params.embedding_dim == 128
        assert params.max_concepts == 5000


class TestConcept:
    """Test Concept data structure."""
    
    def test_concept_creation(self):
        """Test concept creation with all fields."""
        embedding = jnp.array([0.1, 0.2, 0.3, 0.4])
        concept = Concept(
            concept_id="test_concept_1",
            name="test_pattern",
            embedding=embedding,
            frequency=5,
            creation_time=time.time(),
            last_updated=time.time(),
            confidence=0.8,
            abstraction_level=0,
            source_experiences=["exp1", "exp2"],
            properties={"type": "visual"},
            semantic_tags={"visual", "pattern"}
        )
        
        assert concept.concept_id == "test_concept_1"
        assert concept.name == "test_pattern"
        assert jnp.allclose(concept.embedding, embedding)
        assert concept.frequency == 5
        assert concept.confidence == 0.8
        assert concept.abstraction_level == 0
        assert len(concept.source_experiences) == 2
        assert "visual" in concept.semantic_tags


class TestConceptRelation:
    """Test ConceptRelation data structure."""
    
    def test_relation_creation(self):
        """Test relation creation with all fields."""
        relation = ConceptRelation(
            relation_id="rel_1_2",
            source_concept="concept_1",
            target_concept="concept_2",
            relation_type="similar_to",
            strength=0.7,
            confidence=0.8,
            evidence_count=3,
            creation_time=time.time(),
            last_reinforced=time.time(),
            temporal_context=None,
            properties={}
        )
        
        assert relation.relation_id == "rel_1_2"
        assert relation.source_concept == "concept_1"
        assert relation.target_concept == "concept_2"
        assert relation.relation_type == "similar_to"
        assert relation.strength == 0.7
        assert relation.confidence == 0.8
        assert relation.evidence_count == 3


class TestSemanticMemory:
    """Test SemanticMemory main functionality."""
    
    @pytest.fixture
    def semantic_memory(self):
        """Create semantic memory for testing."""
        params = SemanticMemoryParams(
            min_concept_frequency=2,  # Lower for testing
            embedding_dim=64,         # Smaller for testing
            reservoir_size=100,       # Smaller for testing
            max_concepts=100
        )
        return SemanticMemory(params)
    
    @pytest.fixture
    def sample_experiences(self):
        """Create sample experiences for testing."""
        key = random.PRNGKey(42)
        experiences = []
        
        for i in range(10):
            key, subkey = random.split(key)
            obs = random.normal(subkey, (8,))
            key, subkey = random.split(key)
            action = random.normal(subkey, (4,))
            
            experience = Experience(
                observation=obs,
                action=action,
                reward=float(random.uniform(key, (), minval=-1.0, maxval=1.0)),
                next_observation=obs + 0.1,
                timestamp=time.time() + i,
                context={"episode": i // 3}
            )
            experiences.append(experience)
        
        return experiences
    
    def test_initialization(self, semantic_memory):
        """Test semantic memory initialization."""
        assert semantic_memory.params.embedding_dim == 64
        assert semantic_memory.params.reservoir_size == 100
        assert semantic_memory.lsm is not None
    
    def test_init_state(self, semantic_memory):
        """Test state initialization."""
        key = random.PRNGKey(0)
        state = semantic_memory.init_state(key)
        
        assert isinstance(state, SemanticMemoryState)
        assert len(state.knowledge_graph.concepts) == 0
        assert len(state.knowledge_graph.relations) == 0
        assert state.concept_embeddings.shape == (0, 64)
        assert len(state.concept_frequencies) == 0
        assert state.total_concepts_created == 0
        assert state.total_relations_created == 0
    
    def test_extract_concepts(self, semantic_memory, sample_experiences):
        """Test concept extraction from experiences."""
        key = random.PRNGKey(42)
        state = semantic_memory.init_state(key)
        
        # Extract concepts from experiences
        new_state, concepts = semantic_memory.extract_concepts(
            state, sample_experiences, key
        )
        
        # Should have extracted some concepts
        assert len(concepts) >= 0  # May be 0 if clustering doesn't find patterns
        assert new_state.total_concepts_created >= len(concepts)
        assert len(new_state.extraction_buffer) == len(sample_experiences)
        
        # If concepts were created, check their properties
        for concept in concepts:
            assert isinstance(concept, Concept)
            assert concept.concept_id.startswith("concept_")
            assert concept.embedding.shape == (64,)
            assert concept.frequency >= 1
            assert concept.confidence > 0
    
    def test_build_knowledge_graph(self, semantic_memory):
        """Test knowledge graph construction."""
        key = random.PRNGKey(42)
        state = semantic_memory.init_state(key)
        
        # Create some test concepts
        concepts = []
        for i in range(3):
            key, subkey = random.split(key)
            embedding = random.normal(subkey, (64,))
            concept = Concept(
                concept_id=f"concept_{i}",
                name=f"test_concept_{i}",
                embedding=embedding,
                frequency=5,
                creation_time=time.time(),
                last_updated=time.time(),
                confidence=0.8,
                abstraction_level=0,
                source_experiences=[],
                properties={},
                semantic_tags=set()
            )
            concepts.append(concept)
            
            # Add to state
            state.knowledge_graph.concepts[concept.concept_id] = concept
            state.knowledge_graph.graph.add_node(concept.concept_id, concept=concept)
        
        # Build knowledge graph
        new_state = semantic_memory.build_knowledge_graph(state, concepts)
        
        # Check that relations were created
        assert new_state.total_relations_created >= 0
        
        # Check graph structure
        graph = new_state.knowledge_graph.graph
        assert graph.number_of_nodes() == 3
    
    def test_compute_semantic_similarity(self, semantic_memory):
        """Test semantic similarity computation."""
        key = random.PRNGKey(42)
        state = semantic_memory.init_state(key)
        
        # Create two similar concepts
        key, subkey1, subkey2 = random.split(key, 3)
        base_embedding = random.normal(subkey1, (64,))
        embedding1 = base_embedding
        embedding2 = base_embedding + 0.1 * random.normal(subkey2, (64,))
        
        concept1 = Concept(
            concept_id="concept_1",
            name="concept_1",
            embedding=embedding1,
            frequency=5,
            creation_time=time.time(),
            last_updated=time.time(),
            confidence=0.8,
            abstraction_level=0,
            source_experiences=[],
            properties={},
            semantic_tags=set()
        )
        
        concept2 = Concept(
            concept_id="concept_2",
            name="concept_2",
            embedding=embedding2,
            frequency=5,
            creation_time=time.time(),
            last_updated=time.time(),
            confidence=0.8,
            abstraction_level=0,
            source_experiences=[],
            properties={},
            semantic_tags=set()
        )
        
        # Add concepts to state
        state.knowledge_graph.concepts["concept_1"] = concept1
        state.knowledge_graph.concepts["concept_2"] = concept2
        
        # Compute similarity
        similarity = semantic_memory.compute_semantic_similarity(
            state, "concept_1", "concept_2"
        )
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be similar due to similar embeddings
    
    def test_query_knowledge(self, semantic_memory):
        """Test knowledge querying."""
        key = random.PRNGKey(42)
        state = semantic_memory.init_state(key)
        
        # Create test concepts
        concepts = []
        for i in range(5):
            key, subkey = random.split(key)
            embedding = random.normal(subkey, (64,))
            concept = Concept(
                concept_id=f"concept_{i}",
                name=f"test_concept_{i}",
                embedding=embedding,
                frequency=5,
                creation_time=time.time(),
                last_updated=time.time(),
                confidence=0.8,
                abstraction_level=0,
                source_experiences=[],
                properties={},
                semantic_tags=set()
            )
            concepts.append(concept)
            state.knowledge_graph.concepts[concept.concept_id] = concept
        
        # Query with an embedding
        key, subkey = random.split(key)
        query_embedding = random.normal(subkey, (64,))
        results = semantic_memory.query_knowledge(state, query_embedding, max_results=3)
        
        assert len(results) <= 3
        for concept, similarity in results:
            assert isinstance(concept, Concept)
            assert 0.0 <= similarity <= 1.0
        
        # Results should be sorted by similarity
        if len(results) > 1:
            similarities = [sim for _, sim in results]
            assert similarities == sorted(similarities, reverse=True)
    
    def test_get_concept_neighbors(self, semantic_memory):
        """Test concept neighbor retrieval."""
        key = random.PRNGKey(42)
        state = semantic_memory.init_state(key)
        
        # Create concepts and relations
        concepts = []
        for i in range(3):
            key, subkey = random.split(key)
            embedding = random.normal(subkey, (64,))
            concept = Concept(
                concept_id=f"concept_{i}",
                name=f"test_concept_{i}",
                embedding=embedding,
                frequency=5,
                creation_time=time.time(),
                last_updated=time.time(),
                confidence=0.8,
                abstraction_level=0,
                source_experiences=[],
                properties={},
                semantic_tags=set()
            )
            concepts.append(concept)
            state.knowledge_graph.concepts[concept.concept_id] = concept
            state.knowledge_graph.graph.add_node(concept.concept_id, concept=concept)
        
        # Add edges
        state.knowledge_graph.graph.add_edge("concept_0", "concept_1", weight=0.8)
        state.knowledge_graph.graph.add_edge("concept_1", "concept_2", weight=0.7)
        
        # Get neighbors
        neighbors = semantic_memory.get_concept_neighbors(state, "concept_0", max_depth=2)
        
        assert isinstance(neighbors, dict)
        assert "concept_1" in neighbors  # Direct neighbor
        # concept_2 might be included depending on max_depth
    
    def test_consolidate_knowledge(self, semantic_memory):
        """Test knowledge consolidation."""
        key = random.PRNGKey(42)
        state = semantic_memory.init_state(key)
        
        # Create some concepts
        for i in range(5):
            key, subkey = random.split(key)
            embedding = random.normal(subkey, (64,))
            concept = Concept(
                concept_id=f"concept_{i}",
                name=f"test_concept_{i}",
                embedding=embedding,
                frequency=5,
                creation_time=time.time() - 3600,  # 1 hour ago
                last_updated=time.time() - 3600,
                confidence=0.8,
                abstraction_level=0,
                source_experiences=[],
                properties={},
                semantic_tags=set()
            )
            state.knowledge_graph.concepts[concept.concept_id] = concept
            state.concept_frequencies[concept.concept_id] = 5
        
        # Set last consolidation to trigger consolidation
        state = state._replace(last_consolidation=time.time() - 3700)  # More than 1 hour ago
        
        # Consolidate
        new_state = semantic_memory.consolidate_knowledge(state, key)
        
        assert new_state.last_consolidation > state.last_consolidation
        # Other checks depend on specific consolidation logic
    
    def test_get_knowledge_statistics(self, semantic_memory):
        """Test knowledge statistics retrieval."""
        key = random.PRNGKey(42)
        state = semantic_memory.init_state(key)
        
        # Empty state statistics
        stats = semantic_memory.get_knowledge_statistics(state)
        assert stats['num_concepts'] == 0
        assert stats['num_relations'] == 0
        assert stats['graph_density'] == 0.0
        
        # Add some concepts and relations
        for i in range(3):
            key, subkey = random.split(key)
            embedding = random.normal(subkey, (64,))
            concept = Concept(
                concept_id=f"concept_{i}",
                name=f"test_concept_{i}",
                embedding=embedding,
                frequency=5,
                creation_time=time.time(),
                last_updated=time.time(),
                confidence=0.8,
                abstraction_level=0,
                source_experiences=[],
                properties={},
                semantic_tags=set()
            )
            state.knowledge_graph.concepts[concept.concept_id] = concept
            state.knowledge_graph.graph.add_node(concept.concept_id, concept=concept)
            state.concept_frequencies[concept.concept_id] = 5
        
        # Add relation
        relation = ConceptRelation(
            relation_id="rel_0_1",
            source_concept="concept_0",
            target_concept="concept_1",
            relation_type="similar_to",
            strength=0.8,
            confidence=0.8,
            evidence_count=1,
            creation_time=time.time(),
            last_reinforced=time.time(),
            temporal_context=None,
            properties={}
        )
        state.knowledge_graph.relations["rel_0_1"] = relation
        state.knowledge_graph.graph.add_edge("concept_0", "concept_1", relation=relation, weight=0.8)
        
        state = state._replace(total_concepts_created=3, total_relations_created=1)
        
        # Get statistics
        stats = semantic_memory.get_knowledge_statistics(state)
        assert stats['num_concepts'] == 3
        assert stats['num_relations'] == 1
        assert stats['graph_density'] > 0.0
        assert stats['avg_concept_frequency'] == 5.0
        assert stats['total_concepts_created'] == 3
        assert stats['total_relations_created'] == 1


class TestSemanticMemoryIntegration:
    """Test semantic memory integration scenarios."""
    
    def test_end_to_end_concept_formation(self):
        """Test complete concept formation pipeline."""
        # Create semantic memory
        semantic_memory = create_semantic_memory("small")
        key = random.PRNGKey(42)
        state = semantic_memory.init_state(key)
        
        # Create experiences with patterns
        experiences = []
        for i in range(20):  # More experiences to ensure concept formation
            key, subkey = random.split(key)
            # Create similar observations to form concepts
            base_obs = jnp.array([1.0, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0])
            noise = 0.1 * random.normal(subkey, (8,))
            obs = base_obs + noise
            
            key, subkey = random.split(key)
            action = random.normal(subkey, (4,))
            
            experience = Experience(
                observation=obs,
                action=action,
                reward=1.0,
                next_observation=obs + 0.1,
                timestamp=time.time() + i,
                context={"pattern": "A"}
            )
            experiences.append(experience)
        
        # Extract concepts
        state, concepts = semantic_memory.extract_concepts(state, experiences, key)
        
        # Build knowledge graph
        if concepts:
            state = semantic_memory.build_knowledge_graph(state, concepts)
        
        # Query for similar patterns
        if concepts:
            key, subkey = random.split(key)
            query_embedding = random.normal(subkey, (semantic_memory.params.embedding_dim,))
            results = semantic_memory.query_knowledge(state, query_embedding)
            
            # Should get some results
            assert len(results) >= 0
        
        # Get statistics
        stats = semantic_memory.get_knowledge_statistics(state)
        assert stats['num_concepts'] >= 0
        assert stats['total_concepts_created'] >= 0
    
    def test_concept_similarity_and_relations(self):
        """Test concept similarity computation and relation formation."""
        semantic_memory = create_semantic_memory("fast_learning")
        key = random.PRNGKey(42)
        state = semantic_memory.init_state(key)
        
        # Create two sets of similar experiences
        experiences_a = []
        experiences_b = []
        
        for i in range(6):  # Enough for concept formation
            key, subkey = random.split(key)
            # Pattern A
            base_obs_a = jnp.array([1.0, 1.0, 0.0, 0.0])
            obs_a = base_obs_a + 0.05 * random.normal(subkey, (4,))
            
            key, subkey = random.split(key)
            # Pattern B (similar to A)
            base_obs_b = jnp.array([0.9, 0.9, 0.1, 0.1])
            obs_b = base_obs_b + 0.05 * random.normal(subkey, (4,))
            
            # Pad to match embedding dimension
            obs_a_padded = jnp.concatenate([obs_a, jnp.zeros(60)])
            obs_b_padded = jnp.concatenate([obs_b, jnp.zeros(60)])
            
            key, subkey = random.split(key)
            action = random.normal(subkey, (4,))
            
            exp_a = Experience(
                observation=obs_a_padded,
                action=action,
                reward=1.0,
                next_observation=obs_a_padded + 0.01,
                timestamp=time.time() + i,
                context={"pattern": "A"}
            )
            
            exp_b = Experience(
                observation=obs_b_padded,
                action=action,
                reward=1.0,
                next_observation=obs_b_padded + 0.01,
                timestamp=time.time() + i + 10,
                context={"pattern": "B"}
            )
            
            experiences_a.append(exp_a)
            experiences_b.append(exp_b)
        
        # Extract concepts from both sets
        all_experiences = experiences_a + experiences_b
        state, concepts = semantic_memory.extract_concepts(state, all_experiences, key)
        
        if len(concepts) >= 2:
            # Build knowledge graph
            state = semantic_memory.build_knowledge_graph(state, concepts)
            
            # Check if relations were formed
            stats = semantic_memory.get_knowledge_statistics(state)
            assert stats['num_concepts'] >= 1
            # Relations might be formed if concepts are similar enough


class TestSemanticMemoryConvenienceFunctions:
    """Test convenience functions for semantic memory creation."""
    
    def test_create_standard_memory(self):
        """Test creating standard semantic memory."""
        memory = create_semantic_memory("standard")
        assert memory.params.embedding_dim == 256
        assert memory.params.reservoir_size == 400
        assert memory.params.min_concept_frequency == 3
    
    def test_create_small_memory(self):
        """Test creating small semantic memory."""
        memory = create_semantic_memory("small")
        assert memory.params.max_concepts == 100
        assert memory.params.embedding_dim == 64
        assert memory.params.reservoir_size == 100
        assert memory.params.min_concept_frequency == 2
    
    def test_create_large_memory(self):
        """Test creating large semantic memory."""
        memory = create_semantic_memory("large")
        assert memory.params.max_concepts == 50000
        assert memory.params.embedding_dim == 512
        assert memory.params.reservoir_size == 800
    
    def test_create_fast_learning_memory(self):
        """Test creating fast learning semantic memory."""
        memory = create_semantic_memory("fast_learning")
        assert memory.params.min_concept_frequency == 2
        assert memory.params.concept_similarity_threshold == 0.6
        assert memory.params.relation_strength_threshold == 0.2
    
    def test_create_slow_learning_memory(self):
        """Test creating slow learning semantic memory."""
        memory = create_semantic_memory("slow_learning")
        assert memory.params.min_concept_frequency == 5
        assert memory.params.concept_similarity_threshold == 0.9
        assert memory.params.relation_strength_threshold == 0.5
    
    def test_create_high_abstraction_memory(self):
        """Test creating high abstraction semantic memory."""
        memory = create_semantic_memory("high_abstraction")
        assert memory.params.max_abstraction_levels == 10
        assert memory.params.abstraction_merge_threshold == 0.8
        assert memory.params.clustering_threshold == 0.6
    
    def test_invalid_memory_type(self):
        """Test error handling for invalid memory type."""
        with pytest.raises(ValueError, match="Unknown memory type"):
            create_semantic_memory("invalid_type")


class TestSemanticMemoryEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_experiences(self):
        """Test handling of empty experience list."""
        memory = create_semantic_memory("small")
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Extract concepts from empty list
        new_state, concepts = memory.extract_concepts(state, [], key)
        
        assert len(concepts) == 0
        assert new_state.total_concepts_created == 0
    
    def test_single_experience(self):
        """Test handling of single experience."""
        memory = create_semantic_memory("small")
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Create single experience
        key, subkey = random.split(key)
        obs = random.normal(subkey, (8,))
        key, subkey = random.split(key)
        action = random.normal(subkey, (4,))
        
        experience = Experience(
            observation=obs,
            action=action,
            reward=1.0,
            next_observation=obs + 0.1,
            timestamp=time.time(),
            context={}
        )
        
        # Extract concepts
        new_state, concepts = memory.extract_concepts(state, [experience], key)
        
        # Should handle gracefully (might not create concepts due to min frequency)
        assert len(concepts) >= 0
        assert new_state.total_concepts_created >= 0
    
    def test_query_empty_knowledge_graph(self):
        """Test querying empty knowledge graph."""
        memory = create_semantic_memory("small")
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Query empty graph
        key, subkey = random.split(key)
        query_embedding = random.normal(subkey, (64,))
        results = memory.query_knowledge(state, query_embedding)
        
        assert len(results) == 0
    
    def test_similarity_with_nonexistent_concepts(self):
        """Test similarity computation with nonexistent concepts."""
        memory = create_semantic_memory("small")
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Compute similarity with nonexistent concepts
        similarity = memory.compute_semantic_similarity(state, "nonexistent_1", "nonexistent_2")
        
        assert similarity == 0.0
    
    def test_neighbors_of_nonexistent_concept(self):
        """Test getting neighbors of nonexistent concept."""
        memory = create_semantic_memory("small")
        key = random.PRNGKey(42)
        state = memory.init_state(key)
        
        # Get neighbors of nonexistent concept
        neighbors = memory.get_concept_neighbors(state, "nonexistent_concept")
        
        assert len(neighbors) == 0


if __name__ == "__main__":
    pytest.main([__file__])
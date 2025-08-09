#!/usr/bin/env python3
"""
Semantic Memory System Demonstration

This script demonstrates the key features of the semantic memory system:
- Concept extraction from experiences
- Knowledge graph construction
- Semantic similarity computation
- Content-addressable retrieval
- Knowledge consolidation
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.semantic.semantic_memory import create_semantic_memory
from src.memory.episodic.episodic_memory import Experience


def create_sample_experiences(key, n_experiences=50):
    """Create sample experiences with different patterns."""
    experiences = []
    
    # Pattern A: High-reward visual experiences
    for i in range(n_experiences // 3):
        key, subkey = random.split(key)
        # Visual pattern: bright, high-contrast
        obs = jnp.array([0.8, 0.9, 0.1, 0.2, 0.7, 0.8, 0.0, 0.1])
        noise = 0.1 * random.normal(subkey, (8,))
        obs = obs + noise
        
        key, subkey = random.split(key)
        action = jnp.array([1.0, 0.0, 0.5, 0.8]) + 0.1 * random.normal(subkey, (4,))
        
        experience = Experience(
            observation=obs,
            action=action,
            reward=0.8 + 0.2 * random.uniform(key),
            next_observation=obs + 0.05,
            timestamp=time.time() + i,
            context={"pattern_type": "visual_bright", "reward_level": "high"}
        )
        experiences.append(experience)
    
    # Pattern B: Medium-reward auditory experiences
    for i in range(n_experiences // 3):
        key, subkey = random.split(key)
        # Auditory pattern: rhythmic, medium intensity
        obs = jnp.array([0.3, 0.4, 0.6, 0.5, 0.4, 0.3, 0.6, 0.5])
        noise = 0.1 * random.normal(subkey, (8,))
        obs = obs + noise
        
        key, subkey = random.split(key)
        action = jnp.array([0.5, 0.6, 0.4, 0.3]) + 0.1 * random.normal(subkey, (4,))
        
        experience = Experience(
            observation=obs,
            action=action,
            reward=0.5 + 0.2 * random.uniform(key),
            next_observation=obs + 0.03,
            timestamp=time.time() + i + 100,
            context={"pattern_type": "auditory_rhythm", "reward_level": "medium"}
        )
        experiences.append(experience)
    
    # Pattern C: Low-reward tactile experiences
    for i in range(n_experiences - 2 * (n_experiences // 3)):
        key, subkey = random.split(key)
        # Tactile pattern: soft, low intensity
        obs = jnp.array([0.1, 0.2, 0.0, 0.1, 0.2, 0.1, 0.0, 0.2])
        noise = 0.1 * random.normal(subkey, (8,))
        obs = obs + noise
        
        key, subkey = random.split(key)
        action = jnp.array([0.1, 0.2, 0.0, 0.1]) + 0.1 * random.normal(subkey, (4,))
        
        experience = Experience(
            observation=obs,
            action=action,
            reward=0.2 + 0.2 * random.uniform(key),
            next_observation=obs + 0.01,
            timestamp=time.time() + i + 200,
            context={"pattern_type": "tactile_soft", "reward_level": "low"}
        )
        experiences.append(experience)
    
    return experiences


def demonstrate_semantic_memory():
    """Demonstrate the semantic memory system."""
    print("🧠 Semantic Memory System Demonstration")
    print("=" * 50)
    
    # Initialize semantic memory
    print("\n1. Initializing Semantic Memory System...")
    semantic_memory = create_semantic_memory("small")  # Use small config for demo
    key = random.PRNGKey(42)
    state = semantic_memory.init_state(key)
    
    print(f"   - Embedding dimension: {semantic_memory.params.embedding_dim}")
    print(f"   - Reservoir size: {semantic_memory.params.reservoir_size}")
    print(f"   - Min concept frequency: {semantic_memory.params.min_concept_frequency}")
    
    # Create sample experiences
    print("\n2. Creating Sample Experiences...")
    key, subkey = random.split(key)
    experiences = create_sample_experiences(subkey, n_experiences=30)
    print(f"   - Created {len(experiences)} experiences with 3 different patterns")
    
    # Extract concepts
    print("\n3. Extracting Concepts from Experiences...")
    key, subkey = random.split(key)
    state, concepts = semantic_memory.extract_concepts(state, experiences, subkey)
    
    print(f"   - Extracted {len(concepts)} concepts")
    for i, concept in enumerate(concepts):
        print(f"     • Concept {i+1}: {concept.name} (confidence: {concept.confidence:.2f})")
    
    # Build knowledge graph
    print("\n4. Building Knowledge Graph...")
    if concepts:
        state = semantic_memory.build_knowledge_graph(state, concepts)
        stats = semantic_memory.get_knowledge_statistics(state)
        print(f"   - Created {stats['num_relations']} relationships")
        print(f"   - Graph density: {stats['graph_density']:.3f}")
        print(f"   - Knowledge coverage: {stats['knowledge_coverage']:.3f}")
    
    # Demonstrate semantic similarity
    print("\n5. Computing Semantic Similarities...")
    if len(concepts) >= 2:
        concept_ids = list(state.knowledge_graph.concepts.keys())
        for i in range(min(3, len(concept_ids))):
            for j in range(i+1, min(3, len(concept_ids))):
                similarity = semantic_memory.compute_semantic_similarity(
                    state, concept_ids[i], concept_ids[j]
                )
                concept1_name = state.knowledge_graph.concepts[concept_ids[i]].name
                concept2_name = state.knowledge_graph.concepts[concept_ids[j]].name
                print(f"   - {concept1_name} ↔ {concept2_name}: {similarity:.3f}")
    
    # Demonstrate knowledge querying
    print("\n6. Querying Knowledge Graph...")
    if concepts:
        # Query with a random embedding
        key, subkey = random.split(key)
        query_embedding = random.normal(subkey, (semantic_memory.params.embedding_dim,))
        results = semantic_memory.query_knowledge(state, query_embedding, max_results=3)
        
        print(f"   - Query returned {len(results)} relevant concepts:")
        for concept, similarity in results:
            print(f"     • {concept.name}: similarity {similarity:.3f}")
    
    # Demonstrate concept neighbors
    print("\n7. Finding Concept Neighbors...")
    if concepts:
        concept_id = list(state.knowledge_graph.concepts.keys())[0]
        neighbors = semantic_memory.get_concept_neighbors(state, concept_id, max_depth=2)
        concept_name = state.knowledge_graph.concepts[concept_id].name
        
        print(f"   - Neighbors of {concept_name}:")
        for neighbor_id, strength in neighbors.items():
            neighbor_name = state.knowledge_graph.concepts[neighbor_id].name
            print(f"     • {neighbor_name}: connection strength {strength:.3f}")
    
    # Demonstrate knowledge consolidation
    print("\n8. Knowledge Consolidation...")
    # Force consolidation by setting old timestamp
    state = state._replace(last_consolidation=time.time() - 3700)
    key, subkey = random.split(key)
    consolidated_state = semantic_memory.consolidate_knowledge(state, subkey)
    
    old_stats = semantic_memory.get_knowledge_statistics(state)
    new_stats = semantic_memory.get_knowledge_statistics(consolidated_state)
    
    print(f"   - Before consolidation: {old_stats['num_concepts']} concepts, {old_stats['num_relations']} relations")
    print(f"   - After consolidation: {new_stats['num_concepts']} concepts, {new_stats['num_relations']} relations")
    
    # Final statistics
    print("\n9. Final Knowledge Graph Statistics:")
    final_stats = semantic_memory.get_knowledge_statistics(consolidated_state)
    print(f"   - Total concepts: {final_stats['num_concepts']}")
    print(f"   - Total relations: {final_stats['num_relations']}")
    print(f"   - Graph density: {final_stats['graph_density']:.3f}")
    print(f"   - Average concept degree: {final_stats['avg_concept_degree']:.2f}")
    print(f"   - Knowledge coverage: {final_stats['knowledge_coverage']:.3f}")
    print(f"   - Average concept age: {final_stats['avg_concept_age']:.1f} seconds")
    
    print("\n✅ Semantic Memory Demonstration Complete!")
    print("\nKey Features Demonstrated:")
    print("• Concept extraction from experience patterns")
    print("• Knowledge graph construction with relationships")
    print("• Semantic similarity computation")
    print("• Content-addressable concept retrieval")
    print("• Graph-based neighbor finding")
    print("• Knowledge consolidation and optimization")


if __name__ == "__main__":
    demonstrate_semantic_memory()
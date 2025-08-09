"""
Semantic Memory Implementation with Knowledge Graphs

This module implements the semantic memory system using knowledge graphs for
concept extraction, relationship modeling, and semantic similarity computation.
"""

from typing import NamedTuple, Optional, Tuple, Dict, List, Any, Set, Union
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass, field
import numpy as np
import networkx as nx
import hashlib
import time
import pickle
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from ...core.liquid_state_machine import LiquidStateMachine, LSMParams, LSMState
from ..episodic.episodic_memory import Experience


@dataclass
class Concept:
    """Concept representation in semantic memory."""
    
    concept_id: str                    # Unique concept identifier
    name: str                         # Human-readable concept name
    embedding: jnp.ndarray            # Vector representation
    frequency: int                    # How often concept appears
    creation_time: float              # When concept was created
    last_updated: float               # Last update timestamp
    confidence: float                 # Confidence in concept validity (0-1)
    abstraction_level: int            # Level of abstraction (0=concrete, higher=abstract)
    source_experiences: List[str]     # IDs of experiences that contributed
    properties: Dict[str, Any]        # Additional concept properties
    semantic_tags: Set[str]           # Semantic category tags


@dataclass
class ConceptRelation:
    """Relationship between concepts in the knowledge graph."""
    
    relation_id: str                  # Unique relation identifier
    source_concept: str               # Source concept ID
    target_concept: str               # Target concept ID
    relation_type: str                # Type of relationship
    strength: float                   # Relationship strength (0-1)
    confidence: float                 # Confidence in relationship (0-1)
    evidence_count: int               # Number of supporting experiences
    creation_time: float              # When relation was created
    last_reinforced: float            # Last time relation was reinforced
    temporal_context: Optional[float] # Temporal context if relevant
    properties: Dict[str, Any]        # Additional relation properties


@dataclass
class SemanticMemoryParams:
    """Parameters for the Semantic Memory system."""
    
    # Concept extraction
    min_concept_frequency: int = 3     # Minimum frequency to form concept
    concept_similarity_threshold: float = 0.8  # Threshold for concept merging
    max_concepts: int = 10000         # Maximum number of concepts
    
    # Knowledge graph
    max_relations_per_concept: int = 50  # Maximum outgoing relations per concept
    relation_strength_threshold: float = 0.3  # Minimum strength for relation
    relation_decay_rate: float = 0.01  # Decay rate for unused relations
    
    # Embedding parameters
    embedding_dim: int = 256          # Dimensionality of concept embeddings
    context_window: int = 5           # Context window for concept extraction
    
    # LSM encoding
    reservoir_size: int = 400         # Size of concept encoding reservoir
    encoding_timesteps: int = 30      # Timesteps for concept encoding
    
    # Clustering and abstraction
    clustering_threshold: float = 0.7  # Threshold for concept clustering
    max_abstraction_levels: int = 5   # Maximum abstraction hierarchy levels
    abstraction_merge_threshold: float = 0.9  # Threshold for abstraction merging
    
    # Query and retrieval
    query_similarity_threshold: float = 0.6  # Minimum similarity for query results
    max_query_results: int = 20       # Maximum results per query
    semantic_search_depth: int = 3    # Maximum depth for semantic search
    
    # Temporal dynamics
    concept_decay_rate: float = 0.005 # Decay rate for unused concepts
    relation_reinforcement_rate: float = 0.1  # Rate of relation reinforcement
    
    # Minute timescale operation
    minute_timescale: float = 60.0    # Operating timescale in seconds


@dataclass
class KnowledgeGraph:
    """Knowledge graph structure containing concepts and relations."""
    
    concepts: Dict[str, Concept]      # All concepts indexed by ID
    relations: Dict[str, ConceptRelation]  # All relations indexed by ID
    graph: nx.DiGraph                 # NetworkX directed graph
    concept_index: Dict[str, Set[str]]  # Index for fast concept lookup
    relation_index: Dict[str, Set[str]]  # Index for fast relation lookup
    abstraction_hierarchy: Dict[int, Set[str]]  # Concepts by abstraction level
    temporal_index: Dict[float, Set[str]]  # Temporal index for concepts
    
    def __post_init__(self):
        """Initialize indices if not provided."""
        if not self.concept_index:
            self.concept_index = defaultdict(set)
        if not self.relation_index:
            self.relation_index = defaultdict(set)
        if not self.abstraction_hierarchy:
            self.abstraction_hierarchy = defaultdict(set)
        if not self.temporal_index:
            self.temporal_index = defaultdict(set)


class SemanticMemoryState(NamedTuple):
    """State variables for the Semantic Memory system."""
    
    knowledge_graph: KnowledgeGraph   # The main knowledge graph
    concept_embeddings: jnp.ndarray   # Matrix of all concept embeddings
    concept_frequencies: Dict[str, int]  # Frequency count for each concept
    relation_strengths: Dict[str, float]  # Current strength of each relation
    lsm_states: Dict[str, LSMState]   # LSM states for concept encodings
    extraction_buffer: List[Experience]  # Buffer for concept extraction
    last_consolidation: float         # Last consolidation timestamp
    total_concepts_created: int       # Total concepts created
    total_relations_created: int      # Total relations created
    global_time: float                # Current time


class SemanticMemory:
    """
    Semantic Memory system with knowledge graphs.
    
    Implements:
    - Concept extraction from experience patterns
    - Knowledge graph construction and maintenance
    - Semantic similarity computation
    - Content-addressable concept retrieval
    - Hierarchical concept abstraction
    - Temporal concept dynamics
    """
    
    def __init__(self, params: Optional[SemanticMemoryParams] = None):
        """Initialize Semantic Memory system."""
        self.params = params or SemanticMemoryParams()
        
        # Create LSM for concept encoding
        lsm_params = LSMParams(
            reservoir_size=self.params.reservoir_size,
            input_size=self.params.embedding_dim,
            output_size=self.params.embedding_dim,
            reservoir_connectivity=0.1,
            spectral_radius=0.9,
            enable_plasticity=True,
            homeostatic_scaling=True
        )
        self.lsm = LiquidStateMachine(lsm_params)
        
        # Initialize concept extraction components
        self._initialize_extraction_components()
    
    def init_state(self, key: Optional[jax.random.PRNGKey] = None) -> SemanticMemoryState:
        """
        Initialize Semantic Memory state.
        
        Args:
            key: Random key for initialization
            
        Returns:
            Initial semantic memory state
        """
        if key is None:
            key = random.PRNGKey(0)
        
        # Initialize empty knowledge graph
        knowledge_graph = KnowledgeGraph(
            concepts={},
            relations={},
            graph=nx.DiGraph(),
            concept_index=defaultdict(set),
            relation_index=defaultdict(set),
            abstraction_hierarchy=defaultdict(set),
            temporal_index=defaultdict(set)
        )
        
        return SemanticMemoryState(
            knowledge_graph=knowledge_graph,
            concept_embeddings=jnp.zeros((0, self.params.embedding_dim)),
            concept_frequencies={},
            relation_strengths={},
            lsm_states={},
            extraction_buffer=[],
            last_consolidation=0.0,
            total_concepts_created=0,
            total_relations_created=0,
            global_time=0.0
        )
    
    def extract_concepts(
        self, 
        state: SemanticMemoryState, 
        experiences: List[Experience],
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[SemanticMemoryState, List[Concept]]:
        """
        Extract concepts from experience patterns.
        
        Args:
            state: Current semantic memory state
            experiences: List of experiences to process
            key: Random key for LSM operations
            
        Returns:
            Tuple of (updated_state, extracted_concepts)
        """
        if key is None:
            key = random.PRNGKey(int(time.time() * 1000))
        
        if not experiences:
            return state, []
        
        # Add experiences to extraction buffer
        new_buffer = state.extraction_buffer + experiences
        
        # Extract patterns from experiences
        patterns = self._extract_patterns_from_experiences(experiences)
        
        # Cluster patterns to identify potential concepts
        concept_candidates = self._cluster_patterns(patterns, key)
        
        # Validate and create concepts
        new_concepts = []
        updated_knowledge_graph = state.knowledge_graph
        updated_concept_embeddings = state.concept_embeddings
        updated_frequencies = state.concept_frequencies.copy()
        updated_lsm_states = state.lsm_states.copy()
        
        for candidate in concept_candidates:
            # Check if concept already exists
            existing_concept = self._find_similar_concept(
                candidate, updated_knowledge_graph.concepts
            )
            
            if existing_concept:
                # Update existing concept
                updated_concept = self._update_concept(existing_concept, candidate)
                updated_knowledge_graph.concepts[existing_concept.concept_id] = updated_concept
                updated_frequencies[existing_concept.concept_id] += 1
            else:
                # Create new concept
                concept_id = self._generate_concept_id(candidate)
                
                # Encode concept using LSM
                lsm_state = self.lsm.init_state(key)
                concept_encoding = self._encode_concept_with_lsm(candidate, lsm_state, key)
                
                new_concept = Concept(
                    concept_id=concept_id,
                    name=self._generate_concept_name(candidate),
                    embedding=candidate,
                    frequency=1,
                    creation_time=time.time(),
                    last_updated=time.time(),
                    confidence=0.7,  # Initial confidence
                    abstraction_level=0,  # Start at concrete level
                    source_experiences=[exp.timestamp for exp in experiences if hasattr(exp, 'timestamp')],
                    properties={},
                    semantic_tags=set()
                )
                
                # Add to knowledge graph
                updated_knowledge_graph.concepts[concept_id] = new_concept
                updated_knowledge_graph.graph.add_node(concept_id, concept=new_concept)
                
                # Update indices
                self._update_concept_indices(updated_knowledge_graph, new_concept)
                
                # Update embeddings matrix
                if updated_concept_embeddings.shape[0] == 0:
                    updated_concept_embeddings = candidate.reshape(1, -1)
                else:
                    updated_concept_embeddings = jnp.vstack([
                        updated_concept_embeddings, candidate.reshape(1, -1)
                    ])
                
                updated_frequencies[concept_id] = 1
                updated_lsm_states[concept_id] = lsm_state
                new_concepts.append(new_concept)
        
        # Update state
        new_state = SemanticMemoryState(
            knowledge_graph=updated_knowledge_graph,
            concept_embeddings=updated_concept_embeddings,
            concept_frequencies=updated_frequencies,
            relation_strengths=state.relation_strengths,
            lsm_states=updated_lsm_states,
            extraction_buffer=new_buffer[-1000:],  # Keep last 1000 experiences
            last_consolidation=state.last_consolidation,
            total_concepts_created=state.total_concepts_created + len(new_concepts),
            total_relations_created=state.total_relations_created,
            global_time=time.time()
        )
        
        return new_state, new_concepts
    
    def build_knowledge_graph(
        self, 
        state: SemanticMemoryState, 
        concepts: List[Concept]
    ) -> SemanticMemoryState:
        """
        Build knowledge graph from concepts by identifying relationships.
        
        Args:
            state: Current semantic memory state
            concepts: List of concepts to process for relationships
            
        Returns:
            Updated semantic memory state with new relationships
        """
        if len(concepts) < 2:
            return state
        
        updated_knowledge_graph = state.knowledge_graph
        updated_relation_strengths = state.relation_strengths.copy()
        new_relations_count = 0
        
        # Find relationships between concepts
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                # Compute semantic similarity
                similarity = self._compute_concept_similarity(concept1, concept2)
                
                if similarity > self.params.relation_strength_threshold:
                    # Determine relationship type
                    relation_type = self._infer_relation_type(concept1, concept2, similarity)
                    
                    # Create relation
                    relation_id = f"{concept1.concept_id}_{relation_type}_{concept2.concept_id}"
                    
                    # Check if relation already exists
                    if relation_id not in updated_knowledge_graph.relations:
                        relation = ConceptRelation(
                            relation_id=relation_id,
                            source_concept=concept1.concept_id,
                            target_concept=concept2.concept_id,
                            relation_type=relation_type,
                            strength=similarity,
                            confidence=min(concept1.confidence, concept2.confidence),
                            evidence_count=1,
                            creation_time=time.time(),
                            last_reinforced=time.time(),
                            temporal_context=None,
                            properties={}
                        )
                        
                        # Add to knowledge graph
                        updated_knowledge_graph.relations[relation_id] = relation
                        updated_knowledge_graph.graph.add_edge(
                            concept1.concept_id, 
                            concept2.concept_id,
                            relation=relation,
                            weight=similarity
                        )
                        
                        # Update indices
                        self._update_relation_indices(updated_knowledge_graph, relation)
                        
                        updated_relation_strengths[relation_id] = similarity
                        new_relations_count += 1
                    else:
                        # Reinforce existing relation
                        existing_relation = updated_knowledge_graph.relations[relation_id]
                        updated_relation = self._reinforce_relation(existing_relation, similarity)
                        updated_knowledge_graph.relations[relation_id] = updated_relation
                        updated_relation_strengths[relation_id] = updated_relation.strength
        
        # Update state
        return SemanticMemoryState(
            knowledge_graph=updated_knowledge_graph,
            concept_embeddings=state.concept_embeddings,
            concept_frequencies=state.concept_frequencies,
            relation_strengths=updated_relation_strengths,
            lsm_states=state.lsm_states,
            extraction_buffer=state.extraction_buffer,
            last_consolidation=state.last_consolidation,
            total_concepts_created=state.total_concepts_created,
            total_relations_created=state.total_relations_created + new_relations_count,
            global_time=time.time()
        )    

    def compute_semantic_similarity(
        self, 
        state: SemanticMemoryState, 
        concept1_id: str, 
        concept2_id: str
    ) -> float:
        """
        Compute semantic similarity between two concepts.
        
        Args:
            state: Current semantic memory state
            concept1_id: ID of first concept
            concept2_id: ID of second concept
            
        Returns:
            Semantic similarity score (0-1)
        """
        if (concept1_id not in state.knowledge_graph.concepts or 
            concept2_id not in state.knowledge_graph.concepts):
            return 0.0
        
        concept1 = state.knowledge_graph.concepts[concept1_id]
        concept2 = state.knowledge_graph.concepts[concept2_id]
        
        # Direct embedding similarity
        embedding_similarity = self._compute_concept_similarity(concept1, concept2)
        
        # Graph-based similarity (shared neighbors, path distance)
        graph_similarity = self._compute_graph_similarity(
            state.knowledge_graph.graph, concept1_id, concept2_id
        )
        
        # Temporal similarity (concepts active at similar times)
        temporal_similarity = self._compute_temporal_similarity(concept1, concept2)
        
        # Combine similarities
        total_similarity = (
            0.5 * embedding_similarity +
            0.3 * graph_similarity +
            0.2 * temporal_similarity
        )
        
        return float(jnp.clip(total_similarity, 0.0, 1.0))
    
    def query_knowledge(
        self, 
        state: SemanticMemoryState, 
        query: Union[str, jnp.ndarray, Concept],
        max_results: Optional[int] = None
    ) -> List[Tuple[Concept, float]]:
        """
        Query the knowledge graph for relevant concepts.
        
        Args:
            state: Current semantic memory state
            query: Query as string, embedding, or concept
            max_results: Maximum number of results to return
            
        Returns:
            List of (concept, similarity_score) tuples, sorted by similarity
        """
        if max_results is None:
            max_results = self.params.max_query_results
        
        if not state.knowledge_graph.concepts:
            return []
        
        # Convert query to embedding
        query_embedding = self._convert_query_to_embedding(query, state)
        
        if query_embedding is None:
            return []
        
        # Compute similarities with all concepts
        similarities = []
        for concept_id, concept in state.knowledge_graph.concepts.items():
            similarity = self._compute_embedding_similarity(query_embedding, concept.embedding)
            
            if similarity >= self.params.query_similarity_threshold:
                similarities.append((concept, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def get_concept_neighbors(
        self, 
        state: SemanticMemoryState, 
        concept_id: str,
        max_depth: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get neighboring concepts in the knowledge graph.
        
        Args:
            state: Current semantic memory state
            concept_id: ID of the concept to find neighbors for
            max_depth: Maximum depth to search (default: semantic_search_depth)
            
        Returns:
            Dictionary mapping neighbor concept IDs to their connection strengths
        """
        if max_depth is None:
            max_depth = self.params.semantic_search_depth
        
        if concept_id not in state.knowledge_graph.concepts:
            return {}
        
        graph = state.knowledge_graph.graph
        neighbors = {}
        
        # BFS to find neighbors within max_depth
        visited = set()
        queue = [(concept_id, 0, 1.0)]  # (node, depth, strength)
        
        while queue:
            current_id, depth, strength = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            if current_id != concept_id:
                neighbors[current_id] = strength
            
            # Add neighbors to queue
            if depth < max_depth:
                for neighbor_id in graph.neighbors(current_id):
                    if neighbor_id not in visited:
                        edge_data = graph.get_edge_data(current_id, neighbor_id)
                        edge_strength = edge_data.get('weight', 0.5) if edge_data else 0.5
                        new_strength = strength * edge_strength
                        queue.append((neighbor_id, depth + 1, new_strength))
        
        return neighbors
    
    def consolidate_knowledge(
        self, 
        state: SemanticMemoryState,
        key: Optional[jax.random.PRNGKey] = None
    ) -> SemanticMemoryState:
        """
        Consolidate knowledge by merging similar concepts and pruning weak relations.
        
        Args:
            state: Current semantic memory state
            key: Random key for operations
            
        Returns:
            Updated semantic memory state after consolidation
        """
        if key is None:
            key = random.PRNGKey(int(time.time() * 1000))
        
        current_time = time.time()
        
        # Skip if not enough time has passed
        if (current_time - state.last_consolidation < 
            self.params.minute_timescale):
            return state
        
        updated_knowledge_graph = state.knowledge_graph
        
        # 1. Merge similar concepts
        updated_knowledge_graph = self._merge_similar_concepts(
            updated_knowledge_graph, key
        )
        
        # 2. Prune weak relations
        updated_knowledge_graph = self._prune_weak_relations(updated_knowledge_graph)
        
        # 3. Update abstraction hierarchy
        updated_knowledge_graph = self._update_abstraction_hierarchy(
            updated_knowledge_graph, key
        )
        
        # 4. Apply temporal decay
        updated_frequencies, updated_relation_strengths = self._apply_temporal_decay(
            state.concept_frequencies, state.relation_strengths, current_time
        )
        
        # 5. Rebuild concept embeddings matrix
        updated_concept_embeddings = self._rebuild_embeddings_matrix(
            updated_knowledge_graph.concepts
        )
        
        return SemanticMemoryState(
            knowledge_graph=updated_knowledge_graph,
            concept_embeddings=updated_concept_embeddings,
            concept_frequencies=updated_frequencies,
            relation_strengths=updated_relation_strengths,
            lsm_states=state.lsm_states,
            extraction_buffer=state.extraction_buffer,
            last_consolidation=current_time,
            total_concepts_created=state.total_concepts_created,
            total_relations_created=state.total_relations_created,
            global_time=current_time
        )
    
    def get_knowledge_statistics(self, state: SemanticMemoryState) -> Dict[str, Any]:
        """
        Get statistics about the current knowledge graph.
        
        Args:
            state: Current semantic memory state
            
        Returns:
            Dictionary of knowledge graph statistics
        """
        graph = state.knowledge_graph.graph
        concepts = state.knowledge_graph.concepts
        relations = state.knowledge_graph.relations
        
        if not concepts:
            return {
                'num_concepts': 0,
                'num_relations': 0,
                'graph_density': 0.0,
                'avg_concept_degree': 0.0,
                'abstraction_levels': 0,
                'knowledge_coverage': 0.0
            }
        
        # Basic statistics
        num_concepts = len(concepts)
        num_relations = len(relations)
        
        # Graph statistics
        if num_concepts > 1:
            max_possible_edges = num_concepts * (num_concepts - 1)
            graph_density = (2 * graph.number_of_edges()) / max_possible_edges
            avg_concept_degree = sum(dict(graph.degree()).values()) / num_concepts
        else:
            graph_density = 0.0
            avg_concept_degree = 0.0
        
        # Abstraction statistics
        abstraction_levels = len(state.knowledge_graph.abstraction_hierarchy)
        
        # Knowledge coverage (concepts with relations vs isolated concepts)
        connected_concepts = len([c for c in concepts.keys() if graph.degree(c) > 0])
        knowledge_coverage = connected_concepts / num_concepts if num_concepts > 0 else 0.0
        
        # Temporal statistics
        current_time = time.time()
        concept_ages = [current_time - c.creation_time for c in concepts.values()]
        avg_concept_age = float(jnp.mean(jnp.array(concept_ages))) if concept_ages else 0.0
        
        # Frequency statistics
        frequencies = list(state.concept_frequencies.values())
        avg_concept_frequency = float(jnp.mean(jnp.array(frequencies))) if frequencies else 0.0
        
        return {
            'num_concepts': num_concepts,
            'num_relations': num_relations,
            'graph_density': graph_density,
            'avg_concept_degree': avg_concept_degree,
            'abstraction_levels': abstraction_levels,
            'knowledge_coverage': knowledge_coverage,
            'avg_concept_age': avg_concept_age,
            'avg_concept_frequency': avg_concept_frequency,
            'total_concepts_created': state.total_concepts_created,
            'total_relations_created': state.total_relations_created
        }
    
    # Private helper methods
    
    def _initialize_extraction_components(self):
        """Initialize components for concept extraction."""
        # This would initialize any ML models or preprocessing components
        # For now, we'll use simple clustering and pattern matching
        pass
    
    def _extract_patterns_from_experiences(self, experiences: List[Experience]) -> List[jnp.ndarray]:
        """Extract patterns from experiences for concept formation."""
        patterns = []
        
        for experience in experiences:
            # Combine observation and action into a pattern
            if hasattr(experience, 'observation') and hasattr(experience, 'action'):
                # Ensure arrays are JAX arrays
                obs = jnp.array(experience.observation) if not isinstance(experience.observation, jnp.ndarray) else experience.observation
                act = jnp.array(experience.action) if not isinstance(experience.action, jnp.ndarray) else experience.action
                
                # Flatten and concatenate
                obs_flat = obs.flatten()
                act_flat = act.flatten()
                
                # Create fixed-size pattern by padding or truncating
                target_size = self.params.embedding_dim
                combined = jnp.concatenate([obs_flat, act_flat])
                
                if len(combined) > target_size:
                    pattern = combined[:target_size]
                else:
                    padding = jnp.zeros(target_size - len(combined))
                    pattern = jnp.concatenate([combined, padding])
                
                patterns.append(pattern)
        
        return patterns
    
    def _cluster_patterns(
        self, 
        patterns: List[jnp.ndarray], 
        key: jax.random.PRNGKey
    ) -> List[jnp.ndarray]:
        """Cluster patterns to identify concept candidates."""
        if len(patterns) < self.params.min_concept_frequency:
            return []
        
        # Convert to numpy for sklearn
        patterns_np = np.array([np.array(p) for p in patterns])
        
        # Determine number of clusters
        n_clusters = min(len(patterns) // self.params.min_concept_frequency, 10)
        if n_clusters < 1:
            return []
        
        # Perform clustering
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(patterns_np)
            
            # Extract cluster centers as concept candidates
            concept_candidates = []
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                if np.sum(cluster_mask) >= self.params.min_concept_frequency:
                    # Use cluster center as concept representation
                    concept_candidates.append(jnp.array(kmeans.cluster_centers_[i]))
            
            return concept_candidates
        except Exception:
            # Fallback: use mean of all patterns
            if patterns:
                mean_pattern = jnp.mean(jnp.stack(patterns), axis=0)
                return [mean_pattern]
            return []
    
    def _find_similar_concept(
        self, 
        candidate: jnp.ndarray, 
        existing_concepts: Dict[str, Concept]
    ) -> Optional[Concept]:
        """Find existing concept similar to candidate."""
        for concept in existing_concepts.values():
            similarity = self._compute_embedding_similarity(candidate, concept.embedding)
            if similarity > self.params.concept_similarity_threshold:
                return concept
        return None
    
    def _update_concept(self, existing_concept: Concept, new_embedding: jnp.ndarray) -> Concept:
        """Update existing concept with new information."""
        # Weighted average of embeddings
        alpha = 0.1  # Learning rate for concept updates
        updated_embedding = (1 - alpha) * existing_concept.embedding + alpha * new_embedding
        
        return Concept(
            concept_id=existing_concept.concept_id,
            name=existing_concept.name,
            embedding=updated_embedding,
            frequency=existing_concept.frequency + 1,
            creation_time=existing_concept.creation_time,
            last_updated=time.time(),
            confidence=min(1.0, existing_concept.confidence + 0.05),
            abstraction_level=existing_concept.abstraction_level,
            source_experiences=existing_concept.source_experiences,
            properties=existing_concept.properties,
            semantic_tags=existing_concept.semantic_tags
        )
    
    def _generate_concept_id(self, embedding: jnp.ndarray) -> str:
        """Generate unique concept ID."""
        embedding_hash = hashlib.md5(np.array(embedding).tobytes()).hexdigest()
        return f"concept_{embedding_hash[:8]}"
    
    def _generate_concept_name(self, embedding: jnp.ndarray) -> str:
        """Generate human-readable concept name."""
        # Simple naming based on embedding characteristics
        mean_val = float(jnp.mean(embedding))
        std_val = float(jnp.std(embedding))
        
        if mean_val > 0.5:
            prefix = "high"
        elif mean_val < -0.5:
            prefix = "low"
        else:
            prefix = "mid"
        
        if std_val > 0.5:
            suffix = "varied"
        else:
            suffix = "stable"
        
        return f"{prefix}_{suffix}_pattern"
    
    def _encode_concept_with_lsm(
        self, 
        concept_embedding: jnp.ndarray, 
        lsm_state: LSMState, 
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Encode concept using LSM."""
        # Convert embedding to spike train
        spike_train = self._embedding_to_spikes(concept_embedding, key)
        
        # Process through LSM
        reservoir_states, final_state = self.lsm.process_spike_train(
            spike_train, self.params.minute_timescale / self.params.encoding_timesteps, key
        )
        
        return reservoir_states[-1]  # Return final reservoir state
    
    def _embedding_to_spikes(
        self, 
        embedding: jnp.ndarray, 
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Convert embedding to spike train."""
        # Normalize embedding to [0, 1]
        normalized = (embedding - jnp.min(embedding)) / (jnp.max(embedding) - jnp.min(embedding) + 1e-10)
        
        # Create spike train
        n_timesteps = self.params.encoding_timesteps
        spike_train = jnp.zeros((n_timesteps, len(embedding)))
        
        for t in range(n_timesteps):
            key, subkey = random.split(key)
            spike_probs = normalized * 0.2  # Max 20% spike probability
            spikes = random.bernoulli(subkey, spike_probs)
            spike_train = spike_train.at[t].set(spikes)
        
        return spike_train
    
    def _update_concept_indices(self, knowledge_graph: KnowledgeGraph, concept: Concept):
        """Update concept indices for fast lookup."""
        # Add to abstraction hierarchy
        knowledge_graph.abstraction_hierarchy[concept.abstraction_level].add(concept.concept_id)
        
        # Add to temporal index (rounded to nearest minute)
        time_bucket = int(concept.creation_time // 60) * 60
        knowledge_graph.temporal_index[time_bucket].add(concept.concept_id)
        
        # Add semantic tags to concept index
        for tag in concept.semantic_tags:
            knowledge_graph.concept_index[tag].add(concept.concept_id)
    
    def _compute_concept_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """Compute similarity between two concepts."""
        return self._compute_embedding_similarity(concept1.embedding, concept2.embedding)
    
    def _compute_embedding_similarity(self, emb1: jnp.ndarray, emb2: jnp.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        dot_product = jnp.dot(emb1, emb2)
        norm1 = jnp.linalg.norm(emb1)
        norm2 = jnp.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(jnp.clip(similarity, 0.0, 1.0))
    
    def _infer_relation_type(self, concept1: Concept, concept2: Concept, similarity: float) -> str:
        """Infer the type of relationship between concepts."""
        # Simple heuristic based on similarity and concept properties
        if similarity > 0.9:
            return "similar_to"
        elif similarity > 0.7:
            return "related_to"
        elif concept1.abstraction_level > concept2.abstraction_level:
            return "generalizes"
        elif concept1.abstraction_level < concept2.abstraction_level:
            return "specializes"
        else:
            return "associated_with"
    
    def _update_relation_indices(self, knowledge_graph: KnowledgeGraph, relation: ConceptRelation):
        """Update relation indices for fast lookup."""
        knowledge_graph.relation_index[relation.relation_type].add(relation.relation_id)
        knowledge_graph.relation_index[relation.source_concept].add(relation.relation_id)
        knowledge_graph.relation_index[relation.target_concept].add(relation.relation_id)
    
    def _reinforce_relation(self, relation: ConceptRelation, new_strength: float) -> ConceptRelation:
        """Reinforce existing relation with new evidence."""
        # Weighted average of strengths
        alpha = self.params.relation_reinforcement_rate
        updated_strength = (1 - alpha) * relation.strength + alpha * new_strength
        
        return ConceptRelation(
            relation_id=relation.relation_id,
            source_concept=relation.source_concept,
            target_concept=relation.target_concept,
            relation_type=relation.relation_type,
            strength=updated_strength,
            confidence=min(1.0, relation.confidence + 0.05),
            evidence_count=relation.evidence_count + 1,
            creation_time=relation.creation_time,
            last_reinforced=time.time(),
            temporal_context=relation.temporal_context,
            properties=relation.properties
        )
    
    def _compute_graph_similarity(
        self, 
        graph: nx.DiGraph, 
        concept1_id: str, 
        concept2_id: str
    ) -> float:
        """Compute graph-based similarity between concepts."""
        try:
            # Jaccard similarity of neighbors
            neighbors1 = set(graph.neighbors(concept1_id))
            neighbors2 = set(graph.neighbors(concept2_id))
            
            if not neighbors1 and not neighbors2:
                return 0.0
            
            intersection = len(neighbors1.intersection(neighbors2))
            union = len(neighbors1.union(neighbors2))
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            # Path-based similarity (inverse of shortest path distance)
            try:
                path_length = nx.shortest_path_length(graph, concept1_id, concept2_id)
                path_similarity = 1.0 / (1.0 + path_length)
            except nx.NetworkXNoPath:
                path_similarity = 0.0
            
            return 0.7 * jaccard_similarity + 0.3 * path_similarity
        except Exception:
            return 0.0
    
    def _compute_temporal_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """Compute temporal similarity between concepts."""
        time_diff = abs(concept1.creation_time - concept2.creation_time)
        # Exponential decay with time difference
        temporal_similarity = jnp.exp(-time_diff / (24 * 3600))  # 24 hour decay constant
        return float(temporal_similarity)
    
    def _convert_query_to_embedding(
        self, 
        query: Union[str, jnp.ndarray, Concept], 
        state: SemanticMemoryState
    ) -> Optional[jnp.ndarray]:
        """Convert query to embedding for similarity computation."""
        if isinstance(query, Concept):
            return query.embedding
        elif isinstance(query, jnp.ndarray):
            return query
        elif isinstance(query, str):
            # Simple string to embedding conversion
            # In a real implementation, this would use a proper text encoder
            query_hash = hash(query)
            key = random.PRNGKey(abs(query_hash) % (2**31))
            return random.normal(key, (self.params.embedding_dim,))
        else:
            return None
    
    def _merge_similar_concepts(
        self, 
        knowledge_graph: KnowledgeGraph, 
        key: jax.random.PRNGKey
    ) -> KnowledgeGraph:
        """Merge concepts that have become very similar."""
        concepts_to_merge = []
        
        # Find pairs of very similar concepts
        concept_ids = list(knowledge_graph.concepts.keys())
        for i, id1 in enumerate(concept_ids):
            for id2 in concept_ids[i+1:]:
                concept1 = knowledge_graph.concepts[id1]
                concept2 = knowledge_graph.concepts[id2]
                similarity = self._compute_concept_similarity(concept1, concept2)
                
                if similarity > self.params.abstraction_merge_threshold:
                    concepts_to_merge.append((id1, id2, similarity))
        
        # Merge concepts (keep the one with higher frequency)
        for id1, id2, similarity in concepts_to_merge:
            if id1 in knowledge_graph.concepts and id2 in knowledge_graph.concepts:
                concept1 = knowledge_graph.concepts[id1]
                concept2 = knowledge_graph.concepts[id2]
                
                # Keep concept with higher frequency
                if concept1.frequency >= concept2.frequency:
                    keep_id, remove_id = id1, id2
                else:
                    keep_id, remove_id = id2, id1
                
                # Merge properties and update kept concept
                kept_concept = knowledge_graph.concepts[keep_id]
                removed_concept = knowledge_graph.concepts[remove_id]
                
                # Update embedding as weighted average
                total_freq = kept_concept.frequency + removed_concept.frequency
                new_embedding = (
                    (kept_concept.frequency * kept_concept.embedding + 
                     removed_concept.frequency * removed_concept.embedding) / total_freq
                )
                
                updated_concept = Concept(
                    concept_id=kept_concept.concept_id,
                    name=kept_concept.name,
                    embedding=new_embedding,
                    frequency=total_freq,
                    creation_time=min(kept_concept.creation_time, removed_concept.creation_time),
                    last_updated=time.time(),
                    confidence=max(kept_concept.confidence, removed_concept.confidence),
                    abstraction_level=kept_concept.abstraction_level,
                    source_experiences=kept_concept.source_experiences + removed_concept.source_experiences,
                    properties={**kept_concept.properties, **removed_concept.properties},
                    semantic_tags=kept_concept.semantic_tags.union(removed_concept.semantic_tags)
                )
                
                knowledge_graph.concepts[keep_id] = updated_concept
                
                # Remove the merged concept
                del knowledge_graph.concepts[remove_id]
                knowledge_graph.graph.remove_node(remove_id)
        
        return knowledge_graph
    
    def _prune_weak_relations(self, knowledge_graph: KnowledgeGraph) -> KnowledgeGraph:
        """Remove relations that have become too weak."""
        relations_to_remove = []
        
        for relation_id, relation in knowledge_graph.relations.items():
            # Apply decay based on time since last reinforcement
            current_time = time.time()
            time_since_reinforcement = current_time - relation.last_reinforced
            decay_factor = jnp.exp(-self.params.relation_decay_rate * time_since_reinforcement)
            current_strength = relation.strength * decay_factor
            
            if current_strength < self.params.relation_strength_threshold:
                relations_to_remove.append(relation_id)
        
        # Remove weak relations
        for relation_id in relations_to_remove:
            relation = knowledge_graph.relations[relation_id]
            del knowledge_graph.relations[relation_id]
            
            # Remove from graph
            if knowledge_graph.graph.has_edge(relation.source_concept, relation.target_concept):
                knowledge_graph.graph.remove_edge(relation.source_concept, relation.target_concept)
        
        return knowledge_graph
    
    def _update_abstraction_hierarchy(
        self, 
        knowledge_graph: KnowledgeGraph, 
        key: jax.random.PRNGKey
    ) -> KnowledgeGraph:
        """Update the abstraction hierarchy of concepts."""
        # This is a simplified version - in practice, this would be more sophisticated
        for concept_id, concept in knowledge_graph.concepts.items():
            # Concepts with many connections might be more abstract
            if not knowledge_graph.graph.has_node(concept_id):
                continue
            degree = knowledge_graph.graph.degree(concept_id)
            
            if degree > 10 and concept.abstraction_level == 0:
                # Promote to higher abstraction level
                new_level = min(concept.abstraction_level + 1, self.params.max_abstraction_levels - 1)
                
                updated_concept = Concept(
                    concept_id=concept.concept_id,
                    name=concept.name,
                    embedding=concept.embedding,
                    frequency=concept.frequency,
                    creation_time=concept.creation_time,
                    last_updated=time.time(),
                    confidence=concept.confidence,
                    abstraction_level=new_level,
                    source_experiences=concept.source_experiences,
                    properties=concept.properties,
                    semantic_tags=concept.semantic_tags
                )
                
                knowledge_graph.concepts[concept_id] = updated_concept
                
                # Update indices
                knowledge_graph.abstraction_hierarchy[concept.abstraction_level].discard(concept_id)
                knowledge_graph.abstraction_hierarchy[new_level].add(concept_id)
        
        return knowledge_graph
    
    def _apply_temporal_decay(
        self, 
        concept_frequencies: Dict[str, int], 
        relation_strengths: Dict[str, float], 
        current_time: float
    ) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Apply temporal decay to concept frequencies and relation strengths."""
        # Simple decay - in practice, this would be more sophisticated
        updated_frequencies = concept_frequencies.copy()
        updated_strengths = relation_strengths.copy()
        
        # Apply small decay to unused concepts and relations
        for concept_id in updated_frequencies:
            if updated_frequencies[concept_id] > 1:
                updated_frequencies[concept_id] = max(1, int(updated_frequencies[concept_id] * 0.99))
        
        for relation_id in updated_strengths:
            updated_strengths[relation_id] = max(0.1, updated_strengths[relation_id] * 0.995)
        
        return updated_frequencies, updated_strengths
    
    def _rebuild_embeddings_matrix(self, concepts: Dict[str, Concept]) -> jnp.ndarray:
        """Rebuild the concept embeddings matrix."""
        if not concepts:
            return jnp.zeros((0, self.params.embedding_dim))
        
        embeddings = [concept.embedding for concept in concepts.values()]
        return jnp.stack(embeddings)


# Convenience functions
def create_semantic_memory(memory_type: str = "standard") -> SemanticMemory:
    """
    Create semantic memory with predefined parameter sets.
    
    Args:
        memory_type: Type of memory configuration
                    - "standard": Default parameters
                    - "small": Small capacity for testing
                    - "large": Large capacity for complex domains
                    - "fast_learning": Quick concept formation
                    - "slow_learning": Conservative concept formation
                    - "high_abstraction": Strong abstraction capabilities
    
    Returns:
        Configured SemanticMemory
    """
    if memory_type == "standard":
        return SemanticMemory()
    elif memory_type == "small":
        params = SemanticMemoryParams(
            max_concepts=100,
            embedding_dim=64,
            reservoir_size=100,
            min_concept_frequency=2
        )
        return SemanticMemory(params)
    elif memory_type == "large":
        params = SemanticMemoryParams(
            max_concepts=50000,
            embedding_dim=512,
            reservoir_size=800,
            max_relations_per_concept=100
        )
        return SemanticMemory(params)
    elif memory_type == "fast_learning":
        params = SemanticMemoryParams(
            min_concept_frequency=2,
            concept_similarity_threshold=0.6,
            relation_strength_threshold=0.2
        )
        return SemanticMemory(params)
    elif memory_type == "slow_learning":
        params = SemanticMemoryParams(
            min_concept_frequency=5,
            concept_similarity_threshold=0.9,
            relation_strength_threshold=0.5
        )
        return SemanticMemory(params)
    elif memory_type == "high_abstraction":
        params = SemanticMemoryParams(
            max_abstraction_levels=10,
            abstraction_merge_threshold=0.8,
            clustering_threshold=0.6
        )
        return SemanticMemory(params)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
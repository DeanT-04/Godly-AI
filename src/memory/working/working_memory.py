"""
Working Memory Implementation with Dynamic Reservoirs

This module implements the working memory system using dynamic spiking reservoirs
for pattern storage and retrieval with attention-based mechanisms.
"""

from typing import NamedTuple, Optional, Tuple, Dict, List, Any
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
import numpy as np
import hashlib
import time

from ...core.liquid_state_machine import LiquidStateMachine, LSMParams, LSMState


@dataclass
class WorkingMemoryParams:
    """Parameters for the Working Memory system."""
    
    # Memory capacity
    capacity: int = 50              # Maximum number of patterns to store
    decay_rate: float = 0.1         # Memory decay rate per second
    
    # Reservoir parameters
    reservoir_size: int = 200       # Size of each memory reservoir
    input_size: int = 100          # Input pattern size
    
    # Attention mechanism
    attention_temperature: float = 2.0  # Temperature for attention softmax
    attention_decay: float = 0.05   # Attention weight decay rate
    
    # Competitive dynamics
    competition_strength: float = 0.3  # Strength of competitive inhibition
    winner_take_all_threshold: float = 0.7  # Threshold for winner-take-all
    
    # Pattern matching
    similarity_threshold: float = 0.8  # Minimum similarity for pattern match
    retrieval_noise: float = 0.01   # Noise added during retrieval
    
    # Timing
    millisecond_timescale: float = 1e-3  # Operating timescale


@dataclass
class MemoryPattern:
    """Stored memory pattern with metadata."""
    
    pattern_id: str                 # Unique pattern identifier
    pattern: jnp.ndarray           # Original pattern data
    reservoir_state: jnp.ndarray   # Encoded reservoir state
    timestamp: float               # Storage timestamp
    access_count: int              # Number of times accessed
    last_accessed: float           # Last access timestamp
    strength: float                # Memory strength (0-1)
    attention_weight: float        # Current attention weight
    associations: List[str]        # Associated pattern IDs


class WorkingMemoryState(NamedTuple):
    """State variables for the Working Memory system."""
    
    patterns: Dict[str, MemoryPattern]  # Stored patterns
    reservoir_states: Dict[str, LSMState]  # LSM states for each pattern
    attention_weights: jnp.ndarray     # Global attention distribution
    competition_state: jnp.ndarray     # Competitive dynamics state
    global_time: float                 # Current time
    total_patterns: int                # Total patterns stored


class WorkingMemory:
    """
    Working Memory system with dynamic spiking reservoirs.
    
    Implements:
    - Pattern storage using liquid state machines
    - Attention-based retrieval mechanisms
    - Competitive dynamics for capacity management
    - Content-addressable memory access
    - Temporal decay and forgetting
    """
    
    def __init__(self, params: Optional[WorkingMemoryParams] = None):
        """Initialize Working Memory system."""
        self.params = params or WorkingMemoryParams()
        
        # Create LSM template for pattern encoding
        lsm_params = LSMParams(
            reservoir_size=self.params.reservoir_size,
            input_size=self.params.input_size,
            output_size=self.params.input_size,  # Reconstruction output
            reservoir_connectivity=0.15,
            spectral_radius=0.95,
            enable_plasticity=True,
            homeostatic_scaling=True
        )
        self.lsm_template = LiquidStateMachine(lsm_params)
    
    def init_state(self, key: Optional[jax.random.PRNGKey] = None) -> WorkingMemoryState:
        """
        Initialize Working Memory state.
        
        Args:
            key: Random key for initialization
            
        Returns:
            Initial working memory state
        """
        if key is None:
            key = random.PRNGKey(0)
        
        return WorkingMemoryState(
            patterns={},
            reservoir_states={},
            attention_weights=jnp.zeros(self.params.capacity),
            competition_state=jnp.zeros(self.params.capacity),
            global_time=0.0,
            total_patterns=0
        )
    
    def store_pattern(
        self, 
        state: WorkingMemoryState, 
        pattern: jnp.ndarray, 
        timestamp: Optional[float] = None,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[WorkingMemoryState, str]:
        """
        Store a pattern in working memory.
        
        Args:
            state: Current working memory state
            pattern: Pattern to store [input_size]
            timestamp: Storage timestamp (current time if None)
            key: Random key for LSM initialization
            
        Returns:
            Tuple of (updated_state, pattern_id)
        """
        if key is None:
            key = random.PRNGKey(int(time.time() * 1000))
        
        if timestamp is None:
            timestamp = time.time()
        
        # Generate unique pattern ID
        pattern_id = self._generate_pattern_id(pattern, timestamp)
        
        # Check if we need to make space
        if len(state.patterns) >= self.params.capacity:
            state = self._make_space(state)
        
        # Encode pattern using LSM
        lsm_state = self.lsm_template.init_state(key)
        
        # Convert pattern to spike train for encoding
        spike_train = self._pattern_to_spikes(pattern)
        
        # Process through LSM to get reservoir encoding
        reservoir_states, final_lsm_state = self.lsm_template.process_spike_train(
            spike_train, self.params.millisecond_timescale, key
        )
        
        # Create memory pattern
        memory_pattern = MemoryPattern(
            pattern_id=pattern_id,
            pattern=pattern,
            reservoir_state=reservoir_states[-1],  # Final reservoir state
            timestamp=timestamp,
            access_count=0,
            last_accessed=timestamp,
            strength=0.8,  # Initial strength with room to grow
            attention_weight=0.0,  # Will be updated by attention mechanism
            associations=[]
        )
        
        # Update state
        new_patterns = state.patterns.copy()
        new_patterns[pattern_id] = memory_pattern
        
        new_reservoir_states = state.reservoir_states.copy()
        new_reservoir_states[pattern_id] = final_lsm_state
        
        # Update attention weights
        new_attention_weights = self._update_attention_after_storage(
            state.attention_weights, len(new_patterns)
        )
        
        new_state = WorkingMemoryState(
            patterns=new_patterns,
            reservoir_states=new_reservoir_states,
            attention_weights=new_attention_weights,
            competition_state=state.competition_state,
            global_time=timestamp,
            total_patterns=state.total_patterns + 1
        )
        
        return new_state, pattern_id
    
    def retrieve_pattern(
        self, 
        state: WorkingMemoryState, 
        query: jnp.ndarray,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[WorkingMemoryState, Optional[jnp.ndarray], float]:
        """
        Retrieve a pattern from working memory using content-addressable access.
        
        Args:
            state: Current working memory state
            query: Query pattern [input_size]
            key: Random key for noise injection
            
        Returns:
            Tuple of (updated_state, retrieved_pattern, confidence)
        """
        if key is None:
            key = random.PRNGKey(int(time.time() * 1000))
        
        if not state.patterns:
            return state, None, 0.0
        
        # Encode query using LSM
        query_spike_train = self._pattern_to_spikes(query)
        query_lsm_state = self.lsm_template.init_state(key)
        query_reservoir_states, _ = self.lsm_template.process_spike_train(
            query_spike_train, self.params.millisecond_timescale, key
        )
        query_encoding = query_reservoir_states[-1]
        
        # Compute similarities with stored patterns
        similarities = {}
        for pattern_id, memory_pattern in state.patterns.items():
            similarity = self._compute_similarity(
                query_encoding, memory_pattern.reservoir_state
            )
            similarities[pattern_id] = similarity
        
        # Find best match
        best_pattern_id = max(similarities.keys(), key=lambda k: similarities[k])
        best_similarity = similarities[best_pattern_id]
        
        if best_similarity < self.params.similarity_threshold:
            return state, None, 0.0
        
        # Retrieve pattern with competitive dynamics
        retrieved_pattern, confidence = self._competitive_retrieval(
            state, best_pattern_id, similarities, key
        )
        
        # Update pattern access statistics
        updated_patterns = state.patterns.copy()
        memory_pattern = updated_patterns[best_pattern_id]
        updated_patterns[best_pattern_id] = MemoryPattern(
            pattern_id=memory_pattern.pattern_id,
            pattern=memory_pattern.pattern,
            reservoir_state=memory_pattern.reservoir_state,
            timestamp=memory_pattern.timestamp,
            access_count=memory_pattern.access_count + 1,
            last_accessed=time.time(),
            strength=min(1.0, memory_pattern.strength + 0.1),  # Strengthen on access
            attention_weight=memory_pattern.attention_weight,
            associations=memory_pattern.associations
        )
        
        # Update attention weights
        new_attention_weights = self._update_attention_after_retrieval(
            state.attention_weights, best_pattern_id, list(state.patterns.keys())
        )
        
        new_state = WorkingMemoryState(
            patterns=updated_patterns,
            reservoir_states=state.reservoir_states,
            attention_weights=new_attention_weights,
            competition_state=state.competition_state,
            global_time=time.time(),
            total_patterns=state.total_patterns
        )
        
        return new_state, retrieved_pattern, confidence
    
    def update_attention_weights(
        self, 
        state: WorkingMemoryState, 
        relevance_scores: jnp.ndarray
    ) -> WorkingMemoryState:
        """
        Update attention weights based on relevance scores.
        
        Args:
            state: Current working memory state
            relevance_scores: Relevance scores for each pattern [n_patterns]
            
        Returns:
            Updated working memory state
        """
        if not state.patterns:
            return state
        
        # Ensure relevance scores match number of patterns
        n_patterns = len(state.patterns)
        if len(relevance_scores) != n_patterns:
            # Pad or truncate as needed
            if len(relevance_scores) < n_patterns:
                relevance_scores = jnp.concatenate([
                    relevance_scores, 
                    jnp.zeros(n_patterns - len(relevance_scores))
                ])
            else:
                relevance_scores = relevance_scores[:n_patterns]
        
        # Apply softmax with temperature
        attention_weights = jax.nn.softmax(
            relevance_scores / self.params.attention_temperature
        )
        
        # Apply attention decay to previous weights
        current_weights = state.attention_weights[:n_patterns]
        decayed_weights = current_weights * (1 - self.params.attention_decay)
        
        # Combine with new attention
        new_attention_weights = 0.7 * attention_weights + 0.3 * decayed_weights
        
        # Pad to full capacity
        if n_patterns < self.params.capacity:
            new_attention_weights = jnp.concatenate([
                new_attention_weights,
                jnp.zeros(self.params.capacity - n_patterns)
            ])
        
        return state._replace(attention_weights=new_attention_weights)
    
    def decay_memories(
        self, 
        state: WorkingMemoryState, 
        dt: float
    ) -> WorkingMemoryState:
        """
        Apply temporal decay to stored memories.
        
        Args:
            state: Current working memory state
            dt: Time step for decay
            
        Returns:
            Updated working memory state with decayed memories
        """
        if not state.patterns:
            return state
        
        current_time = time.time()
        updated_patterns = {}
        patterns_to_remove = []
        
        for pattern_id, memory_pattern in state.patterns.items():
            # Calculate age-based decay
            age = current_time - memory_pattern.timestamp
            decay_factor = jnp.exp(-self.params.decay_rate * age)
            
            # Calculate access-based strengthening
            access_factor = 1.0 + 0.1 * memory_pattern.access_count
            
            # Update strength
            new_strength = memory_pattern.strength * decay_factor * access_factor
            
            if new_strength < 0.1:  # Remove very weak memories
                patterns_to_remove.append(pattern_id)
            else:
                updated_patterns[pattern_id] = MemoryPattern(
                    pattern_id=memory_pattern.pattern_id,
                    pattern=memory_pattern.pattern,
                    reservoir_state=memory_pattern.reservoir_state,
                    timestamp=memory_pattern.timestamp,
                    access_count=memory_pattern.access_count,
                    last_accessed=memory_pattern.last_accessed,
                    strength=new_strength,
                    attention_weight=memory_pattern.attention_weight,
                    associations=memory_pattern.associations
                )
        
        # Remove weak patterns
        updated_reservoir_states = state.reservoir_states.copy()
        for pattern_id in patterns_to_remove:
            if pattern_id in updated_reservoir_states:
                del updated_reservoir_states[pattern_id]
        
        # Update attention weights
        if updated_patterns:
            n_patterns = len(updated_patterns)
            new_attention_weights = state.attention_weights[:n_patterns]
            if n_patterns < self.params.capacity:
                new_attention_weights = jnp.concatenate([
                    new_attention_weights,
                    jnp.zeros(self.params.capacity - n_patterns)
                ])
        else:
            new_attention_weights = jnp.zeros(self.params.capacity)
        
        return WorkingMemoryState(
            patterns=updated_patterns,
            reservoir_states=updated_reservoir_states,
            attention_weights=new_attention_weights,
            competition_state=state.competition_state,
            global_time=current_time,
            total_patterns=state.total_patterns
        )
    
    def get_memory_statistics(self, state: WorkingMemoryState) -> Dict[str, Any]:
        """
        Get statistics about current memory state.
        
        Args:
            state: Current working memory state
            
        Returns:
            Dictionary of memory statistics
        """
        if not state.patterns:
            return {
                'num_patterns': 0,
                'capacity_utilization': 0.0,
                'mean_strength': 0.0,
                'mean_age': 0.0,
                'total_accesses': 0,
                'attention_entropy': 0.0
            }
        
        patterns = list(state.patterns.values())
        current_time = time.time()
        
        # Basic statistics
        num_patterns = len(patterns)
        capacity_utilization = num_patterns / self.params.capacity
        
        # Strength statistics
        strengths = [p.strength for p in patterns]
        mean_strength = float(jnp.mean(jnp.array(strengths)))
        
        # Age statistics
        ages = [current_time - p.timestamp for p in patterns]
        mean_age = float(jnp.mean(jnp.array(ages)))
        
        # Access statistics
        total_accesses = sum(p.access_count for p in patterns)
        
        # Attention entropy
        active_attention = state.attention_weights[:num_patterns]
        if jnp.sum(active_attention) > 0:
            normalized_attention = active_attention / jnp.sum(active_attention)
            attention_entropy = float(-jnp.sum(
                normalized_attention * jnp.log(normalized_attention + 1e-10)
            ))
        else:
            attention_entropy = 0.0
        
        return {
            'num_patterns': num_patterns,
            'capacity_utilization': capacity_utilization,
            'mean_strength': mean_strength,
            'mean_age': mean_age,
            'total_accesses': total_accesses,
            'attention_entropy': attention_entropy,
            'patterns_stored_total': state.total_patterns
        }
    
    def _generate_pattern_id(self, pattern: jnp.ndarray, timestamp: float) -> str:
        """Generate unique pattern ID."""
        pattern_hash = hashlib.md5(
            jnp.array(pattern).tobytes() + str(timestamp).encode()
        ).hexdigest()
        return f"pattern_{pattern_hash[:8]}"
    
    def _pattern_to_spikes(self, pattern: jnp.ndarray) -> jnp.ndarray:
        """Convert pattern to spike train for LSM encoding."""
        # Simple rate coding: pattern values -> spike probabilities
        # Normalize pattern to [0, 1]
        normalized_pattern = (pattern - jnp.min(pattern)) / (
            jnp.max(pattern) - jnp.min(pattern) + 1e-10
        )
        
        # Create spike train over time
        n_timesteps = 50  # 50ms encoding window
        spike_train = jnp.zeros((n_timesteps, len(pattern)))
        
        # Generate spikes based on pattern values
        # Use pattern hash for deterministic but pattern-specific randomness
        pattern_hash = hash(tuple(pattern.tolist()))
        key = random.PRNGKey(pattern_hash % (2**31))  # Ensure positive int32
        
        for t in range(n_timesteps):
            spike_probs = normalized_pattern * 0.1  # Max 10% spike probability
            key, subkey = random.split(key)
            spikes = random.bernoulli(subkey, spike_probs)
            spike_train = spike_train.at[t].set(spikes)
        
        return spike_train
    
    def _compute_similarity(
        self, 
        encoding1: jnp.ndarray, 
        encoding2: jnp.ndarray
    ) -> float:
        """Compute similarity between two reservoir encodings."""
        # Cosine similarity
        dot_product = jnp.dot(encoding1, encoding2)
        norm1 = jnp.linalg.norm(encoding1)
        norm2 = jnp.linalg.norm(encoding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(jnp.clip(similarity, 0.0, 1.0))
    
    def _competitive_retrieval(
        self,
        state: WorkingMemoryState,
        best_pattern_id: str,
        similarities: Dict[str, float],
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, float]:
        """Perform competitive retrieval with winner-take-all dynamics."""
        # Get the best pattern
        best_pattern = state.patterns[best_pattern_id]
        retrieved_pattern = best_pattern.pattern
        
        # Apply competitive dynamics
        similarity_values = jnp.array(list(similarities.values()))
        competition_weights = jax.nn.softmax(
            similarity_values * self.params.competition_strength
        )
        
        # Winner-take-all if best similarity is strong enough
        best_similarity = similarities[best_pattern_id]
        if best_similarity > self.params.winner_take_all_threshold:
            # Scale confidence to be less than 1.0 for non-perfect matches
            confidence = best_similarity * 0.95  # Slight reduction for realism
        else:
            # Weighted combination of top patterns
            confidence = float(jnp.max(competition_weights)) * 0.8
            
            # Add small amount of noise for biological realism
            noise = random.normal(key, retrieved_pattern.shape) * self.params.retrieval_noise
            retrieved_pattern = retrieved_pattern + noise
        
        return retrieved_pattern, confidence
    
    def _make_space(self, state: WorkingMemoryState) -> WorkingMemoryState:
        """Remove least important patterns to make space."""
        if not state.patterns:
            return state
        
        # Score patterns for removal (lower is more likely to be removed)
        removal_scores = {}
        current_time = time.time()
        
        for pattern_id, pattern in state.patterns.items():
            # Combine recency, strength, and access count
            recency_score = 1.0 / (current_time - pattern.last_accessed + 1.0)
            strength_score = pattern.strength
            access_score = jnp.log(pattern.access_count + 1.0)
            
            removal_scores[pattern_id] = (
                0.4 * recency_score + 
                0.4 * strength_score + 
                0.2 * access_score
            )
        
        # Remove pattern with lowest score
        pattern_to_remove = min(removal_scores.keys(), key=lambda k: removal_scores[k])
        
        new_patterns = state.patterns.copy()
        del new_patterns[pattern_to_remove]
        
        new_reservoir_states = state.reservoir_states.copy()
        if pattern_to_remove in new_reservoir_states:
            del new_reservoir_states[pattern_to_remove]
        
        return WorkingMemoryState(
            patterns=new_patterns,
            reservoir_states=new_reservoir_states,
            attention_weights=state.attention_weights,
            competition_state=state.competition_state,
            global_time=state.global_time,
            total_patterns=state.total_patterns
        )
    
    def _update_attention_after_storage(
        self, 
        current_attention: jnp.ndarray, 
        n_patterns: int
    ) -> jnp.ndarray:
        """Update attention weights after storing a new pattern."""
        if n_patterns == 1:
            # First pattern gets full attention
            new_attention = jnp.zeros(self.params.capacity)
            new_attention = new_attention.at[0].set(1.0)
            return new_attention
        
        # Redistribute attention to include new pattern
        active_attention = current_attention[:n_patterns-1]
        
        # New pattern gets some attention, others are slightly reduced
        new_pattern_attention = 0.2
        existing_attention = active_attention * 0.9
        
        new_attention = jnp.concatenate([
            existing_attention,
            jnp.array([new_pattern_attention])
        ])
        
        # Normalize
        new_attention = new_attention / jnp.sum(new_attention)
        
        # Pad to full capacity
        if n_patterns < self.params.capacity:
            new_attention = jnp.concatenate([
                new_attention,
                jnp.zeros(self.params.capacity - n_patterns)
            ])
        
        return new_attention
    
    def _update_attention_after_retrieval(
        self,
        current_attention: jnp.ndarray,
        retrieved_pattern_id: str,
        pattern_ids: List[str]
    ) -> jnp.ndarray:
        """Update attention weights after retrieving a pattern."""
        try:
            retrieved_index = pattern_ids.index(retrieved_pattern_id)
        except ValueError:
            return current_attention
        
        # Boost attention for retrieved pattern
        new_attention = current_attention.copy()
        n_patterns = len(pattern_ids)
        
        if retrieved_index < len(new_attention):
            # Increase attention for retrieved pattern
            boost = 0.1
            new_attention = new_attention.at[retrieved_index].add(boost)
            
            # Normalize active attention
            active_attention = new_attention[:n_patterns]
            if jnp.sum(active_attention) > 0:
                active_attention = active_attention / jnp.sum(active_attention)
                new_attention = new_attention.at[:n_patterns].set(active_attention)
        
        return new_attention


# Convenience functions
def create_working_memory(memory_type: str = "standard") -> WorkingMemory:
    """
    Create working memory with predefined parameter sets.
    
    Args:
        memory_type: Type of memory configuration
                    - "standard": Default parameters
                    - "small": Small capacity for testing
                    - "large": Large capacity for complex tasks
                    - "fast_decay": Rapid memory decay
                    - "slow_decay": Slow memory decay
                    - "high_attention": Strong attention mechanisms
    
    Returns:
        Configured WorkingMemory
    """
    if memory_type == "standard":
        return WorkingMemory()
    elif memory_type == "small":
        params = WorkingMemoryParams(
            capacity=10,
            reservoir_size=50,
            input_size=20,
            similarity_threshold=0.3  # Lower threshold for LSM-based encoding
        )
        return WorkingMemory(params)
    elif memory_type == "large":
        params = WorkingMemoryParams(
            capacity=200,
            reservoir_size=500,
            input_size=200
        )
        return WorkingMemory(params)
    elif memory_type == "fast_decay":
        params = WorkingMemoryParams(
            decay_rate=0.5,  # Fast decay
            attention_decay=0.2
        )
        return WorkingMemory(params)
    elif memory_type == "slow_decay":
        params = WorkingMemoryParams(
            decay_rate=0.01,  # Slow decay
            attention_decay=0.01
        )
        return WorkingMemory(params)
    elif memory_type == "high_attention":
        params = WorkingMemoryParams(
            attention_temperature=0.5,  # Sharp attention
            competition_strength=0.8,   # Strong competition
            winner_take_all_threshold=0.5
        )
        return WorkingMemory(params)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
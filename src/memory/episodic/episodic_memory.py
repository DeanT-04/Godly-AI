"""
Episodic Memory Implementation

This module implements the episodic memory system for experience storage and replay
with temporal context, hierarchical compression, and associative retrieval.
"""

from typing import NamedTuple, Optional, Tuple, Dict, List, Any, Union
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass, field
import numpy as np
import hashlib
import time
import pickle
import gzip
from collections import defaultdict

from ...core.liquid_state_machine import LiquidStateMachine, LSMParams, LSMState


@dataclass
class Experience:
    """Experience data structure for episodic memory storage."""
    
    observation: jnp.ndarray           # Current observation
    action: jnp.ndarray               # Action taken
    reward: float                     # Reward received
    next_observation: jnp.ndarray     # Next observation
    timestamp: float                  # When experience occurred
    context: Dict[str, Any]           # Additional context information


@dataclass
class EpisodicMemoryParams:
    """Parameters for the Episodic Memory system."""
    
    # Storage capacity
    max_episodes: int = 10000         # Maximum number of episodes to store
    max_episode_length: int = 1000    # Maximum steps per episode
    
    # Compression settings
    compression_threshold: float = 0.8  # Similarity threshold for compression
    compression_ratio: float = 0.5     # Target compression ratio
    
    # Temporal context
    temporal_window: float = 10.0      # Seconds for temporal context
    context_decay: float = 0.1         # Context strength decay rate
    
    # Replay mechanisms
    replay_batch_size: int = 32        # Number of experiences per replay batch
    replay_priority_alpha: float = 0.6 # Priority exponent for replay sampling
    replay_beta: float = 0.4           # Importance sampling exponent
    
    # Consolidation
    consolidation_interval: float = 3600.0  # Consolidation every hour
    consolidation_threshold: int = 100      # Min episodes before consolidation
    
    # Retrieval
    similarity_threshold: float = 0.7   # Minimum similarity for retrieval
    max_retrieval_results: int = 10     # Maximum results per query
    
    # LSM encoding
    reservoir_size: int = 300          # Size of encoding reservoir
    encoding_timesteps: int = 20       # Timesteps for experience encoding


@dataclass
class Episode:
    """Stored episode with metadata."""
    
    episode_id: str                    # Unique episode identifier
    experiences: List[Experience]      # Sequence of experiences
    start_time: float                 # Episode start timestamp
    end_time: float                   # Episode end timestamp
    total_reward: float               # Cumulative reward
    episode_length: int               # Number of steps
    context_summary: Dict[str, Any]   # Summarized context
    encoding: jnp.ndarray            # LSM encoding of episode
    compression_level: float          # Level of compression applied
    access_count: int                 # Number of times accessed
    last_accessed: float             # Last access timestamp
    importance_weight: float          # Importance for replay
    consolidation_level: int          # Level of consolidation (0=raw, higher=more compressed)


@dataclass
class ConsolidationNode:
    """Node in the hierarchical consolidation tree."""
    
    node_id: str                      # Unique node identifier
    child_episodes: List[str]         # Child episode IDs
    parent_node: Optional[str]        # Parent node ID
    level: int                        # Consolidation level
    summary_encoding: jnp.ndarray     # Compressed representation
    temporal_span: Tuple[float, float] # Time span covered
    access_frequency: float           # How often accessed
    creation_time: float              # When node was created


class EpisodicMemoryState(NamedTuple):
    """State variables for the Episodic Memory system."""
    
    episodes: Dict[str, Episode]                    # Stored episodes
    consolidation_tree: Dict[str, ConsolidationNode] # Hierarchical consolidation
    current_episode: Optional[List[Experience]]     # Currently building episode
    current_episode_start: float                   # Current episode start time
    current_episode_context: Dict[str, Any]        # Current episode initial context
    replay_priorities: Dict[str, float]            # Episode replay priorities
    temporal_index: Dict[float, List[str]]         # Time-based episode index
    context_index: Dict[str, List[str]]            # Context-based episode index
    total_episodes: int                            # Total episodes stored
    last_consolidation: float                      # Last consolidation time
    global_time: float                             # Current system time


class EpisodicMemory:
    """
    Episodic Memory system for experience storage and replay.
    
    Implements:
    - Experience storage with temporal context
    - Memory replay mechanisms for learning enhancement
    - Hierarchical compression for long-term storage
    - Associative retrieval based on context similarity
    - Priority-based replay sampling
    - Automatic memory consolidation
    """
    
    def __init__(self, params: Optional[EpisodicMemoryParams] = None):
        """Initialize Episodic Memory system."""
        self.params = params or EpisodicMemoryParams()
        
        # Create LSM for experience encoding - will be initialized when first used
        self.encoding_lsm = None
        self._input_size = None
    
    def init_state(self, key: Optional[jax.random.PRNGKey] = None) -> EpisodicMemoryState:
        """
        Initialize Episodic Memory state.
        
        Args:
            key: Random key for initialization
            
        Returns:
            Initial episodic memory state
        """
        current_time = time.time()
        
        return EpisodicMemoryState(
            episodes={},
            consolidation_tree={},
            current_episode=None,
            current_episode_start=current_time,
            current_episode_context={},
            replay_priorities={},
            temporal_index=defaultdict(list),
            context_index=defaultdict(list),
            total_episodes=0,
            last_consolidation=current_time,
            global_time=current_time
        )
    
    def start_episode(
        self, 
        state: EpisodicMemoryState,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> EpisodicMemoryState:
        """
        Start a new episode.
        
        Args:
            state: Current episodic memory state
            initial_context: Initial context for the episode
            
        Returns:
            Updated state with new episode started
        """
        current_time = time.time()
        
        # Finalize previous episode if exists
        if state.current_episode is not None:
            state = self._finalize_current_episode(state)
        
        # Store initial context for this episode
        episode_context = initial_context or {}
        
        return state._replace(
            current_episode=[],
            current_episode_start=current_time,
            current_episode_context=episode_context,
            global_time=current_time
        )
    
    def store_experience(
        self,
        state: EpisodicMemoryState,
        experience: Experience
    ) -> EpisodicMemoryState:
        """
        Store an experience in the current episode.
        
        Args:
            state: Current episodic memory state
            experience: Experience to store
            
        Returns:
            Updated state with experience stored
        """
        # Start episode if none exists
        if state.current_episode is None:
            state = self.start_episode(state)
        
        # Add experience to current episode
        current_episode = state.current_episode.copy()
        current_episode.append(experience)
        
        # Check if episode is getting too long
        if len(current_episode) >= self.params.max_episode_length:
            # Finalize current episode and start new one
            state = state._replace(current_episode=current_episode)
            state = self._finalize_current_episode(state)
            state = self.start_episode(state)
        else:
            state = state._replace(
                current_episode=current_episode,
                global_time=experience.timestamp
            )
        
        return state
    
    def store_episode(
        self,
        state: EpisodicMemoryState,
        experience: Experience,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[EpisodicMemoryState, str]:
        """
        Store a complete experience as an episode.
        
        Args:
            state: Current episodic memory state
            experience: Experience to store as episode
            key: Random key for encoding
            
        Returns:
            Tuple of (updated_state, episode_id)
        """
        if key is None:
            key = random.PRNGKey(int(time.time() * 1000))
        
        # Create single-experience episode
        experiences = [experience]
        episode_id = self._generate_episode_id(experiences)
        
        # Encode episode
        encoding = self._encode_episode(experiences, key)
        
        # Create episode object
        episode = Episode(
            episode_id=episode_id,
            experiences=experiences,
            start_time=experience.timestamp,
            end_time=experience.timestamp,
            total_reward=experience.reward,
            episode_length=1,
            context_summary=experience.context.copy(),
            encoding=encoding,
            compression_level=0.0,
            access_count=0,
            last_accessed=experience.timestamp,
            importance_weight=abs(experience.reward),  # Use reward magnitude as importance
            consolidation_level=0
        )
        
        # Check capacity and make space if needed
        if len(state.episodes) >= self.params.max_episodes:
            state = self._make_space(state)
        
        # Store episode
        new_episodes = state.episodes.copy()
        new_episodes[episode_id] = episode
        
        # Update indices
        new_temporal_index = dict(state.temporal_index)
        time_key = int(experience.timestamp)
        if time_key not in new_temporal_index:
            new_temporal_index[time_key] = []
        new_temporal_index[time_key].append(episode_id)
        
        new_context_index = dict(state.context_index)
        for context_key, context_value in experience.context.items():
            context_str = f"{context_key}:{str(context_value)}"
            if context_str not in new_context_index:
                new_context_index[context_str] = []
            new_context_index[context_str].append(episode_id)
        
        # Update replay priorities
        new_replay_priorities = state.replay_priorities.copy()
        new_replay_priorities[episode_id] = episode.importance_weight
        
        new_state = state._replace(
            episodes=new_episodes,
            temporal_index=new_temporal_index,
            context_index=new_context_index,
            replay_priorities=new_replay_priorities,
            total_episodes=state.total_episodes + 1,
            global_time=experience.timestamp
        )
        
        # Check if consolidation is needed
        if self._should_consolidate(new_state):
            new_state = self.consolidate_memories(new_state, key)
        
        return new_state, episode_id
    
    def replay_episode(
        self,
        state: EpisodicMemoryState,
        episode_id: str
    ) -> Tuple[EpisodicMemoryState, Optional[Experience]]:
        """
        Replay an episode by retrieving its experiences.
        
        Args:
            state: Current episodic memory state
            episode_id: ID of episode to replay
            
        Returns:
            Tuple of (updated_state, experience) or (state, None) if not found
        """
        if episode_id not in state.episodes:
            return state, None
        
        episode = state.episodes[episode_id]
        
        # Update access statistics
        updated_episode = Episode(
            episode_id=episode.episode_id,
            experiences=episode.experiences,
            start_time=episode.start_time,
            end_time=episode.end_time,
            total_reward=episode.total_reward,
            episode_length=episode.episode_length,
            context_summary=episode.context_summary,
            encoding=episode.encoding,
            compression_level=episode.compression_level,
            access_count=episode.access_count + 1,
            last_accessed=time.time(),
            importance_weight=episode.importance_weight * 1.1,  # Boost importance on access
            consolidation_level=episode.consolidation_level
        )
        
        # Update state
        new_episodes = state.episodes.copy()
        new_episodes[episode_id] = updated_episode
        
        new_replay_priorities = state.replay_priorities.copy()
        new_replay_priorities[episode_id] = updated_episode.importance_weight
        
        new_state = state._replace(
            episodes=new_episodes,
            replay_priorities=new_replay_priorities,
            global_time=time.time()
        )
        
        # Return first experience (for single-experience episodes) or random experience
        if episode.experiences:
            # For now, return the first experience
            # In a more sophisticated implementation, this could return the full sequence
            return new_state, episode.experiences[0]
        
        return new_state, None
    
    def sample_replay_batch(
        self,
        state: EpisodicMemoryState,
        batch_size: Optional[int] = None,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[EpisodicMemoryState, List[Experience]]:
        """
        Sample a batch of experiences for replay using priority sampling.
        
        Args:
            state: Current episodic memory state
            batch_size: Size of batch to sample (uses default if None)
            key: Random key for sampling
            
        Returns:
            Tuple of (updated_state, batch_of_experiences)
        """
        if key is None:
            key = random.PRNGKey(int(time.time() * 1000))
        
        if batch_size is None:
            batch_size = self.params.replay_batch_size
        
        if not state.episodes:
            return state, []
        
        # Get episode IDs and priorities
        episode_ids = list(state.episodes.keys())
        priorities = jnp.array([
            state.replay_priorities.get(eid, 1.0) for eid in episode_ids
        ])
        
        # Apply priority exponent
        priorities = jnp.power(priorities, self.params.replay_priority_alpha)
        
        # Normalize to probabilities
        probabilities = priorities / jnp.sum(priorities)
        
        # Sample episodes
        batch_size = min(batch_size, len(episode_ids))
        sampled_indices = random.choice(
            key, len(episode_ids), shape=(batch_size,), 
            replace=True, p=probabilities
        )
        
        # Collect experiences and update access counts
        batch_experiences = []
        new_episodes = state.episodes.copy()
        new_replay_priorities = state.replay_priorities.copy()
        
        for idx in sampled_indices:
            episode_id = episode_ids[idx]
            episode = state.episodes[episode_id]
            
            # Add experience to batch
            if episode.experiences:
                batch_experiences.append(episode.experiences[0])
            
            # Update access count
            updated_episode = Episode(
                episode_id=episode.episode_id,
                experiences=episode.experiences,
                start_time=episode.start_time,
                end_time=episode.end_time,
                total_reward=episode.total_reward,
                episode_length=episode.episode_length,
                context_summary=episode.context_summary,
                encoding=episode.encoding,
                compression_level=episode.compression_level,
                access_count=episode.access_count + 1,
                last_accessed=time.time(),
                importance_weight=episode.importance_weight,
                consolidation_level=episode.consolidation_level
            )
            
            new_episodes[episode_id] = updated_episode
        
        new_state = state._replace(
            episodes=new_episodes,
            replay_priorities=new_replay_priorities,
            global_time=time.time()
        )
        
        return new_state, batch_experiences
    
    def consolidate_memories(
        self,
        state: EpisodicMemoryState,
        key: Optional[jax.random.PRNGKey] = None
    ) -> EpisodicMemoryState:
        """
        Perform hierarchical memory consolidation.
        
        Args:
            state: Current episodic memory state
            key: Random key for consolidation operations
            
        Returns:
            Updated state with consolidated memories
        """
        if key is None:
            key = random.PRNGKey(int(time.time() * 1000))
        
        current_time = time.time()
        
        # Find episodes that can be consolidated
        consolidation_candidates = self._find_consolidation_candidates(state)
        
        if len(consolidation_candidates) < 2:
            return state._replace(
                last_consolidation=current_time,
                global_time=current_time
            )
        
        # Group similar episodes for consolidation
        consolidation_groups = self._group_similar_episodes(
            state, consolidation_candidates, key
        )
        
        new_consolidation_tree = state.consolidation_tree.copy()
        new_episodes = state.episodes.copy()
        
        # Create consolidation nodes for each group
        for group_episodes in consolidation_groups:
            if len(group_episodes) >= 2:
                # Create consolidation node
                node_id = self._generate_consolidation_node_id(group_episodes)
                
                # Compute summary encoding
                summary_encoding = self._compute_summary_encoding(
                    [state.episodes[eid] for eid in group_episodes], key
                )
                
                # Find temporal span
                start_times = [state.episodes[eid].start_time for eid in group_episodes]
                end_times = [state.episodes[eid].end_time for eid in group_episodes]
                temporal_span = (min(start_times), max(end_times))
                
                # Create consolidation node
                consolidation_node = ConsolidationNode(
                    node_id=node_id,
                    child_episodes=group_episodes,
                    parent_node=None,  # Top-level for now
                    level=1,
                    summary_encoding=summary_encoding,
                    temporal_span=temporal_span,
                    access_frequency=0.0,
                    creation_time=current_time
                )
                
                new_consolidation_tree[node_id] = consolidation_node
                
                # Update episodes to reference consolidation
                for episode_id in group_episodes:
                    episode = new_episodes[episode_id]
                    new_episodes[episode_id] = Episode(
                        episode_id=episode.episode_id,
                        experiences=episode.experiences,
                        start_time=episode.start_time,
                        end_time=episode.end_time,
                        total_reward=episode.total_reward,
                        episode_length=episode.episode_length,
                        context_summary=episode.context_summary,
                        encoding=episode.encoding,
                        compression_level=min(1.0, episode.compression_level + 0.2),
                        access_count=episode.access_count,
                        last_accessed=episode.last_accessed,
                        importance_weight=episode.importance_weight * 0.9,  # Reduce importance
                        consolidation_level=episode.consolidation_level + 1
                    )
        
        return state._replace(
            consolidation_tree=new_consolidation_tree,
            episodes=new_episodes,
            last_consolidation=current_time,
            global_time=current_time
        )
    
    def query_episodes(
        self,
        state: EpisodicMemoryState,
        query_context: Dict[str, Any],
        temporal_range: Optional[Tuple[float, float]] = None,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[EpisodicMemoryState, List[str]]:
        """
        Query episodes based on context and temporal constraints.
        
        Args:
            state: Current episodic memory state
            query_context: Context to match against
            temporal_range: Optional time range (start, end)
            key: Random key for similarity computation
            
        Returns:
            Tuple of (updated_state, list_of_episode_ids)
        """
        if key is None:
            key = random.PRNGKey(int(time.time() * 1000))
        
        if not state.episodes:
            return state, []
        
        # Filter by temporal range if specified
        candidate_episodes = []
        for episode_id, episode in state.episodes.items():
            if temporal_range is not None:
                start_time, end_time = temporal_range
                if not (start_time <= episode.start_time <= end_time):
                    continue
            candidate_episodes.append(episode_id)
        
        if not candidate_episodes:
            return state, []
        
        # Compute context similarities
        similarities = {}
        for episode_id in candidate_episodes:
            episode = state.episodes[episode_id]
            similarity = self._compute_context_similarity(
                query_context, episode.context_summary
            )
            if similarity >= self.params.similarity_threshold:
                similarities[episode_id] = similarity
        
        # Sort by similarity and return top results
        sorted_episodes = sorted(
            similarities.keys(), 
            key=lambda k: similarities[k], 
            reverse=True
        )
        
        result_episodes = sorted_episodes[:self.params.max_retrieval_results]
        
        return state, result_episodes
    
    def get_memory_statistics(self, state: EpisodicMemoryState) -> Dict[str, Any]:
        """
        Get statistics about current episodic memory state.
        
        Args:
            state: Current episodic memory state
            
        Returns:
            Dictionary of memory statistics
        """
        if not state.episodes:
            return {
                'num_episodes': 0,
                'total_experiences': 0,
                'memory_utilization': 0.0,
                'average_episode_length': 0.0,
                'consolidation_nodes': 0,
                'oldest_episode_age': 0.0,
                'most_accessed_episode': None,
                'total_reward_stored': 0.0
            }
        
        episodes = list(state.episodes.values())
        current_time = time.time()
        
        # Basic statistics
        num_episodes = len(episodes)
        total_experiences = sum(ep.episode_length for ep in episodes)
        memory_utilization = num_episodes / self.params.max_episodes
        average_episode_length = total_experiences / num_episodes
        
        # Consolidation statistics
        consolidation_nodes = len(state.consolidation_tree)
        
        # Age statistics
        oldest_episode_age = max(
            current_time - ep.start_time for ep in episodes
        )
        
        # Access statistics
        most_accessed_episode = max(
            episodes, key=lambda ep: ep.access_count
        ).episode_id
        
        # Reward statistics
        total_reward_stored = sum(ep.total_reward for ep in episodes)
        
        return {
            'num_episodes': num_episodes,
            'total_experiences': total_experiences,
            'memory_utilization': memory_utilization,
            'average_episode_length': average_episode_length,
            'consolidation_nodes': consolidation_nodes,
            'oldest_episode_age': oldest_episode_age,
            'most_accessed_episode': most_accessed_episode,
            'total_reward_stored': total_reward_stored,
            'episodes_stored_total': state.total_episodes
        }
    
    # Private helper methods
    
    def _generate_episode_id(self, experiences: List[Experience]) -> str:
        """Generate unique episode ID."""
        # Create hash from first and last experience
        if not experiences:
            return f"episode_{int(time.time() * 1000)}"
        
        first_exp = experiences[0]
        last_exp = experiences[-1]
        
        content = (
            str(first_exp.timestamp) + 
            str(last_exp.timestamp) + 
            str(len(experiences))
        )
        
        episode_hash = hashlib.md5(content.encode()).hexdigest()
        return f"episode_{episode_hash[:8]}"
    
    def _encode_episode(
        self, 
        experiences: List[Experience], 
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Encode episode using LSM."""
        if not experiences:
            return jnp.zeros(self.params.reservoir_size)
        
        # Convert experiences to input sequence
        input_sequence = []
        for exp in experiences:
            # Flatten experience into input vector
            obs_flat = exp.observation.flatten()
            action_flat = exp.action.flatten()
            reward_vec = jnp.array([exp.reward])
            
            # Combine into single input vector
            exp_vector = jnp.concatenate([obs_flat, action_flat, reward_vec])
            input_sequence.append(exp_vector)
        
        # Initialize LSM if not done yet
        if self.encoding_lsm is None:
            self._input_size = len(input_sequence[0])
            lsm_params = LSMParams(
                reservoir_size=self.params.reservoir_size,
                input_size=self._input_size,
                output_size=self.params.reservoir_size,
                reservoir_connectivity=0.1,
                spectral_radius=0.9,
                enable_plasticity=False,  # Static encoding for consistency
                homeostatic_scaling=False
            )
            self.encoding_lsm = LiquidStateMachine(lsm_params)
        
        # Pad or truncate to fixed length
        max_length = self.params.encoding_timesteps
        if len(input_sequence) > max_length:
            input_sequence = input_sequence[:max_length]
        else:
            # Pad with zeros
            while len(input_sequence) < max_length:
                input_sequence.append(jnp.zeros(self._input_size))
        
        # Stack into input matrix
        input_matrix = jnp.stack(input_sequence)
        
        # Process through LSM
        lsm_state = self.encoding_lsm.init_state(key)
        reservoir_states, _ = self.encoding_lsm.process_spike_train(
            input_matrix, dt=0.001, key=key
        )
        
        # Return final reservoir state as encoding
        return reservoir_states[-1]
    
    def _finalize_current_episode(self, state: EpisodicMemoryState) -> EpisodicMemoryState:
        """Finalize the current episode and store it."""
        if state.current_episode is None or not state.current_episode:
            return state
        
        # Create episode from current experiences
        experiences = state.current_episode
        episode_id = self._generate_episode_id(experiences)
        
        # Encode episode
        key = random.PRNGKey(int(time.time() * 1000))
        encoding = self._encode_episode(experiences, key)
        
        # Calculate episode statistics
        total_reward = sum(exp.reward for exp in experiences)
        start_time = experiences[0].timestamp
        end_time = experiences[-1].timestamp
        
        # Create context summary by merging episode context with experience contexts
        context_summary = state.current_episode_context.copy()  # Start with episode context
        
        # Add experience contexts
        for exp in experiences:
            for key, value in exp.context.items():
                if key not in context_summary:
                    context_summary[key] = []
                elif not isinstance(context_summary[key], list):
                    # Convert single value to list for aggregation
                    context_summary[key] = [context_summary[key]]
                context_summary[key].append(value)
        
        # Summarize context (take most common values for lists, keep single values)
        for key, values in context_summary.items():
            if isinstance(values, list):
                if len(values) == 1:
                    context_summary[key] = values[0]
                elif isinstance(values[0], (int, float)):
                    context_summary[key] = float(jnp.mean(jnp.array(values)))
                else:
                    # Take most common value
                    context_summary[key] = max(set(values), key=values.count)
        
        # Create episode
        episode = Episode(
            episode_id=episode_id,
            experiences=experiences,
            start_time=start_time,
            end_time=end_time,
            total_reward=total_reward,
            episode_length=len(experiences),
            context_summary=context_summary,
            encoding=encoding,
            compression_level=0.0,
            access_count=0,
            last_accessed=end_time,
            importance_weight=abs(total_reward),
            consolidation_level=0
        )
        
        # Store episode
        new_episodes = state.episodes.copy()
        new_episodes[episode_id] = episode
        
        # Update indices and priorities
        new_replay_priorities = state.replay_priorities.copy()
        new_replay_priorities[episode_id] = episode.importance_weight
        
        return state._replace(
            episodes=new_episodes,
            current_episode=None,
            current_episode_context={},
            replay_priorities=new_replay_priorities,
            total_episodes=state.total_episodes + 1
        )
    
    def _should_consolidate(self, state: EpisodicMemoryState) -> bool:
        """Check if memory consolidation should be performed."""
        current_time = time.time()
        
        # Check time-based consolidation
        time_since_last = current_time - state.last_consolidation
        if time_since_last < self.params.consolidation_interval:
            return False
        
        # Check if we have enough episodes
        if len(state.episodes) < self.params.consolidation_threshold:
            return False
        
        return True
    
    def _find_consolidation_candidates(self, state: EpisodicMemoryState) -> List[str]:
        """Find episodes that are candidates for consolidation."""
        candidates = []
        current_time = time.time()
        
        for episode_id, episode in state.episodes.items():
            # Consider episodes that are old enough and not recently accessed
            age = current_time - episode.end_time
            time_since_access = current_time - episode.last_accessed
            
            if age > 3600.0 and time_since_access > 1800.0:  # 1 hour old, 30 min since access
                candidates.append(episode_id)
        
        return candidates
    
    def _group_similar_episodes(
        self, 
        state: EpisodicMemoryState, 
        episode_ids: List[str],
        key: jax.random.PRNGKey
    ) -> List[List[str]]:
        """Group similar episodes for consolidation."""
        if len(episode_ids) < 2:
            return []
        
        # Compute pairwise similarities
        similarities = {}
        for i, id1 in enumerate(episode_ids):
            for j, id2 in enumerate(episode_ids[i+1:], i+1):
                episode1 = state.episodes[id1]
                episode2 = state.episodes[id2]
                
                # Compute encoding similarity
                encoding_sim = self._compute_encoding_similarity(
                    episode1.encoding, episode2.encoding
                )
                
                # Compute context similarity
                context_sim = self._compute_context_similarity(
                    episode1.context_summary, episode2.context_summary
                )
                
                # Combined similarity
                combined_sim = 0.6 * encoding_sim + 0.4 * context_sim
                similarities[(id1, id2)] = combined_sim
        
        # Simple greedy grouping
        groups = []
        used_episodes = set()
        
        for (id1, id2), similarity in sorted(
            similarities.items(), key=lambda x: x[1], reverse=True
        ):
            if similarity >= self.params.compression_threshold:
                if id1 not in used_episodes and id2 not in used_episodes:
                    groups.append([id1, id2])
                    used_episodes.add(id1)
                    used_episodes.add(id2)
        
        return groups
    
    def _compute_summary_encoding(
        self, 
        episodes: List[Episode], 
        key: jax.random.PRNGKey
    ) -> jnp.ndarray:
        """Compute summary encoding for a group of episodes."""
        if not episodes:
            return jnp.zeros(self.params.reservoir_size)
        
        # Simple average of encodings
        encodings = jnp.stack([ep.encoding for ep in episodes])
        return jnp.mean(encodings, axis=0)
    
    def _generate_consolidation_node_id(self, episode_ids: List[str]) -> str:
        """Generate unique consolidation node ID."""
        content = "_".join(sorted(episode_ids))
        node_hash = hashlib.md5(content.encode()).hexdigest()
        return f"consolidation_{node_hash[:8]}"
    
    def _compute_encoding_similarity(
        self, 
        encoding1: jnp.ndarray, 
        encoding2: jnp.ndarray
    ) -> float:
        """Compute similarity between two episode encodings."""
        # Cosine similarity
        dot_product = jnp.dot(encoding1, encoding2)
        norm1 = jnp.linalg.norm(encoding1)
        norm2 = jnp.linalg.norm(encoding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(jnp.clip(similarity, 0.0, 1.0))
    
    def _compute_context_similarity(
        self, 
        context1: Dict[str, Any], 
        context2: Dict[str, Any]
    ) -> float:
        """Compute similarity between two context dictionaries."""
        if not context1 or not context2:
            return 0.0
        
        # Find common keys
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        # Compute similarity for each common key
        similarities = []
        for key in common_keys:
            val1, val2 = context1[key], context2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1.0)
                sim = 1.0 - abs(val1 - val2) / max_val
            elif val1 == val2:
                # Exact match
                sim = 1.0
            else:
                # No match
                sim = 0.0
            
            similarities.append(sim)
        
        return float(jnp.mean(jnp.array(similarities)))
    
    def _make_space(self, state: EpisodicMemoryState) -> EpisodicMemoryState:
        """Remove least important episodes to make space."""
        if not state.episodes:
            return state
        
        # Score episodes for removal (lower is more likely to be removed)
        removal_scores = {}
        current_time = time.time()
        
        for episode_id, episode in state.episodes.items():
            # Combine age, access frequency, and importance
            age = current_time - episode.end_time
            recency_score = 1.0 / (current_time - episode.last_accessed + 1.0)
            importance_score = episode.importance_weight
            access_score = jnp.log(episode.access_count + 1.0)
            
            removal_scores[episode_id] = (
                0.3 * recency_score + 
                0.4 * importance_score + 
                0.3 * access_score
            )
        
        # Remove episodes with lowest scores
        episodes_to_remove = sorted(
            removal_scores.keys(), 
            key=lambda k: removal_scores[k]
        )[:int(self.params.max_episodes * 0.1)]  # Remove 10%
        
        new_episodes = state.episodes.copy()
        new_replay_priorities = state.replay_priorities.copy()
        
        for episode_id in episodes_to_remove:
            if episode_id in new_episodes:
                del new_episodes[episode_id]
            if episode_id in new_replay_priorities:
                del new_replay_priorities[episode_id]
        
        return state._replace(
            episodes=new_episodes,
            replay_priorities=new_replay_priorities
        )


# Convenience functions
def create_episodic_memory(memory_type: str = "standard") -> EpisodicMemory:
    """
    Create episodic memory with predefined parameter sets.
    
    Args:
        memory_type: Type of memory configuration
                    - "standard": Default parameters
                    - "small": Small capacity for testing
                    - "large": Large capacity for complex tasks
                    - "fast_consolidation": Rapid memory consolidation
                    - "slow_consolidation": Slow memory consolidation
                    - "high_priority": Strong priority-based replay
    
    Returns:
        Configured EpisodicMemory
    """
    if memory_type == "standard":
        return EpisodicMemory()
    elif memory_type == "small":
        params = EpisodicMemoryParams(
            max_episodes=100,
            max_episode_length=50,
            reservoir_size=100,
            consolidation_threshold=10
        )
        return EpisodicMemory(params)
    elif memory_type == "large":
        params = EpisodicMemoryParams(
            max_episodes=50000,
            max_episode_length=5000,
            reservoir_size=500,
            consolidation_threshold=1000
        )
        return EpisodicMemory(params)
    elif memory_type == "fast_consolidation":
        params = EpisodicMemoryParams(
            consolidation_interval=600.0,  # 10 minutes
            consolidation_threshold=20,
            compression_threshold=0.6
        )
        return EpisodicMemory(params)
    elif memory_type == "slow_consolidation":
        params = EpisodicMemoryParams(
            consolidation_interval=86400.0,  # 24 hours
            consolidation_threshold=500,
            compression_threshold=0.9
        )
        return EpisodicMemory(params)
    elif memory_type == "high_priority":
        params = EpisodicMemoryParams(
            replay_priority_alpha=1.0,  # Strong priority
            replay_beta=0.8,           # Strong importance sampling
            replay_batch_size=64
        )
        return EpisodicMemory(params)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
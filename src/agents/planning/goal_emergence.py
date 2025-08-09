"""
Goal Emergence and Planning System

This module implements pattern recognition in behavior history and goal formulation
algorithms that emerge from behavioral patterns and resource constraints.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
import time
from collections import defaultdict, deque

from ..exploration.curiosity_engine import ExplorationGoal, ExplorationStrategy


class GoalType(Enum):
    """Types of emergent goals."""
    EXPLORATION = "exploration"
    SKILL_ACQUISITION = "skill_acquisition"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    PATTERN_COMPLETION = "pattern_completion"
    NOVELTY_SEEKING = "novelty_seeking"
    COMPETENCE_BUILDING = "competence_building"


class GoalPriority(Enum):
    """Goal priority levels."""
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    BACKGROUND = 0.2


@dataclass
class BehaviorPattern:
    """Represents a recognized pattern in behavior history."""
    pattern_id: str
    pattern_type: str
    observations: List[jnp.ndarray]
    actions: List[jnp.ndarray]
    rewards: List[float]
    frequency: int
    success_rate: float
    last_occurrence: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmergentGoal:
    """Represents a goal that emerged from behavioral patterns."""
    goal_id: str
    goal_type: GoalType
    priority: GoalPriority
    target_state: jnp.ndarray
    success_criteria: Dict[str, float]
    resource_requirements: Dict[str, float]
    estimated_duration: float
    parent_patterns: List[str]
    subgoals: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    attempts: int = 0
    success_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceState:
    """Represents current resource availability."""
    computational_budget: float
    memory_usage: float
    attention_capacity: float
    exploration_time: float
    learning_capacity: float
    energy_level: float = 1.0


class PatternRecognizer:
    """
    Recognizes patterns in behavior history for goal emergence.
    
    Uses sequence analysis and clustering to identify recurring behavioral patterns
    that can inform goal generation.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        pattern_memory_size: int = 1000,
        min_pattern_frequency: int = 3,
        pattern_similarity_threshold: float = 0.8
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.pattern_memory_size = pattern_memory_size
        self.min_pattern_frequency = min_pattern_frequency
        self.pattern_similarity_threshold = pattern_similarity_threshold
        
        # Behavior history storage
        self.behavior_history: deque = deque(maxlen=pattern_memory_size)
        self.recognized_patterns: Dict[str, BehaviorPattern] = {}
        self.pattern_counter = 0
        
        # Pattern analysis parameters
        self.sequence_length = 5  # Length of behavioral sequences to analyze
        self.clustering_threshold = 0.7
    
    def add_behavior_sample(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        reward: float,
        timestamp: float
    ) -> None:
        """Add a new behavior sample to the history."""
        sample = {
            'observation': observation,
            'action': action,
            'reward': reward,
            'timestamp': timestamp
        }
        self.behavior_history.append(sample)
    
    def _extract_sequences(self, sequence_length: int) -> List[List[Dict]]:
        """Extract behavioral sequences of specified length."""
        sequences = []
        history_list = list(self.behavior_history)
        
        for i in range(len(history_list) - sequence_length + 1):
            sequence = history_list[i:i + sequence_length]
            sequences.append(sequence)
        
        return sequences
    
    def _compute_sequence_similarity(self, seq1: List[Dict], seq2: List[Dict]) -> float:
        """Compute similarity between two behavioral sequences."""
        if len(seq1) != len(seq2):
            return 0.0
        
        obs_similarities = []
        action_similarities = []
        
        for s1, s2 in zip(seq1, seq2):
            # Observation similarity (cosine similarity)
            obs1, obs2 = s1['observation'], s2['observation']
            obs_sim = float(jnp.dot(obs1, obs2) / (jnp.linalg.norm(obs1) * jnp.linalg.norm(obs2) + 1e-8))
            obs_similarities.append(obs_sim)
            
            # Action similarity
            act1, act2 = s1['action'], s2['action']
            act_sim = float(jnp.dot(act1, act2) / (jnp.linalg.norm(act1) * jnp.linalg.norm(act2) + 1e-8))
            action_similarities.append(act_sim)
        
        # Combined similarity
        obs_sim_mean = np.mean(obs_similarities)
        act_sim_mean = np.mean(action_similarities)
        
        return 0.6 * obs_sim_mean + 0.4 * act_sim_mean
    
    def _cluster_sequences(self, sequences: List[List[Dict]]) -> List[List[int]]:
        """Cluster similar behavioral sequences."""
        if not sequences:
            return []
        
        # Compute pairwise similarities
        n_sequences = len(sequences)
        similarity_matrix = np.zeros((n_sequences, n_sequences))
        
        for i in range(n_sequences):
            for j in range(i + 1, n_sequences):
                sim = self._compute_sequence_similarity(sequences[i], sequences[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        # Simple clustering based on similarity threshold
        clusters = []
        assigned = set()
        
        for i in range(n_sequences):
            if i in assigned:
                continue
            
            cluster = [i]
            assigned.add(i)
            
            for j in range(i + 1, n_sequences):
                if j not in assigned and similarity_matrix[i, j] > self.clustering_threshold:
                    cluster.append(j)
                    assigned.add(j)
            
            if len(cluster) >= self.min_pattern_frequency:
                clusters.append(cluster)
        
        return clusters
    
    def recognize_patterns(self) -> List[BehaviorPattern]:
        """Recognize patterns in the current behavior history."""
        if len(self.behavior_history) < self.sequence_length:
            return []
        
        # Extract sequences
        sequences = self._extract_sequences(self.sequence_length)
        
        # Cluster similar sequences
        clusters = self._cluster_sequences(sequences)
        
        new_patterns = []
        current_time = time.time()
        
        for cluster_indices in clusters:
            # Create pattern from cluster
            cluster_sequences = [sequences[i] for i in cluster_indices]
            
            # Compute pattern statistics
            all_observations = []
            all_actions = []
            all_rewards = []
            
            for sequence in cluster_sequences:
                for sample in sequence:
                    all_observations.append(sample['observation'])
                    all_actions.append(sample['action'])
                    all_rewards.append(sample['reward'])
            
            # Pattern characteristics
            frequency = len(cluster_sequences)
            success_rate = np.mean([r for r in all_rewards if r > 0]) if all_rewards else 0.0
            confidence = min(1.0, frequency / 10.0)  # Confidence increases with frequency
            
            # Determine pattern type based on characteristics
            avg_reward = np.mean(all_rewards) if all_rewards else 0.0
            if avg_reward > 0.5:
                pattern_type = "successful_behavior"
            elif avg_reward < -0.5:
                pattern_type = "unsuccessful_behavior"
            else:
                pattern_type = "exploratory_behavior"
            
            pattern = BehaviorPattern(
                pattern_id=f"pattern_{self.pattern_counter}",
                pattern_type=pattern_type,
                observations=all_observations,
                actions=all_actions,
                rewards=all_rewards,
                frequency=frequency,
                success_rate=success_rate,
                last_occurrence=current_time,
                confidence=confidence,
                metadata={
                    'sequence_length': self.sequence_length,
                    'cluster_size': len(cluster_indices),
                    'avg_reward': avg_reward
                }
            )
            
            self.recognized_patterns[pattern.pattern_id] = pattern
            new_patterns.append(pattern)
            self.pattern_counter += 1
        
        return new_patterns
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about recognized patterns."""
        if not self.recognized_patterns:
            return {'total_patterns': 0}
        
        patterns = list(self.recognized_patterns.values())
        
        return {
            'total_patterns': len(patterns),
            'avg_frequency': np.mean([p.frequency for p in patterns]),
            'avg_success_rate': np.mean([p.success_rate for p in patterns]),
            'avg_confidence': np.mean([p.confidence for p in patterns]),
            'pattern_types': {
                ptype: len([p for p in patterns if p.pattern_type == ptype])
                for ptype in set(p.pattern_type for p in patterns)
            }
        }


class GoalFormulator:
    """
    Formulates goals from recognized behavioral patterns.
    
    Analyzes patterns to generate meaningful goals that can drive
    autonomous behavior and learning.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        max_active_goals: int = 10,
        goal_timeout: float = 300.0
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_active_goals = max_active_goals
        self.goal_timeout = goal_timeout
        
        # Goal management
        self.active_goals: Dict[str, EmergentGoal] = {}
        self.completed_goals: Dict[str, EmergentGoal] = {}
        self.goal_counter = 0
        
        # Goal generation parameters
        self.success_threshold = 0.6
        self.novelty_weight = 0.4
        self.competence_weight = 0.6
    
    def formulate_goals_from_patterns(
        self,
        patterns: List[BehaviorPattern],
        current_observation: jnp.ndarray,
        resource_state: ResourceState
    ) -> List[EmergentGoal]:
        """
        Formulate goals based on recognized behavioral patterns.
        
        Args:
            patterns: List of recognized behavioral patterns
            current_observation: Current system observation
            resource_state: Current resource availability
            
        Returns:
            List of formulated emergent goals
        """
        new_goals = []
        current_time = time.time()
        
        # Clean up expired goals
        self._cleanup_expired_goals(current_time)
        
        # Don't generate more goals if at capacity
        if len(self.active_goals) >= self.max_active_goals:
            return new_goals
        
        for pattern in patterns:
            if len(self.active_goals) >= self.max_active_goals:
                break
            
            # Generate different types of goals based on pattern characteristics
            goal_candidates = []
            
            # 1. Skill acquisition goals from successful patterns
            if pattern.success_rate > self.success_threshold:
                skill_goal = self._create_skill_acquisition_goal(pattern, current_observation)
                if skill_goal:
                    goal_candidates.append(skill_goal)
            
            # 2. Exploration goals from novel patterns
            if pattern.confidence < 0.5:  # Low confidence indicates novelty
                exploration_goal = self._create_exploration_goal(pattern, current_observation)
                if exploration_goal:
                    goal_candidates.append(exploration_goal)
            
            # 3. Pattern completion goals
            completion_goal = self._create_pattern_completion_goal(pattern, current_observation)
            if completion_goal:
                goal_candidates.append(completion_goal)
            
            # 4. Resource optimization goals
            if resource_state.computational_budget < 0.5:
                optimization_goal = self._create_resource_optimization_goal(pattern, resource_state)
                if optimization_goal:
                    goal_candidates.append(optimization_goal)
            
            # Select best goal candidate based on priority and resources
            if goal_candidates:
                best_goal = self._select_best_goal(goal_candidates, resource_state)
                if best_goal:
                    self.active_goals[best_goal.goal_id] = best_goal
                    new_goals.append(best_goal)
        
        return new_goals
    
    def _create_skill_acquisition_goal(
        self,
        pattern: BehaviorPattern,
        current_observation: jnp.ndarray
    ) -> Optional[EmergentGoal]:
        """Create a skill acquisition goal from a successful pattern."""
        if not pattern.observations:
            return None
        
        # Target state is the average successful observation
        successful_obs = [
            obs for obs, reward in zip(pattern.observations, pattern.rewards)
            if reward > 0
        ]
        
        if not successful_obs:
            return None
        
        target_state = jnp.mean(jnp.array(successful_obs), axis=0)
        
        goal = EmergentGoal(
            goal_id=f"skill_goal_{self.goal_counter}",
            goal_type=GoalType.SKILL_ACQUISITION,
            priority=GoalPriority.HIGH,
            target_state=target_state,
            success_criteria={
                'min_success_rate': 0.8,
                'min_reward': 0.5,
                'consistency_threshold': 0.9
            },
            resource_requirements={
                'computational_budget': 0.3,
                'learning_capacity': 0.4,
                'attention_capacity': 0.5
            },
            estimated_duration=100.0,
            parent_patterns=[pattern.pattern_id],
            metadata={
                'pattern_success_rate': pattern.success_rate,
                'pattern_frequency': pattern.frequency,
                'skill_type': 'behavioral_replication'
            }
        )
        
        self.goal_counter += 1
        return goal
    
    def _create_exploration_goal(
        self,
        pattern: BehaviorPattern,
        current_observation: jnp.ndarray
    ) -> Optional[EmergentGoal]:
        """Create an exploration goal from a novel pattern."""
        if not pattern.observations:
            return None
        
        # Target state is in the direction of the novel pattern
        pattern_center = jnp.mean(jnp.array(pattern.observations), axis=0)
        direction = pattern_center - current_observation
        direction_norm = jnp.linalg.norm(direction)
        
        if direction_norm < 1e-6:
            return None
        
        # Target is in the direction of the pattern but further out
        target_state = current_observation + (direction / direction_norm) * 2.0
        
        goal = EmergentGoal(
            goal_id=f"explore_goal_{self.goal_counter}",
            goal_type=GoalType.EXPLORATION,
            priority=GoalPriority.MEDIUM,
            target_state=target_state,
            success_criteria={
                'novelty_threshold': 0.7,
                'exploration_distance': 1.5,
                'information_gain': 0.5
            },
            resource_requirements={
                'computational_budget': 0.2,
                'exploration_time': 0.6,
                'attention_capacity': 0.3
            },
            estimated_duration=50.0,
            parent_patterns=[pattern.pattern_id],
            metadata={
                'pattern_confidence': pattern.confidence,
                'exploration_direction': direction.tolist(),
                'novelty_score': 1.0 - pattern.confidence
            }
        )
        
        self.goal_counter += 1
        return goal
    
    def _create_pattern_completion_goal(
        self,
        pattern: BehaviorPattern,
        current_observation: jnp.ndarray
    ) -> Optional[EmergentGoal]:
        """Create a goal to complete an interrupted pattern."""
        if len(pattern.observations) < 2:
            return None
        
        # Find the most common ending state in the pattern
        ending_states = pattern.observations[-len(pattern.observations)//3:]  # Last third
        target_state = jnp.mean(jnp.array(ending_states), axis=0)
        
        goal = EmergentGoal(
            goal_id=f"completion_goal_{self.goal_counter}",
            goal_type=GoalType.PATTERN_COMPLETION,
            priority=GoalPriority.MEDIUM,
            target_state=target_state,
            success_criteria={
                'completion_accuracy': 0.8,
                'sequence_fidelity': 0.7,
                'reward_threshold': 0.3
            },
            resource_requirements={
                'computational_budget': 0.25,
                'learning_capacity': 0.2,
                'attention_capacity': 0.4
            },
            estimated_duration=30.0,
            parent_patterns=[pattern.pattern_id],
            metadata={
                'pattern_length': len(pattern.observations),
                'completion_type': 'sequence_ending',
                'expected_reward': np.mean(pattern.rewards) if pattern.rewards else 0.0
            }
        )
        
        self.goal_counter += 1
        return goal
    
    def _create_resource_optimization_goal(
        self,
        pattern: BehaviorPattern,
        resource_state: ResourceState
    ) -> Optional[EmergentGoal]:
        """Create a goal to optimize resource usage."""
        # Find the most efficient actions in the pattern
        if not pattern.actions or not pattern.rewards:
            return None
        
        # Efficiency is reward per action magnitude
        efficiencies = []
        for action, reward in zip(pattern.actions, pattern.rewards):
            action_cost = float(jnp.linalg.norm(action))
            efficiency = reward / (action_cost + 1e-6)
            efficiencies.append(efficiency)
        
        if not efficiencies:
            return None
        
        # Target the most efficient action
        best_idx = np.argmax(efficiencies)
        target_action = pattern.actions[best_idx]
        
        # Convert action to target state (simplified)
        target_state = jnp.array(target_action)  # Simplified mapping
        
        goal = EmergentGoal(
            goal_id=f"optimize_goal_{self.goal_counter}",
            goal_type=GoalType.RESOURCE_OPTIMIZATION,
            priority=GoalPriority.HIGH if resource_state.computational_budget < 0.3 else GoalPriority.MEDIUM,
            target_state=target_state,
            success_criteria={
                'efficiency_improvement': 0.2,
                'resource_reduction': 0.15,
                'performance_maintenance': 0.9
            },
            resource_requirements={
                'computational_budget': 0.1,  # Low resource requirement
                'learning_capacity': 0.3,
                'attention_capacity': 0.2
            },
            estimated_duration=20.0,
            parent_patterns=[pattern.pattern_id],
            metadata={
                'target_efficiency': max(efficiencies),
                'current_efficiency': np.mean(efficiencies),
                'optimization_type': 'action_efficiency'
            }
        )
        
        self.goal_counter += 1
        return goal
    
    def _select_best_goal(
        self,
        goal_candidates: List[EmergentGoal],
        resource_state: ResourceState
    ) -> Optional[EmergentGoal]:
        """Select the best goal from candidates based on priority and resources."""
        if not goal_candidates:
            return None
        
        # Score goals based on priority and resource availability
        goal_scores = []
        
        for goal in goal_candidates:
            priority_score = goal.priority.value
            
            # Resource feasibility score
            resource_score = 1.0
            for resource, requirement in goal.resource_requirements.items():
                available = getattr(resource_state, resource, 1.0)
                if available < requirement:
                    resource_score *= available / requirement
            
            # Combined score
            total_score = priority_score * resource_score
            goal_scores.append((goal, total_score))
        
        # Return highest scoring goal
        goal_scores.sort(key=lambda x: x[1], reverse=True)
        return goal_scores[0][0]
    
    def _cleanup_expired_goals(self, current_time: float) -> None:
        """Remove expired goals from active list."""
        expired_goals = []
        
        for goal_id, goal in self.active_goals.items():
            if current_time - goal.created_at > self.goal_timeout:
                expired_goals.append(goal_id)
        
        for goal_id in expired_goals:
            goal = self.active_goals.pop(goal_id)
            # Don't add to completed since it expired
    
    def evaluate_goal_progress(
        self,
        goal_id: str,
        current_observation: jnp.ndarray,
        current_reward: float
    ) -> Dict[str, Any]:
        """
        Evaluate progress towards a specific goal.
        
        Args:
            goal_id: ID of the goal to evaluate
            current_observation: Current system observation
            current_reward: Current reward signal
            
        Returns:
            Dictionary containing progress metrics
        """
        if goal_id not in self.active_goals:
            return {'error': 'Goal not found'}
        
        goal = self.active_goals[goal_id]
        goal.attempts += 1
        
        # Compute distance to target
        distance_to_target = float(jnp.linalg.norm(current_observation - goal.target_state))
        
        # Evaluate success criteria
        success_metrics = {}
        overall_success = True
        
        for criterion, threshold in goal.success_criteria.items():
            if criterion == 'min_reward':
                success_metrics[criterion] = current_reward >= threshold
            elif criterion == 'novelty_threshold':
                # Simplified novelty check
                success_metrics[criterion] = distance_to_target > threshold
            elif criterion == 'completion_accuracy':
                success_metrics[criterion] = distance_to_target < (1.0 - threshold)
            else:
                # Default: assume met for unknown criteria
                success_metrics[criterion] = True
            
            overall_success = overall_success and success_metrics[criterion]
        
        # Update goal statistics
        if overall_success:
            goal.success_count += 1
        
        progress = {
            'goal_id': goal_id,
            'distance_to_target': distance_to_target,
            'success_metrics': success_metrics,
            'overall_success': overall_success,
            'attempts': goal.attempts,
            'success_rate': goal.success_count / goal.attempts if goal.attempts > 0 else 0.0,
            'progress_percentage': max(0.0, min(1.0, 1.0 - distance_to_target / 5.0))  # Normalized progress
        }
        
        # Move to completed if successful
        if overall_success:
            self.completed_goals[goal_id] = self.active_goals.pop(goal_id)
        
        return progress
    
    def get_goal_statistics(self) -> Dict[str, Any]:
        """Get comprehensive goal statistics."""
        active_goals = list(self.active_goals.values())
        completed_goals = list(self.completed_goals.values())
        all_goals = active_goals + completed_goals
        
        if not all_goals:
            return {'total_goals': 0}
        
        stats = {
            'total_goals': len(all_goals),
            'active_goals': len(active_goals),
            'completed_goals': len(completed_goals),
            'completion_rate': len(completed_goals) / len(all_goals),
            'goal_types': {
                goal_type.value: len([g for g in all_goals if g.goal_type == goal_type])
                for goal_type in GoalType
            },
            'priority_distribution': {
                priority.name: len([g for g in all_goals if g.priority == priority])
                for priority in GoalPriority
            }
        }
        
        if completed_goals:
            stats['avg_completion_time'] = np.mean([
                time.time() - g.created_at for g in completed_goals
            ])
            stats['avg_success_rate'] = np.mean([
                g.success_count / g.attempts if g.attempts > 0 else 0.0
                for g in completed_goals
            ])
        
        return stats
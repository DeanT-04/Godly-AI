"""
Integrated Planning System

This module provides the main interface for goal emergence and planning,
integrating pattern recognition, goal formulation, and resource management.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
import time

from .goal_emergence import (
    PatternRecognizer,
    GoalFormulator,
    BehaviorPattern,
    EmergentGoal,
    GoalType,
    GoalPriority,
    ResourceState
)


@dataclass
class PlanningConfig:
    """Configuration for the planning system."""
    observation_dim: int
    action_dim: int
    pattern_memory_size: int = 1000
    max_active_goals: int = 10
    goal_timeout: float = 300.0
    min_pattern_frequency: int = 3
    pattern_similarity_threshold: float = 0.8
    resource_update_interval: float = 10.0


class ResourceManager:
    """
    Manages system resources and constraints for goal prioritization.
    
    Tracks computational budget, memory usage, attention capacity, and other
    resources to inform goal selection and prioritization.
    """
    
    def __init__(
        self,
        initial_computational_budget: float = 1.0,
        initial_memory_capacity: float = 1.0,
        initial_attention_capacity: float = 1.0,
        decay_rate: float = 0.01
    ):
        self.initial_computational_budget = initial_computational_budget
        self.initial_memory_capacity = initial_memory_capacity
        self.initial_attention_capacity = initial_attention_capacity
        self.decay_rate = decay_rate
        
        # Current resource state
        self.current_state = ResourceState(
            computational_budget=initial_computational_budget,
            memory_usage=0.0,
            attention_capacity=initial_attention_capacity,
            exploration_time=1.0,
            learning_capacity=1.0,
            energy_level=1.0
        )
        
        # Resource usage history
        self.usage_history: List[Tuple[float, ResourceState]] = []
        self.last_update_time = time.time()
    
    def update_resources(
        self,
        computational_usage: float = 0.0,
        memory_usage: float = 0.0,
        attention_usage: float = 0.0,
        exploration_usage: float = 0.0,
        learning_usage: float = 0.0
    ) -> None:
        """
        Update resource usage and availability.
        
        Args:
            computational_usage: Amount of computational budget used
            memory_usage: Amount of memory used
            attention_usage: Amount of attention capacity used
            exploration_usage: Amount of exploration time used
            learning_usage: Amount of learning capacity used
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # Update resource usage
        self.current_state.computational_budget = max(0.0, 
            self.current_state.computational_budget - computational_usage)
        self.current_state.memory_usage = min(1.0, 
            self.current_state.memory_usage + memory_usage)
        self.current_state.attention_capacity = max(0.0, 
            self.current_state.attention_capacity - attention_usage)
        self.current_state.exploration_time = max(0.0, 
            self.current_state.exploration_time - exploration_usage)
        self.current_state.learning_capacity = max(0.0, 
            self.current_state.learning_capacity - learning_usage)
        
        # Natural resource regeneration over time
        regeneration_factor = 1.0 - np.exp(-dt * self.decay_rate)
        
        self.current_state.computational_budget = min(1.0,
            self.current_state.computational_budget + regeneration_factor * 0.1)
        self.current_state.attention_capacity = min(1.0,
            self.current_state.attention_capacity + regeneration_factor * 0.05)
        self.current_state.exploration_time = min(1.0,
            self.current_state.exploration_time + regeneration_factor * 0.02)
        self.current_state.learning_capacity = min(1.0,
            self.current_state.learning_capacity + regeneration_factor * 0.03)
        
        # Energy level affects all other resources
        self.current_state.energy_level = max(0.1, 
            self.current_state.energy_level - dt * self.decay_rate * 0.001)
        
        # Store history
        self.usage_history.append((current_time, self.current_state))
        if len(self.usage_history) > 1000:
            self.usage_history.pop(0)
        
        self.last_update_time = current_time
    
    def get_resource_state(self) -> ResourceState:
        """Get current resource state."""
        return self.current_state
    
    def can_afford_goal(self, goal: EmergentGoal) -> bool:
        """Check if current resources can support a goal."""
        for resource, requirement in goal.resource_requirements.items():
            available = getattr(self.current_state, resource, 0.0)
            if available < requirement:
                return False
        return True
    
    def allocate_resources_for_goal(self, goal: EmergentGoal) -> bool:
        """
        Allocate resources for a goal if available.
        
        Returns True if allocation successful, False otherwise.
        """
        if not self.can_afford_goal(goal):
            return False
        
        # Allocate resources
        for resource, requirement in goal.resource_requirements.items():
            current_value = getattr(self.current_state, resource, 0.0)
            setattr(self.current_state, resource, current_value - requirement)
        
        return True
    
    def get_resource_pressure(self) -> Dict[str, float]:
        """Get resource pressure metrics (how constrained each resource is)."""
        return {
            'computational_pressure': 1.0 - self.current_state.computational_budget,
            'memory_pressure': self.current_state.memory_usage,
            'attention_pressure': 1.0 - self.current_state.attention_capacity,
            'exploration_pressure': 1.0 - self.current_state.exploration_time,
            'learning_pressure': 1.0 - self.current_state.learning_capacity,
            'energy_pressure': 1.0 - self.current_state.energy_level
        }


class PlanningSystem:
    """
    Main planning system that coordinates pattern recognition, goal formulation,
    and resource management for autonomous behavior planning.
    """
    
    def __init__(self, config: PlanningConfig):
        self.config = config
        
        # Initialize components
        self.pattern_recognizer = PatternRecognizer(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            pattern_memory_size=config.pattern_memory_size,
            min_pattern_frequency=config.min_pattern_frequency,
            pattern_similarity_threshold=config.pattern_similarity_threshold
        )
        
        self.goal_formulator = GoalFormulator(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            max_active_goals=config.max_active_goals,
            goal_timeout=config.goal_timeout
        )
        
        self.resource_manager = ResourceManager()
        
        # Planning state
        self.planning_active = False
        self.planning_episode = 0
        self.last_pattern_recognition = 0.0
        self.last_resource_update = 0.0
        
        # Performance tracking
        self.planning_history: List[Dict[str, Any]] = []
        self.goal_achievement_rates: List[float] = []
        
        # Callbacks for external integration
        self.action_callback: Optional[Callable[[EmergentGoal], jnp.ndarray]] = None
        self.feedback_callback: Optional[Callable[[str, float], None]] = None
    
    def set_action_callback(self, callback: Callable[[EmergentGoal], jnp.ndarray]) -> None:
        """Set callback function for executing goal-directed actions."""
        self.action_callback = callback
    
    def set_feedback_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set callback function for receiving goal achievement feedback."""
        self.feedback_callback = callback
    
    def start_planning(self, initial_observation: jnp.ndarray, initial_action: jnp.ndarray) -> None:
        """
        Start a new planning episode.
        
        Args:
            initial_observation: Initial observation to start planning from
            initial_action: Initial action taken
        """
        self.planning_active = True
        self.planning_episode += 1
        
        # Add initial behavior sample
        current_time = time.time()
        self.pattern_recognizer.add_behavior_sample(
            observation=initial_observation,
            action=initial_action,
            reward=0.0,  # No reward for initial sample
            timestamp=current_time
        )
        
        self.last_pattern_recognition = current_time
        self.last_resource_update = current_time
        
        print(f"Started planning episode {self.planning_episode}")
    
    def step_planning(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        reward: float
    ) -> Dict[str, Any]:
        """
        Perform one step of planning.
        
        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            
        Returns:
            Dictionary containing planning step results
        """
        if not self.planning_active:
            return {'error': 'Planning not active'}
        
        current_time = time.time()
        step_results = {
            'patterns_recognized': 0,
            'goals_generated': 0,
            'active_goals': 0,
            'resource_pressure': {},
            'recommended_goal': None
        }
        
        # Add behavior sample
        self.pattern_recognizer.add_behavior_sample(
            observation=observation,
            action=action,
            reward=reward,
            timestamp=current_time
        )
        
        # Update resources based on action
        action_magnitude = float(jnp.linalg.norm(action))
        self.resource_manager.update_resources(
            computational_usage=action_magnitude * 0.01,
            memory_usage=0.001,  # Small constant memory usage
            attention_usage=0.005,
            exploration_usage=0.002 if reward < 0.1 else 0.0,  # More exploration if low reward
            learning_usage=0.01 if reward > 0.5 else 0.005
        )
        
        # Periodic pattern recognition
        if current_time - self.last_pattern_recognition > 30.0:  # Every 30 seconds
            new_patterns = self.pattern_recognizer.recognize_patterns()
            step_results['patterns_recognized'] = len(new_patterns)
            
            if new_patterns:
                # Generate goals from new patterns
                resource_state = self.resource_manager.get_resource_state()
                new_goals = self.goal_formulator.formulate_goals_from_patterns(
                    patterns=new_patterns,
                    current_observation=observation,
                    resource_state=resource_state
                )
                step_results['goals_generated'] = len(new_goals)
            
            self.last_pattern_recognition = current_time
        
        # Get current planning state
        step_results['active_goals'] = len(self.goal_formulator.active_goals)
        step_results['resource_pressure'] = self.resource_manager.get_resource_pressure()
        
        # Get recommended goal
        recommended_goal = self._get_recommended_goal()
        if recommended_goal:
            step_results['recommended_goal'] = {
                'goal_id': recommended_goal.goal_id,
                'goal_type': recommended_goal.goal_type.value,
                'priority': recommended_goal.priority.name,
                'target_state': recommended_goal.target_state.tolist(),
                'estimated_duration': recommended_goal.estimated_duration
            }
        
        return step_results
    
    def _get_recommended_goal(self) -> Optional[EmergentGoal]:
        """Get the highest priority goal that can be afforded."""
        if not self.goal_formulator.active_goals:
            return None
        
        # Sort goals by priority and resource feasibility
        goals = list(self.goal_formulator.active_goals.values())
        affordable_goals = [
            goal for goal in goals 
            if self.resource_manager.can_afford_goal(goal)
        ]
        
        if not affordable_goals:
            return None
        
        # Return highest priority affordable goal
        affordable_goals.sort(key=lambda g: g.priority.value, reverse=True)
        return affordable_goals[0]
    
    def execute_goal(self, goal_id: str) -> Dict[str, Any]:
        """
        Execute a specific goal.
        
        Args:
            goal_id: ID of the goal to execute
            
        Returns:
            Execution results
        """
        if goal_id not in self.goal_formulator.active_goals:
            return {'error': 'Goal not found'}
        
        goal = self.goal_formulator.active_goals[goal_id]
        
        # Check resource availability
        if not self.resource_manager.can_afford_goal(goal):
            return {'error': 'Insufficient resources'}
        
        # Allocate resources
        if not self.resource_manager.allocate_resources_for_goal(goal):
            return {'error': 'Resource allocation failed'}
        
        # Execute goal action if callback is available
        if self.action_callback:
            try:
                action = self.action_callback(goal)
                return {
                    'goal_id': goal_id,
                    'action': action.tolist(),
                    'resources_allocated': goal.resource_requirements,
                    'status': 'executing'
                }
            except Exception as e:
                return {'error': f'Action execution failed: {str(e)}'}
        
        return {
            'goal_id': goal_id,
            'resources_allocated': goal.resource_requirements,
            'status': 'ready_for_execution'
        }
    
    def evaluate_goal_achievement(
        self,
        goal_id: str,
        current_observation: jnp.ndarray,
        reward: float
    ) -> Dict[str, Any]:
        """
        Evaluate achievement of a specific goal.
        
        Args:
            goal_id: ID of the goal to evaluate
            current_observation: Current observation
            reward: Current reward signal
            
        Returns:
            Goal achievement evaluation
        """
        progress = self.goal_formulator.evaluate_goal_progress(
            goal_id=goal_id,
            current_observation=current_observation,
            current_reward=reward
        )
        
        # Provide feedback if callback is available
        if self.feedback_callback and 'overall_success' in progress:
            self.feedback_callback(goal_id, 1.0 if progress['overall_success'] else 0.0)
        
        return progress
    
    def get_planning_recommendations(self, current_observation: jnp.ndarray) -> Dict[str, Any]:
        """
        Get planning recommendations based on current state.
        
        Args:
            current_observation: Current observation
            
        Returns:
            Planning recommendations
        """
        recommendations = {
            'recommended_goal': None,
            'alternative_goals': [],
            'resource_constraints': [],
            'pattern_insights': {},
            'planning_advice': []
        }
        
        # Get recommended goal
        recommended_goal = self._get_recommended_goal()
        if recommended_goal:
            recommendations['recommended_goal'] = {
                'goal_id': recommended_goal.goal_id,
                'goal_type': recommended_goal.goal_type.value,
                'priority': recommended_goal.priority.name,
                'success_criteria': recommended_goal.success_criteria,
                'estimated_duration': recommended_goal.estimated_duration
            }
        
        # Get alternative goals
        all_goals = list(self.goal_formulator.active_goals.values())
        if recommended_goal:
            alternatives = [g for g in all_goals if g.goal_id != recommended_goal.goal_id]
        else:
            alternatives = all_goals
        
        alternatives.sort(key=lambda g: g.priority.value, reverse=True)
        recommendations['alternative_goals'] = [
            {
                'goal_id': g.goal_id,
                'goal_type': g.goal_type.value,
                'priority': g.priority.name,
                'affordable': self.resource_manager.can_afford_goal(g)
            }
            for g in alternatives[:3]  # Top 3 alternatives
        ]
        
        # Resource constraints
        resource_pressure = self.resource_manager.get_resource_pressure()
        high_pressure_resources = [
            resource for resource, pressure in resource_pressure.items()
            if pressure > 0.7
        ]
        recommendations['resource_constraints'] = high_pressure_resources
        
        # Pattern insights
        pattern_stats = self.pattern_recognizer.get_pattern_statistics()
        recommendations['pattern_insights'] = pattern_stats
        
        # Planning advice
        advice = []
        if not all_goals:
            advice.append("No active goals - consider exploring to discover new patterns")
        elif len(high_pressure_resources) > 2:
            advice.append("High resource pressure - consider resource optimization goals")
        elif recommended_goal and recommended_goal.goal_type == GoalType.EXPLORATION:
            advice.append("Exploration recommended - seek novel experiences")
        elif recommended_goal and recommended_goal.goal_type == GoalType.SKILL_ACQUISITION:
            advice.append("Skill building opportunity identified - focus on competence development")
        
        recommendations['planning_advice'] = advice
        
        return recommendations
    
    def end_planning_episode(self) -> Dict[str, Any]:
        """
        End the current planning episode and return summary.
        
        Returns:
            Episode summary statistics
        """
        if not self.planning_active:
            return {'error': 'No active planning episode'}
        
        self.planning_active = False
        
        # Collect episode statistics
        goal_stats = self.goal_formulator.get_goal_statistics()
        pattern_stats = self.pattern_recognizer.get_pattern_statistics()
        resource_state = self.resource_manager.get_resource_state()
        
        episode_summary = {
            'episode': self.planning_episode,
            'goal_statistics': goal_stats,
            'pattern_statistics': pattern_stats,
            'final_resource_state': {
                'computational_budget': resource_state.computational_budget,
                'memory_usage': resource_state.memory_usage,
                'attention_capacity': resource_state.attention_capacity,
                'energy_level': resource_state.energy_level
            },
            'planning_effectiveness': goal_stats.get('completion_rate', 0.0)
        }
        
        # Store episode history
        self.planning_history.append(episode_summary)
        if goal_stats.get('completion_rate') is not None:
            self.goal_achievement_rates.append(goal_stats['completion_rate'])
        
        return episode_summary
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive planning system statistics."""
        base_stats = {
            'planning_active': self.planning_active,
            'current_episode': self.planning_episode,
            'total_episodes': len(self.planning_history),
            'goal_statistics': self.goal_formulator.get_goal_statistics(),
            'pattern_statistics': self.pattern_recognizer.get_pattern_statistics(),
            'resource_state': self.resource_manager.get_resource_state().__dict__,
            'resource_pressure': self.resource_manager.get_resource_pressure()
        }
        
        # Performance trends
        if self.goal_achievement_rates:
            base_stats['mean_achievement_rate'] = np.mean(self.goal_achievement_rates)
            base_stats['achievement_trend'] = np.polyfit(
                range(len(self.goal_achievement_rates)), 
                self.goal_achievement_rates, 
                1
            )[0] if len(self.goal_achievement_rates) > 1 else 0.0
        
        return base_stats
    
    def reset_system(self) -> None:
        """Reset the entire planning system."""
        self.planning_active = False
        self.planning_episode = 0
        self.last_pattern_recognition = 0.0
        self.last_resource_update = 0.0
        
        # Reset components
        self.pattern_recognizer = PatternRecognizer(
            observation_dim=self.config.observation_dim,
            action_dim=self.config.action_dim,
            pattern_memory_size=self.config.pattern_memory_size,
            min_pattern_frequency=self.config.min_pattern_frequency,
            pattern_similarity_threshold=self.config.pattern_similarity_threshold
        )
        
        self.goal_formulator = GoalFormulator(
            observation_dim=self.config.observation_dim,
            action_dim=self.config.action_dim,
            max_active_goals=self.config.max_active_goals,
            goal_timeout=self.config.goal_timeout
        )
        
        self.resource_manager = ResourceManager()
        
        # Clear history
        self.planning_history.clear()
        self.goal_achievement_rates.clear()


def create_planning_system(observation_dim: int, action_dim: int, **kwargs) -> PlanningSystem:
    """
    Factory function to create a planning system with default configuration.
    
    Args:
        observation_dim: Dimensionality of observations
        action_dim: Dimensionality of actions
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured PlanningSystem instance
    """
    config = PlanningConfig(
        observation_dim=observation_dim,
        action_dim=action_dim,
        **kwargs
    )
    return PlanningSystem(config)
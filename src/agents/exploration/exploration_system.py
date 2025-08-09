"""
Integrated Exploration System

This module provides the main interface for the curiosity-driven exploration system,
integrating novelty detection, curiosity engine, and interest modeling.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
import time

from .novelty_detection import NoveltyDetector, NoveltyScore, PredictionErrorNoveltyDetector, EnsembleNoveltyDetector
from .curiosity_engine import CuriosityEngine, ExplorationGoal, ExplorationStrategy, InterestModel


@dataclass
class ExplorationConfig:
    """Configuration for the exploration system."""
    observation_dim: int
    novelty_threshold: float = 0.3
    exploration_rate: float = 0.1
    max_goals: int = 10
    goal_timeout: float = 100.0
    use_ensemble_detector: bool = False
    num_ensemble_models: int = 3
    interest_decay: float = 0.95
    learning_rate: float = 0.01


class ExplorationSystem:
    """
    Main exploration system that coordinates curiosity-driven exploration.
    
    This system integrates novelty detection, interest modeling, and goal generation
    to drive autonomous exploration behavior in the Godly AI system.
    """
    
    def __init__(self, config: ExplorationConfig):
        self.config = config
        
        # Initialize novelty detector
        if config.use_ensemble_detector:
            self.novelty_detector = EnsembleNoveltyDetector(
                input_dim=config.observation_dim,
                num_models=config.num_ensemble_models,
                learning_rate=config.learning_rate,
                novelty_threshold=config.novelty_threshold
            )
        else:
            self.novelty_detector = PredictionErrorNoveltyDetector(
                input_dim=config.observation_dim,
                learning_rate=config.learning_rate,
                novelty_threshold=config.novelty_threshold
            )
        
        # Initialize curiosity engine
        self.curiosity_engine = CuriosityEngine(
            observation_dim=config.observation_dim,
            novelty_detector=self.novelty_detector,
            novelty_threshold=config.novelty_threshold,
            exploration_rate=config.exploration_rate,
            max_goals=config.max_goals,
            goal_timeout=config.goal_timeout
        )
        
        # Exploration state
        self.current_observation: Optional[jnp.ndarray] = None
        self.exploration_active = False
        self.exploration_episode = 0
        
        # Performance tracking
        self.episode_performances: List[float] = []
        self.exploration_efficiency: List[float] = []
        
        # Callbacks for external integration
        self.action_callback: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
        self.reward_callback: Optional[Callable[[jnp.ndarray, jnp.ndarray], float]] = None
    
    def set_action_callback(self, callback: Callable[[jnp.ndarray], jnp.ndarray]) -> None:
        """Set callback function for taking actions in the environment."""
        self.action_callback = callback
    
    def set_reward_callback(self, callback: Callable[[jnp.ndarray, jnp.ndarray], float]) -> None:
        """Set callback function for computing rewards."""
        self.reward_callback = callback
    
    def start_exploration(self, initial_observation: jnp.ndarray) -> None:
        """
        Start a new exploration episode.
        
        Args:
            initial_observation: Initial observation to start exploration from
        """
        self.current_observation = initial_observation
        self.exploration_active = True
        self.exploration_episode += 1
        
        # Generate initial exploration goals
        initial_goals = self.curiosity_engine.generate_exploration_goals(
            current_observation=initial_observation,
            num_goals=min(3, self.config.max_goals)
        )
        
        print(f"Started exploration episode {self.exploration_episode} with {len(initial_goals)} initial goals")
    
    def step_exploration(self, current_observation: jnp.ndarray, action_taken: Optional[jnp.ndarray] = None) -> Dict[str, Any]:
        """
        Perform one step of exploration.
        
        Args:
            current_observation: Current observation from environment
            action_taken: Action that was taken (optional)
            
        Returns:
            Dictionary containing exploration step results
        """
        if not self.exploration_active:
            return {'error': 'Exploration not active'}
        
        step_results = {
            'novelty_score': 0.0,
            'exploration_goal': None,
            'goal_achieved': False,
            'new_goals_generated': 0,
            'interest_level': 0.0
        }
        
        # Compute novelty score
        novelty_score = self.curiosity_engine.compute_novelty_score(current_observation)
        step_results['novelty_score'] = novelty_score.score
        
        # Update models if we have a previous observation
        if self.current_observation is not None:
            # Compute performance/reward if callback is available
            performance = 0.0
            if self.reward_callback is not None:
                performance = self.reward_callback(self.current_observation, current_observation)
            
            # Update curiosity engine models
            self.curiosity_engine.update_models(
                observation=self.current_observation,
                next_observation=current_observation,
                performance=performance
            )
        
        # Get current exploration target
        current_goal = self.curiosity_engine.get_current_exploration_target()
        if current_goal is not None:
            step_results['exploration_goal'] = {
                'goal_id': current_goal.goal_id,
                'priority': current_goal.priority,
                'strategy': current_goal.strategy.value,
                'expected_novelty': current_goal.expected_novelty
            }
            
            # Check if goal was achieved
            if action_taken is not None:
                performance = self.reward_callback(self.current_observation, current_observation) if self.reward_callback else 0.0
                goal_achieved = self.curiosity_engine.evaluate_goal_achievement(
                    goal=current_goal,
                    actual_observation=current_observation,
                    performance=performance
                )
                step_results['goal_achieved'] = goal_achieved
        
        # Generate new goals if needed
        if len(self.curiosity_engine.active_goals) < self.config.max_goals // 2:
            new_goals = self.curiosity_engine.generate_exploration_goals(
                current_observation=current_observation,
                num_goals=2
            )
            step_results['new_goals_generated'] = len(new_goals)
        
        # Get interest level for current observation
        interest_level = self.curiosity_engine.interest_model.get_interest_level(current_observation)
        step_results['interest_level'] = interest_level
        
        # Update current observation
        self.current_observation = current_observation
        
        return step_results
    
    def get_exploration_action(self, current_observation: jnp.ndarray) -> Optional[jnp.ndarray]:
        """
        Get suggested action for exploration based on current goals.
        
        Args:
            current_observation: Current observation
            
        Returns:
            Suggested action or None if no specific action is recommended
        """
        current_goal = self.curiosity_engine.get_current_exploration_target()
        if current_goal is None:
            return None
        
        # Compute direction towards goal
        direction = current_goal.target_observation - current_observation
        direction_norm = jnp.linalg.norm(direction)
        
        if direction_norm > 1e-6:
            # Normalize and scale by exploration rate
            action = direction / direction_norm * self.config.exploration_rate
            return action
        
        return None
    
    def should_explore(self, current_observation: jnp.ndarray) -> bool:
        """
        Determine if exploration should be prioritized over exploitation.
        
        Args:
            current_observation: Current observation
            
        Returns:
            True if exploration should be prioritized
        """
        if not self.exploration_active:
            return False
        
        # Compute novelty score
        novelty_score = self.curiosity_engine.compute_novelty_score(current_observation)
        
        # Explore if novelty is above threshold
        if novelty_score.score > self.config.novelty_threshold:
            return True
        
        # Explore if we have high-priority goals
        current_goal = self.curiosity_engine.get_current_exploration_target()
        if current_goal is not None and current_goal.priority > 0.7:
            return True
        
        # Explore based on interest level
        interest_level = self.curiosity_engine.interest_model.get_interest_level(current_observation)
        if interest_level > 0.6:
            return True
        
        return False
    
    def end_exploration_episode(self, final_performance: float = 0.0) -> Dict[str, Any]:
        """
        End the current exploration episode and return summary statistics.
        
        Args:
            final_performance: Final performance measure for the episode
            
        Returns:
            Episode summary statistics
        """
        if not self.exploration_active:
            return {'error': 'No active exploration episode'}
        
        self.exploration_active = False
        self.episode_performances.append(final_performance)
        
        # Compute exploration efficiency
        stats = self.curiosity_engine.get_exploration_statistics()
        completed_goals = stats.get('completed_goals', 0)
        total_goals = stats.get('total_goals_generated', 1)
        efficiency = completed_goals / total_goals if total_goals > 0 else 0.0
        self.exploration_efficiency.append(efficiency)
        
        # Adapt interest model regions
        self.curiosity_engine.interest_model.adapt_regions()
        
        episode_summary = {
            'episode': self.exploration_episode,
            'final_performance': final_performance,
            'exploration_efficiency': efficiency,
            'goals_completed': completed_goals,
            'goals_generated': total_goals,
            'mean_novelty': stats.get('recent_mean_novelty', 0.0),
            'novelty_trend': stats.get('recent_novelty_trend', 0.0)
        }
        
        # Add performance trends
        if len(self.episode_performances) > 1:
            episode_summary['performance_trend'] = np.polyfit(
                range(len(self.episode_performances)), 
                self.episode_performances, 
                1
            )[0]
        
        if len(self.exploration_efficiency) > 1:
            episode_summary['efficiency_trend'] = np.polyfit(
                range(len(self.exploration_efficiency)), 
                self.exploration_efficiency, 
                1
            )[0]
        
        return episode_summary
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        base_stats = self.curiosity_engine.get_exploration_statistics()
        
        system_stats = {
            **base_stats,
            'exploration_active': self.exploration_active,
            'current_episode': self.exploration_episode,
            'total_episodes': len(self.episode_performances),
            'config': {
                'observation_dim': self.config.observation_dim,
                'novelty_threshold': self.config.novelty_threshold,
                'exploration_rate': self.config.exploration_rate,
                'max_goals': self.config.max_goals,
                'use_ensemble_detector': self.config.use_ensemble_detector
            }
        }
        
        # Performance statistics
        if self.episode_performances:
            system_stats['mean_episode_performance'] = np.mean(self.episode_performances)
            system_stats['performance_std'] = np.std(self.episode_performances)
            system_stats['best_performance'] = np.max(self.episode_performances)
        
        if self.exploration_efficiency:
            system_stats['mean_exploration_efficiency'] = np.mean(self.exploration_efficiency)
            system_stats['efficiency_std'] = np.std(self.exploration_efficiency)
            system_stats['best_efficiency'] = np.max(self.exploration_efficiency)
        
        return system_stats
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state of the exploration system."""
        return {
            'config': self.config.__dict__,
            'exploration_episode': self.exploration_episode,
            'episode_performances': self.episode_performances,
            'exploration_efficiency': self.exploration_efficiency,
            'novelty_detector_stats': self.novelty_detector.get_statistics(),
            'curiosity_stats': self.curiosity_engine.get_exploration_statistics()
        }
    
    def reset_system(self) -> None:
        """Reset the entire exploration system."""
        self.exploration_active = False
        self.exploration_episode = 0
        self.current_observation = None
        self.episode_performances.clear()
        self.exploration_efficiency.clear()
        
        # Reset curiosity engine
        self.curiosity_engine.reset_exploration_state()
        
        # Reset novelty detector
        if hasattr(self.novelty_detector, 'reset_statistics'):
            self.novelty_detector.reset_statistics()


def create_exploration_system(observation_dim: int, **kwargs) -> ExplorationSystem:
    """
    Factory function to create an exploration system with default configuration.
    
    Args:
        observation_dim: Dimensionality of observations
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ExplorationSystem instance
    """
    config = ExplorationConfig(observation_dim=observation_dim, **kwargs)
    return ExplorationSystem(config)
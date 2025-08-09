"""
Integrated Reward System

This module provides the main interface for the internal reward system,
integrating intrinsic reward generation with learning and other system components.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
import time

from .internal_reward import (
    IntrinsicRewardGenerator,
    RewardPredictor,
    SurpriseDetector,
    RewardSignal,
    RewardType,
    SurpriseEvent
)
from .novelty_detection import NoveltyScore


@dataclass
class RewardSystemConfig:
    """Configuration for the reward system."""
    observation_dim: int
    action_dim: int = 0
    reward_weights: Optional[Dict[RewardType, float]] = None
    learning_rate: float = 0.01
    surprise_threshold: float = 2.0
    reward_integration_window: int = 50
    learning_integration: bool = True


class RewardLearningIntegrator:
    """
    Integrates reward signals with learning processes.
    
    Provides mechanisms to use intrinsic rewards for reinforcement learning,
    policy updates, and experience prioritization.
    """
    
    def __init__(
        self,
        integration_window: int = 50,
        reward_decay: float = 0.95,
        learning_rate_adaptation: bool = True
    ):
        self.integration_window = integration_window
        self.reward_decay = reward_decay
        self.learning_rate_adaptation = learning_rate_adaptation
        
        # Learning integration state
        self.reward_buffer: List[Tuple[float, Dict[RewardType, RewardSignal]]] = []
        self.learning_signals: List[float] = []
        self.policy_updates: List[Dict[str, Any]] = []
        
        # Adaptive learning parameters
        self.base_learning_rate = 0.01
        self.current_learning_rate = 0.01
        self.learning_momentum = 0.0
    
    def integrate_rewards_with_learning(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray],
        reward_signals: Dict[RewardType, RewardSignal],
        external_reward: float,
        learning_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Integrate intrinsic rewards with learning processes.
        
        Args:
            observation: Current observation
            action: Action taken
            reward_signals: Generated intrinsic reward signals
            external_reward: External reward received
            learning_context: Additional learning context
            
        Returns:
            Learning integration results
        """
        if learning_context is None:
            learning_context = {}
        
        current_time = time.time()
        
        # Compute total intrinsic reward
        total_intrinsic_reward = sum(signal.magnitude for signal in reward_signals.values())
        
        # Combine with external reward
        combined_reward = external_reward + total_intrinsic_reward
        
        # Store in buffer
        self.reward_buffer.append((current_time, reward_signals))
        if len(self.reward_buffer) > self.integration_window:
            self.reward_buffer.pop(0)
        
        # Compute learning signal strength
        learning_signal = self._compute_learning_signal(reward_signals, external_reward)
        self.learning_signals.append(learning_signal)
        
        # Adapt learning rate if enabled
        if self.learning_rate_adaptation:
            self._adapt_learning_rate(learning_signal, reward_signals)
        
        # Generate policy update suggestions
        policy_update = self._generate_policy_update(
            observation, action, reward_signals, combined_reward
        )
        
        integration_results = {
            'total_intrinsic_reward': total_intrinsic_reward,
            'combined_reward': combined_reward,
            'learning_signal_strength': learning_signal,
            'adapted_learning_rate': self.current_learning_rate,
            'policy_update': policy_update,
            'reward_composition': {
                reward_type.value: signal.magnitude
                for reward_type, signal in reward_signals.items()
            }
        }
        
        return integration_results
    
    def _compute_learning_signal(
        self,
        reward_signals: Dict[RewardType, RewardSignal],
        external_reward: float
    ) -> float:
        """Compute learning signal strength based on reward composition."""
        # Different reward types contribute differently to learning
        learning_contributions = {
            RewardType.SURPRISE: 0.8,  # High learning from surprise
            RewardType.CURIOSITY: 0.7,  # High learning from curiosity
            RewardType.COMPETENCE: 0.6,  # Moderate learning from competence
            RewardType.PROGRESS: 0.5,  # Moderate learning from progress
            RewardType.NOVELTY: 0.4,  # Lower learning from pure novelty
            RewardType.EXPLORATION: 0.3  # Lower learning from exploration
        }
        
        learning_signal = 0.0
        
        for reward_type, signal in reward_signals.items():
            contribution = learning_contributions.get(reward_type, 0.5)
            learning_signal += signal.magnitude * contribution * signal.confidence
        
        # Add external reward contribution
        learning_signal += abs(external_reward) * 0.9
        
        return min(1.0, learning_signal)  # Cap at 1.0
    
    def _adapt_learning_rate(
        self,
        learning_signal: float,
        reward_signals: Dict[RewardType, RewardSignal]
    ) -> None:
        """Adapt learning rate based on reward signals."""
        # Higher learning rate for high learning signals
        target_learning_rate = self.base_learning_rate * (1.0 + learning_signal)
        
        # Apply momentum
        self.learning_momentum = 0.9 * self.learning_momentum + 0.1 * (target_learning_rate - self.current_learning_rate)
        self.current_learning_rate += self.learning_momentum
        
        # Clamp learning rate
        self.current_learning_rate = max(0.001, min(0.1, self.current_learning_rate))
    
    def _generate_policy_update(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray],
        reward_signals: Dict[RewardType, RewardSignal],
        combined_reward: float
    ) -> Dict[str, Any]:
        """Generate policy update suggestions based on rewards."""
        policy_update = {
            'update_strength': combined_reward,
            'exploration_bias': 0.0,
            'exploitation_bias': 0.0,
            'learning_focus': 'balanced'
        }
        
        # Adjust based on reward composition
        if RewardType.EXPLORATION in reward_signals or RewardType.NOVELTY in reward_signals:
            policy_update['exploration_bias'] = 0.3
            policy_update['learning_focus'] = 'exploration'
        
        if RewardType.COMPETENCE in reward_signals or RewardType.PROGRESS in reward_signals:
            policy_update['exploitation_bias'] = 0.3
            policy_update['learning_focus'] = 'exploitation'
        
        if RewardType.SURPRISE in reward_signals or RewardType.CURIOSITY in reward_signals:
            policy_update['learning_focus'] = 'learning'
        
        return policy_update
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning integration statistics."""
        if not self.learning_signals:
            return {'learning_signals_count': 0}
        
        recent_signals = self.learning_signals[-20:]
        
        return {
            'learning_signals_count': len(self.learning_signals),
            'current_learning_rate': self.current_learning_rate,
            'base_learning_rate': self.base_learning_rate,
            'learning_momentum': self.learning_momentum,
            'recent_avg_learning_signal': np.mean(recent_signals),
            'learning_signal_trend': np.polyfit(range(len(recent_signals)), recent_signals, 1)[0] if len(recent_signals) > 1 else 0.0
        }


class RewardSystem:
    """
    Main reward system that coordinates intrinsic reward generation and learning integration.
    
    This system provides a unified interface for generating intrinsic rewards,
    detecting surprises, and integrating rewards with learning processes.
    """
    
    def __init__(self, config: RewardSystemConfig):
        self.config = config
        
        # Initialize components
        self.reward_generator = IntrinsicRewardGenerator(
            observation_dim=config.observation_dim,
            action_dim=config.action_dim,
            reward_weights=config.reward_weights,
            learning_rate=config.learning_rate
        )
        
        if config.learning_integration:
            self.learning_integrator = RewardLearningIntegrator(
                integration_window=config.reward_integration_window
            )
        else:
            self.learning_integrator = None
        
        # System state
        self.system_active = False
        self.total_rewards_generated = 0
        self.reward_history: List[Tuple[float, float, Dict[RewardType, float]]] = []
        
        # Callbacks for external integration
        self.learning_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.reward_callback: Optional[Callable[[float, Dict[RewardType, float]], None]] = None
    
    def set_learning_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Set callback for learning integration updates."""
        self.learning_callback = callback
    
    def set_reward_callback(self, callback: Callable[[float, Dict[RewardType, float]], None]) -> None:
        """Set callback for reward signal updates."""
        self.reward_callback = callback
    
    def activate_system(self) -> None:
        """Activate the reward system."""
        self.system_active = True
        print("Reward system activated")
    
    def deactivate_system(self) -> None:
        """Deactivate the reward system."""
        self.system_active = False
        print("Reward system deactivated")
    
    def process_experience(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        external_reward: float = 0.0,
        novelty_score: Optional[NoveltyScore] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an experience and generate intrinsic rewards.
        
        Args:
            observation: Current observation
            action: Action taken (optional)
            external_reward: External reward received
            novelty_score: Novelty score from novelty detector
            context: Additional context information
            
        Returns:
            Processing results including rewards and learning signals
        """
        if not self.system_active:
            return {'error': 'Reward system not active'}
        
        current_time = time.time()
        
        # Generate intrinsic rewards
        reward_signals = self.reward_generator.generate_intrinsic_reward(
            observation=observation,
            action=action,
            external_reward=external_reward,
            novelty_score=novelty_score,
            context=context
        )
        
        # Compute total intrinsic reward
        total_intrinsic_reward = self.reward_generator.compute_total_intrinsic_reward(reward_signals)
        
        # Prepare results
        results = {
            'timestamp': current_time,
            'total_intrinsic_reward': total_intrinsic_reward,
            'external_reward': external_reward,
            'combined_reward': external_reward + total_intrinsic_reward,
            'reward_signals': {
                reward_type.value: {
                    'magnitude': signal.magnitude,
                    'confidence': signal.confidence,
                    'metadata': signal.metadata
                }
                for reward_type, signal in reward_signals.items()
            },
            'reward_composition': {
                reward_type.value: signal.magnitude
                for reward_type, signal in reward_signals.items()
            }
        }
        
        # Integrate with learning if enabled
        if self.learning_integrator is not None:
            learning_results = self.learning_integrator.integrate_rewards_with_learning(
                observation=observation,
                action=action,
                reward_signals=reward_signals,
                external_reward=external_reward,
                learning_context=context
            )
            results['learning_integration'] = learning_results
            
            # Call learning callback if set
            if self.learning_callback:
                self.learning_callback(learning_results)
        
        # Update system state
        self.total_rewards_generated += len(reward_signals)
        reward_magnitudes = {reward_type: signal.magnitude for reward_type, signal in reward_signals.items()}
        self.reward_history.append((current_time, total_intrinsic_reward, reward_magnitudes))
        
        # Maintain history size
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-800:]
        
        # Call reward callback if set
        if self.reward_callback:
            self.reward_callback(total_intrinsic_reward, reward_magnitudes)
        
        return results
    
    def get_reward_prediction(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray] = None
    ) -> float:
        """Get predicted reward for given observation and action."""
        return self.reward_generator.reward_predictor.predict_reward(observation, action)
    
    def update_reward_weights(self, new_weights: Dict[RewardType, float]) -> None:
        """Update reward weights for different types."""
        self.reward_generator.update_reward_weights(new_weights)
    
    def get_recent_surprises(self, time_window: float = 60.0) -> List[SurpriseEvent]:
        """Get recent surprise events."""
        return self.reward_generator.surprise_detector.get_recent_surprises(time_window)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward system statistics."""
        base_stats = {
            'system_active': self.system_active,
            'total_rewards_generated': self.total_rewards_generated,
            'reward_generator_stats': self.reward_generator.get_reward_statistics(),
            'config': {
                'observation_dim': self.config.observation_dim,
                'action_dim': self.config.action_dim,
                'learning_rate': self.config.learning_rate,
                'surprise_threshold': self.config.surprise_threshold,
                'learning_integration': self.config.learning_integration
            }
        }
        
        # Add learning integration statistics
        if self.learning_integrator is not None:
            base_stats['learning_integration_stats'] = self.learning_integrator.get_learning_statistics()
        
        # Add reward history statistics
        if self.reward_history:
            recent_rewards = [reward for _, reward, _ in self.reward_history[-50:]]
            base_stats['recent_avg_intrinsic_reward'] = np.mean(recent_rewards)
            base_stats['intrinsic_reward_trend'] = np.polyfit(
                range(len(recent_rewards)), recent_rewards, 1
            )[0] if len(recent_rewards) > 1 else 0.0
            
            # Reward type distribution
            all_reward_types = set()
            for _, _, reward_magnitudes in self.reward_history[-100:]:
                all_reward_types.update(reward_magnitudes.keys())
            
            for reward_type in all_reward_types:
                type_rewards = [
                    reward_magnitudes.get(reward_type, 0.0)
                    for _, _, reward_magnitudes in self.reward_history[-100:]
                ]
                base_stats[f'{reward_type}_frequency'] = sum(1 for r in type_rewards if r > 0) / len(type_rewards)
                base_stats[f'{reward_type}_avg_magnitude'] = np.mean([r for r in type_rewards if r > 0]) if any(type_rewards) else 0.0
        
        return base_stats
    
    def reset_system(self) -> None:
        """Reset the entire reward system."""
        self.system_active = False
        self.total_rewards_generated = 0
        self.reward_history.clear()
        
        # Reset components
        self.reward_generator.reset_reward_history()
        
        if self.learning_integrator is not None:
            self.learning_integrator.reward_buffer.clear()
            self.learning_integrator.learning_signals.clear()
            self.learning_integrator.policy_updates.clear()
    
    def save_system_state(self) -> Dict[str, Any]:
        """Save the current state of the reward system."""
        return {
            'config': self.config.__dict__,
            'system_active': self.system_active,
            'total_rewards_generated': self.total_rewards_generated,
            'reward_weights': self.reward_generator.reward_weights,
            'reward_statistics': self.get_system_statistics()
        }


def create_reward_system(observation_dim: int, action_dim: int = 0, **kwargs) -> RewardSystem:
    """
    Factory function to create a reward system with default configuration.
    
    Args:
        observation_dim: Dimensionality of observations
        action_dim: Dimensionality of actions
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured RewardSystem instance
    """
    config = RewardSystemConfig(
        observation_dim=observation_dim,
        action_dim=action_dim,
        **kwargs
    )
    return RewardSystem(config)
"""
Internal Reward System

This module implements intrinsic reward signal generation, reward prediction,
and surprise detection mechanisms for autonomous motivation.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
import time
from collections import deque

from .novelty_detection import NoveltyScore


class RewardType(Enum):
    """Types of intrinsic rewards."""
    NOVELTY = "novelty"
    COMPETENCE = "competence"
    CURIOSITY = "curiosity"
    SURPRISE = "surprise"
    PROGRESS = "progress"
    EXPLORATION = "exploration"
    LEARNING = "learning"


@dataclass
class RewardSignal:
    """Represents an intrinsic reward signal."""
    reward_type: RewardType
    magnitude: float
    confidence: float
    timestamp: float
    source_observation: jnp.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SurpriseEvent:
    """Represents a surprise detection event."""
    surprise_magnitude: float
    prediction_error: float
    expected_value: float
    actual_value: float
    observation: jnp.ndarray
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


class RewardPredictor:
    """
    Predicts expected rewards based on observations and actions.
    
    Uses a simple neural network to learn reward prediction and detect
    prediction errors for surprise-based rewards.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 0,
        hidden_dim: int = 64,
        learning_rate: float = 0.01,
        prediction_window: int = 100
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.prediction_window = prediction_window
        
        # Initialize prediction network
        key = jax.random.PRNGKey(42)
        input_dim = observation_dim + action_dim
        
        key1, key2, key3 = jax.random.split(key, 3)
        self.w1 = jax.random.normal(key1, (input_dim, hidden_dim)) * 0.1
        self.b1 = jax.random.normal(key2, (hidden_dim,)) * 0.01
        self.w2 = jax.random.normal(key3, (hidden_dim, 1)) * 0.1
        self.b2 = jnp.zeros(1)
        
        # Prediction history
        self.prediction_history: deque = deque(maxlen=prediction_window)
        self.reward_history: deque = deque(maxlen=prediction_window)
        self.prediction_errors: deque = deque(maxlen=prediction_window)
        
        # Running statistics
        self.mean_prediction_error = 0.0
        self.std_prediction_error = 1.0
        self.update_count = 0
    
    def _forward_pass(self, observation: jnp.ndarray, action: Optional[jnp.ndarray] = None) -> float:
        """Forward pass through prediction network."""
        if action is not None:
            input_vec = jnp.concatenate([observation, action])
        else:
            # Pad with zeros if no action provided
            input_vec = jnp.concatenate([observation, jnp.zeros(self.action_dim)])
        
        h = jnp.tanh(jnp.dot(input_vec, self.w1) + self.b1)
        output = jnp.dot(h, self.w2) + self.b2
        return float(output[0])
    
    def predict_reward(self, observation: jnp.ndarray, action: Optional[jnp.ndarray] = None) -> float:
        """Predict expected reward for given observation and action."""
        return self._forward_pass(observation, action)
    
    def update_predictor(
        self,
        observation: jnp.ndarray,
        actual_reward: float,
        action: Optional[jnp.ndarray] = None
    ) -> float:
        """
        Update predictor with actual reward and return prediction error.
        
        Args:
            observation: Input observation
            actual_reward: Actual reward received
            action: Action taken (optional)
            
        Returns:
            Prediction error magnitude
        """
        # Make prediction
        predicted_reward = self.predict_reward(observation, action)
        prediction_error = abs(actual_reward - predicted_reward)
        
        # Store history
        self.prediction_history.append(predicted_reward)
        self.reward_history.append(actual_reward)
        self.prediction_errors.append(prediction_error)
        
        # Update network using gradient descent
        if action is not None:
            input_vec = jnp.concatenate([observation, action])
        else:
            # Pad with zeros if no action provided
            input_vec = jnp.concatenate([observation, jnp.zeros(self.action_dim)])
        
        # Forward pass
        h = jnp.tanh(jnp.dot(input_vec, self.w1) + self.b1)
        prediction = jnp.dot(h, self.w2) + self.b2
        
        # Compute loss and gradients
        loss = (prediction[0] - actual_reward) ** 2
        
        # Backward pass (simplified gradient computation)
        d_output = 2 * (prediction[0] - actual_reward)
        d_w2 = jnp.outer(h, jnp.array([d_output]))
        d_b2 = jnp.array([d_output])
        
        d_h = d_output * self.w2.flatten()
        d_h_pre = d_h * (1 - jnp.tanh(jnp.dot(input_vec, self.w1) + self.b1) ** 2)
        d_w1 = jnp.outer(input_vec, d_h_pre)
        d_b1 = d_h_pre
        
        # Update weights
        self.w1 -= self.learning_rate * d_w1
        self.b1 -= self.learning_rate * d_b1
        self.w2 -= self.learning_rate * d_w2
        self.b2 -= self.learning_rate * d_b2
        
        # Update statistics
        self.update_count += 1
        delta = prediction_error - self.mean_prediction_error
        self.mean_prediction_error += delta / self.update_count
        
        if self.update_count > 1:
            delta2 = prediction_error - self.mean_prediction_error
            variance_update = delta * delta2 / (self.update_count - 1)
            current_variance = self.std_prediction_error ** 2
            new_variance = ((self.update_count - 2) * current_variance + variance_update) / (self.update_count - 1)
            self.std_prediction_error = max(jnp.sqrt(new_variance), 1e-6)
        
        return prediction_error
    
    def get_prediction_statistics(self) -> Dict[str, float]:
        """Get prediction statistics."""
        return {
            'mean_prediction_error': self.mean_prediction_error,
            'std_prediction_error': self.std_prediction_error,
            'update_count': self.update_count,
            'recent_accuracy': 1.0 - np.mean(list(self.prediction_errors)[-10:]) if self.prediction_errors else 0.0
        }


class SurpriseDetector:
    """
    Detects surprise events based on prediction errors and unexpected outcomes.
    
    Surprise is computed as the magnitude of prediction error relative to
    expected prediction accuracy.
    """
    
    def __init__(
        self,
        surprise_threshold: float = 2.0,
        adaptation_rate: float = 0.1,
        memory_size: int = 200
    ):
        self.surprise_threshold = surprise_threshold
        self.adaptation_rate = adaptation_rate
        self.memory_size = memory_size
        
        # Surprise detection state
        self.surprise_events: deque = deque(maxlen=memory_size)
        self.baseline_error = 1.0
        self.surprise_sensitivity = 1.0
        
        # Adaptation parameters
        self.false_positive_count = 0
        self.true_positive_count = 0
    
    def detect_surprise(
        self,
        prediction_error: float,
        expected_error: float,
        observation: jnp.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[SurpriseEvent]:
        """
        Detect surprise based on prediction error.
        
        Args:
            prediction_error: Actual prediction error
            expected_error: Expected prediction error
            observation: Current observation
            context: Additional context information
            
        Returns:
            SurpriseEvent if surprise detected, None otherwise
        """
        if context is None:
            context = {}
        
        # Compute surprise magnitude
        if expected_error > 1e-6:
            surprise_magnitude = (prediction_error - expected_error) / expected_error
        else:
            surprise_magnitude = prediction_error * 10.0  # High surprise if no baseline
        
        # Apply sensitivity adjustment
        adjusted_surprise = surprise_magnitude * self.surprise_sensitivity
        
        # Check if surprise exceeds threshold
        if adjusted_surprise > self.surprise_threshold:
            surprise_event = SurpriseEvent(
                surprise_magnitude=adjusted_surprise,
                prediction_error=prediction_error,
                expected_value=expected_error,
                actual_value=prediction_error,
                observation=observation,
                timestamp=time.time(),
                context=context
            )
            
            self.surprise_events.append(surprise_event)
            self.true_positive_count += 1
            
            # Adapt threshold based on frequency
            self._adapt_sensitivity()
            
            return surprise_event
        
        return None
    
    def _adapt_sensitivity(self) -> None:
        """Adapt surprise sensitivity based on detection frequency."""
        total_detections = self.true_positive_count + self.false_positive_count
        
        if total_detections > 10:
            # If too many surprises, reduce sensitivity
            surprise_rate = self.true_positive_count / total_detections
            if surprise_rate > 0.3:  # More than 30% surprise rate
                self.surprise_sensitivity *= 0.95
            elif surprise_rate < 0.05:  # Less than 5% surprise rate
                self.surprise_sensitivity *= 1.05
            
            # Reset counters periodically
            if total_detections > 100:
                self.true_positive_count = int(self.true_positive_count * 0.8)
                self.false_positive_count = int(self.false_positive_count * 0.8)
    
    def get_recent_surprises(self, time_window: float = 60.0) -> List[SurpriseEvent]:
        """Get surprise events within the specified time window."""
        current_time = time.time()
        recent_surprises = []
        
        for event in self.surprise_events:
            if current_time - event.timestamp <= time_window:
                recent_surprises.append(event)
        
        return recent_surprises
    
    def get_surprise_statistics(self) -> Dict[str, Any]:
        """Get surprise detection statistics."""
        recent_surprises = self.get_recent_surprises(300.0)  # Last 5 minutes
        
        return {
            'total_surprises': len(self.surprise_events),
            'recent_surprises': len(recent_surprises),
            'surprise_rate': len(recent_surprises) / 300.0 if recent_surprises else 0.0,
            'surprise_sensitivity': self.surprise_sensitivity,
            'surprise_threshold': self.surprise_threshold,
            'avg_surprise_magnitude': np.mean([s.surprise_magnitude for s in recent_surprises]) if recent_surprises else 0.0
        }


class IntrinsicRewardGenerator:
    """
    Generates intrinsic reward signals based on various internal motivations.
    
    Combines novelty, competence, curiosity, surprise, and learning progress
    to create comprehensive intrinsic motivation signals.
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 0,
        reward_weights: Optional[Dict[RewardType, float]] = None,
        learning_rate: float = 0.01
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Default reward weights
        if reward_weights is None:
            self.reward_weights = {
                RewardType.NOVELTY: 0.3,
                RewardType.COMPETENCE: 0.2,
                RewardType.CURIOSITY: 0.2,
                RewardType.SURPRISE: 0.15,
                RewardType.PROGRESS: 0.1,
                RewardType.EXPLORATION: 0.05
            }
        else:
            self.reward_weights = reward_weights
        
        # Initialize components
        self.reward_predictor = RewardPredictor(
            observation_dim=observation_dim,
            action_dim=action_dim,
            learning_rate=learning_rate
        )
        
        self.surprise_detector = SurpriseDetector()
        
        # Reward generation state
        self.generated_rewards: List[RewardSignal] = []
        self.competence_history: deque = deque(maxlen=100)
        self.progress_history: deque = deque(maxlen=50)
        
        # Learning progress tracking
        self.skill_levels: Dict[str, float] = {}
        self.learning_curves: Dict[str, List[float]] = {}
    
    def generate_intrinsic_reward(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        external_reward: float = 0.0,
        novelty_score: Optional[NoveltyScore] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[RewardType, RewardSignal]:
        """
        Generate intrinsic reward signals based on current state.
        
        Args:
            observation: Current observation
            action: Action taken (optional)
            external_reward: External reward received
            novelty_score: Novelty score from novelty detector
            context: Additional context information
            
        Returns:
            Dictionary of reward signals by type
        """
        if context is None:
            context = {}
        
        current_time = time.time()
        reward_signals = {}
        
        # 1. Novelty-based reward
        if novelty_score is not None:
            novelty_reward = self._generate_novelty_reward(
                observation, novelty_score, current_time
            )
            reward_signals[RewardType.NOVELTY] = novelty_reward
        
        # 2. Surprise-based reward
        prediction_error = self.reward_predictor.update_predictor(
            observation, external_reward, action
        )
        
        surprise_event = self.surprise_detector.detect_surprise(
            prediction_error=prediction_error,
            expected_error=self.reward_predictor.mean_prediction_error,
            observation=observation,
            context=context
        )
        
        if surprise_event:
            surprise_reward = self._generate_surprise_reward(
                observation, surprise_event, current_time
            )
            reward_signals[RewardType.SURPRISE] = surprise_reward
        
        # 3. Competence-based reward
        competence_reward = self._generate_competence_reward(
            observation, external_reward, prediction_error, current_time
        )
        if competence_reward:
            reward_signals[RewardType.COMPETENCE] = competence_reward
        
        # 4. Progress-based reward
        progress_reward = self._generate_progress_reward(
            observation, external_reward, current_time
        )
        if progress_reward:
            reward_signals[RewardType.PROGRESS] = progress_reward
        
        # 5. Curiosity-based reward (based on information gain)
        curiosity_reward = self._generate_curiosity_reward(
            observation, prediction_error, current_time
        )
        if curiosity_reward:
            reward_signals[RewardType.CURIOSITY] = curiosity_reward
        
        # Store generated rewards
        for reward_signal in reward_signals.values():
            self.generated_rewards.append(reward_signal)
        
        # Maintain reward history size
        if len(self.generated_rewards) > 1000:
            self.generated_rewards = self.generated_rewards[-800:]
        
        return reward_signals
    
    def _generate_novelty_reward(
        self,
        observation: jnp.ndarray,
        novelty_score: NoveltyScore,
        timestamp: float
    ) -> RewardSignal:
        """Generate reward based on novelty."""
        # Higher reward for moderate novelty (not too high, not too low)
        optimal_novelty = 0.6
        novelty_factor = 1.0 - abs(novelty_score.score - optimal_novelty)
        reward_magnitude = max(0.0, novelty_factor * self.reward_weights[RewardType.NOVELTY])
        
        return RewardSignal(
            reward_type=RewardType.NOVELTY,
            magnitude=reward_magnitude,
            confidence=novelty_score.confidence,
            timestamp=timestamp,
            source_observation=observation,
            metadata={
                'novelty_score': novelty_score.score,
                'prediction_error': novelty_score.prediction_error,
                'optimal_novelty': optimal_novelty
            }
        )
    
    def _generate_surprise_reward(
        self,
        observation: jnp.ndarray,
        surprise_event: SurpriseEvent,
        timestamp: float
    ) -> RewardSignal:
        """Generate reward based on surprise."""
        # Moderate surprise is rewarded more than extreme surprise
        normalized_surprise = min(1.0, surprise_event.surprise_magnitude / 5.0)
        reward_magnitude = normalized_surprise * self.reward_weights[RewardType.SURPRISE]
        
        return RewardSignal(
            reward_type=RewardType.SURPRISE,
            magnitude=reward_magnitude,
            confidence=0.8,  # High confidence in surprise detection
            timestamp=timestamp,
            source_observation=observation,
            metadata={
                'surprise_magnitude': surprise_event.surprise_magnitude,
                'prediction_error': surprise_event.prediction_error,
                'expected_error': surprise_event.expected_value
            }
        )
    
    def _generate_competence_reward(
        self,
        observation: jnp.ndarray,
        external_reward: float,
        prediction_error: float,
        timestamp: float
    ) -> Optional[RewardSignal]:
        """Generate reward based on competence development."""
        # Track competence as inverse of prediction error
        current_competence = 1.0 / (1.0 + prediction_error)
        self.competence_history.append(current_competence)
        
        if len(self.competence_history) < 10:
            return None
        
        # Reward improvement in competence
        recent_competence = np.mean(list(self.competence_history)[-5:])
        older_competence = np.mean(list(self.competence_history)[-10:-5])
        
        competence_improvement = recent_competence - older_competence
        
        if competence_improvement > 0.01:  # Significant improvement
            reward_magnitude = competence_improvement * self.reward_weights[RewardType.COMPETENCE] * 10.0
            
            return RewardSignal(
                reward_type=RewardType.COMPETENCE,
                magnitude=reward_magnitude,
                confidence=0.7,
                timestamp=timestamp,
                source_observation=observation,
                metadata={
                    'competence_improvement': competence_improvement,
                    'current_competence': current_competence,
                    'recent_competence': recent_competence
                }
            )
        
        return None
    
    def _generate_progress_reward(
        self,
        observation: jnp.ndarray,
        external_reward: float,
        timestamp: float
    ) -> Optional[RewardSignal]:
        """Generate reward based on learning progress."""
        self.progress_history.append(external_reward)
        
        if len(self.progress_history) < 20:
            return None
        
        # Compute progress as improvement in external rewards
        recent_rewards = list(self.progress_history)[-10:]
        older_rewards = list(self.progress_history)[-20:-10]
        
        recent_avg = np.mean(recent_rewards)
        older_avg = np.mean(older_rewards)
        
        progress = recent_avg - older_avg
        
        if progress > 0.05:  # Significant progress
            reward_magnitude = progress * self.reward_weights[RewardType.PROGRESS] * 5.0
            
            return RewardSignal(
                reward_type=RewardType.PROGRESS,
                magnitude=reward_magnitude,
                confidence=0.6,
                timestamp=timestamp,
                source_observation=observation,
                metadata={
                    'progress': progress,
                    'recent_avg_reward': recent_avg,
                    'older_avg_reward': older_avg
                }
            )
        
        return None
    
    def _generate_curiosity_reward(
        self,
        observation: jnp.ndarray,
        prediction_error: float,
        timestamp: float
    ) -> Optional[RewardSignal]:
        """Generate reward based on curiosity and information gain."""
        # Curiosity reward is based on prediction error (information gain)
        # Moderate prediction error indicates good learning opportunity
        optimal_error = 0.5
        curiosity_factor = 1.0 - abs(prediction_error - optimal_error) / optimal_error
        
        if curiosity_factor > 0.3:  # Sufficient curiosity value
            reward_magnitude = curiosity_factor * self.reward_weights[RewardType.CURIOSITY]
            
            return RewardSignal(
                reward_type=RewardType.CURIOSITY,
                magnitude=reward_magnitude,
                confidence=0.5,
                timestamp=timestamp,
                source_observation=observation,
                metadata={
                    'prediction_error': prediction_error,
                    'optimal_error': optimal_error,
                    'curiosity_factor': curiosity_factor
                }
            )
        
        return None
    
    def compute_total_intrinsic_reward(
        self,
        reward_signals: Dict[RewardType, RewardSignal]
    ) -> float:
        """Compute total intrinsic reward from individual signals."""
        total_reward = 0.0
        
        for reward_type, signal in reward_signals.items():
            weighted_reward = signal.magnitude * self.reward_weights.get(reward_type, 0.0)
            total_reward += weighted_reward
        
        return total_reward
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward generation statistics."""
        if not self.generated_rewards:
            return {'total_rewards_generated': 0}
        
        # Group rewards by type
        rewards_by_type = {}
        for reward in self.generated_rewards[-100:]:  # Last 100 rewards
            if reward.reward_type not in rewards_by_type:
                rewards_by_type[reward.reward_type] = []
            rewards_by_type[reward.reward_type].append(reward.magnitude)
        
        stats = {
            'total_rewards_generated': len(self.generated_rewards),
            'reward_weights': self.reward_weights,
            'predictor_stats': self.reward_predictor.get_prediction_statistics(),
            'surprise_stats': self.surprise_detector.get_surprise_statistics()
        }
        
        # Add statistics for each reward type
        for reward_type, magnitudes in rewards_by_type.items():
            stats[f'{reward_type.value}_count'] = len(magnitudes)
            stats[f'{reward_type.value}_mean'] = np.mean(magnitudes)
            stats[f'{reward_type.value}_std'] = np.std(magnitudes)
        
        return stats
    
    def update_reward_weights(self, new_weights: Dict[RewardType, float]) -> None:
        """Update reward weights for different types."""
        # Normalize weights to sum to 1.0
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.reward_weights = {
                reward_type: weight / total_weight
                for reward_type, weight in new_weights.items()
            }
    
    def reset_reward_history(self) -> None:
        """Reset reward generation history."""
        self.generated_rewards.clear()
        self.competence_history.clear()
        self.progress_history.clear()
        self.skill_levels.clear()
        self.learning_curves.clear()
"""
Novelty Detection System for Curiosity-Driven Exploration

This module implements novelty detection algorithms using prediction error
to drive curiosity-based exploration in the Godly AI system.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class NoveltyScore:
    """Represents a novelty score for an observation."""
    score: float
    prediction_error: float
    confidence: float
    timestamp: float
    observation_hash: str


@dataclass
class PredictionModel:
    """Simple prediction model for novelty detection."""
    weights: jnp.ndarray
    bias: jnp.ndarray
    learning_rate: float
    prediction_history: List[float]


class NoveltyDetector(ABC):
    """Abstract base class for novelty detection algorithms."""
    
    @abstractmethod
    def compute_novelty(self, observation: jnp.ndarray) -> NoveltyScore:
        """Compute novelty score for an observation."""
        pass
    
    @abstractmethod
    def update_model(self, observation: jnp.ndarray, target: jnp.ndarray) -> None:
        """Update the internal prediction model."""
        pass


class PredictionErrorNoveltyDetector(NoveltyDetector):
    """
    Novelty detector based on prediction error.
    
    Uses a simple neural network to predict next observations and
    computes novelty as the prediction error magnitude.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 0.01,
        novelty_threshold: float = 0.5,
        memory_size: int = 1000
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.novelty_threshold = novelty_threshold
        self.memory_size = memory_size
        
        # Initialize prediction network
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        
        self.w1 = jax.random.normal(key1, (input_dim, hidden_dim)) * 0.1
        self.b1 = jax.random.normal(key2, (hidden_dim,)) * 0.01
        self.w2 = jax.random.normal(key3, (hidden_dim, input_dim)) * 0.1
        self.b2 = jnp.zeros(input_dim)
        
        # Observation history for computing baselines
        self.observation_history: List[jnp.ndarray] = []
        self.prediction_errors: List[float] = []
        self.novelty_scores: List[NoveltyScore] = []
        
        # Running statistics
        self.mean_prediction_error = 0.0
        self.std_prediction_error = 1.0
        self.update_count = 0
    
    def _forward_pass(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through prediction network."""
        h = jnp.tanh(jnp.dot(x, self.w1) + self.b1)
        output = jnp.dot(h, self.w2) + self.b2
        return output
    
    def _compute_prediction_error(self, observation: jnp.ndarray, prediction: jnp.ndarray) -> float:
        """Compute prediction error between observation and prediction."""
        error = jnp.mean(jnp.square(observation - prediction))
        return float(error)
    
    def _update_statistics(self, prediction_error: float) -> None:
        """Update running statistics for normalization."""
        self.update_count += 1
        delta = prediction_error - self.mean_prediction_error
        self.mean_prediction_error += delta / self.update_count
        
        if self.update_count > 1:
            delta2 = prediction_error - self.mean_prediction_error
            variance_update = delta * delta2 / (self.update_count - 1)
            current_variance = self.std_prediction_error ** 2
            new_variance = ((self.update_count - 2) * current_variance + variance_update) / (self.update_count - 1)
            self.std_prediction_error = max(jnp.sqrt(new_variance), 1e-6)
    
    def compute_novelty(self, observation: jnp.ndarray) -> NoveltyScore:
        """
        Compute novelty score based on prediction error.
        
        Args:
            observation: Input observation to evaluate for novelty
            
        Returns:
            NoveltyScore containing novelty metrics
        """
        # Make prediction
        prediction = self._forward_pass(observation)
        
        # Compute prediction error
        prediction_error = self._compute_prediction_error(observation, prediction)
        
        # Normalize prediction error to get novelty score
        if self.std_prediction_error > 0:
            normalized_error = (prediction_error - self.mean_prediction_error) / self.std_prediction_error
            novelty_score = float(jnp.tanh(normalized_error))  # Squash to [-1, 1]
        else:
            novelty_score = 1.0  # High novelty if no baseline
        
        # Compute confidence based on prediction consistency
        confidence = 1.0 / (1.0 + prediction_error)
        
        # Create observation hash for tracking
        obs_hash = str(hash(observation.tobytes()))
        
        score = NoveltyScore(
            score=max(0.0, novelty_score),  # Ensure non-negative
            prediction_error=prediction_error,
            confidence=confidence,
            timestamp=float(jnp.array(len(self.novelty_scores))),
            observation_hash=obs_hash
        )
        
        # Store for history
        self.novelty_scores.append(score)
        self.prediction_errors.append(prediction_error)
        
        # Maintain memory size limit
        if len(self.novelty_scores) > self.memory_size:
            self.novelty_scores.pop(0)
            self.prediction_errors.pop(0)
        
        return score
    
    def update_model(self, observation: jnp.ndarray, target: jnp.ndarray) -> None:
        """
        Update prediction model using gradient descent.
        
        Args:
            observation: Input observation
            target: Target prediction (usually next observation)
        """
        # Forward pass
        h = jnp.tanh(jnp.dot(observation, self.w1) + self.b1)
        prediction = jnp.dot(h, self.w2) + self.b2
        
        # Compute loss and gradients
        loss = jnp.mean(jnp.square(prediction - target))
        
        # Backward pass (simplified gradient computation)
        d_output = 2 * (prediction - target) / len(prediction)
        d_w2 = jnp.outer(h, d_output)
        d_b2 = d_output
        
        d_h = jnp.dot(d_output, self.w2.T)
        d_h_pre = d_h * (1 - jnp.tanh(jnp.dot(observation, self.w1) + self.b1) ** 2)
        d_w1 = jnp.outer(observation, d_h_pre)
        d_b1 = d_h_pre
        
        # Update weights
        self.w1 -= self.learning_rate * d_w1
        self.b1 -= self.learning_rate * d_b1
        self.w2 -= self.learning_rate * d_w2
        self.b2 -= self.learning_rate * d_b2
        
        # Update statistics
        prediction_error = float(loss)
        self._update_statistics(prediction_error)
        
        # Store observation
        self.observation_history.append(observation)
        if len(self.observation_history) > self.memory_size:
            self.observation_history.pop(0)
    
    def get_novelty_threshold(self) -> float:
        """Get current novelty threshold."""
        return self.novelty_threshold
    
    def set_novelty_threshold(self, threshold: float) -> None:
        """Set novelty threshold."""
        self.novelty_threshold = max(0.0, min(1.0, threshold))
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current detector statistics."""
        return {
            'mean_prediction_error': self.mean_prediction_error,
            'std_prediction_error': self.std_prediction_error,
            'update_count': self.update_count,
            'novelty_threshold': self.novelty_threshold,
            'recent_novelty_mean': np.mean([s.score for s in self.novelty_scores[-100:]]) if self.novelty_scores else 0.0
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics and history."""
        self.observation_history.clear()
        self.prediction_errors.clear()
        self.novelty_scores.clear()
        self.mean_prediction_error = 0.0
        self.std_prediction_error = 1.0
        self.update_count = 0


class EnsembleNoveltyDetector(NoveltyDetector):
    """
    Ensemble novelty detector combining multiple detection methods.
    
    Uses multiple prediction models and combines their novelty scores
    for more robust novelty detection.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_models: int = 3,
        hidden_dim: int = 64,
        learning_rate: float = 0.01,
        novelty_threshold: float = 0.5
    ):
        self.input_dim = input_dim
        self.num_models = num_models
        self.novelty_threshold = novelty_threshold
        
        # Create ensemble of detectors
        self.detectors = []
        for i in range(num_models):
            detector = PredictionErrorNoveltyDetector(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate * (0.8 + 0.4 * i / num_models),  # Vary learning rates
                novelty_threshold=novelty_threshold
            )
            self.detectors.append(detector)
    
    def compute_novelty(self, observation: jnp.ndarray) -> NoveltyScore:
        """Compute ensemble novelty score."""
        scores = []
        prediction_errors = []
        confidences = []
        
        for detector in self.detectors:
            score = detector.compute_novelty(observation)
            scores.append(score.score)
            prediction_errors.append(score.prediction_error)
            confidences.append(score.confidence)
        
        # Combine scores (weighted average by confidence)
        total_confidence = sum(confidences)
        if total_confidence > 0:
            weighted_score = sum(s * c for s, c in zip(scores, confidences)) / total_confidence
            weighted_error = sum(e * c for e, c in zip(prediction_errors, confidences)) / total_confidence
            ensemble_confidence = np.mean(confidences)
        else:
            weighted_score = np.mean(scores)
            weighted_error = np.mean(prediction_errors)
            ensemble_confidence = 0.0
        
        obs_hash = str(hash(observation.tobytes()))
        
        return NoveltyScore(
            score=weighted_score,
            prediction_error=weighted_error,
            confidence=ensemble_confidence,
            timestamp=float(jnp.array(len(self.detectors[0].novelty_scores))),
            observation_hash=obs_hash
        )
    
    def update_model(self, observation: jnp.ndarray, target: jnp.ndarray) -> None:
        """Update all detectors in the ensemble."""
        for detector in self.detectors:
            detector.update_model(observation, target)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        stats = {}
        for i, detector in enumerate(self.detectors):
            detector_stats = detector.get_statistics()
            for key, value in detector_stats.items():
                stats[f'detector_{i}_{key}'] = value
        
        # Add ensemble-level statistics
        recent_scores = []
        for detector in self.detectors:
            if detector.novelty_scores:
                recent_scores.extend([s.score for s in detector.novelty_scores[-50:]])
        
        stats['ensemble_mean_novelty'] = np.mean(recent_scores) if recent_scores else 0.0
        stats['ensemble_std_novelty'] = np.std(recent_scores) if recent_scores else 0.0
        stats['num_detectors'] = len(self.detectors)
        
        return stats
"""
Competitive learning algorithms for pattern discovery.

This module implements competitive learning mechanisms that allow the system
to discover patterns in data without supervision through winner-take-all dynamics.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, List, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class CompetitiveLearningConfig:
    """Configuration for competitive learning algorithm."""
    n_units: int = 100
    learning_rate: float = 0.01
    decay_rate: float = 0.999
    min_learning_rate: float = 0.001
    neighborhood_radius: float = 2.0
    neighborhood_decay: float = 0.99
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6


class CompetitiveLearning:
    """
    Competitive learning algorithm for unsupervised pattern discovery.
    
    Uses winner-take-all dynamics where competing units learn to represent
    different patterns in the input space through competitive inhibition.
    """
    
    def __init__(self, config: CompetitiveLearningConfig, input_dim: int, key: jax.random.PRNGKey):
        self.config = config
        self.input_dim = input_dim
        self.key = key
        
        # Initialize weight matrix
        self.weights = self._initialize_weights()
        self.learning_rate = config.learning_rate
        self.neighborhood_radius = config.neighborhood_radius
        
        # Track learning statistics
        self.iteration = 0
        self.convergence_history = []
        
    def _initialize_weights(self) -> jnp.ndarray:
        """Initialize competitive unit weights randomly."""
        key, subkey = random.split(self.key)
        self.key = key
        
        # Initialize weights with small random values
        weights = random.normal(subkey, (self.config.n_units, self.input_dim)) * 0.1
        
        # Normalize weights
        weights = weights / jnp.linalg.norm(weights, axis=1, keepdims=True)
        
        return weights
    
    def _find_winner(self, input_pattern: jnp.ndarray) -> int:
        """Find the winning unit (best matching unit) for input pattern."""
        # Compute distances to all units
        distances = jnp.linalg.norm(self.weights - input_pattern, axis=1)
        
        # Return index of minimum distance (winner)
        return jnp.argmin(distances)
    
    def _compute_neighborhood(self, winner_idx: int) -> jnp.ndarray:
        """Compute neighborhood function around winner."""
        # Create unit positions (1D topology for simplicity)
        # Use actual current number of units, not config value
        current_n_units = len(self.weights)
        positions = jnp.arange(current_n_units)
        
        # Compute distances from winner
        distances = jnp.abs(positions - winner_idx)
        
        # Apply Gaussian neighborhood function
        neighborhood = jnp.exp(-(distances ** 2) / (2 * self.neighborhood_radius ** 2))
        
        return neighborhood
    
    def _update_weights(self, input_pattern: jnp.ndarray, winner_idx: int) -> jnp.ndarray:
        """Update weights using competitive learning rule."""
        # Compute neighborhood influence
        neighborhood = self._compute_neighborhood(winner_idx)
        
        # Ensure neighborhood has correct shape for broadcasting
        neighborhood = neighborhood.reshape(-1, 1)  # Shape: (n_units, 1)
        
        # Update weights for all units based on neighborhood
        weight_updates = self.learning_rate * neighborhood * (input_pattern - self.weights)
        new_weights = self.weights + weight_updates
        
        # Normalize weights to prevent unbounded growth
        new_weights = new_weights / jnp.linalg.norm(new_weights, axis=1, keepdims=True)
        
        return new_weights
    
    def train_step(self, input_pattern: jnp.ndarray) -> Dict[str, float]:
        """Perform one training step with input pattern."""
        # Find winner
        winner_idx = self._find_winner(input_pattern)
        
        # Store old weights for convergence tracking
        old_weights = self.weights.copy()
        
        # Update weights
        self.weights = self._update_weights(input_pattern, winner_idx)
        
        # Compute weight change for convergence tracking
        weight_change = jnp.mean(jnp.linalg.norm(self.weights - old_weights, axis=1))
        
        # Update learning parameters
        self.learning_rate = max(
            self.config.min_learning_rate,
            self.learning_rate * self.config.decay_rate
        )
        self.neighborhood_radius = max(
            0.5,
            self.neighborhood_radius * self.config.neighborhood_decay
        )
        
        self.iteration += 1
        self.convergence_history.append(weight_change)
        
        return {
            'winner_idx': winner_idx,
            'weight_change': weight_change,
            'learning_rate': self.learning_rate,
            'neighborhood_radius': self.neighborhood_radius
        }
    
    def train(self, data: jnp.ndarray) -> Dict[str, List[float]]:
        """
        Train competitive learning on dataset.
        
        Args:
            data: Input data of shape (n_samples, input_dim)
            
        Returns:
            Training statistics dictionary
        """
        n_samples = data.shape[0]
        training_stats = {
            'weight_changes': [],
            'winners': [],
            'learning_rates': [],
            'neighborhood_radii': []
        }
        
        # Training loop
        for iteration in range(self.config.max_iterations):
            # Select random sample
            key, subkey = random.split(self.key)
            self.key = key
            sample_idx = random.randint(subkey, (), 0, n_samples)
            input_pattern = data[sample_idx]
            
            # Perform training step
            step_stats = self.train_step(input_pattern)
            
            # Store statistics
            training_stats['weight_changes'].append(step_stats['weight_change'])
            training_stats['winners'].append(step_stats['winner_idx'])
            training_stats['learning_rates'].append(step_stats['learning_rate'])
            training_stats['neighborhood_radii'].append(step_stats['neighborhood_radius'])
            
            # Check convergence
            if step_stats['weight_change'] < self.config.convergence_threshold:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return training_stats
    
    def get_pattern_assignments(self, data: jnp.ndarray) -> jnp.ndarray:
        """Get pattern assignments for data samples."""
        assignments = []
        for sample in data:
            winner = self._find_winner(sample)
            assignments.append(winner)
        return jnp.array(assignments)
    
    def get_learned_patterns(self) -> jnp.ndarray:
        """Get the learned pattern prototypes (weight vectors)."""
        return self.weights
    
    def compute_quantization_error(self, data: jnp.ndarray) -> float:
        """Compute average quantization error on dataset."""
        total_error = 0.0
        for sample in data:
            winner_idx = self._find_winner(sample)
            error = jnp.linalg.norm(sample - self.weights[winner_idx])
            total_error += error
        
        return total_error / len(data)
    
    def get_unit_activation_counts(self, data: jnp.ndarray) -> jnp.ndarray:
        """Get activation counts for each competitive unit."""
        assignments = self.get_pattern_assignments(data)
        activation_counts = jnp.bincount(assignments, length=self.config.n_units)
        return activation_counts


class AdaptiveCompetitiveLearning(CompetitiveLearning):
    """
    Adaptive competitive learning with dynamic unit creation and pruning.
    
    Extends basic competitive learning with mechanisms to add new units
    for novel patterns and remove unused units.
    """
    
    def __init__(self, config: CompetitiveLearningConfig, input_dim: int, key: jax.random.PRNGKey):
        super().__init__(config, input_dim, key)
        
        # Adaptive parameters
        self.activation_threshold = 10  # Minimum activations to keep unit
        self.novelty_threshold = 0.5    # Threshold for creating new unit
        self.max_units = config.n_units * 2  # Maximum number of units
        
        # Track unit statistics
        self.unit_activations = jnp.zeros(config.n_units)
        self.active_units = jnp.ones(config.n_units, dtype=bool)
    
    def _should_create_new_unit(self, input_pattern: jnp.ndarray) -> bool:
        """Determine if a new unit should be created for novel pattern."""
        # Find distance to closest unit
        distances = jnp.linalg.norm(self.weights - input_pattern, axis=1)
        min_distance = jnp.min(distances)
        
        # Create new unit if pattern is sufficiently novel
        return (min_distance > self.novelty_threshold and 
                jnp.sum(self.active_units) < self.max_units)
    
    def _create_new_unit(self, input_pattern: jnp.ndarray) -> None:
        """Create a new competitive unit for novel pattern."""
        # Find first inactive unit or expand if needed
        inactive_indices = jnp.where(~self.active_units)[0]
        
        if len(inactive_indices) > 0:
            # Reactivate an inactive unit
            new_unit_idx = inactive_indices[0]
            self.weights = self.weights.at[new_unit_idx].set(input_pattern)
            self.active_units = self.active_units.at[new_unit_idx].set(True)
            self.unit_activations = self.unit_activations.at[new_unit_idx].set(0)
        else:
            # Expand network (if under max_units)
            if len(self.weights) < self.max_units:
                new_weight = input_pattern.reshape(1, -1)
                self.weights = jnp.concatenate([self.weights, new_weight], axis=0)
                self.active_units = jnp.concatenate([self.active_units, jnp.array([True])])
                self.unit_activations = jnp.concatenate([self.unit_activations, jnp.array([0.0])])
                
                # Update config to reflect new size
                self.config.n_units = len(self.weights)
    
    def _prune_inactive_units(self) -> None:
        """Remove units that haven't been activated enough."""
        # Mark units with low activation as inactive
        low_activation = self.unit_activations < self.activation_threshold
        self.active_units = self.active_units & ~low_activation
        
        # Reset activation counts for pruned units
        self.unit_activations = jnp.where(self.active_units, self.unit_activations, 0.0)
    
    def train_step(self, input_pattern: jnp.ndarray) -> Dict[str, float]:
        """Adaptive training step with unit creation and pruning."""
        # Check if new unit should be created
        if self._should_create_new_unit(input_pattern):
            self._create_new_unit(input_pattern)
        
        # Perform standard competitive learning step
        step_stats = super().train_step(input_pattern)
        
        # Update unit activation counts
        winner_idx = step_stats['winner_idx']
        self.unit_activations = self.unit_activations.at[winner_idx].add(1.0)
        
        # Periodically prune inactive units
        if self.iteration % 100 == 0:
            self._prune_inactive_units()
        
        # Add adaptive statistics
        step_stats['n_active_units'] = jnp.sum(self.active_units)
        step_stats['total_units'] = len(self.weights)
        
        return step_stats
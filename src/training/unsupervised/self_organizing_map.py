"""
Self-Organizing Map (SOM) algorithms for experience clustering.

This module implements Kohonen's Self-Organizing Map for clustering experiences
and creating topological representations of high-dimensional data.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Dict, List, Optional, NamedTuple
import numpy as np
from dataclasses import dataclass


@dataclass
class SOMConfig:
    """Configuration for Self-Organizing Map."""
    map_width: int = 10
    map_height: int = 10
    input_dim: int = 100
    initial_learning_rate: float = 0.1
    final_learning_rate: float = 0.01
    initial_radius: float = 5.0
    final_radius: float = 0.5
    max_iterations: int = 1000
    topology: str = "rectangular"  # "rectangular" or "hexagonal"


class Experience(NamedTuple):
    """Experience data structure for clustering."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: float
    context: Dict[str, jnp.ndarray]
    timestamp: float


class SelfOrganizingMap:
    """
    Self-Organizing Map for experience clustering and topological learning.
    
    Creates a 2D topological map that preserves neighborhood relationships
    in high-dimensional experience space.
    """
    
    def __init__(self, config: SOMConfig, key: jax.random.PRNGKey):
        self.config = config
        self.key = key
        
        # Initialize map weights
        self.weights = self._initialize_weights()
        
        # Create coordinate grid
        self.coordinates = self._create_coordinate_grid()
        
        # Training state
        self.iteration = 0
        self.learning_rate = config.initial_learning_rate
        self.neighborhood_radius = config.initial_radius
        
        # Statistics tracking
        self.quantization_errors = []
        self.topographic_errors = []
        
    def _initialize_weights(self) -> jnp.ndarray:
        """Initialize SOM weights with small random values."""
        key, subkey = random.split(self.key)
        self.key = key
        
        shape = (self.config.map_height, self.config.map_width, self.config.input_dim)
        weights = random.normal(subkey, shape) * 0.1
        
        return weights
    
    def _create_coordinate_grid(self) -> jnp.ndarray:
        """Create coordinate grid for map topology."""
        if self.config.topology == "rectangular":
            x_coords = jnp.arange(self.config.map_width)
            y_coords = jnp.arange(self.config.map_height)
            xx, yy = jnp.meshgrid(x_coords, y_coords)
            coordinates = jnp.stack([yy, xx], axis=-1)
        elif self.config.topology == "hexagonal":
            # Hexagonal topology with offset rows
            coordinates = []
            for y in range(self.config.map_height):
                for x in range(self.config.map_width):
                    if y % 2 == 0:
                        coord = [y, x]
                    else:
                        coord = [y, x + 0.5]
                    coordinates.append(coord)
            coordinates = jnp.array(coordinates).reshape(
                self.config.map_height, self.config.map_width, 2
            )
        else:
            raise ValueError(f"Unknown topology: {self.config.topology}")
        
        return coordinates
    
    def _find_best_matching_unit(self, input_vector: jnp.ndarray) -> Tuple[int, int]:
        """Find the best matching unit (BMU) for input vector."""
        # Compute distances to all units
        distances = jnp.linalg.norm(self.weights - input_vector, axis=2)
        
        # Find coordinates of minimum distance
        flat_idx = jnp.argmin(distances)
        bmu_y, bmu_x = jnp.unravel_index(flat_idx, distances.shape)
        
        return int(bmu_y), int(bmu_x)
    
    def _compute_neighborhood_function(self, bmu_y: int, bmu_x: int) -> jnp.ndarray:
        """Compute neighborhood function around BMU."""
        bmu_coord = self.coordinates[bmu_y, bmu_x]
        
        # Compute distances from BMU to all units
        distances = jnp.linalg.norm(self.coordinates - bmu_coord, axis=2)
        
        # Apply Gaussian neighborhood function
        neighborhood = jnp.exp(-(distances ** 2) / (2 * self.neighborhood_radius ** 2))
        
        return neighborhood
    
    def _update_weights(self, input_vector: jnp.ndarray, bmu_y: int, bmu_x: int) -> jnp.ndarray:
        """Update SOM weights using learning rule."""
        # Compute neighborhood influence
        neighborhood = self._compute_neighborhood_function(bmu_y, bmu_x)
        
        # Update weights
        weight_diff = input_vector - self.weights
        weight_updates = self.learning_rate * neighborhood[:, :, None] * weight_diff
        new_weights = self.weights + weight_updates
        
        return new_weights
    
    def _update_learning_parameters(self) -> None:
        """Update learning rate and neighborhood radius."""
        # Exponential decay
        progress = self.iteration / self.config.max_iterations
        
        self.learning_rate = (
            self.config.final_learning_rate + 
            (self.config.initial_learning_rate - self.config.final_learning_rate) * 
            jnp.exp(-progress * 3)
        )
        
        self.neighborhood_radius = (
            self.config.final_radius + 
            (self.config.initial_radius - self.config.final_radius) * 
            jnp.exp(-progress * 3)
        )
    
    def train_step(self, input_vector: jnp.ndarray) -> Dict[str, float]:
        """Perform one training step."""
        # Find best matching unit
        bmu_y, bmu_x = self._find_best_matching_unit(input_vector)
        
        # Update weights
        old_weights = self.weights.copy()
        self.weights = self._update_weights(input_vector, bmu_y, bmu_x)
        
        # Compute weight change
        weight_change = jnp.mean(jnp.linalg.norm(self.weights - old_weights, axis=2))
        
        # Update learning parameters
        self._update_learning_parameters()
        
        self.iteration += 1
        
        return {
            'bmu_y': bmu_y,
            'bmu_x': bmu_x,
            'weight_change': weight_change,
            'learning_rate': float(self.learning_rate),
            'neighborhood_radius': float(self.neighborhood_radius)
        }
    
    def train(self, data: jnp.ndarray, verbose: bool = False) -> Dict[str, List[float]]:
        """
        Train SOM on dataset.
        
        Args:
            data: Input data of shape (n_samples, input_dim)
            verbose: Whether to print training progress
            
        Returns:
            Training statistics
        """
        n_samples = data.shape[0]
        training_stats = {
            'weight_changes': [],
            'quantization_errors': [],
            'learning_rates': [],
            'neighborhood_radii': []
        }
        
        for iteration in range(self.config.max_iterations):
            # Select random sample
            key, subkey = random.split(self.key)
            self.key = key
            sample_idx = random.randint(subkey, (), 0, n_samples)
            input_vector = data[sample_idx]
            
            # Training step
            step_stats = self.train_step(input_vector)
            
            # Store statistics
            training_stats['weight_changes'].append(step_stats['weight_change'])
            training_stats['learning_rates'].append(step_stats['learning_rate'])
            training_stats['neighborhood_radii'].append(step_stats['neighborhood_radius'])
            
            # Compute quantization error periodically
            if iteration % 100 == 0:
                qe = self.compute_quantization_error(data)
                training_stats['quantization_errors'].append(qe)
                
                if verbose:
                    print(f"Iteration {iteration}: QE = {qe:.4f}, LR = {step_stats['learning_rate']:.4f}")
        
        return training_stats
    
    def compute_quantization_error(self, data: jnp.ndarray) -> float:
        """Compute average quantization error."""
        total_error = 0.0
        for sample in data:
            bmu_y, bmu_x = self._find_best_matching_unit(sample)
            error = jnp.linalg.norm(sample - self.weights[bmu_y, bmu_x])
            total_error += error
        
        return float(total_error / len(data))
    
    def compute_topographic_error(self, data: jnp.ndarray) -> float:
        """Compute topographic error (preservation of topology)."""
        topographic_errors = 0
        
        for sample in data:
            # Find two best matching units
            distances = jnp.linalg.norm(self.weights - sample, axis=2)
            flat_distances = distances.flatten()
            sorted_indices = jnp.argsort(flat_distances)
            
            # Get coordinates of two best units
            bmu1_idx = sorted_indices[0]
            bmu2_idx = sorted_indices[1]
            
            bmu1_y, bmu1_x = jnp.unravel_index(bmu1_idx, distances.shape)
            bmu2_y, bmu2_x = jnp.unravel_index(bmu2_idx, distances.shape)
            
            # Check if they are neighbors
            coord_dist = jnp.linalg.norm(
                self.coordinates[bmu1_y, bmu1_x] - self.coordinates[bmu2_y, bmu2_x]
            )
            
            if coord_dist > 1.5:  # Not neighbors
                topographic_errors += 1
        
        return topographic_errors / len(data)
    
    def get_activation_map(self, data: jnp.ndarray) -> jnp.ndarray:
        """Get activation frequency map for dataset."""
        activation_map = jnp.zeros((self.config.map_height, self.config.map_width))
        
        for sample in data:
            bmu_y, bmu_x = self._find_best_matching_unit(sample)
            activation_map = activation_map.at[bmu_y, bmu_x].add(1)
        
        return activation_map
    
    def cluster_experiences(self, experiences: List[Experience]) -> Dict[Tuple[int, int], List[int]]:
        """
        Cluster experiences based on their feature representations.
        
        Args:
            experiences: List of Experience objects
            
        Returns:
            Dictionary mapping BMU coordinates to experience indices
        """
        clusters = {}
        
        for i, exp in enumerate(experiences):
            # Create feature vector from experience
            feature_vector = self._experience_to_vector(exp)
            
            # Find BMU
            bmu_y, bmu_x = self._find_best_matching_unit(feature_vector)
            
            # Add to cluster
            if (bmu_y, bmu_x) not in clusters:
                clusters[(bmu_y, bmu_x)] = []
            clusters[(bmu_y, bmu_x)].append(i)
        
        return clusters
    
    def _experience_to_vector(self, experience: Experience) -> jnp.ndarray:
        """Convert experience to feature vector for SOM input."""
        # Combine observation, action, and context into single vector
        features = [experience.observation.flatten()]
        
        if experience.action.size > 0:
            features.append(experience.action.flatten())
        
        # Add reward as feature
        features.append(jnp.array([experience.reward]))
        
        # Add context features if available
        for key, value in experience.context.items():
            if isinstance(value, jnp.ndarray):
                features.append(value.flatten())
        
        # Concatenate all features
        feature_vector = jnp.concatenate(features)
        
        # Pad or truncate to match input_dim
        if len(feature_vector) > self.config.input_dim:
            feature_vector = feature_vector[:self.config.input_dim]
        elif len(feature_vector) < self.config.input_dim:
            padding = jnp.zeros(self.config.input_dim - len(feature_vector))
            feature_vector = jnp.concatenate([feature_vector, padding])
        
        return feature_vector
    
    def get_cluster_prototypes(self) -> jnp.ndarray:
        """Get learned cluster prototypes (weight vectors)."""
        return self.weights
    
    def visualize_u_matrix(self) -> jnp.ndarray:
        """Compute U-matrix for visualization of cluster boundaries."""
        u_matrix = jnp.zeros((self.config.map_height, self.config.map_width))
        
        for y in range(self.config.map_height):
            for x in range(self.config.map_width):
                # Get neighbors
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < self.config.map_height and 
                            0 <= nx < self.config.map_width and 
                            (dy != 0 or dx != 0)):
                            neighbors.append(self.weights[ny, nx])
                
                # Compute average distance to neighbors
                if neighbors:
                    neighbors = jnp.stack(neighbors)
                    distances = jnp.linalg.norm(neighbors - self.weights[y, x], axis=1)
                    u_matrix = u_matrix.at[y, x].set(jnp.mean(distances))
        
        return u_matrix


class HierarchicalSOM:
    """
    Hierarchical Self-Organizing Map for multi-scale experience clustering.
    
    Creates multiple SOM layers with different resolutions to capture
    both fine-grained and coarse-grained patterns in experiences.
    """
    
    def __init__(self, configs: List[SOMConfig], key: jax.random.PRNGKey):
        self.configs = configs
        self.key = key
        
        # Create SOM layers
        self.som_layers = []
        for i, config in enumerate(configs):
            key, subkey = random.split(key)
            som = SelfOrganizingMap(config, subkey)
            self.som_layers.append(som)
    
    def train_hierarchical(self, data: jnp.ndarray, verbose: bool = False) -> List[Dict]:
        """Train all SOM layers hierarchically."""
        training_stats = []
        
        for i, som in enumerate(self.som_layers):
            if verbose:
                print(f"Training SOM layer {i+1}/{len(self.som_layers)}")
            
            stats = som.train(data, verbose=verbose)
            training_stats.append(stats)
            
            # Use current layer's prototypes as input for next layer
            if i < len(self.som_layers) - 1:
                # Flatten weight vectors as new data
                data = som.weights.reshape(-1, som.config.input_dim)
        
        return training_stats
    
    def get_hierarchical_clusters(self, experiences: List[Experience]) -> List[Dict]:
        """Get clusters at all hierarchical levels."""
        hierarchical_clusters = []
        
        for som in self.som_layers:
            clusters = som.cluster_experiences(experiences)
            hierarchical_clusters.append(clusters)
        
        return hierarchical_clusters
    
    def get_multi_scale_representation(self, experience: Experience) -> List[Tuple[int, int]]:
        """Get multi-scale representation of experience across all layers."""
        representations = []
        
        for som in self.som_layers:
            feature_vector = som._experience_to_vector(experience)
            bmu_y, bmu_x = som._find_best_matching_unit(feature_vector)
            representations.append((bmu_y, bmu_x))
        
        return representations
"""
Unsupervised feature extraction for multi-modal inputs.

This module implements various unsupervised feature extraction techniques
that can discover meaningful representations from multi-modal sensory data
without requiring labeled examples.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Tuple, Optional, Union, NamedTuple
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction algorithms."""
    n_components: int = 50
    learning_rate: float = 0.01
    batch_size: int = 32
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    regularization: float = 0.01


class MultiModalInput(NamedTuple):
    """Multi-modal input data structure."""
    visual: Optional[jnp.ndarray] = None
    audio: Optional[jnp.ndarray] = None
    text: Optional[jnp.ndarray] = None
    proprioceptive: Optional[jnp.ndarray] = None
    timestamp: float = 0.0


class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    @abstractmethod
    def fit(self, data: jnp.ndarray) -> Dict[str, List[float]]:
        """Fit the feature extractor to data."""
        pass
    
    @abstractmethod
    def transform(self, data: jnp.ndarray) -> jnp.ndarray:
        """Transform data to feature representation."""
        pass
    
    @abstractmethod
    def fit_transform(self, data: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, List[float]]]:
        """Fit and transform data in one step."""
        pass


class SparseAutoencoder(FeatureExtractor):
    """
    Sparse autoencoder for unsupervised feature learning.
    
    Learns sparse representations by adding sparsity constraints
    to the hidden layer activations.
    """
    
    def __init__(self, config: FeatureExtractionConfig, input_dim: int, key: jax.random.PRNGKey):
        self.config = config
        self.input_dim = input_dim
        self.key = key
        
        # Initialize parameters
        self.encoder_weights, self.decoder_weights = self._initialize_weights()
        self.encoder_bias = jnp.zeros(config.n_components)
        self.decoder_bias = jnp.zeros(input_dim)
        
        # Sparsity parameters
        self.sparsity_target = 0.05  # Target activation probability
        self.sparsity_weight = 3.0   # Weight for sparsity penalty
        
        # Training state
        self.iteration = 0
        self.loss_history = []
    
    def _initialize_weights(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Initialize encoder and decoder weights."""
        key1, key2 = random.split(self.key)
        
        # Xavier initialization
        encoder_scale = jnp.sqrt(2.0 / (self.input_dim + self.config.n_components))
        decoder_scale = jnp.sqrt(2.0 / (self.config.n_components + self.input_dim))
        
        encoder_weights = random.normal(key1, (self.input_dim, self.config.n_components)) * encoder_scale
        decoder_weights = random.normal(key2, (self.config.n_components, self.input_dim)) * decoder_scale
        
        return encoder_weights, decoder_weights
    
    def _encode(self, x: jnp.ndarray) -> jnp.ndarray:
        """Encode input to hidden representation."""
        hidden = jnp.dot(x, self.encoder_weights) + self.encoder_bias
        return jax.nn.sigmoid(hidden)
    
    def _decode(self, h: jnp.ndarray) -> jnp.ndarray:
        """Decode hidden representation to reconstruction."""
        reconstruction = jnp.dot(h, self.decoder_weights) + self.decoder_bias
        return jax.nn.sigmoid(reconstruction)
    
    def _compute_loss(self, x: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
        """Compute reconstruction and sparsity loss."""
        # Forward pass
        hidden = self._encode(x)
        reconstruction = self._decode(hidden)
        
        # Reconstruction loss
        reconstruction_loss = jnp.mean((x - reconstruction) ** 2)
        
        # Sparsity loss (KL divergence)
        mean_activation = jnp.mean(hidden, axis=0)
        sparsity_loss = jnp.sum(
            self.sparsity_target * jnp.log(self.sparsity_target / (mean_activation + 1e-8)) +
            (1 - self.sparsity_target) * jnp.log((1 - self.sparsity_target) / (1 - mean_activation + 1e-8))
        )
        
        # Weight decay
        weight_decay = (jnp.sum(self.encoder_weights ** 2) + jnp.sum(self.decoder_weights ** 2))
        
        # Total loss
        total_loss = (reconstruction_loss + 
                     self.sparsity_weight * sparsity_loss + 
                     self.config.regularization * weight_decay)
        
        return total_loss, hidden
    
    def _update_parameters(self, x: jnp.ndarray) -> float:
        """Update parameters using gradient descent."""
        # Compute gradients
        def loss_fn(params):
            encoder_w, decoder_w, encoder_b, decoder_b = params
            
            # Temporarily update parameters
            old_encoder_w, old_decoder_w = self.encoder_weights, self.decoder_weights
            old_encoder_b, old_decoder_b = self.encoder_bias, self.decoder_bias
            
            self.encoder_weights, self.decoder_weights = encoder_w, decoder_w
            self.encoder_bias, self.decoder_bias = encoder_b, decoder_b
            
            loss, _ = self._compute_loss(x)
            
            # Restore parameters
            self.encoder_weights, self.decoder_weights = old_encoder_w, old_decoder_w
            self.encoder_bias, self.decoder_bias = old_encoder_b, old_decoder_b
            
            return loss
        
        params = (self.encoder_weights, self.decoder_weights, self.encoder_bias, self.decoder_bias)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        
        # Update parameters
        encoder_w_grad, decoder_w_grad, encoder_b_grad, decoder_b_grad = grads
        
        self.encoder_weights -= self.config.learning_rate * encoder_w_grad
        self.decoder_weights -= self.config.learning_rate * decoder_w_grad
        self.encoder_bias -= self.config.learning_rate * encoder_b_grad
        self.decoder_bias -= self.config.learning_rate * decoder_b_grad
        
        return float(loss)
    
    def fit(self, data: jnp.ndarray) -> Dict[str, List[float]]:
        """Fit sparse autoencoder to data."""
        n_samples = data.shape[0]
        training_stats = {'losses': [], 'sparsity_levels': []}
        
        for iteration in range(self.config.max_iterations):
            # Sample batch
            key, subkey = random.split(self.key)
            self.key = key
            batch_indices = random.choice(subkey, n_samples, (self.config.batch_size,), replace=False)
            batch = data[batch_indices]
            
            # Update parameters
            loss = self._update_parameters(batch)
            training_stats['losses'].append(loss)
            
            # Compute sparsity level
            if iteration % 100 == 0:
                hidden = self._encode(data)
                sparsity_level = jnp.mean(jnp.mean(hidden, axis=0))
                training_stats['sparsity_levels'].append(float(sparsity_level))
            
            # Check convergence
            if len(training_stats['losses']) > 10:
                recent_losses = training_stats['losses'][-10:]
                if max(recent_losses) - min(recent_losses) < self.config.convergence_threshold:
                    print(f"Converged after {iteration + 1} iterations")
                    break
            
            self.iteration += 1
        
        return training_stats
    
    def transform(self, data: jnp.ndarray) -> jnp.ndarray:
        """Transform data to sparse feature representation."""
        return self._encode(data)
    
    def fit_transform(self, data: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, List[float]]]:
        """Fit and transform data."""
        training_stats = self.fit(data)
        features = self.transform(data)
        return features, training_stats
    
    def reconstruct(self, data: jnp.ndarray) -> jnp.ndarray:
        """Reconstruct data from input."""
        hidden = self._encode(data)
        return self._decode(hidden)


class IndependentComponentAnalysis(FeatureExtractor):
    """
    Independent Component Analysis (ICA) for blind source separation.
    
    Discovers statistically independent components in multi-modal data
    using the FastICA algorithm.
    """
    
    def __init__(self, config: FeatureExtractionConfig, key: jax.random.PRNGKey):
        self.config = config
        self.key = key
        
        # ICA parameters
        self.mixing_matrix = None
        self.unmixing_matrix = None
        self.mean = None
        self.whitening_matrix = None
        
        # Training state
        self.converged = False
        self.iteration = 0
    
    def _whiten_data(self, data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Whiten the data (zero mean, unit covariance)."""
        # Center data
        self.mean = jnp.mean(data, axis=0)
        centered_data = data - self.mean
        
        # Compute covariance matrix
        cov_matrix = jnp.cov(centered_data.T)
        
        # Eigendecomposition
        eigenvals, eigenvecs = jnp.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = jnp.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Select top components
        eigenvals = eigenvals[:self.config.n_components]
        eigenvecs = eigenvecs[:, :self.config.n_components]
        
        # Whitening matrix
        self.whitening_matrix = eigenvecs @ jnp.diag(1.0 / jnp.sqrt(eigenvals + 1e-8))
        
        # Whiten data
        whitened_data = centered_data @ self.whitening_matrix
        
        return whitened_data, self.whitening_matrix
    
    def _g_function(self, x: jnp.ndarray) -> jnp.ndarray:
        """Nonlinear function for ICA (tanh)."""
        return jnp.tanh(x)
    
    def _g_derivative(self, x: jnp.ndarray) -> jnp.ndarray:
        """Derivative of nonlinear function."""
        return 1.0 - jnp.tanh(x) ** 2
    
    def _fastica_one_unit(self, whitened_data: jnp.ndarray, w_init: jnp.ndarray) -> jnp.ndarray:
        """FastICA algorithm for one component."""
        w = w_init / jnp.linalg.norm(w_init)
        
        for _ in range(100):  # Max iterations for one component
            # Compute projection
            projection = whitened_data @ w
            
            # Apply nonlinearity
            g_proj = self._g_function(projection)
            g_prime_proj = self._g_derivative(projection)
            
            # Update rule
            w_new = jnp.mean(whitened_data * g_proj[:, None], axis=0) - jnp.mean(g_prime_proj) * w
            
            # Normalize
            w_new = w_new / jnp.linalg.norm(w_new)
            
            # Check convergence
            if jnp.abs(jnp.abs(jnp.dot(w, w_new)) - 1.0) < 1e-6:
                break
            
            w = w_new
        
        return w
    
    def fit(self, data: jnp.ndarray) -> Dict[str, List[float]]:
        """Fit ICA to data."""
        # Whiten data
        whitened_data, _ = self._whiten_data(data)
        
        # Initialize unmixing matrix
        key, subkey = random.split(self.key)
        self.key = key
        self.unmixing_matrix = random.normal(subkey, (self.config.n_components, self.config.n_components))
        
        training_stats = {'convergence': []}
        
        # Extract components one by one
        for i in range(self.config.n_components):
            # Initialize weight vector
            w_init = self.unmixing_matrix[i]
            
            # Orthogonalize against previous components
            if i > 0:
                for j in range(i):
                    w_init = w_init - jnp.dot(w_init, self.unmixing_matrix[j]) * self.unmixing_matrix[j]
                w_init = w_init / jnp.linalg.norm(w_init)
            
            # Run FastICA for this component
            w = self._fastica_one_unit(whitened_data, w_init)
            self.unmixing_matrix = self.unmixing_matrix.at[i].set(w)
            
            # Track convergence
            training_stats['convergence'].append(float(jnp.linalg.norm(w - w_init)))
        
        # Compute mixing matrix (pseudo-inverse)
        self.mixing_matrix = jnp.linalg.pinv(self.unmixing_matrix)
        self.converged = True
        
        return training_stats
    
    def transform(self, data: jnp.ndarray) -> jnp.ndarray:
        """Transform data to independent components."""
        if not self.converged:
            raise ValueError("ICA must be fitted before transform")
        
        # Center and whiten
        centered_data = data - self.mean
        whitened_data = centered_data @ self.whitening_matrix
        
        # Apply unmixing matrix
        independent_components = whitened_data @ self.unmixing_matrix.T
        
        return independent_components
    
    def fit_transform(self, data: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, List[float]]]:
        """Fit and transform data."""
        training_stats = self.fit(data)
        components = self.transform(data)
        return components, training_stats


class MultiModalFeatureExtractor:
    """
    Multi-modal feature extractor that handles different types of sensory input.
    
    Combines multiple feature extraction techniques to create unified
    representations from heterogeneous multi-modal data.
    """
    
    def __init__(self, config: Dict[str, FeatureExtractionConfig], key: jax.random.PRNGKey):
        self.config = config
        self.key = key
        
        # Feature extractors for each modality
        self.extractors = {}
        self.fitted = {}
        
        # Cross-modal integration parameters
        self.integration_weights = None
        self.unified_dim = None
    
    def add_modality(self, modality: str, extractor_type: str, input_dim: int) -> None:
        """Add a feature extractor for a specific modality."""
        key, subkey = random.split(self.key)
        self.key = key
        
        config = self.config.get(modality, FeatureExtractionConfig())
        
        if extractor_type == "sparse_autoencoder":
            extractor = SparseAutoencoder(config, input_dim, subkey)
        elif extractor_type == "ica":
            extractor = IndependentComponentAnalysis(config, subkey)
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
        
        self.extractors[modality] = extractor
        self.fitted[modality] = False
    
    def fit_modality(self, modality: str, data: jnp.ndarray) -> Dict[str, List[float]]:
        """Fit feature extractor for specific modality."""
        if modality not in self.extractors:
            raise ValueError(f"Modality {modality} not found")
        
        training_stats = self.extractors[modality].fit(data)
        self.fitted[modality] = True
        
        return training_stats
    
    def transform_modality(self, modality: str, data: jnp.ndarray) -> jnp.ndarray:
        """Transform data for specific modality."""
        if modality not in self.extractors:
            raise ValueError(f"Modality {modality} not found")
        
        if not self.fitted[modality]:
            raise ValueError(f"Extractor for {modality} must be fitted first")
        
        return self.extractors[modality].transform(data)
    
    def fit_multimodal(self, multimodal_data: List[MultiModalInput]) -> Dict[str, Dict[str, List[float]]]:
        """Fit all modality extractors on multi-modal data."""
        # Separate data by modality
        modality_data = {}
        
        for sample in multimodal_data:
            for modality in ['visual', 'audio', 'text', 'proprioceptive']:
                data = getattr(sample, modality)
                if data is not None:
                    if modality not in modality_data:
                        modality_data[modality] = []
                    modality_data[modality].append(data)
        
        # Convert to arrays and fit extractors
        training_stats = {}
        for modality, data_list in modality_data.items():
            if modality in self.extractors:
                data_array = jnp.stack(data_list)
                stats = self.fit_modality(modality, data_array)
                training_stats[modality] = stats
        
        # Learn cross-modal integration weights
        self._learn_integration_weights(multimodal_data)
        
        return training_stats
    
    def _learn_integration_weights(self, multimodal_data: List[MultiModalInput]) -> None:
        """Learn weights for integrating features across modalities."""
        # Extract features for each modality
        modality_features = {}
        
        for sample in multimodal_data:
            for modality in ['visual', 'audio', 'text', 'proprioceptive']:
                data = getattr(sample, modality)
                if data is not None and modality in self.extractors and self.fitted[modality]:
                    if modality not in modality_features:
                        modality_features[modality] = []
                    
                    features = self.extractors[modality].transform(data.reshape(1, -1))
                    modality_features[modality].append(features[0])
        
        # Convert to arrays
        for modality in modality_features:
            modality_features[modality] = jnp.stack(modality_features[modality])
        
        # Compute integration weights based on feature variance
        self.integration_weights = {}
        total_variance = 0.0
        
        for modality, features in modality_features.items():
            variance = jnp.var(features, axis=0)
            total_var = jnp.sum(variance)
            self.integration_weights[modality] = total_var
            total_variance += total_var
        
        # Normalize weights
        for modality in self.integration_weights:
            self.integration_weights[modality] /= total_variance
        
        # Set unified dimension
        self.unified_dim = sum(self.config[mod].n_components for mod in self.integration_weights)
    
    def transform_multimodal(self, multimodal_input: MultiModalInput) -> jnp.ndarray:
        """Transform multi-modal input to unified feature representation."""
        unified_features = []
        
        for modality in ['visual', 'audio', 'text', 'proprioceptive']:
            data = getattr(multimodal_input, modality)
            if (data is not None and 
                modality in self.extractors and 
                self.fitted[modality] and 
                modality in self.integration_weights):
                
                features = self.extractors[modality].transform(data.reshape(1, -1))[0]
                weighted_features = features * self.integration_weights[modality]
                unified_features.append(weighted_features)
        
        if not unified_features:
            raise ValueError("No valid modalities found in input")
        
        return jnp.concatenate(unified_features)
    
    def get_modality_importance(self) -> Dict[str, float]:
        """Get importance weights for each modality."""
        return self.integration_weights.copy() if self.integration_weights else {}
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get feature dimensions for each modality."""
        dimensions = {}
        for modality, extractor in self.extractors.items():
            if hasattr(extractor, 'config'):
                dimensions[modality] = extractor.config.n_components
        return dimensions
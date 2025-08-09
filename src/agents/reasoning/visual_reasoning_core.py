"""
Visual Reasoning Core Implementation

This module implements the visual reasoning core with convolutional spike processing
for spatial pattern recognition and visual information processing.
"""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from .base_reasoning_core import (
    BaseReasoningCore, 
    ReasoningCoreParams, 
    ModalityType
)


class VisualReasoningCore(BaseReasoningCore):
    """
    Visual reasoning core specialized for processing visual information.
    
    Features:
    - Convolutional spike processing for spatial patterns
    - Edge detection and feature extraction
    - Hierarchical visual processing
    - Spatial attention mechanisms
    """
    
    def __init__(
        self, 
        params: Optional[ReasoningCoreParams] = None,
        image_height: int = 32,
        image_width: int = 32,
        num_channels: int = 1
    ):
        """Initialize visual reasoning core."""
        if params is None:
            params = ReasoningCoreParams(
                modality=ModalityType.VISUAL,
                core_id="visual_core",
                reservoir_size=800,  # Larger reservoir for visual processing
                input_size=image_height * image_width * num_channels,
                output_size=64,      # Visual feature output
                processing_layers=4,  # Multiple processing layers
                temporal_window=0.05  # 50ms for visual processing
            )
        
        # Visual-specific parameters
        self.image_height = image_height
        self.image_width = image_width
        self.num_channels = num_channels
        
        # Initialize convolutional filters for spike processing
        self._init_conv_filters()
        
        super().__init__(params)
    
    def _init_conv_filters(self):
        """Initialize convolutional filters for visual processing."""
        # Edge detection filters
        self.edge_filters = {
            'horizontal': jnp.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
            'vertical': jnp.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            'diagonal1': jnp.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
            'diagonal2': jnp.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
        }
        
        # Gabor-like filters for texture detection
        self.texture_filters = self._create_gabor_filters()
    
    def _create_gabor_filters(self) -> dict:
        """Create Gabor-like filters for texture detection."""
        filters = {}
        
        # Create filters at different orientations
        for i, theta in enumerate([0, 45, 90, 135]):
            # Simple approximation of Gabor filter
            x, y = jnp.meshgrid(jnp.arange(-1, 2), jnp.arange(-1, 2))
            
            # Rotate coordinates
            theta_rad = jnp.radians(theta)
            x_rot = x * jnp.cos(theta_rad) + y * jnp.sin(theta_rad)
            y_rot = -x * jnp.sin(theta_rad) + y * jnp.cos(theta_rad)
            
            # Create oriented filter
            gabor = jnp.exp(-(x_rot**2 + y_rot**2) / 0.5) * jnp.cos(2 * jnp.pi * x_rot)
            filters[f'gabor_{theta}'] = gabor
        
        return filters
    
    def _get_connectivity_pattern(self) -> float:
        """Get visual-specific connectivity pattern."""
        # Visual processing benefits from moderate connectivity
        # to maintain spatial relationships
        return 0.15
    
    def _get_spectral_radius(self) -> float:
        """Get visual-specific spectral radius."""
        # Slightly higher spectral radius for rich visual dynamics
        return 0.97
    
    def preprocess_input(self, raw_input: jnp.ndarray) -> jnp.ndarray:
        """
        Preprocess visual input with convolutional spike processing.
        
        Args:
            raw_input: Raw visual input [height, width, channels] or flattened
            
        Returns:
            Processed spike-encoded visual features
        """
        # Reshape input if flattened
        if raw_input.ndim == 1:
            visual_input = raw_input.reshape(
                self.image_height, self.image_width, self.num_channels
            )
        else:
            visual_input = raw_input
        
        # Handle single channel case
        if visual_input.ndim == 2:
            visual_input = visual_input[:, :, None]
        
        # Normalize input
        visual_input = (visual_input - jnp.mean(visual_input)) / (jnp.std(visual_input) + 1e-8)
        
        # Apply convolutional filters to extract features
        features = []
        
        # Process each channel
        for c in range(visual_input.shape[2]):
            channel_data = visual_input[:, :, c]
            
            # Apply edge detection filters
            for filter_name, filter_kernel in self.edge_filters.items():
                feature_map = self._apply_conv_filter(channel_data, filter_kernel)
                features.append(feature_map.flatten())
            
            # Apply texture filters
            for filter_name, filter_kernel in self.texture_filters.items():
                feature_map = self._apply_conv_filter(channel_data, filter_kernel)
                features.append(feature_map.flatten())
        
        # Concatenate all features
        all_features = jnp.concatenate(features)
        
        # Convert to spike encoding using threshold-based approach
        spike_encoded = self._encode_as_spikes(all_features)
        
        # Pad or truncate to match input size
        if len(spike_encoded) > self.params.input_size:
            spike_encoded = spike_encoded[:self.params.input_size]
        elif len(spike_encoded) < self.params.input_size:
            padding = jnp.zeros(self.params.input_size - len(spike_encoded))
            spike_encoded = jnp.concatenate([spike_encoded, padding])
        
        return spike_encoded
    
    def _apply_conv_filter(
        self, 
        image: jnp.ndarray, 
        filter_kernel: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply a convolutional filter to an image."""
        # Simple convolution implementation
        kernel_h, kernel_w = filter_kernel.shape
        img_h, img_w = image.shape
        
        # Calculate output dimensions
        out_h = img_h - kernel_h + 1
        out_w = img_w - kernel_w + 1
        
        if out_h <= 0 or out_w <= 0:
            # Return zeros if kernel is too large
            return jnp.zeros((max(1, out_h), max(1, out_w)))
        
        # Apply convolution
        output = jnp.zeros((out_h, out_w))
        
        for i in range(out_h):
            for j in range(out_w):
                patch = image[i:i+kernel_h, j:j+kernel_w]
                output = output.at[i, j].set(jnp.sum(patch * filter_kernel))
        
        return output
    
    def _encode_as_spikes(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Encode features as spike patterns.
        
        Args:
            features: Feature vector to encode
            
        Returns:
            Spike-encoded features
        """
        # Threshold-based spike encoding
        # Positive values above threshold generate spikes
        threshold = jnp.std(features) * 0.5
        spikes = (features > threshold).astype(float)
        
        # Add some temporal dynamics by considering feature magnitude
        spike_probability = jax.nn.sigmoid(features / (threshold + 1e-8))
        
        # Generate spikes based on probability (deterministic for now)
        enhanced_spikes = jnp.where(
            spike_probability > 0.5,
            spike_probability,
            0.0
        )
        
        return enhanced_spikes
    
    def postprocess_output(self, raw_output: jnp.ndarray) -> jnp.ndarray:
        """
        Postprocess LSM output for visual interpretation.
        
        Args:
            raw_output: Raw LSM output
            
        Returns:
            Processed visual features
        """
        # Apply visual-specific postprocessing
        
        # 1. Normalize output
        normalized_output = (raw_output - jnp.mean(raw_output)) / (jnp.std(raw_output) + 1e-8)
        
        # 2. Apply spatial attention mechanism
        attention_weights = jax.nn.softmax(jnp.abs(normalized_output))
        attended_output = normalized_output * attention_weights
        
        # 3. Apply non-linear activation for feature enhancement
        enhanced_output = jnp.tanh(attended_output * 2.0)
        
        # 4. Create hierarchical feature representation
        # Split output into different feature levels
        feature_levels = []
        chunk_size = len(enhanced_output) // 4
        
        for i in range(4):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < 3 else len(enhanced_output)
            level_features = enhanced_output[start_idx:end_idx]
            
            # Apply level-specific processing
            if i == 0:  # Low-level features (edges, textures)
                processed_level = jnp.abs(level_features)  # Edge strength
            elif i == 1:  # Mid-level features (shapes, patterns)
                processed_level = jnp.maximum(level_features, 0)  # ReLU-like
            elif i == 2:  # High-level features (objects, scenes)
                processed_level = jax.nn.sigmoid(level_features)  # Bounded activation
            else:  # Abstract features (concepts, relationships)
                processed_level = jnp.tanh(level_features)  # Bipolar activation
            
            feature_levels.append(processed_level)
        
        # Combine all levels
        final_output = jnp.concatenate(feature_levels)
        
        return final_output
    
    def extract_visual_features(
        self, 
        image: jnp.ndarray,
        feature_type: str = "all"
    ) -> dict:
        """
        Extract specific visual features from an image.
        
        Args:
            image: Input image
            feature_type: Type of features to extract ("edges", "textures", "all")
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Normalize image
        normalized_image = (image - jnp.mean(image)) / (jnp.std(image) + 1e-8)
        
        if feature_type in ["edges", "all"]:
            # Extract edge features
            edge_features = {}
            for filter_name, filter_kernel in self.edge_filters.items():
                edge_map = self._apply_conv_filter(normalized_image, filter_kernel)
                edge_features[filter_name] = edge_map
            features["edges"] = edge_features
        
        if feature_type in ["textures", "all"]:
            # Extract texture features
            texture_features = {}
            for filter_name, filter_kernel in self.texture_filters.items():
                texture_map = self._apply_conv_filter(normalized_image, filter_kernel)
                texture_features[filter_name] = texture_map
            features["textures"] = texture_features
        
        return features
    
    def compute_spatial_attention(
        self, 
        feature_map: jnp.ndarray,
        attention_type: str = "saliency"
    ) -> jnp.ndarray:
        """
        Compute spatial attention over a feature map.
        
        Args:
            feature_map: Input feature map
            attention_type: Type of attention ("saliency", "center", "uniform")
            
        Returns:
            Attention weights
        """
        if attention_type == "saliency":
            # Saliency-based attention (high variance regions)
            # Apply local variance computation
            attention = jnp.abs(feature_map - jnp.mean(feature_map))
            attention = attention / (jnp.sum(attention) + 1e-8)
        
        elif attention_type == "center":
            # Center-biased attention
            h, w = feature_map.shape
            y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
            center_y, center_x = h // 2, w // 2
            
            # Gaussian attention centered on image center
            attention = jnp.exp(-((y - center_y)**2 + (x - center_x)**2) / (2 * (min(h, w) / 4)**2))
            attention = attention / jnp.sum(attention)
        
        else:  # uniform
            # Uniform attention
            attention = jnp.ones_like(feature_map) / feature_map.size
        
        return attention


def create_visual_reasoning_core(
    image_height: int = 32,
    image_width: int = 32,
    num_channels: int = 1,
    core_id: str = "visual_core"
) -> VisualReasoningCore:
    """
    Create a visual reasoning core with specified parameters.
    
    Args:
        image_height: Height of input images
        image_width: Width of input images
        num_channels: Number of image channels
        core_id: Unique identifier for this core
        
    Returns:
        Configured visual reasoning core
    """
    params = ReasoningCoreParams(
        modality=ModalityType.VISUAL,
        core_id=core_id,
        reservoir_size=800,
        input_size=image_height * image_width * num_channels,
        output_size=64,
        processing_layers=4,
        temporal_window=0.05
    )
    
    return VisualReasoningCore(
        params=params,
        image_height=image_height,
        image_width=image_width,
        num_channels=num_channels
    )
"""
Text Reasoning Core Implementation

This module implements the text reasoning core with sequential spike encoding
for natural language processing and textual information analysis.
"""

from typing import Optional, Dict, List, Tuple
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import string

from .base_reasoning_core import (
    BaseReasoningCore, 
    ReasoningCoreParams, 
    ModalityType
)


class TextReasoningCore(BaseReasoningCore):
    """
    Text reasoning core specialized for processing textual information.
    
    Features:
    - Sequential spike encoding for text sequences
    - Character and word-level processing
    - Syntactic pattern recognition
    - Semantic feature extraction
    - Attention mechanisms for important tokens
    """
    
    def __init__(
        self, 
        params: Optional[ReasoningCoreParams] = None,
        vocab_size: int = 256,  # ASCII character set
        max_sequence_length: int = 128,
        embedding_dim: int = 64
    ):
        """Initialize text reasoning core."""
        if params is None:
            params = ReasoningCoreParams(
                modality=ModalityType.TEXT,
                core_id="text_core",
                reservoir_size=700,  # Large reservoir for sequential processing
                input_size=embedding_dim,
                output_size=56,      # Text feature output
                processing_layers=3,  # Sequential processing layers
                temporal_window=0.3   # 300ms for text processing
            )
        
        # Text-specific parameters
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        
        # Initialize text processing components
        self._init_text_processors()
        
        super().__init__(params)
    
    def _init_text_processors(self):
        """Initialize text processing components."""
        # Character vocabulary (ASCII)
        self.char_to_idx = {chr(i): i for i in range(min(256, self.vocab_size))}
        self.idx_to_char = {i: chr(i) for i in range(min(256, self.vocab_size))}
        
        # Simple character embeddings (random initialization)
        key = random.PRNGKey(42)
        self.char_embeddings = random.normal(
            key, (self.vocab_size, self.embedding_dim)
        ) * 0.1
        
        # Syntactic patterns for recognition
        self.syntactic_patterns = {
            'punctuation': set(string.punctuation),
            'digits': set(string.digits),
            'uppercase': set(string.ascii_uppercase),
            'lowercase': set(string.ascii_lowercase),
            'whitespace': set(string.whitespace)
        }
        
        # Common word patterns (simplified)
        self.word_patterns = {
            'short_words': ['a', 'an', 'the', 'is', 'in', 'on', 'at', 'to', 'of'],
            'question_words': ['what', 'when', 'where', 'who', 'why', 'how'],
            'connectives': ['and', 'or', 'but', 'if', 'then', 'because']
        }
    
    def _get_connectivity_pattern(self) -> float:
        """Get text-specific connectivity pattern."""
        # Text processing benefits from moderate connectivity
        # for sequential pattern recognition
        return 0.18
    
    def _get_spectral_radius(self) -> float:
        """Get text-specific spectral radius."""
        # Moderate spectral radius for stable sequential processing
        return 0.95
    
    def preprocess_input(self, raw_input: jnp.ndarray) -> jnp.ndarray:
        """
        Preprocess text input with sequential spike encoding.
        
        Args:
            raw_input: Raw text input (character indices or string)
            
        Returns:
            Processed spike-encoded text features
        """
        # Handle different input formats
        if isinstance(raw_input, str):
            # Convert string to character indices
            char_indices = self._string_to_indices(raw_input)
        elif raw_input.dtype == jnp.int32 or raw_input.dtype == jnp.int64:
            # Already character indices
            char_indices = raw_input.astype(int)
        else:
            # Assume it's already processed features
            return self._normalize_features(raw_input)
        
        # Convert to embeddings
        embeddings = self._indices_to_embeddings(char_indices)
        
        # Extract sequential features
        sequential_features = self._extract_sequential_features(embeddings, char_indices)
        
        # Extract syntactic features
        syntactic_features = self._extract_syntactic_features(char_indices)
        
        # Extract semantic features
        semantic_features = self._extract_semantic_features(embeddings)
        
        # Combine all features
        combined_features = jnp.concatenate([
            sequential_features.flatten(),
            syntactic_features.flatten(),
            semantic_features.flatten()
        ])
        
        # Convert to spike encoding
        spike_encoded = self._encode_as_spikes(combined_features)
        
        # Pad or truncate to match input size
        if len(spike_encoded) > self.params.input_size:
            spike_encoded = spike_encoded[:self.params.input_size]
        elif len(spike_encoded) < self.params.input_size:
            padding = jnp.zeros(self.params.input_size - len(spike_encoded))
            spike_encoded = jnp.concatenate([spike_encoded, padding])
        
        return spike_encoded
    
    def _string_to_indices(self, text: str) -> jnp.ndarray:
        """Convert string to character indices."""
        # Truncate or pad to max sequence length
        text = text[:self.max_sequence_length]
        
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(0)  # Unknown character
        
        # Pad with zeros if necessary
        while len(indices) < self.max_sequence_length:
            indices.append(0)
        
        return jnp.array(indices)
    
    def _indices_to_embeddings(self, indices: jnp.ndarray) -> jnp.ndarray:
        """Convert character indices to embeddings."""
        # Clip indices to valid range
        clipped_indices = jnp.clip(indices, 0, self.vocab_size - 1)
        
        # Look up embeddings
        embeddings = self.char_embeddings[clipped_indices]
        
        return embeddings
    
    def _extract_sequential_features(
        self, 
        embeddings: jnp.ndarray, 
        indices: jnp.ndarray
    ) -> jnp.ndarray:
        """Extract sequential features from text."""
        seq_features = []
        
        # 1. Positional encoding
        seq_length = len(embeddings)
        positions = jnp.arange(seq_length)
        pos_encoding = jnp.sin(positions / 10.0)  # Simple positional encoding
        seq_features.append(pos_encoding)
        
        # 2. Character transition patterns
        if len(indices) > 1:
            transitions = jnp.diff(indices.astype(float))
            # Pad to match sequence length
            transitions = jnp.concatenate([transitions, jnp.array([0.0])])
            seq_features.append(transitions)
        else:
            seq_features.append(jnp.zeros(seq_length))
        
        # 3. Local context features (n-gram like)
        context_features = jnp.zeros(seq_length)
        for i in range(1, min(seq_length, len(indices))):
            # Simple bigram feature
            context_features = context_features.at[i].set(
                float(indices[i-1] * 256 + indices[i]) / (256 * 256)
            )
        seq_features.append(context_features)
        
        # 4. Sequence statistics
        char_frequency = jnp.zeros(seq_length)
        for i, idx in enumerate(indices):
            if i < seq_length:
                # Character frequency in sequence
                freq = jnp.sum(indices == idx) / len(indices)
                char_frequency = char_frequency.at[i].set(freq)
        seq_features.append(char_frequency)
        
        # Stack all sequential features
        sequential_features = jnp.stack(seq_features, axis=1)
        
        return sequential_features
    
    def _extract_syntactic_features(self, indices: jnp.ndarray) -> jnp.ndarray:
        """Extract syntactic features from character indices."""
        syntactic_features = []
        
        # Convert indices back to characters for pattern matching
        chars = [self.idx_to_char.get(int(idx), '\0') for idx in indices]
        
        # Extract pattern features
        for pattern_name, pattern_set in self.syntactic_patterns.items():
            pattern_vector = jnp.array([
                1.0 if char in pattern_set else 0.0 for char in chars
            ])
            syntactic_features.append(pattern_vector)
        
        # Word boundary detection (simplified)
        word_boundaries = jnp.array([
            1.0 if char in self.syntactic_patterns['whitespace'] else 0.0 
            for char in chars
        ])
        syntactic_features.append(word_boundaries)
        
        # Sentence structure features
        sentence_features = jnp.zeros(len(chars))
        for i, char in enumerate(chars):
            if char in '.!?':
                sentence_features = sentence_features.at[i].set(1.0)  # Sentence end
            elif char in ',;:':
                sentence_features = sentence_features.at[i].set(0.5)  # Clause boundary
        syntactic_features.append(sentence_features)
        
        # Stack syntactic features
        syntactic_matrix = jnp.stack(syntactic_features, axis=1)
        
        return syntactic_matrix
    
    def _extract_semantic_features(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        """Extract semantic features from embeddings."""
        semantic_features = []
        
        # 1. Embedding statistics
        embedding_mean = jnp.mean(embeddings, axis=0)
        embedding_std = jnp.std(embeddings, axis=0)
        semantic_features.extend([embedding_mean, embedding_std])
        
        # 2. Embedding dynamics
        if len(embeddings) > 1:
            embedding_diff = jnp.diff(embeddings, axis=0)
            embedding_velocity = jnp.mean(jnp.abs(embedding_diff), axis=0)
            semantic_features.append(embedding_velocity)
        else:
            semantic_features.append(jnp.zeros(self.embedding_dim))
        
        # 3. Attention-like features
        # Compute self-attention weights (simplified)
        attention_weights = jnp.zeros(len(embeddings))
        for i in range(len(embeddings)):
            # Compute similarity with all other positions
            similarities = jnp.dot(embeddings, embeddings[i])
            attention_weights = attention_weights.at[i].set(jnp.mean(similarities))
        
        # Normalize attention weights
        attention_weights = jax.nn.softmax(attention_weights)
        
        # Weighted embedding representation
        attended_embedding = jnp.sum(
            embeddings * attention_weights[:, None], axis=0
        )
        semantic_features.append(attended_embedding)
        
        # Concatenate all semantic features
        semantic_vector = jnp.concatenate(semantic_features)
        
        return semantic_vector
    
    def _encode_as_spikes(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Encode features as spike patterns with sequential dynamics.
        
        Args:
            features: Feature vector to encode
            
        Returns:
            Spike-encoded features
        """
        # Sequential spike encoding for text
        # Use temporal coding with sequence-aware modulation
        
        # Normalize features
        normalized_features = (features - jnp.mean(features)) / (jnp.std(features) + 1e-8)
        
        # Convert to spike rates with sequential bias
        spike_rates = jax.nn.relu(normalized_features)
        
        # Add sequential modulation
        # Earlier features get slightly higher rates (recency bias)
        sequence_weights = jnp.exp(-0.01 * jnp.arange(len(spike_rates)))
        modulated_rates = spike_rates * sequence_weights
        
        # Convert to spike probabilities
        spike_probabilities = jax.nn.sigmoid(modulated_rates * 2.0)
        
        # Generate spikes with temporal structure
        spikes = jnp.where(
            spike_probabilities > 0.4,  # Lower threshold for text
            spike_probabilities,
            0.0
        )
        
        return spikes
    
    def _normalize_features(self, features: jnp.ndarray) -> jnp.ndarray:
        """Normalize pre-processed features."""
        normalized = (features - jnp.mean(features)) / (jnp.std(features) + 1e-8)
        
        # Ensure correct size
        if len(normalized) > self.params.input_size:
            normalized = normalized[:self.params.input_size]
        elif len(normalized) < self.params.input_size:
            padding = jnp.zeros(self.params.input_size - len(normalized))
            normalized = jnp.concatenate([normalized, padding])
        
        return normalized
    
    def postprocess_output(self, raw_output: jnp.ndarray) -> jnp.ndarray:
        """
        Postprocess LSM output for text interpretation.
        
        Args:
            raw_output: Raw LSM output
            
        Returns:
            Processed text features
        """
        # Apply text-specific postprocessing
        
        # 1. Normalize output
        normalized_output = (raw_output - jnp.mean(raw_output)) / (jnp.std(raw_output) + 1e-8)
        
        # 2. Apply sequential attention mechanism
        # Weight features based on their position in the sequence
        position_weights = jnp.exp(-0.05 * jnp.arange(len(normalized_output)))
        position_weights = position_weights / jnp.sum(position_weights)
        attended_output = normalized_output * position_weights
        
        # 3. Apply linguistic feature interpretation
        # Split output into different linguistic levels
        n_levels = 4
        level_size = len(attended_output) // n_levels
        
        processed_levels = []
        for i in range(n_levels):
            start_idx = i * level_size
            end_idx = start_idx + level_size if i < n_levels - 1 else len(attended_output)
            level_data = attended_output[start_idx:end_idx]
            
            # Apply level-specific processing
            if i == 0:  # Character-level features
                processed_level = jax.nn.sigmoid(level_data)  # Bounded activation
            elif i == 1:  # Word-level features
                processed_level = jnp.tanh(level_data)  # Bipolar activation
            elif i == 2:  # Phrase-level features
                processed_level = jax.nn.relu(level_data)  # Rectified activation
            else:  # Sentence-level features
                processed_level = jax.nn.softmax(jnp.abs(level_data))  # Probability-like
            
            processed_levels.append(processed_level)
        
        # Combine processed levels
        final_output = jnp.concatenate(processed_levels)
        
        # 4. Apply semantic smoothing
        # Smooth transitions between semantic features
        if len(final_output) > 5:
            smoothing_kernel = jnp.array([0.1, 0.2, 0.4, 0.2, 0.1])
            smoothed_output = jnp.convolve(final_output, smoothing_kernel, mode='same')
        else:
            smoothed_output = final_output
        
        return smoothed_output
    
    def extract_text_features(
        self, 
        text: str,
        feature_type: str = "all"
    ) -> dict:
        """
        Extract specific text features from input text.
        
        Args:
            text: Input text string
            feature_type: Type of features to extract ("syntactic", "semantic", "all")
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Convert text to indices
        char_indices = self._string_to_indices(text)
        embeddings = self._indices_to_embeddings(char_indices)
        
        if feature_type in ["syntactic", "all"]:
            # Extract syntactic features
            syntactic_features = self._extract_syntactic_features(char_indices)
            features["syntactic"] = syntactic_features
            
            # Word-level analysis
            words = text.lower().split()
            word_features = {
                'word_count': len(words),
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'unique_words': len(set(words)),
                'question_words': sum(1 for word in words if word in self.word_patterns['question_words']),
                'connectives': sum(1 for word in words if word in self.word_patterns['connectives'])
            }
            features["word_features"] = word_features
        
        if feature_type in ["semantic", "all"]:
            # Extract semantic features
            semantic_features = self._extract_semantic_features(embeddings)
            features["semantic"] = semantic_features
            
            # Text statistics
            text_stats = {
                'character_count': len(text),
                'sentence_count': text.count('.') + text.count('!') + text.count('?'),
                'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / len(text) if text else 0,
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
            }
            features["text_stats"] = text_stats
        
        return features
    
    def compute_text_similarity(
        self, 
        text1: str, 
        text2: str,
        method: str = "embedding"
    ) -> float:
        """
        Compute similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            method: Similarity method ("embedding", "character", "word")
            
        Returns:
            Similarity score between 0 and 1
        """
        if method == "embedding":
            # Embedding-based similarity
            indices1 = self._string_to_indices(text1)
            indices2 = self._string_to_indices(text2)
            
            embeddings1 = self._indices_to_embeddings(indices1)
            embeddings2 = self._indices_to_embeddings(indices2)
            
            # Compute mean embeddings
            mean_emb1 = jnp.mean(embeddings1, axis=0)
            mean_emb2 = jnp.mean(embeddings2, axis=0)
            
            # Cosine similarity
            similarity = jnp.dot(mean_emb1, mean_emb2) / (
                jnp.linalg.norm(mean_emb1) * jnp.linalg.norm(mean_emb2) + 1e-8
            )
            
        elif method == "character":
            # Character-level similarity
            set1 = set(text1.lower())
            set2 = set(text2.lower())
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            similarity = intersection / union if union > 0 else 0.0
            
        else:  # word
            # Word-level similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            similarity = intersection / union if union > 0 else 0.0
        
        return float(jnp.clip(similarity, 0.0, 1.0))


def create_text_reasoning_core(
    vocab_size: int = 256,
    max_sequence_length: int = 128,
    embedding_dim: int = 64,
    core_id: str = "text_core"
) -> TextReasoningCore:
    """
    Create a text reasoning core with specified parameters.
    
    Args:
        vocab_size: Size of character vocabulary
        max_sequence_length: Maximum sequence length to process
        embedding_dim: Dimension of character embeddings
        core_id: Unique identifier for this core
        
    Returns:
        Configured text reasoning core
    """
    params = ReasoningCoreParams(
        modality=ModalityType.TEXT,
        core_id=core_id,
        reservoir_size=700,
        input_size=embedding_dim,
        output_size=56,
        processing_layers=3,
        temporal_window=0.3
    )
    
    return TextReasoningCore(
        params=params,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        embedding_dim=embedding_dim
    )
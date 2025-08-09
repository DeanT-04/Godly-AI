"""
Audio Reasoning Core Implementation

This module implements the audio reasoning core with temporal pattern recognition
for auditory information processing and sound analysis.
"""

from typing import Optional, Tuple, List
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from .base_reasoning_core import (
    BaseReasoningCore, 
    ReasoningCoreParams, 
    ModalityType
)


class AudioReasoningCore(BaseReasoningCore):
    """
    Audio reasoning core specialized for processing auditory information.
    
    Features:
    - Temporal pattern recognition for audio sequences
    - Frequency domain analysis
    - Spectral feature extraction
    - Temporal attention mechanisms
    - Rhythm and beat detection
    """
    
    def __init__(
        self, 
        params: Optional[ReasoningCoreParams] = None,
        sample_rate: int = 16000,
        window_size: int = 512,
        hop_length: int = 256,
        n_mels: int = 64
    ):
        """Initialize audio reasoning core."""
        if params is None:
            params = ReasoningCoreParams(
                modality=ModalityType.AUDIO,
                core_id="audio_core",
                reservoir_size=600,  # Medium reservoir for temporal processing
                input_size=n_mels,   # Mel-spectrogram features
                output_size=48,      # Audio feature output
                processing_layers=3,  # Temporal processing layers
                temporal_window=0.2   # 200ms for audio processing
            )
        
        # Audio-specific parameters
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Initialize audio processing components
        self._init_audio_filters()
        
        super().__init__(params)
    
    def _init_audio_filters(self):
        """Initialize audio processing filters."""
        # Mel filter bank for spectral analysis
        self.mel_filters = self._create_mel_filterbank()
        
        # Temporal filters for pattern detection
        self.temporal_filters = {
            'onset': jnp.array([1, -1]),  # Simple onset detection
            'rhythm': jnp.array([1, 0, -1, 0, 1]),  # Rhythm pattern
            'decay': jnp.array([1, 0.7, 0.5, 0.3, 0.1])  # Decay pattern
        }
        
        # Frequency bands for analysis
        self.freq_bands = {
            'low': (0, self.n_mels // 4),
            'mid_low': (self.n_mels // 4, self.n_mels // 2),
            'mid_high': (self.n_mels // 2, 3 * self.n_mels // 4),
            'high': (3 * self.n_mels // 4, self.n_mels)
        }
    
    def _create_mel_filterbank(self) -> jnp.ndarray:
        """Create mel-scale filter bank."""
        # Simplified mel filter bank creation
        # In practice, this would use proper mel-scale computation
        
        n_fft = self.window_size // 2 + 1
        mel_filters = jnp.zeros((self.n_mels, n_fft))
        
        # Create triangular filters
        mel_points = jnp.linspace(0, n_fft - 1, self.n_mels + 2)
        
        for i in range(self.n_mels):
            left = int(mel_points[i])
            center = int(mel_points[i + 1])
            right = int(mel_points[i + 2])
            
            # Left slope
            for j in range(left, center):
                if center > left:
                    mel_filters = mel_filters.at[i, j].set((j - left) / (center - left))
            
            # Right slope
            for j in range(center, right):
                if right > center:
                    mel_filters = mel_filters.at[i, j].set((right - j) / (right - center))
        
        return mel_filters
    
    def _get_connectivity_pattern(self) -> float:
        """Get audio-specific connectivity pattern."""
        # Audio processing benefits from higher connectivity
        # for temporal pattern recognition
        return 0.2
    
    def _get_spectral_radius(self) -> float:
        """Get audio-specific spectral radius."""
        # Higher spectral radius for rich temporal dynamics
        return 0.98
    
    def preprocess_input(self, raw_input: jnp.ndarray) -> jnp.ndarray:
        """
        Preprocess audio input with temporal pattern recognition.
        
        Args:
            raw_input: Raw audio input (time series or spectrogram)
            
        Returns:
            Processed spike-encoded audio features
        """
        # Handle different input formats
        if raw_input.ndim == 1:
            # Time series audio - convert to spectrogram
            spectrogram = self._compute_spectrogram(raw_input)
        else:
            # Already a spectrogram or feature matrix
            spectrogram = raw_input
        
        # Extract mel-scale features
        mel_features = self._extract_mel_features(spectrogram)
        
        # Apply temporal filtering for pattern detection
        temporal_features = self._apply_temporal_filters(mel_features)
        
        # Extract frequency band features
        band_features = self._extract_frequency_band_features(mel_features)
        
        # Combine all features
        combined_features = jnp.concatenate([
            mel_features.flatten(),
            temporal_features.flatten(),
            band_features.flatten()
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
    
    def _compute_spectrogram(self, audio: jnp.ndarray) -> jnp.ndarray:
        """Compute spectrogram from time series audio."""
        # Simplified spectrogram computation
        # In practice, this would use FFT-based methods
        
        n_frames = (len(audio) - self.window_size) // self.hop_length + 1
        n_freqs = self.window_size // 2 + 1
        
        spectrogram = jnp.zeros((n_frames, n_freqs))
        
        # Simple windowed analysis
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.window_size
            
            if end <= len(audio):
                window = audio[start:end]
                # Apply Hanning window
                hann_window = 0.5 * (1 - jnp.cos(2 * jnp.pi * jnp.arange(self.window_size) / self.window_size))
                windowed = window * hann_window
                
                # Compute magnitude spectrum (simplified)
                # In practice, would use FFT
                spectrum = jnp.abs(jnp.fft.fft(windowed))[:n_freqs]
                spectrogram = spectrogram.at[i, :].set(spectrum)
        
        return spectrogram
    
    def _extract_mel_features(self, spectrogram: jnp.ndarray) -> jnp.ndarray:
        """Extract mel-scale features from spectrogram."""
        # Apply mel filter bank
        if spectrogram.shape[1] == self.mel_filters.shape[1]:
            mel_features = jnp.dot(spectrogram, self.mel_filters.T)
        else:
            # Adapt to different spectrogram sizes
            mel_features = spectrogram[:, :self.n_mels] if spectrogram.shape[1] >= self.n_mels else spectrogram
        
        # Apply log compression
        mel_features = jnp.log(mel_features + 1e-8)
        
        # Normalize
        mel_features = (mel_features - jnp.mean(mel_features)) / (jnp.std(mel_features) + 1e-8)
        
        return mel_features
    
    def _apply_temporal_filters(self, features: jnp.ndarray) -> jnp.ndarray:
        """Apply temporal filters for pattern detection."""
        temporal_outputs = []
        
        # Apply each temporal filter
        for filter_name, filter_kernel in self.temporal_filters.items():
            filtered = self._convolve_temporal(features, filter_kernel)
            temporal_outputs.append(filtered)
        
        # Ensure all outputs have the same shape by padding
        if temporal_outputs:
            max_time_steps = max(output.shape[0] for output in temporal_outputs)
            max_features = max(output.shape[1] for output in temporal_outputs)
            
            padded_outputs = []
            for output in temporal_outputs:
                # Pad time dimension
                time_pad = max_time_steps - output.shape[0]
                feature_pad = max_features - output.shape[1]
                
                if time_pad > 0 or feature_pad > 0:
                    padded = jnp.pad(output, ((0, time_pad), (0, feature_pad)), mode='constant')
                else:
                    padded = output
                padded_outputs.append(padded)
            
            # Stack temporal features
            temporal_features = jnp.stack(padded_outputs, axis=-1)
        else:
            # Fallback if no temporal outputs
            temporal_features = jnp.zeros((features.shape[0], features.shape[1], 1))
        
        return temporal_features
    
    def _convolve_temporal(
        self, 
        features: jnp.ndarray, 
        kernel: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply temporal convolution."""
        # Simple 1D convolution along time axis
        n_frames, n_features = features.shape
        kernel_size = len(kernel)
        
        if n_frames < kernel_size:
            return jnp.zeros_like(features)
        
        output = jnp.zeros((n_frames - kernel_size + 1, n_features))
        
        for i in range(n_frames - kernel_size + 1):
            window = features[i:i+kernel_size, :]
            # Apply kernel to each frequency bin
            for j in range(n_features):
                output = output.at[i, j].set(jnp.dot(window[:, j], kernel))
        
        return output
    
    def _extract_frequency_band_features(self, mel_features: jnp.ndarray) -> jnp.ndarray:
        """Extract features from different frequency bands."""
        band_features = []
        
        for band_name, (start, end) in self.freq_bands.items():
            band_data = mel_features[:, start:end]
            
            # Compute band statistics
            band_mean = jnp.mean(band_data, axis=1)
            band_std = jnp.std(band_data, axis=1)
            band_max = jnp.max(band_data, axis=1)
            
            # Combine band features
            band_feature = jnp.stack([band_mean, band_std, band_max], axis=1)
            band_features.append(band_feature)
        
        # Concatenate all band features
        all_band_features = jnp.concatenate(band_features, axis=1)
        
        return all_band_features
    
    def _encode_as_spikes(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Encode features as spike patterns with temporal dynamics.
        
        Args:
            features: Feature vector to encode
            
        Returns:
            Spike-encoded features
        """
        # Temporal spike encoding for audio
        # Use rate coding with temporal modulation
        
        # Normalize features
        normalized_features = (features - jnp.mean(features)) / (jnp.std(features) + 1e-8)
        
        # Convert to spike rates (positive values)
        spike_rates = jax.nn.relu(normalized_features)
        
        # Add temporal modulation based on feature dynamics
        # Higher rates for more dynamic features
        feature_dynamics = jnp.abs(jnp.gradient(spike_rates))
        modulated_rates = spike_rates * (1 + 0.5 * feature_dynamics)
        
        # Convert rates to spike probabilities
        spike_probabilities = jax.nn.sigmoid(modulated_rates)
        
        # Generate deterministic spikes based on probabilities
        spikes = jnp.where(spike_probabilities > 0.5, spike_probabilities, 0.0)
        
        return spikes
    
    def postprocess_output(self, raw_output: jnp.ndarray) -> jnp.ndarray:
        """
        Postprocess LSM output for audio interpretation.
        
        Args:
            raw_output: Raw LSM output
            
        Returns:
            Processed audio features
        """
        # Apply audio-specific postprocessing
        
        # 1. Normalize output
        normalized_output = (raw_output - jnp.mean(raw_output)) / (jnp.std(raw_output) + 1e-8)
        
        # 2. Apply temporal attention mechanism
        # Weight recent information more heavily
        temporal_weights = jnp.exp(-0.1 * jnp.arange(len(normalized_output)))
        temporal_weights = temporal_weights / jnp.sum(temporal_weights)
        attended_output = normalized_output * temporal_weights
        
        # 3. Apply frequency-domain interpretation
        # Split output into frequency-like bands
        n_bands = 4
        band_size = len(attended_output) // n_bands
        
        processed_bands = []
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = start_idx + band_size if i < n_bands - 1 else len(attended_output)
            band_data = attended_output[start_idx:end_idx]
            
            # Apply band-specific processing
            if i == 0:  # Low frequency band
                processed_band = jnp.tanh(band_data)  # Smooth activation
            elif i == 1:  # Mid-low frequency band
                processed_band = jax.nn.relu(band_data)  # Rectified activation
            elif i == 2:  # Mid-high frequency band
                processed_band = jax.nn.sigmoid(band_data)  # Bounded activation
            else:  # High frequency band
                processed_band = jnp.abs(band_data)  # Magnitude activation
            
            processed_bands.append(processed_band)
        
        # Combine processed bands
        final_output = jnp.concatenate(processed_bands)
        
        # 4. Apply temporal smoothing
        # Simple moving average for stability
        if len(final_output) > 3:
            smoothed_output = jnp.convolve(
                final_output, 
                jnp.array([0.25, 0.5, 0.25]), 
                mode='same'
            )
        else:
            smoothed_output = final_output
        
        return smoothed_output
    
    def detect_onset(self, audio_features: jnp.ndarray) -> jnp.ndarray:
        """
        Detect onset events in audio features.
        
        Args:
            audio_features: Time series of audio features
            
        Returns:
            Onset detection function
        """
        # Compute spectral flux for onset detection
        if audio_features.ndim == 1:
            # Simple difference-based onset detection
            onset_strength = jnp.maximum(jnp.diff(audio_features), 0)
        else:
            # Multi-dimensional features
            spectral_flux = jnp.sum(jnp.maximum(jnp.diff(audio_features, axis=0), 0), axis=1)
            onset_strength = spectral_flux
        
        # Apply smoothing
        if len(onset_strength) > 3:
            onset_strength = jnp.convolve(
                onset_strength, 
                jnp.array([0.1, 0.8, 0.1]), 
                mode='same'
            )
        
        return onset_strength
    
    def extract_rhythm_features(
        self, 
        audio_features: jnp.ndarray,
        tempo_range: Tuple[int, int] = (60, 200)
    ) -> dict:
        """
        Extract rhythm and tempo features.
        
        Args:
            audio_features: Time series of audio features
            tempo_range: Range of tempos to analyze (BPM)
            
        Returns:
            Dictionary of rhythm features
        """
        rhythm_features = {}
        
        # Detect onsets
        onset_strength = self.detect_onset(audio_features)
        rhythm_features['onset_strength'] = onset_strength
        
        # Estimate tempo using autocorrelation
        if len(onset_strength) > 10:
            # Simple autocorrelation-based tempo estimation
            autocorr = jnp.correlate(onset_strength, onset_strength, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks in autocorrelation
            peak_indices = []
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    peak_indices.append(i)
            
            if peak_indices:
                # Convert to tempo (simplified)
                dominant_period = peak_indices[0] if peak_indices else 10
                estimated_tempo = 60.0 / (dominant_period * 0.01)  # Assuming 10ms hop
                rhythm_features['estimated_tempo'] = float(jnp.clip(
                    estimated_tempo, tempo_range[0], tempo_range[1]
                ))
            else:
                rhythm_features['estimated_tempo'] = 120.0  # Default tempo
        else:
            rhythm_features['estimated_tempo'] = 120.0
        
        # Compute rhythmic regularity
        if len(onset_strength) > 1:
            rhythm_features['rhythmic_regularity'] = float(
                1.0 / (jnp.std(jnp.diff(onset_strength)) + 1e-8)
            )
        else:
            rhythm_features['rhythmic_regularity'] = 0.0
        
        return rhythm_features


def create_audio_reasoning_core(
    sample_rate: int = 16000,
    n_mels: int = 64,
    core_id: str = "audio_core"
) -> AudioReasoningCore:
    """
    Create an audio reasoning core with specified parameters.
    
    Args:
        sample_rate: Audio sample rate
        n_mels: Number of mel-scale features
        core_id: Unique identifier for this core
        
    Returns:
        Configured audio reasoning core
    """
    params = ReasoningCoreParams(
        modality=ModalityType.AUDIO,
        core_id=core_id,
        reservoir_size=600,
        input_size=n_mels,
        output_size=48,
        processing_layers=3,
        temporal_window=0.2
    )
    
    return AudioReasoningCore(
        params=params,
        sample_rate=sample_rate,
        n_mels=n_mels
    )
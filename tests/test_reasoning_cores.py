"""
Test suite for multi-modal reasoning cores.

This module tests the specialized reservoir modules for different cognitive domains
including visual, audio, text, and motor reasoning capabilities.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from src.agents.reasoning import (
    BaseReasoningCore,
    ReasoningCoreParams,
    ReasoningCoreState,
    ModalityType,
    VisualReasoningCore,
    AudioReasoningCore,
    TextReasoningCore,
    MotorReasoningCore,
    create_visual_reasoning_core,
    create_audio_reasoning_core,
    create_text_reasoning_core,
    create_motor_reasoning_core
)


class TestBaseReasoningCore:
    """Test the base reasoning core functionality."""
    
    def test_base_reasoning_core_initialization(self):
        """Test base reasoning core initialization."""
        # Create a concrete implementation for testing
        class TestReasoningCore(BaseReasoningCore):
            def _get_connectivity_pattern(self):
                return 0.1
            
            def _get_spectral_radius(self):
                return 0.9
            
            def preprocess_input(self, raw_input):
                return raw_input
            
            def postprocess_output(self, raw_output):
                return raw_output
        
        params = ReasoningCoreParams(
            modality=ModalityType.VISUAL,
            core_id="test_core",
            reservoir_size=100,
            input_size=50,
            output_size=25
        )
        
        core = TestReasoningCore(params)
        
        assert core.params.modality == ModalityType.VISUAL
        assert core.params.core_id == "test_core"
        assert core.params.reservoir_size == 100
        assert core.params.input_size == 50
        assert core.params.output_size == 25
        assert len(core.peer_cores) == 0
        assert len(core.performance_history) == 0
    
    def test_state_initialization(self):
        """Test reasoning core state initialization."""
        class TestReasoningCore(BaseReasoningCore):
            def _get_connectivity_pattern(self):
                return 0.1
            
            def _get_spectral_radius(self):
                return 0.9
            
            def preprocess_input(self, raw_input):
                return raw_input
            
            def postprocess_output(self, raw_output):
                return raw_output
        
        params = ReasoningCoreParams(
            modality=ModalityType.VISUAL,
            core_id="test_core"
        )
        
        core = TestReasoningCore(params)
        key = random.PRNGKey(42)
        state = core.init_state(key)
        
        assert isinstance(state, ReasoningCoreState)
        assert state.activity_level == 0.0
        assert state.resource_allocation == params.base_resource_allocation
        assert len(state.sync_signals) == 0
        assert len(state.processing_history) == 0
    
    def test_peer_registration(self):
        """Test peer core registration."""
        class TestReasoningCore(BaseReasoningCore):
            def _get_connectivity_pattern(self):
                return 0.1
            
            def _get_spectral_radius(self):
                return 0.9
            
            def preprocess_input(self, raw_input):
                return raw_input
            
            def postprocess_output(self, raw_output):
                return raw_output
        
        core1 = TestReasoningCore(ReasoningCoreParams(
            modality=ModalityType.VISUAL, core_id="core1"
        ))
        core2 = TestReasoningCore(ReasoningCoreParams(
            modality=ModalityType.AUDIO, core_id="core2"
        ))
        
        core1.register_peer("core2", core2)
        
        assert "core2" in core1.peer_cores
        assert core1.peer_cores["core2"] == core2
    
    def test_performance_tracking(self):
        """Test performance history tracking."""
        class TestReasoningCore(BaseReasoningCore):
            def _get_connectivity_pattern(self):
                return 0.1
            
            def _get_spectral_radius(self):
                return 0.9
            
            def preprocess_input(self, raw_input):
                return raw_input
            
            def postprocess_output(self, raw_output):
                return raw_output
        
        core = TestReasoningCore(ReasoningCoreParams(
            modality=ModalityType.VISUAL, core_id="test_core"
        ))
        
        # Add performance scores
        scores = [0.8, 0.9, 0.7, 0.95, 0.85]
        for score in scores:
            core.update_performance(score)
        
        assert len(core.performance_history) == 5
        assert core.performance_history == scores


class TestVisualReasoningCore:
    """Test the visual reasoning core."""
    
    def test_visual_core_initialization(self):
        """Test visual reasoning core initialization."""
        core = create_visual_reasoning_core(
            image_height=32,
            image_width=32,
            num_channels=1,
            core_id="visual_test"
        )
        
        assert core.params.modality == ModalityType.VISUAL
        assert core.params.core_id == "visual_test"
        assert core.image_height == 32
        assert core.image_width == 32
        assert core.num_channels == 1
        assert len(core.edge_filters) == 4
        assert len(core.texture_filters) == 4
    
    def test_visual_preprocessing(self):
        """Test visual input preprocessing."""
        core = create_visual_reasoning_core(32, 32, 1)
        
        # Test with random image
        key = random.PRNGKey(42)
        image = random.normal(key, (32, 32))
        
        processed = core.preprocess_input(image)
        
        assert processed.shape == (core.params.input_size,)
        assert jnp.all(jnp.isfinite(processed))
        assert jnp.any(processed > 0)  # Should have some spikes
    
    def test_visual_postprocessing(self):
        """Test visual output postprocessing."""
        core = create_visual_reasoning_core(32, 32, 1)
        
        # Test with random LSM output
        key = random.PRNGKey(42)
        lsm_output = random.normal(key, (core.params.output_size,))
        
        processed = core.postprocess_output(lsm_output)
        
        assert processed.shape == lsm_output.shape
        assert jnp.all(jnp.isfinite(processed))
    
    def test_visual_feature_extraction(self):
        """Test visual feature extraction."""
        core = create_visual_reasoning_core(16, 16, 1)
        
        # Create test image with edges
        image = jnp.zeros((16, 16))
        image = image.at[8, :].set(1.0)  # Horizontal edge
        image = image.at[:, 8].set(1.0)  # Vertical edge
        
        features = core.extract_visual_features(image, "edges")
        
        assert "edges" in features
        assert "horizontal" in features["edges"]
        assert "vertical" in features["edges"]
        
        # Check that edges are detected
        h_edge = features["edges"]["horizontal"]
        v_edge = features["edges"]["vertical"]
        
        assert jnp.max(jnp.abs(h_edge)) > 0
        assert jnp.max(jnp.abs(v_edge)) > 0
    
    def test_spatial_attention(self):
        """Test spatial attention computation."""
        core = create_visual_reasoning_core(16, 16, 1)
        
        # Create feature map with salient region
        feature_map = jnp.zeros((16, 16))
        feature_map = feature_map.at[6:10, 6:10].set(1.0)  # Salient square
        
        attention = core.compute_spatial_attention(feature_map, "saliency")
        
        assert attention.shape == feature_map.shape
        assert jnp.sum(attention) > 0
        assert jnp.all(attention >= 0)
        
        # Check that attention is higher in salient region
        salient_attention = jnp.mean(attention[6:10, 6:10])
        background_attention = jnp.mean(attention[:6, :6])
        assert salient_attention > background_attention
    
    def test_visual_processing_pipeline(self):
        """Test complete visual processing pipeline."""
        core = create_visual_reasoning_core(16, 16, 1)
        key = random.PRNGKey(42)
        
        # Initialize state
        state = core.init_state(key)
        
        # Create test image
        image = random.normal(key, (16, 16))
        
        # Process through pipeline
        output, new_state = core.process_input(state, image, 0.01, 0.0, key)
        
        assert output.shape == (core.params.output_size,)
        assert isinstance(new_state, ReasoningCoreState)
        assert new_state.activity_level >= 0
        assert len(new_state.processing_history) == 1


class TestAudioReasoningCore:
    """Test the audio reasoning core."""
    
    def test_audio_core_initialization(self):
        """Test audio reasoning core initialization."""
        core = create_audio_reasoning_core(
            sample_rate=16000,
            n_mels=64,
            core_id="audio_test"
        )
        
        assert core.params.modality == ModalityType.AUDIO
        assert core.params.core_id == "audio_test"
        assert core.sample_rate == 16000
        assert core.n_mels == 64
        assert len(core.temporal_filters) == 3
        assert len(core.freq_bands) == 4
    
    def test_audio_preprocessing(self):
        """Test audio input preprocessing."""
        core = create_audio_reasoning_core(16000, 64)
        
        # Test with random audio signal
        key = random.PRNGKey(42)
        audio_signal = random.normal(key, (1024,))
        
        processed = core.preprocess_input(audio_signal)
        
        assert processed.shape == (core.params.input_size,)
        assert jnp.all(jnp.isfinite(processed))
        assert jnp.any(processed > 0)  # Should have some spikes
    
    def test_audio_postprocessing(self):
        """Test audio output postprocessing."""
        core = create_audio_reasoning_core(16000, 64)
        
        # Test with random LSM output
        key = random.PRNGKey(42)
        lsm_output = random.normal(key, (core.params.output_size,))
        
        processed = core.postprocess_output(lsm_output)
        
        assert processed.shape == lsm_output.shape
        assert jnp.all(jnp.isfinite(processed))
    
    def test_onset_detection(self):
        """Test audio onset detection."""
        core = create_audio_reasoning_core(16000, 64)
        
        # Create audio features with clear onset
        features = jnp.zeros(100)
        features = features.at[50:55].set(1.0)  # Sharp onset
        
        onset_strength = core.detect_onset(features)
        
        assert len(onset_strength) == len(features) - 1
        assert jnp.max(onset_strength) > 0
        
        # Check that onset is detected around position 50
        onset_peak = jnp.argmax(onset_strength)
        assert 48 <= onset_peak <= 52
    
    def test_rhythm_feature_extraction(self):
        """Test rhythm feature extraction."""
        core = create_audio_reasoning_core(16000, 64)
        
        # Create periodic audio features
        t = jnp.arange(200)
        periodic_features = jnp.sin(2 * jnp.pi * t / 20)  # Period of 20 samples
        
        rhythm_features = core.extract_rhythm_features(periodic_features)
        
        assert "onset_strength" in rhythm_features
        assert "estimated_tempo" in rhythm_features
        assert "rhythmic_regularity" in rhythm_features
        
        assert rhythm_features["estimated_tempo"] > 0
        assert rhythm_features["rhythmic_regularity"] >= 0
    
    def test_audio_processing_pipeline(self):
        """Test complete audio processing pipeline."""
        core = create_audio_reasoning_core(16000, 64)
        key = random.PRNGKey(42)
        
        # Initialize state
        state = core.init_state(key)
        
        # Create test spectrogram
        spectrogram = random.normal(key, (10, 64))
        
        # Process through pipeline
        output, new_state = core.process_input(state, spectrogram, 0.01, 0.0, key)
        
        assert output.shape == (core.params.output_size,)
        assert isinstance(new_state, ReasoningCoreState)
        assert new_state.activity_level >= 0
        assert len(new_state.processing_history) == 1


class TestTextReasoningCore:
    """Test the text reasoning core."""
    
    def test_text_core_initialization(self):
        """Test text reasoning core initialization."""
        core = create_text_reasoning_core(
            vocab_size=256,
            max_sequence_length=128,
            embedding_dim=64,
            core_id="text_test"
        )
        
        assert core.params.modality == ModalityType.TEXT
        assert core.params.core_id == "text_test"
        assert core.vocab_size == 256
        assert core.max_sequence_length == 128
        assert core.embedding_dim == 64
        assert len(core.char_to_idx) > 0
        assert len(core.syntactic_patterns) == 5
    
    def test_text_preprocessing(self):
        """Test text input preprocessing."""
        core = create_text_reasoning_core(256, 64, 32)
        
        # Test with string input
        text = "Hello, world! This is a test."
        processed = core.preprocess_input(text)
        
        assert processed.shape == (core.params.input_size,)
        assert jnp.all(jnp.isfinite(processed))
        assert jnp.any(processed > 0)  # Should have some spikes
    
    def test_text_postprocessing(self):
        """Test text output postprocessing."""
        core = create_text_reasoning_core(256, 64, 32)
        
        # Test with random LSM output
        key = random.PRNGKey(42)
        lsm_output = random.normal(key, (core.params.output_size,))
        
        processed = core.postprocess_output(lsm_output)
        
        assert processed.shape == lsm_output.shape
        assert jnp.all(jnp.isfinite(processed))
    
    def test_text_feature_extraction(self):
        """Test text feature extraction."""
        core = create_text_reasoning_core(256, 64, 32)
        
        text = "Hello, world! How are you?"
        features = core.extract_text_features(text, "all")
        
        assert "syntactic" in features
        assert "semantic" in features
        assert "word_features" in features
        assert "text_stats" in features
        
        word_features = features["word_features"]
        assert word_features["word_count"] > 0
        assert word_features["unique_words"] > 0
        
        text_stats = features["text_stats"]
        assert text_stats["character_count"] == len(text)
        assert text_stats["sentence_count"] >= 1
    
    def test_text_similarity(self):
        """Test text similarity computation."""
        core = create_text_reasoning_core(256, 64, 32)
        
        text1 = "Hello world"
        text2 = "Hello world"
        text3 = "Goodbye moon"
        
        # Identical texts should have high similarity
        sim_identical = core.compute_text_similarity(text1, text2, "character")
        assert sim_identical > 0.8
        
        # Different texts should have lower similarity
        sim_different = core.compute_text_similarity(text1, text3, "character")
        assert sim_different < sim_identical
    
    def test_text_processing_pipeline(self):
        """Test complete text processing pipeline."""
        core = create_text_reasoning_core(256, 64, 32)
        key = random.PRNGKey(42)
        
        # Initialize state
        state = core.init_state(key)
        
        # Create test text
        text = "This is a test sentence."
        
        # Process through pipeline
        output, new_state = core.process_input(state, text, 0.01, 0.0, key)
        
        assert output.shape == (core.params.output_size,)
        assert isinstance(new_state, ReasoningCoreState)
        assert new_state.activity_level >= 0
        assert len(new_state.processing_history) == 1


class TestMotorReasoningCore:
    """Test the motor reasoning core."""
    
    def test_motor_core_initialization(self):
        """Test motor reasoning core initialization."""
        core = create_motor_reasoning_core(
            action_dim=6,
            control_frequency=100.0,
            core_id="motor_test"
        )
        
        assert core.params.modality == ModalityType.MOTOR
        assert core.params.core_id == "motor_test"
        assert core.action_dim == 6
        assert core.control_frequency == 100.0
        assert len(core.pid_params) == 3
        assert len(core.action_primitives) == 4
    
    def test_motor_preprocessing(self):
        """Test motor input preprocessing."""
        core = create_motor_reasoning_core(6, 100.0)
        
        # Test with current and target states
        key = random.PRNGKey(42)
        current_state = random.normal(key, (6,))
        target_state = random.normal(key, (6,))
        motor_input = jnp.concatenate([current_state, target_state])
        
        processed = core.preprocess_input(motor_input)
        
        assert processed.shape == (core.params.input_size,)
        assert jnp.all(jnp.isfinite(processed))
        assert jnp.any(processed > 0)  # Should have some spikes
    
    def test_motor_postprocessing(self):
        """Test motor output postprocessing."""
        core = create_motor_reasoning_core(6, 100.0)
        
        # Test with random LSM output
        key = random.PRNGKey(42)
        lsm_output = random.normal(key, (core.params.output_size,))
        
        processed = core.postprocess_output(lsm_output)
        
        assert processed.shape == (core.action_dim,)
        assert jnp.all(jnp.isfinite(processed))
        
        # Check that output is within motor limits
        assert jnp.all(jnp.abs(processed) <= core.motor_dynamics['max_force'])
    
    def test_trajectory_planning(self):
        """Test trajectory planning."""
        core = create_motor_reasoning_core(6, 100.0)
        
        start_state = jnp.zeros(6)
        goal_state = jnp.ones(6)
        
        trajectory = core.plan_trajectory(start_state, goal_state, num_steps=10)
        
        assert trajectory.shape == (10, 6)
        assert jnp.allclose(trajectory[0], start_state, atol=1e-6)
        assert jnp.allclose(trajectory[-1], goal_state, atol=1e-6)
        
        # Check that trajectory is smooth (monotonic progression)
        for i in range(1, len(trajectory)):
            progress = jnp.linalg.norm(trajectory[i] - start_state)
            prev_progress = jnp.linalg.norm(trajectory[i-1] - start_state)
            assert progress >= prev_progress
    
    def test_motor_command_computation(self):
        """Test motor command computation."""
        core = create_motor_reasoning_core(6, 100.0)
        
        current_state = jnp.zeros(6)
        target_state = jnp.ones(6)
        
        commands = core.compute_motor_commands(current_state, target_state)
        
        assert commands.shape == (6,)
        assert jnp.all(jnp.isfinite(commands))
        
        # Commands should be proportional to error
        error = target_state - current_state
        expected_commands = error * core.pid_params['kp']
        assert jnp.allclose(commands, expected_commands, atol=1e-6)
    
    def test_motor_processing_pipeline(self):
        """Test complete motor processing pipeline."""
        core = create_motor_reasoning_core(6, 100.0)
        key = random.PRNGKey(42)
        
        # Initialize state
        state = core.init_state(key)
        
        # Create test motor input
        motor_input = random.normal(key, (18,))  # 6 * 3 for current, target, error
        
        # Process through pipeline
        output, new_state = core.process_input(state, motor_input, 0.01, 0.0, key)
        
        assert output.shape == (core.params.output_size,)
        assert isinstance(new_state, ReasoningCoreState)
        assert new_state.activity_level >= 0
        assert len(new_state.processing_history) == 1


class TestMultiModalIntegration:
    """Test integration between different reasoning cores."""
    
    def test_cross_modal_synchronization(self):
        """Test synchronization between different modality cores."""
        # Create cores
        visual_core = create_visual_reasoning_core(16, 16, 1, "visual")
        audio_core = create_audio_reasoning_core(16000, 32, "audio")
        
        # Register as peers
        visual_core.register_peer("audio", audio_core)
        audio_core.register_peer("visual", visual_core)
        
        # Initialize states
        key = random.PRNGKey(42)
        visual_state = visual_core.init_state(key)
        audio_state = audio_core.init_state(key)
        
        # Test synchronization
        peer_states = {"audio": audio_state}
        synced_visual_state = visual_core.synchronize_with_peers(visual_state, peer_states)
        
        assert isinstance(synced_visual_state, ReasoningCoreState)
        assert "audio" in synced_visual_state.sync_signals
        assert synced_visual_state.sync_signals["audio"].shape == (1,)
    
    def test_resource_competition(self):
        """Test resource competition between cores."""
        # Create cores with different activity levels
        visual_core = create_visual_reasoning_core(16, 16, 1, "visual")
        audio_core = create_audio_reasoning_core(16000, 32, "audio")
        
        # Register as peers
        visual_core.register_peer("audio", audio_core)
        audio_core.register_peer("visual", visual_core)
        
        # Create states with different activity levels
        key = random.PRNGKey(42)
        visual_state = visual_core.init_state(key)
        audio_state = audio_core.init_state(key)
        
        # Set different activity levels
        visual_state.activity_level = 0.8  # High activity
        audio_state.activity_level = 0.2   # Low activity
        
        # Test resource competition
        peer_states = {"audio": audio_state}
        visual_allocation = visual_core.compete_for_resources(visual_state, peer_states)
        
        audio_peer_states = {"visual": visual_state}
        audio_allocation = audio_core.compete_for_resources(audio_state, audio_peer_states)
        
        # Visual core should get more resources due to higher activity
        assert visual_allocation > audio_allocation
        assert 0.1 <= visual_allocation <= visual_core.params.max_resource_allocation
        assert 0.1 <= audio_allocation <= audio_core.params.max_resource_allocation
    
    def test_modality_coordination(self):
        """Test coordination between multiple reasoning modalities."""
        # Create all modality cores
        visual_core = create_visual_reasoning_core(8, 8, 1, "visual")
        audio_core = create_audio_reasoning_core(16000, 16, "audio")
        text_core = create_text_reasoning_core(128, 32, 16, "text")
        motor_core = create_motor_reasoning_core(3, 100.0, "motor")
        
        cores = {
            "visual": visual_core,
            "audio": audio_core,
            "text": text_core,
            "motor": motor_core
        }
        
        # Register all cores as peers of each other
        for core_id, core in cores.items():
            for peer_id, peer_core in cores.items():
                if core_id != peer_id:
                    core.register_peer(peer_id, peer_core)
        
        # Initialize all states
        key = random.PRNGKey(42)
        states = {}
        for core_id, core in cores.items():
            states[core_id] = core.init_state(key)
        
        # Test that all cores can synchronize with all peers
        for core_id, core in cores.items():
            peer_states = {pid: state for pid, state in states.items() if pid != core_id}
            synced_state = core.synchronize_with_peers(states[core_id], peer_states)
            
            assert len(synced_state.sync_signals) == len(peer_states)
            for peer_id in peer_states.keys():
                assert peer_id in synced_state.sync_signals
    
    def test_performance_tracking_across_modalities(self):
        """Test performance tracking across different modalities."""
        cores = {
            "visual": create_visual_reasoning_core(8, 8, 1, "visual"),
            "audio": create_audio_reasoning_core(16000, 16, "audio"),
            "text": create_text_reasoning_core(128, 32, 16, "text"),
            "motor": create_motor_reasoning_core(3, 100.0, "motor")
        }
        
        # Add performance scores to each core
        performance_scores = {
            "visual": [0.9, 0.85, 0.92],
            "audio": [0.7, 0.75, 0.8],
            "text": [0.95, 0.9, 0.88],
            "motor": [0.6, 0.65, 0.7]
        }
        
        for core_id, scores in performance_scores.items():
            for score in scores:
                cores[core_id].update_performance(score)
        
        # Verify performance tracking
        for core_id, expected_scores in performance_scores.items():
            assert cores[core_id].performance_history == expected_scores
        
        # Test resource competition based on performance
        # Text core should get more resources due to higher performance
        text_state = cores["text"].init_state(random.PRNGKey(42))
        audio_state = cores["audio"].init_state(random.PRNGKey(42))
        
        text_state.activity_level = 0.5
        audio_state.activity_level = 0.5  # Same activity level
        
        cores["text"].register_peer("audio", cores["audio"])
        cores["audio"].register_peer("text", cores["text"])
        
        text_allocation = cores["text"].compete_for_resources(
            text_state, {"audio": audio_state}
        )
        audio_allocation = cores["audio"].compete_for_resources(
            audio_state, {"text": text_state}
        )
        
        # Text should get more resources due to better performance history
        assert text_allocation > audio_allocation


if __name__ == "__main__":
    pytest.main([__file__])
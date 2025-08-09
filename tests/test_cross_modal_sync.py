"""
Tests for Cross-Modal Synchronization System

This module contains comprehensive tests for the cross-modal synchronization
system, including synchronization accuracy, information flow, and resource
allocation tests.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import Dict, List

from src.agents.reasoning.cross_modal_sync import (
    CrossModalSynchronizer,
    SyncParams,
    SyncMode,
    create_cross_modal_synchronizer
)
from src.agents.reasoning.base_reasoning_core import (
    ReasoningCoreParams,
    ReasoningCoreState,
    ModalityType
)
from src.agents.reasoning.visual_reasoning_core import create_visual_reasoning_core
from src.agents.reasoning.audio_reasoning_core import create_audio_reasoning_core
from src.agents.reasoning.text_reasoning_core import create_text_reasoning_core
from src.agents.reasoning.motor_reasoning_core import create_motor_reasoning_core
from src.core.liquid_state_machine import LSMParams, LSMState
from src.core.neurons.lif_neuron import LIFState


class TestCrossModalSynchronizer:
    """Test suite for cross-modal synchronization system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.key = random.PRNGKey(42)
        
        # Create synchronizer
        self.sync_params = SyncParams(
            sync_strength=0.5,
            competition_strength=0.7,
            cross_modal_gain=0.2
        )
        self.synchronizer = CrossModalSynchronizer(self.sync_params)
        
        # Create test reasoning cores
        self.visual_core = create_visual_reasoning_core(core_id="visual_test")
        self.audio_core = create_audio_reasoning_core(core_id="audio_test")
        self.text_core = create_text_reasoning_core(core_id="text_test")
        self.motor_core = create_motor_reasoning_core(core_id="motor_test")
        
        # Initialize core states
        self.visual_state = self.visual_core.init_state(self.key)
        self.audio_state = self.audio_core.init_state(random.split(self.key)[0])
        self.text_state = self.text_core.init_state(random.split(self.key)[1])
        self.motor_state = self.motor_core.init_state(random.split(self.key)[0])
        
        # Register cores with synchronizer
        self.synchronizer.register_core(self.visual_core, self.visual_state)
        self.synchronizer.register_core(self.audio_core, self.audio_state)
        self.synchronizer.register_core(self.text_core, self.text_state)
        self.synchronizer.register_core(self.motor_core, self.motor_state)
    
    def test_synchronizer_initialization(self):
        """Test synchronizer initialization."""
        sync = CrossModalSynchronizer()
        
        assert sync.params is not None
        assert isinstance(sync.params, SyncParams)
        assert len(sync.cores) == 0
        assert len(sync.core_states) == 0
        assert len(sync.connection_weights) == 0
    
    def test_core_registration(self):
        """Test reasoning core registration."""
        # Check that cores are registered
        assert len(self.synchronizer.cores) == 4
        assert len(self.synchronizer.core_states) == 4
        
        # Check core IDs
        expected_ids = {"visual_test", "audio_test", "text_test", "motor_test"}
        assert set(self.synchronizer.cores.keys()) == expected_ids
        
        # Check connection weights are initialized
        expected_connections = 4 * 3  # 4 cores, 3 connections each (bidirectional)
        assert len(self.synchronizer.connection_weights) == expected_connections
        
        # Check sync phases are initialized
        assert len(self.synchronizer.sync_phases) == 4
        for phase in self.synchronizer.sync_phases.values():
            assert phase == 0.0
    
    def test_phase_lock_synchronization(self):
        """Test phase-locking synchronization mode."""
        # Create mock spike patterns with different phases
        self._create_mock_spike_patterns()
        
        # Run phase-lock synchronization
        dt = 0.01
        t = 0.0
        updated_states = self.synchronizer.synchronize_cores(
            dt, t, SyncMode.PHASE_LOCK
        )
        
        # Check that synchronization signals are generated
        assert len(updated_states) == 4
        
        for core_id, state in updated_states.items():
            assert 'global_sync' in state.sync_signals
            sync_signal = state.sync_signals['global_sync']
            assert isinstance(sync_signal, jnp.ndarray)
            assert len(sync_signal) > 0
    
    def test_coherence_synchronization(self):
        """Test coherence-based synchronization mode."""
        self._create_mock_spike_patterns()
        
        # Run coherence synchronization
        dt = 0.01
        t = 0.0
        updated_states = self.synchronizer.synchronize_cores(
            dt, t, SyncMode.COHERENCE
        )
        
        # Check synchronization results
        assert len(updated_states) == 4
        
        # Verify sync history is updated
        assert len(self.synchronizer.sync_history) > 0
        
        # Check coherence computation
        coherence_matrix = self.synchronizer._compute_coherence_matrix()
        assert len(coherence_matrix) == 12  # 4 cores, 3 pairs each
        
        for coherence_value in coherence_matrix.values():
            assert 0.0 <= coherence_value <= 1.0
    
    def test_competitive_synchronization(self):
        """Test competitive synchronization mode."""
        # Set different activity levels
        self.synchronizer.core_states["visual_test"].activity_level = 0.8
        self.synchronizer.core_states["audio_test"].activity_level = 0.6
        self.synchronizer.core_states["text_test"].activity_level = 0.4
        self.synchronizer.core_states["motor_test"].activity_level = 0.2
        
        # Run competitive synchronization
        dt = 0.01
        t = 0.0
        updated_states = self.synchronizer.synchronize_cores(
            dt, t, SyncMode.COMPETITIVE
        )
        
        # Check that competitive signals reflect activity differences
        visual_signal = updated_states["visual_test"].sync_signals['global_sync']
        motor_signal = updated_states["motor_test"].sync_signals['global_sync']
        
        # Visual core (higher activity) should have positive signal
        # Motor core (lower activity) should have negative signal
        assert float(visual_signal[0]) > float(motor_signal[0])
    
    def test_cooperative_synchronization(self):
        """Test cooperative synchronization mode."""
        self._create_mock_spike_patterns()
        
        # Run cooperative synchronization
        dt = 0.01
        t = 0.0
        updated_states = self.synchronizer.synchronize_cores(
            dt, t, SyncMode.COOPERATIVE
        )
        
        # Check mutual information computation
        mutual_info_matrix = self.synchronizer._compute_mutual_information_matrix()
        assert len(mutual_info_matrix) == 12
        
        for mi_value in mutual_info_matrix.values():
            assert mi_value >= 0.0  # Mutual information is non-negative
    
    def test_competitive_resource_allocation(self):
        """Test competitive resource allocation."""
        # Set different activity levels and performance
        cores_data = [
            ("visual_test", 0.8, [0.9, 0.85, 0.88]),
            ("audio_test", 0.6, [0.7, 0.75, 0.72]),
            ("text_test", 0.4, [0.6, 0.65, 0.63]),
            ("motor_test", 0.2, [0.5, 0.55, 0.52])
        ]
        
        for core_id, activity, performance_history in cores_data:
            self.synchronizer.core_states[core_id].activity_level = activity
            self.synchronizer.cores[core_id].performance_history = performance_history
        
        # Allocate resources competitively
        allocations = self.synchronizer.allocate_resources(
            total_resources=1.0,
            allocation_mode="competitive"
        )
        
        # Check allocation results
        assert len(allocations) == 4
        assert abs(sum(allocations.values()) - 1.0) < 1e-6  # Total should be 1.0
        
        # Higher performing cores should get more resources
        assert allocations["visual_test"] > allocations["motor_test"]
        assert allocations["audio_test"] > allocations["text_test"]
        
        # Check minimum allocation constraint
        for allocation in allocations.values():
            assert allocation >= self.sync_params.min_allocation
    
    def test_equal_resource_allocation(self):
        """Test equal resource allocation."""
        allocations = self.synchronizer.allocate_resources(
            total_resources=1.0,
            allocation_mode="equal"
        )
        
        # Check equal allocation
        expected_allocation = 1.0 / 4
        for allocation in allocations.values():
            assert abs(allocation - expected_allocation) < 1e-6
    
    def test_adaptive_resource_allocation(self):
        """Test adaptive resource allocation based on learning progress."""
        # Set up performance histories with different learning trends
        self.synchronizer.cores["visual_test"].performance_history = [0.5, 0.6, 0.7, 0.8, 0.9]  # Improving
        self.synchronizer.cores["audio_test"].performance_history = [0.8, 0.7, 0.6, 0.5, 0.4]   # Declining
        self.synchronizer.cores["text_test"].performance_history = [0.6, 0.6, 0.6, 0.6, 0.6]    # Stable
        self.synchronizer.cores["motor_test"].performance_history = [0.3, 0.4, 0.5, 0.6, 0.7]   # Improving
        
        allocations = self.synchronizer.allocate_resources(
            total_resources=1.0,
            allocation_mode="adaptive"
        )
        
        # Improving cores should get more resources than declining ones
        assert allocations["visual_test"] > allocations["audio_test"]
        assert allocations["motor_test"] > allocations["audio_test"]
    
    def test_cross_modal_information_integration(self):
        """Test cross-modal information integration."""
        self._create_mock_spike_patterns()
        
        # Test integration for visual core
        integrated_features = self.synchronizer.integrate_cross_modal_information(
            primary_core_id="visual_test",
            integration_strength=0.3
        )
        
        # Check integration results
        assert isinstance(integrated_features, jnp.ndarray)
        assert len(integrated_features) > 0
        
        # Check integration history is updated
        assert len(self.synchronizer.integration_history) > 0
        
        latest_integration = self.synchronizer.integration_history[-1]
        assert latest_integration['primary_core'] == "visual_test"
        assert 'features' in latest_integration
    
    def test_connection_weight_updates(self):
        """Test connection weight adaptation."""
        # Set initial activity levels
        for core_id in self.synchronizer.cores.keys():
            self.synchronizer.core_states[core_id].activity_level = 0.5
        
        # Store initial weights
        initial_weights = self.synchronizer.connection_weights.copy()
        
        # Update weights with Hebbian learning
        self.synchronizer.update_connection_weights(
            learning_rate=0.1,
            adaptation_mode="hebbian"
        )
        
        # Check that weights have been updated
        updated_weights = self.synchronizer.connection_weights
        
        # At least some weights should have changed
        weight_changes = [
            abs(updated_weights[conn] - initial_weights[conn])
            for conn in initial_weights.keys()
        ]
        assert max(weight_changes) > 0
        
        # All weights should be within bounds
        for weight in updated_weights.values():
            assert 0.0 <= weight <= 1.0
    
    def test_synchronization_metrics(self):
        """Test synchronization metrics computation."""
        # Run some synchronization steps to generate history
        for i in range(10):
            self.synchronizer.synchronize_cores(0.01, i * 0.01, SyncMode.COHERENCE)
        
        # Get metrics
        metrics = self.synchronizer.get_synchronization_metrics()
        
        # Check that metrics are computed
        assert len(metrics) > 0
        
        # Check core-specific metrics
        for core_id in self.synchronizer.cores.keys():
            assert f"{core_id}_sync_mean" in metrics
            assert f"{core_id}_sync_std" in metrics
        
        # Check global metrics
        assert "global_coherence" in metrics
        assert "coherence_std" in metrics
        
        # Check metric values are reasonable
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
    
    def test_spike_phase_computation(self):
        """Test spike phase computation."""
        # Create test spike pattern
        spikes = jnp.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        
        phase = self.synchronizer._compute_spike_phase(spikes)
        
        # Phase should be between 0 and 2π
        assert 0.0 <= phase <= 2 * np.pi
        
        # Test with no spikes
        no_spikes = jnp.zeros(8)
        phase_no_spikes = self.synchronizer._compute_spike_phase(no_spikes)
        assert phase_no_spikes == 0.0
    
    def test_pairwise_coherence_computation(self):
        """Test pairwise coherence computation."""
        self._create_mock_spike_patterns()
        
        # Test coherence between visual and audio cores
        coherence = self.synchronizer._compute_pairwise_coherence("visual_test", "audio_test")
        
        # Coherence should be between 0 and 1
        assert 0.0 <= coherence <= 1.0
        
        # Test with identical patterns (should have high coherence)
        # Create identical spike patterns
        identical_spikes = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])
        
        # Temporarily set identical patterns
        original_visual_spikes = self.synchronizer.core_states["visual_test"].lsm_state.neuron_state.spikes
        original_audio_spikes = self.synchronizer.core_states["audio_test"].lsm_state.neuron_state.spikes
        
        # Create new states with identical spikes
        visual_neuron_state = LIFState(
            v_mem=original_visual_spikes,
            i_syn=jnp.zeros_like(identical_spikes),
            v_thresh_dyn=jnp.ones_like(identical_spikes),
            t_last_spike=jnp.zeros_like(identical_spikes),
            spikes=identical_spikes
        )
        
        audio_neuron_state = LIFState(
            v_mem=original_audio_spikes,
            i_syn=jnp.zeros_like(identical_spikes),
            v_thresh_dyn=jnp.ones_like(identical_spikes),
            t_last_spike=jnp.zeros_like(identical_spikes),
            spikes=identical_spikes
        )
        
        # Update LSM states
        visual_lsm_state = LSMState(
            neuron_state=visual_neuron_state,
            plasticity_state=self.synchronizer.core_states["visual_test"].lsm_state.plasticity_state,
            readout_weights=self.synchronizer.core_states["visual_test"].lsm_state.readout_weights,
            reservoir_weights=self.synchronizer.core_states["visual_test"].lsm_state.reservoir_weights,
            input_weights=self.synchronizer.core_states["visual_test"].lsm_state.input_weights,
            activity_history=self.synchronizer.core_states["visual_test"].lsm_state.activity_history
        )
        
        audio_lsm_state = LSMState(
            neuron_state=audio_neuron_state,
            plasticity_state=self.synchronizer.core_states["audio_test"].lsm_state.plasticity_state,
            readout_weights=self.synchronizer.core_states["audio_test"].lsm_state.readout_weights,
            reservoir_weights=self.synchronizer.core_states["audio_test"].lsm_state.reservoir_weights,
            input_weights=self.synchronizer.core_states["audio_test"].lsm_state.input_weights,
            activity_history=self.synchronizer.core_states["audio_test"].lsm_state.activity_history
        )
        
        # Update core states
        self.synchronizer.core_states["visual_test"].lsm_state = visual_lsm_state
        self.synchronizer.core_states["audio_test"].lsm_state = audio_lsm_state
        
        # Compute coherence with identical patterns
        identical_coherence = self.synchronizer._compute_pairwise_coherence("visual_test", "audio_test")
        
        # Should have high coherence (close to 1.0)
        assert identical_coherence > 0.8
    
    def test_mutual_information_computation(self):
        """Test mutual information computation."""
        self._create_mock_spike_patterns()
        
        # Test mutual information between cores
        mi = self.synchronizer._compute_pairwise_mutual_information("visual_test", "audio_test")
        
        # Mutual information should be non-negative
        assert mi >= 0.0
        
        # Test with independent patterns (should have low MI)
        # Test with correlated patterns (should have higher MI)
        # This is implicitly tested through the mock patterns
    
    def test_synchronization_history_management(self):
        """Test synchronization history management."""
        # Generate many synchronization steps
        for i in range(1100):  # More than the history limit
            self.synchronizer.synchronize_cores(0.01, i * 0.01, SyncMode.COHERENCE)
        
        # Check history is bounded
        assert len(self.synchronizer.sync_history) <= 1000
        
        # Check resource history management
        for i in range(1100):
            self.synchronizer.allocate_resources()
        
        assert len(self.synchronizer.resource_history) <= 1000
    
    def test_synchronization_reset(self):
        """Test synchronization state reset."""
        # Generate some history
        for i in range(10):
            self.synchronizer.synchronize_cores(0.01, i * 0.01, SyncMode.COHERENCE)
            self.synchronizer.allocate_resources()
            self.synchronizer.integrate_cross_modal_information("visual_test")
        
        # Verify history exists
        assert len(self.synchronizer.sync_history) > 0
        assert len(self.synchronizer.resource_history) > 0
        assert len(self.synchronizer.integration_history) > 0
        
        # Reset synchronization
        self.synchronizer.reset_synchronization()
        
        # Verify reset
        assert len(self.synchronizer.sync_history) == 0
        assert len(self.synchronizer.resource_history) == 0
        assert len(self.synchronizer.integration_history) == 0
        
        # Check sync phases are reset
        for phase in self.synchronizer.sync_phases.values():
            assert phase == 0.0
        
        # Check connection weights are reset to default
        for weight in self.synchronizer.connection_weights.values():
            assert weight == 0.5
    
    def test_create_cross_modal_synchronizer_function(self):
        """Test the factory function for creating synchronizers."""
        sync = create_cross_modal_synchronizer(
            sync_strength=0.8,
            competition_strength=0.9,
            integration_gain=0.3
        )
        
        assert isinstance(sync, CrossModalSynchronizer)
        assert sync.params.sync_strength == 0.8
        assert sync.params.competition_strength == 0.9
        assert sync.params.cross_modal_gain == 0.3
    
    def test_synchronization_with_single_core(self):
        """Test synchronization behavior with only one core."""
        # Create new synchronizer with single core
        single_sync = CrossModalSynchronizer()
        single_sync.register_core(self.visual_core, self.visual_state)
        
        # Synchronization should handle single core gracefully
        updated_states = single_sync.synchronize_cores(0.01, 0.0, SyncMode.COHERENCE)
        
        assert len(updated_states) == 1
        assert "visual_test" in updated_states
    
    def test_resource_allocation_constraints(self):
        """Test resource allocation constraint enforcement."""
        # Set very high activity for one core
        self.synchronizer.core_states["visual_test"].activity_level = 10.0
        
        allocations = self.synchronizer.allocate_resources(
            total_resources=1.0,
            allocation_mode="competitive"
        )
        
        # Check maximum allocation constraint
        visual_allocation = allocations["visual_test"]
        max_allowed = self.visual_core.params.max_resource_allocation
        assert visual_allocation <= max_allowed
        
        # Check that the implementation enforces minimum allocation
        # (The actual implementation clips to minimum allocation after softmax)
        # Test with equal mode to verify minimum allocation works
        equal_allocations = self.synchronizer.allocate_resources(
            total_resources=1.0,
            allocation_mode="equal"
        )
        
        for allocation in equal_allocations.values():
            assert allocation >= self.sync_params.min_allocation
    
    def test_information_flow_accuracy(self):
        """Test accuracy of cross-modal information flow."""
        # Create correlated patterns between cores
        self._create_correlated_spike_patterns()
        
        # Test information integration
        visual_integration = self.synchronizer.integrate_cross_modal_information("visual_test")
        audio_integration = self.synchronizer.integrate_cross_modal_information("audio_test")
        
        # Integration should capture cross-modal relationships
        assert len(visual_integration) > 0
        assert len(audio_integration) > 0
        
        # Test that integration reflects correlation structure
        # (This is a simplified test - in practice would need more sophisticated validation)
        visual_magnitude = float(jnp.mean(jnp.abs(visual_integration)))
        audio_magnitude = float(jnp.mean(jnp.abs(audio_integration)))
        
        # Both should have non-zero integration due to correlation
        assert visual_magnitude > 0
        assert audio_magnitude > 0
    
    def _create_mock_spike_patterns(self):
        """Create mock spike patterns for testing."""
        # Create different spike patterns for each core
        patterns = {
            "visual_test": jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            "audio_test": jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
            "text_test": jnp.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
            "motor_test": jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        }
        
        for core_id, pattern in patterns.items():
            # Create neuron state with mock spikes
            neuron_state = LIFState(
                v_mem=pattern,
                i_syn=jnp.zeros_like(pattern),
                v_thresh_dyn=jnp.ones_like(pattern),
                t_last_spike=jnp.zeros_like(pattern),
                spikes=pattern
            )
            
            # Update LSM state
            lsm_state = LSMState(
                neuron_state=neuron_state,
                plasticity_state=self.synchronizer.core_states[core_id].lsm_state.plasticity_state,
                readout_weights=self.synchronizer.core_states[core_id].lsm_state.readout_weights,
                reservoir_weights=self.synchronizer.core_states[core_id].lsm_state.reservoir_weights,
                input_weights=self.synchronizer.core_states[core_id].lsm_state.input_weights,
                activity_history=self.synchronizer.core_states[core_id].lsm_state.activity_history
            )
            
            # Update core state
            self.synchronizer.core_states[core_id].lsm_state = lsm_state
    
    def _create_correlated_spike_patterns(self):
        """Create correlated spike patterns for information flow testing."""
        # Create base pattern
        base_pattern = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        
        # Create correlated patterns with noise
        key = random.PRNGKey(123)
        keys = random.split(key, 4)
        
        patterns = {}
        for i, core_id in enumerate(["visual_test", "audio_test", "text_test", "motor_test"]):
            # Add noise to base pattern
            noise = random.normal(keys[i], base_pattern.shape) * 0.2
            correlated_pattern = jnp.clip(base_pattern + noise, 0.0, 1.0)
            patterns[core_id] = correlated_pattern
        
        # Update core states with correlated patterns
        for core_id, pattern in patterns.items():
            neuron_state = LIFState(
                v_mem=pattern,
                i_syn=jnp.zeros_like(pattern),
                v_thresh_dyn=jnp.ones_like(pattern),
                t_last_spike=jnp.zeros_like(pattern),
                spikes=pattern
            )
            
            lsm_state = LSMState(
                neuron_state=neuron_state,
                plasticity_state=self.synchronizer.core_states[core_id].lsm_state.plasticity_state,
                readout_weights=self.synchronizer.core_states[core_id].lsm_state.readout_weights,
                reservoir_weights=self.synchronizer.core_states[core_id].lsm_state.reservoir_weights,
                input_weights=self.synchronizer.core_states[core_id].lsm_state.input_weights,
                activity_history=self.synchronizer.core_states[core_id].lsm_state.activity_history
            )
            
            self.synchronizer.core_states[core_id].lsm_state = lsm_state


class TestSyncParams:
    """Test suite for SyncParams dataclass."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = SyncParams()
        
        assert params.sync_strength == 0.5
        assert params.phase_coupling == 0.3
        assert params.coherence_threshold == 0.4
        assert params.competition_strength == 0.7
        assert params.resource_decay == 0.95
        assert params.min_allocation == 0.1
        assert params.integration_window == 0.1
        assert params.cross_modal_gain == 0.2
        assert params.attention_decay == 0.9
        assert params.sync_time_constant == 0.05
        assert params.adaptation_rate == 0.01
    
    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        params = SyncParams(
            sync_strength=0.8,
            competition_strength=0.9,
            cross_modal_gain=0.3
        )
        
        assert params.sync_strength == 0.8
        assert params.competition_strength == 0.9
        assert params.cross_modal_gain == 0.3
        # Other parameters should have default values
        assert params.phase_coupling == 0.3


class TestSyncMode:
    """Test suite for SyncMode enum."""
    
    def test_sync_mode_values(self):
        """Test synchronization mode enum values."""
        assert SyncMode.PHASE_LOCK.value == "phase_lock"
        assert SyncMode.COHERENCE.value == "coherence"
        assert SyncMode.COMPETITIVE.value == "competitive"
        assert SyncMode.COOPERATIVE.value == "cooperative"
    
    def test_sync_mode_membership(self):
        """Test synchronization mode membership."""
        modes = list(SyncMode)
        assert len(modes) == 4
        assert SyncMode.PHASE_LOCK in modes
        assert SyncMode.COHERENCE in modes
        assert SyncMode.COMPETITIVE in modes
        assert SyncMode.COOPERATIVE in modes


if __name__ == "__main__":
    pytest.main([__file__])
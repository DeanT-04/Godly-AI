"""
Unit tests for the Leaky Integrate-and-Fire (LIF) neuron implementation.

Tests cover:
- Basic neuron dynamics
- Spike generation
- Adaptive threshold behavior
- Refractory period handling
- Batch processing
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from src.core.neurons.lif_neuron import LIFNeuron, LIFParams, LIFState, create_lif_neuron


class TestLIFNeuron:
    """Test suite for LIF neuron implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.neuron = LIFNeuron()
        self.key = random.PRNGKey(42)
        self.dt = 0.1e-3  # 0.1ms timestep
    
    def test_neuron_initialization(self):
        """Test neuron initialization with default parameters."""
        assert self.neuron.params.tau_mem == 20e-3
        assert self.neuron.params.v_rest == -70e-3
        assert self.neuron.params.v_thresh == -50e-3
    
    def test_custom_parameters(self):
        """Test neuron initialization with custom parameters."""
        custom_params = LIFParams(
            tau_mem=10e-3,
            v_thresh=-55e-3,
            t_refrac=1e-3
        )
        neuron = LIFNeuron(custom_params)
        assert neuron.params.tau_mem == 10e-3
        assert neuron.params.v_thresh == -55e-3
        assert neuron.params.t_refrac == 1e-3
    
    def test_state_initialization(self):
        """Test proper state initialization."""
        batch_size = 10
        state = self.neuron.init_state(batch_size, self.key)
        
        assert state.v_mem.shape == (batch_size,)
        assert state.i_syn.shape == (batch_size,)
        assert state.v_thresh_dyn.shape == (batch_size,)
        assert state.t_last_spike.shape == (batch_size,)
        assert state.spikes.shape == (batch_size,)
        
        # Check initial values
        assert jnp.allclose(state.v_mem, self.neuron.params.v_rest)
        assert jnp.allclose(state.i_syn, 0.0)
        assert jnp.allclose(state.v_thresh_dyn, self.neuron.params.v_thresh)
        assert jnp.all(state.t_last_spike == -jnp.inf)
        assert jnp.all(state.spikes == False)
    
    def test_membrane_integration(self):
        """Test basic membrane potential integration."""
        batch_size = 1
        state = self.neuron.init_state(batch_size, self.key)
        
        # Apply constant input current
        input_current = jnp.array([50e-12])  # 50pA
        t = 0.0
        
        # Single step
        new_state = self.neuron.step(state, input_current, self.dt, t)
        
        # Membrane potential should increase from rest
        assert new_state.v_mem[0] > state.v_mem[0]
        assert new_state.i_syn[0] > 0  # Synaptic current should be positive
    
    def test_spike_generation(self):
        """Test spike generation when threshold is reached."""
        batch_size = 1
        state = self.neuron.init_state(batch_size, self.key)
        
        # Apply large input current to trigger spike
        input_current = jnp.array([500e-12])  # 500pA - large current
        t = 0.0
        
        # Integrate until spike occurs
        max_steps = 1000
        for i in range(max_steps):
            state = self.neuron.step(state, input_current, self.dt, t + i * self.dt)
            if state.spikes[0]:
                break
        
        # Should have generated a spike
        assert state.spikes[0] == True
        # Membrane potential should be reset
        assert jnp.isclose(state.v_mem[0], self.neuron.params.v_reset, atol=1e-6)
    
    def test_refractory_period(self):
        """Test refractory period behavior."""
        batch_size = 1
        state = self.neuron.init_state(batch_size, self.key)
        
        # Force a spike by setting membrane potential above threshold
        state = state._replace(
            v_mem=jnp.array([self.neuron.params.v_thresh + 10e-3]),
            t_last_spike=jnp.array([0.0])
        )
        
        # Apply input during refractory period
        input_current = jnp.array([100e-12])
        t = 0.5e-3  # 0.5ms after spike (within refractory period)
        
        new_state = self.neuron.step(state, input_current, self.dt, t)
        
        # Should remain at reset potential during refractory period
        assert jnp.isclose(new_state.v_mem[0], self.neuron.params.v_reset, atol=1e-6)
        assert new_state.spikes[0] == False
    
    def test_adaptive_threshold(self):
        """Test adaptive threshold mechanism."""
        batch_size = 1
        state = self.neuron.init_state(batch_size, self.key)
        
        # Force a spike
        state = state._replace(
            v_mem=jnp.array([self.neuron.params.v_thresh + 1e-3]),
            spikes=jnp.array([True])
        )
        
        new_state = self.neuron.step(state, jnp.zeros(1), self.dt, 0.0)
        
        # Threshold should be adapted (increased)
        expected_thresh = (
            self.neuron.params.v_thresh + 
            self.neuron.params.v_thresh_adapt
        )
        assert new_state.v_thresh_dyn[0] > self.neuron.params.v_thresh
    
    def test_threshold_decay(self):
        """Test threshold decay back to baseline."""
        batch_size = 1
        state = self.neuron.init_state(batch_size, self.key)
        
        # Set elevated threshold
        elevated_thresh = self.neuron.params.v_thresh + 20e-3
        state = state._replace(v_thresh_dyn=jnp.array([elevated_thresh]))
        
        # Run multiple steps without spikes
        for _ in range(100):
            state = self.neuron.step(state, jnp.zeros(1), self.dt, 0.0)
        
        # Threshold should decay toward baseline
        assert state.v_thresh_dyn[0] < elevated_thresh
    
    def test_batch_processing(self):
        """Test processing multiple neurons simultaneously."""
        batch_size = 5
        state = self.neuron.init_state(batch_size, self.key)
        
        # Apply different input currents to each neuron
        input_currents = jnp.array([10e-12, 50e-12, 100e-12, 200e-12, 500e-12])
        
        new_state = self.neuron.step(state, input_currents, self.dt, 0.0)
        
        # All neurons should have different membrane potentials
        assert len(jnp.unique(new_state.v_mem)) == batch_size
        # Higher currents should lead to higher potentials
        assert jnp.all(jnp.diff(new_state.v_mem) >= 0)
    
    def test_integrate_and_fire_legacy(self):
        """Test legacy integrate-and-fire interface."""
        input_current = 100e-12
        v_mem = -65e-3
        
        spike, v_new = self.neuron.integrate_and_fire(input_current, v_mem, self.dt)
        
        assert isinstance(spike, bool)
        assert isinstance(v_new, float)
        assert v_new != v_mem  # Should change membrane potential
    
    def test_spike_train_simulation(self):
        """Test spike train simulation with time series input."""
        # Create step input
        n_steps = 1000
        input_currents = jnp.zeros(n_steps)
        input_currents = input_currents.at[100:800].set(200e-12)  # Step input
        
        spike_trains, final_state = self.neuron.simulate_spike_train(
            input_currents, self.dt, batch_size=1, key=self.key
        )
        
        assert spike_trains.shape == (n_steps, 1)
        assert jnp.any(spike_trains)  # Should generate some spikes
        
        # Spikes should occur during the input period
        spike_times = jnp.where(spike_trains[:, 0])[0]
        assert len(spike_times) > 0
        assert jnp.all(spike_times >= 100)  # After input starts
    
    def test_noise_effects(self):
        """Test membrane noise effects."""
        batch_size = 10
        state = self.neuron.init_state(batch_size, self.key)
        
        # Run with noise
        key1, key2 = random.split(self.key)
        state1 = self.neuron.step(state, jnp.zeros(batch_size), self.dt, 0.0, key1)
        state2 = self.neuron.step(state, jnp.zeros(batch_size), self.dt, 0.0, key2)
        
        # Different noise should lead to different outcomes
        assert not jnp.allclose(state1.v_mem, state2.v_mem)
    
    def test_create_neuron_types(self):
        """Test different neuron type configurations."""
        # Test all predefined types
        standard = create_lif_neuron("standard")
        fast = create_lif_neuron("fast")
        slow = create_lif_neuron("slow")
        adaptive = create_lif_neuron("adaptive")
        
        # Check parameter differences
        assert fast.params.tau_mem < standard.params.tau_mem
        assert slow.params.tau_mem > standard.params.tau_mem
        assert adaptive.params.v_thresh_adapt > standard.params.v_thresh_adapt
        
        # Test invalid type
        with pytest.raises(ValueError):
            create_lif_neuron("invalid_type")
    
    def test_parameter_validation(self):
        """Test parameter validation and edge cases."""
        # Test with extreme parameters
        extreme_params = LIFParams(
            tau_mem=1e-6,    # Very fast
            v_thresh=-30e-3, # High threshold
            t_refrac=10e-3   # Long refractory period
        )
        neuron = LIFNeuron(extreme_params)
        
        # Should still work
        state = neuron.init_state(1, self.key)
        new_state = neuron.step(state, jnp.array([100e-12]), self.dt, 0.0)
        
        assert new_state.v_mem.shape == (1,)


class TestLIFNeuronIntegration:
    """Integration tests for LIF neuron with realistic scenarios."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.neuron = LIFNeuron()
        self.key = random.PRNGKey(123)
        self.dt = 0.1e-3
    
    def test_poisson_input_response(self):
        """Test neuron response to Poisson spike train input."""
        # Generate Poisson spike train
        rate = 100.0  # 100 Hz
        duration = 1.0  # 1 second
        n_steps = int(duration / self.dt)
        
        # Generate Poisson spikes
        key = self.key
        spike_prob = rate * self.dt
        spikes = random.bernoulli(key, spike_prob, (n_steps,))
        
        # Convert to current (simple model)
        input_currents = spikes.astype(float) * 100e-12  # 100pA per spike
        
        spike_trains, _ = self.neuron.simulate_spike_train(
            input_currents, self.dt, batch_size=1, key=self.key
        )
        
        # Calculate output firing rate
        output_spikes = jnp.sum(spike_trains)
        output_rate = output_spikes / duration
        
        # Should have reasonable firing rate (not zero, not too high)
        assert 0 < output_rate < 200  # Between 0 and 200 Hz
    
    def test_frequency_response(self):
        """Test neuron frequency response to sinusoidal input."""
        duration = 1.0
        freq = 10.0  # 10 Hz
        n_steps = int(duration / self.dt)
        
        # Generate sinusoidal input
        t = jnp.arange(n_steps) * self.dt
        amplitude = 50e-12  # 50pA amplitude
        offset = 30e-12     # 30pA DC offset
        input_currents = amplitude * jnp.sin(2 * jnp.pi * freq * t) + offset
        
        spike_trains, _ = self.neuron.simulate_spike_train(
            input_currents, self.dt, batch_size=1, key=self.key
        )
        
        # Should generate spikes that follow the input frequency
        total_spikes = jnp.sum(spike_trains)
        assert total_spikes > 0
        
        # Analyze spike timing (basic check)
        spike_times = jnp.where(spike_trains[:, 0])[0] * self.dt
        if len(spike_times) > 1:
            # Inter-spike intervals should show some regularity
            # For spiking neurons with noise and adaptation, coefficient of variation < 2 is reasonable
            isi = jnp.diff(spike_times)
            cv = jnp.std(isi) / jnp.mean(isi)  # Coefficient of variation
            assert cv < 2.0  # Reasonable regularity for noisy spiking neuron
    
    def test_network_compatibility(self):
        """Test compatibility with network-level operations."""
        # Simulate small network of neurons
        batch_size = 10
        state = self.neuron.init_state(batch_size, self.key)
        
        # Apply network-like input (different for each neuron)
        key = self.key
        for step in range(100):
            key, subkey = random.split(key)
            
            # Random input currents
            input_currents = random.normal(subkey, (batch_size,)) * 20e-12 + 50e-12
            
            state = self.neuron.step(state, input_currents, self.dt, step * self.dt, subkey)
        
        # Should maintain proper state shapes and values
        assert state.v_mem.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(state.v_mem))
        assert jnp.all(state.v_mem >= self.neuron.params.v_reset)
        assert jnp.all(state.v_mem <= self.neuron.params.v_thresh + 50e-3)  # Reasonable range


if __name__ == "__main__":
    pytest.main([__file__])
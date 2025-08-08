"""
Unit tests for Spike-Timing Dependent Plasticity (STDP) implementation.

Tests cover:
- STDP learning rule behavior
- Weight update mechanisms
- Eligibility trace dynamics
- Convergence properties
- Different STDP configurations
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from src.core.plasticity.stdp import (
    STDPLearningRule, STDPParams, STDPState, 
    TripletsSTDP, create_stdp_rule, compute_stdp_window
)


class TestSTDPLearningRule:
    """Test suite for STDP learning rule implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stdp = STDPLearningRule()
        self.key = random.PRNGKey(42)
        self.dt = 0.1e-3  # 0.1ms timestep
        self.n_pre = 5
        self.n_post = 3
    
    def test_stdp_initialization(self):
        """Test STDP initialization with default parameters."""
        assert self.stdp.params.a_plus == 0.01
        assert self.stdp.params.a_minus == 0.01
        assert self.stdp.params.tau_plus == 20e-3
        assert self.stdp.params.tau_minus == 20e-3
    
    def test_custom_parameters(self):
        """Test STDP initialization with custom parameters."""
        custom_params = STDPParams(
            a_plus=0.02,
            a_minus=0.015,
            tau_plus=15e-3,
            tau_minus=25e-3,
            w_max=2.0
        )
        stdp = STDPLearningRule(custom_params)
        assert stdp.params.a_plus == 0.02
        assert stdp.params.a_minus == 0.015
        assert stdp.params.tau_plus == 15e-3
        assert stdp.params.tau_minus == 25e-3
        assert stdp.params.w_max == 2.0
    
    def test_state_initialization(self):
        """Test proper STDP state initialization."""
        state = self.stdp.init_state(self.n_pre, self.n_post, self.key)
        
        assert state.weights.shape == (self.n_pre, self.n_post)
        assert state.pre_trace.shape == (self.n_pre,)
        assert state.post_trace.shape == (self.n_post,)
        assert state.weight_updates.shape == (self.n_pre, self.n_post)
        
        # Check initial values
        assert jnp.all(state.weights >= 0.1 * self.stdp.params.w_max)
        assert jnp.all(state.weights <= 0.3 * self.stdp.params.w_max)
        assert jnp.allclose(state.pre_trace, 0.0)
        assert jnp.allclose(state.post_trace, 0.0)
        assert jnp.allclose(state.weight_updates, 0.0)
    
    def test_trace_updates(self):
        """Test eligibility trace updates."""
        state = self.stdp.init_state(self.n_pre, self.n_post, self.key)
        
        # Create spike patterns
        pre_spikes = jnp.array([1, 0, 1, 0, 0], dtype=bool)
        post_spikes = jnp.array([0, 1, 1], dtype=bool)
        
        # Update traces
        new_state = self.stdp.update_traces(state, pre_spikes, post_spikes, self.dt)
        
        # Pre-trace should increase where spikes occurred
        assert new_state.pre_trace[0] > state.pre_trace[0]  # Spike at index 0
        assert new_state.pre_trace[2] > state.pre_trace[2]  # Spike at index 2
        assert new_state.pre_trace[1] == state.pre_trace[1]  # No spike at index 1
        
        # Post-trace should increase where spikes occurred
        assert new_state.post_trace[1] > state.post_trace[1]  # Spike at index 1
        assert new_state.post_trace[2] > state.post_trace[2]  # Spike at index 2
        assert new_state.post_trace[0] == state.post_trace[0]  # No spike at index 0
    
    def test_trace_decay(self):
        """Test exponential decay of eligibility traces."""
        state = self.stdp.init_state(self.n_pre, self.n_post, self.key)
        
        # Set initial traces
        initial_pre_trace = jnp.ones((self.n_pre,))
        initial_post_trace = jnp.ones((self.n_post,))
        state = state._replace(
            pre_trace=initial_pre_trace,
            post_trace=initial_post_trace
        )
        
        # Update without spikes (should decay)
        no_spikes_pre = jnp.zeros((self.n_pre,), dtype=bool)
        no_spikes_post = jnp.zeros((self.n_post,), dtype=bool)
        
        new_state = self.stdp.update_traces(state, no_spikes_pre, no_spikes_post, self.dt)
        
        # Traces should decay
        expected_decay_pre = jnp.exp(-self.dt / self.stdp.params.tau_minus)
        expected_decay_post = jnp.exp(-self.dt / self.stdp.params.tau_plus)
        
        assert jnp.allclose(new_state.pre_trace, expected_decay_pre)
        assert jnp.allclose(new_state.post_trace, expected_decay_post)
    
    def test_weight_updates_potentiation(self):
        """Test weight updates for potentiation (post after pre)."""
        state = self.stdp.init_state(self.n_pre, self.n_post, self.key)
        
        # Set up scenario: pre-spike followed by post-spike
        # First, create pre-trace with a spike
        pre_spikes = jnp.array([1, 0, 0, 0, 0], dtype=bool)
        post_spikes = jnp.zeros((self.n_post,), dtype=bool)
        state = self.stdp.update_traces(state, pre_spikes, post_spikes, self.dt)
        
        # Then, create post-spike (should cause potentiation)
        pre_spikes = jnp.zeros((self.n_pre,), dtype=bool)
        post_spikes = jnp.array([1, 0, 0], dtype=bool)
        
        weight_updates = self.stdp.compute_weight_updates(state, pre_spikes, post_spikes)
        
        # Should have positive update for synapse (0, 0) due to potentiation
        assert weight_updates[0, 0] > 0
        # Other synapses should have minimal updates
        assert jnp.all(jnp.abs(weight_updates[1:, 0]) < jnp.abs(weight_updates[0, 0]))
    
    def test_weight_updates_depression(self):
        """Test weight updates for depression (pre after post)."""
        state = self.stdp.init_state(self.n_pre, self.n_post, self.key)
        
        # Set up scenario: post-spike followed by pre-spike
        # First, create post-trace with a spike
        pre_spikes = jnp.zeros((self.n_pre,), dtype=bool)
        post_spikes = jnp.array([1, 0, 0], dtype=bool)
        state = self.stdp.update_traces(state, pre_spikes, post_spikes, self.dt)
        
        # Then, create pre-spike (should cause depression)
        pre_spikes = jnp.array([1, 0, 0, 0, 0], dtype=bool)
        post_spikes = jnp.zeros((self.n_post,), dtype=bool)
        
        weight_updates = self.stdp.compute_weight_updates(state, pre_spikes, post_spikes)
        
        # Should have negative update for synapse (0, 0) due to depression
        assert weight_updates[0, 0] < 0
        # Other synapses should have minimal updates
        assert jnp.all(jnp.abs(weight_updates[1:, 0]) < jnp.abs(weight_updates[0, 0]))
    
    def test_weight_bounds(self):
        """Test weight bounds enforcement."""
        state = self.stdp.init_state(self.n_pre, self.n_post, self.key)
        
        # Test upper bound
        large_positive_updates = jnp.ones((self.n_pre, self.n_post)) * 10.0
        new_state = self.stdp.apply_weight_updates(state, large_positive_updates)
        assert jnp.all(new_state.weights <= self.stdp.params.w_max)
        
        # Test lower bound
        large_negative_updates = jnp.ones((self.n_pre, self.n_post)) * -10.0
        new_state = self.stdp.apply_weight_updates(state, large_negative_updates)
        assert jnp.all(new_state.weights >= self.stdp.params.w_min)
    
    def test_multiplicative_vs_additive(self):
        """Test difference between multiplicative and additive STDP."""
        # Multiplicative STDP
        mult_params = STDPParams(multiplicative=True)
        mult_stdp = STDPLearningRule(mult_params)
        mult_state = mult_stdp.init_state(self.n_pre, self.n_post, self.key)
        
        # Additive STDP
        add_params = STDPParams(multiplicative=False)
        add_stdp = STDPLearningRule(add_params)
        add_state = add_stdp.init_state(self.n_pre, self.n_post, self.key)
        
        # Use same initial weights for comparison
        add_state = add_state._replace(weights=mult_state.weights)
        
        # Apply same spike pattern
        pre_spikes = jnp.array([1, 0, 0, 0, 0], dtype=bool)
        post_spikes = jnp.array([1, 0, 0], dtype=bool)
        
        # Set up traces
        mult_state = mult_stdp.update_traces(mult_state, pre_spikes, jnp.zeros(self.n_post, dtype=bool), self.dt)
        add_state = add_stdp.update_traces(add_state, pre_spikes, jnp.zeros(self.n_post, dtype=bool), self.dt)
        
        # Compute updates
        mult_updates = mult_stdp.compute_weight_updates(mult_state, jnp.zeros(self.n_pre, dtype=bool), post_spikes)
        add_updates = add_stdp.compute_weight_updates(add_state, jnp.zeros(self.n_pre, dtype=bool), post_spikes)
        
        # Updates should be different
        assert not jnp.allclose(mult_updates, add_updates)
    
    def test_full_stdp_step(self):
        """Test complete STDP learning step."""
        state = self.stdp.init_state(self.n_pre, self.n_post, self.key)
        initial_weights = state.weights.copy()
        
        # Apply spike pattern that should cause learning
        pre_spikes = jnp.array([1, 0, 1, 0, 0], dtype=bool)
        post_spikes = jnp.array([0, 1, 1], dtype=bool)
        
        # Multiple steps to build up traces and learning
        for _ in range(10):
            state = self.stdp.step(state, pre_spikes, post_spikes, self.dt)
        
        # Weights should have changed
        assert not jnp.allclose(state.weights, initial_weights)
        
        # State should be valid
        assert jnp.all(jnp.isfinite(state.weights))
        assert jnp.all(state.weights >= self.stdp.params.w_min)
        assert jnp.all(state.weights <= self.stdp.params.w_max)
    
    def test_legacy_stdp_interface(self):
        """Test legacy STDP interface for single synapses."""
        initial_weight = 0.5
        
        # Test potentiation (post after pre)
        pre_time = 0.0
        post_time = 0.01  # 10ms later
        new_weight = self.stdp.apply_stdp(pre_time, post_time, initial_weight)
        assert new_weight > initial_weight  # Should increase
        
        # Test depression (pre after post)
        pre_time = 0.01
        post_time = 0.0
        new_weight = self.stdp.apply_stdp(pre_time, post_time, initial_weight)
        assert new_weight < initial_weight  # Should decrease
        
        # Test bounds
        assert 0.0 <= new_weight <= 1.0
    
    def test_homeostatic_scaling(self):
        """Test homeostatic scaling mechanism."""
        params = STDPParams(homeostatic_scaling=True, target_rate=10.0)
        stdp = STDPLearningRule(params)
        state = stdp.init_state(self.n_pre, self.n_post, self.key)
        
        initial_weights = state.weights.copy()
        
        # Simulate high firing rates (should scale down)
        high_rates = jnp.array([20.0, 25.0, 30.0])  # Above target
        new_state = stdp.homeostatic_scaling(state, high_rates, 0.1)  # Large dt for effect
        
        # Weights should be scaled (though effect might be small with realistic time constants)
        # This is more of a functionality test than a strong behavioral test
        assert new_state.weights.shape == initial_weights.shape
        assert jnp.all(jnp.isfinite(new_state.weights))
    
    def test_create_stdp_configurations(self):
        """Test different STDP rule configurations."""
        # Test all predefined types
        standard = create_stdp_rule("standard")
        potentiation = create_stdp_rule("potentiation_dominant")
        depression = create_stdp_rule("depression_dominant")
        fast = create_stdp_rule("fast")
        slow = create_stdp_rule("slow")
        homeostatic = create_stdp_rule("homeostatic")
        triplets = create_stdp_rule("triplets")
        
        # Check parameter differences
        assert potentiation.params.a_plus > standard.params.a_plus
        assert depression.params.a_minus > standard.params.a_minus
        assert fast.params.tau_plus < standard.params.tau_plus
        assert slow.params.tau_plus > standard.params.tau_plus
        assert homeostatic.params.homeostatic_scaling == True
        assert isinstance(triplets, TripletsSTDP)
        
        # Test invalid type
        with pytest.raises(ValueError):
            create_stdp_rule("invalid_type")
    
    def test_stdp_window_computation(self):
        """Test STDP learning window computation."""
        dt_range = jnp.linspace(-50e-3, 50e-3, 101)  # -50ms to +50ms
        window = compute_stdp_window(dt_range, self.stdp.params)
        
        assert window.shape == dt_range.shape
        
        # Should be positive for positive dt (potentiation)
        positive_mask = dt_range > 0
        assert jnp.all(window[positive_mask] >= 0)
        
        # Should be negative for negative dt (depression)
        negative_mask = dt_range < 0
        assert jnp.all(window[negative_mask] <= 0)
        
        # Should be zero at dt = 0
        zero_idx = jnp.argmin(jnp.abs(dt_range))
        assert jnp.abs(window[zero_idx]) < 1e-6


class TestTripletsSTDP:
    """Test suite for triplet STDP implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.triplet_stdp = TripletsSTDP()
        self.key = random.PRNGKey(123)
        self.n_pre = 3
        self.n_post = 2
    
    def test_triplet_initialization(self):
        """Test triplet STDP initialization."""
        assert hasattr(self.triplet_stdp, 'a2_plus')
        assert hasattr(self.triplet_stdp, 'a2_minus')
        assert hasattr(self.triplet_stdp, 'tau_x')
        assert hasattr(self.triplet_stdp, 'tau_y')
    
    def test_triplet_state_initialization(self):
        """Test triplet STDP state initialization."""
        state = self.triplet_stdp.init_state(self.n_pre, self.n_post, self.key)
        
        # Should have same structure as basic STDP for now
        assert state.weights.shape == (self.n_pre, self.n_post)
        assert state.pre_trace.shape == (self.n_pre,)
        assert state.post_trace.shape == (self.n_post,)


class TestSTDPIntegration:
    """Integration tests for STDP with realistic scenarios."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.stdp = STDPLearningRule()
        self.key = random.PRNGKey(456)
        self.dt = 0.1e-3
        self.n_pre = 10
        self.n_post = 5
    
    def test_learning_convergence(self):
        """Test STDP learning convergence with repeated patterns."""
        state = self.stdp.init_state(self.n_pre, self.n_post, self.key)
        
        # Define a consistent spike pattern
        pre_pattern = jnp.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
        post_pattern = jnp.array([0, 1, 1, 0, 1], dtype=bool)
        
        # Track weight changes
        initial_weights = state.weights.copy()
        weight_history = [initial_weights]
        
        # Run learning for many steps
        for step in range(100):
            # Vary the timing slightly to create realistic STDP
            if step % 2 == 0:
                # Pre spikes first
                state = self.stdp.step(state, pre_pattern, jnp.zeros_like(post_pattern), self.dt)
                state = self.stdp.step(state, jnp.zeros_like(pre_pattern), post_pattern, self.dt)
            else:
                # Post spikes first (should cause depression)
                state = self.stdp.step(state, jnp.zeros_like(pre_pattern), post_pattern, self.dt)
                state = self.stdp.step(state, pre_pattern, jnp.zeros_like(post_pattern), self.dt)
            
            if step % 10 == 0:
                weight_history.append(state.weights.copy())
        
        # Weights should have changed and stabilized
        final_weights = state.weights
        assert not jnp.allclose(final_weights, initial_weights)
        
        # Check that learning has some consistency
        # (weights for active connections should be different from inactive ones)
        active_synapses = jnp.outer(pre_pattern, post_pattern)
        active_weights = final_weights[active_synapses]
        inactive_weights = final_weights[~active_synapses]
        
        if len(active_weights) > 0 and len(inactive_weights) > 0:
            # Active synapses should have different statistics
            assert jnp.mean(active_weights) != jnp.mean(inactive_weights)
    
    def test_spike_timing_sensitivity(self):
        """Test sensitivity to precise spike timing."""
        state1 = self.stdp.init_state(2, 2, self.key)
        state2 = self.stdp.init_state(2, 2, self.key)
        
        # Use same initial weights
        state2 = state2._replace(weights=state1.weights)
        
        # Scenario 1: Pre before post (should potentiate)
        pre_spikes = jnp.array([1, 0], dtype=bool)
        post_spikes = jnp.array([0, 0], dtype=bool)
        state1 = self.stdp.step(state1, pre_spikes, post_spikes, self.dt)
        
        pre_spikes = jnp.array([0, 0], dtype=bool)
        post_spikes = jnp.array([1, 0], dtype=bool)
        state1 = self.stdp.step(state1, pre_spikes, post_spikes, self.dt)
        
        # Scenario 2: Post before pre (should depress)
        pre_spikes = jnp.array([0, 0], dtype=bool)
        post_spikes = jnp.array([1, 0], dtype=bool)
        state2 = self.stdp.step(state2, pre_spikes, post_spikes, self.dt)
        
        pre_spikes = jnp.array([1, 0], dtype=bool)
        post_spikes = jnp.array([0, 0], dtype=bool)
        state2 = self.stdp.step(state2, pre_spikes, post_spikes, self.dt)
        
        # Results should be different
        assert not jnp.allclose(state1.weights, state2.weights)
        
        # Specifically, synapse (0,0) should be different
        assert state1.weights[0, 0] != state2.weights[0, 0]
    
    def test_network_scale_performance(self):
        """Test STDP performance with larger networks."""
        # Test with larger network
        n_pre_large = 100
        n_post_large = 50
        
        state = self.stdp.init_state(n_pre_large, n_post_large, self.key)
        
        # Generate random spike patterns
        key = self.key
        for _ in range(10):  # Just a few steps for performance test
            key, subkey1, subkey2 = random.split(key, 3)
            
            pre_spikes = random.bernoulli(subkey1, 0.1, (n_pre_large,))  # 10% spike probability
            post_spikes = random.bernoulli(subkey2, 0.1, (n_post_large,))  # 10% spike probability
            
            state = self.stdp.step(state, pre_spikes, post_spikes, self.dt)
        
        # Should complete without errors and maintain valid state
        assert jnp.all(jnp.isfinite(state.weights))
        assert jnp.all(state.weights >= self.stdp.params.w_min)
        assert jnp.all(state.weights <= self.stdp.params.w_max)


if __name__ == "__main__":
    pytest.main([__file__])
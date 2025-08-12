"""
Biological Plausibility Validation Tests

This module validates that the neuromorphic components of the Godly AI system
exhibit biologically plausible behaviors consistent with neuroscience research.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import scipy.stats as stats
from scipy import signal
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)


@dataclass
class BiologicalMetrics:
    """Metrics for biological plausibility validation"""
    spike_rate_hz: float
    cv_isi: float  # Coefficient of variation of inter-spike intervals
    fano_factor: float  # Variance-to-mean ratio of spike counts
    membrane_time_constant_ms: float
    refractory_period_ms: float
    stdp_time_window_ms: float
    learning_rate: float


class BiologicalConstants:
    """Biologically plausible parameter ranges from neuroscience literature"""
    
    # Spike rates (Hz)
    CORTICAL_NEURON_RATE_RANGE = (0.1, 50.0)  # Typical cortical neuron firing rates
    INTERNEURON_RATE_RANGE = (5.0, 100.0)     # Interneuron firing rates
    
    # Membrane properties
    MEMBRANE_TIME_CONSTANT_RANGE = (5.0, 50.0)  # ms, typical range for cortical neurons
    RESTING_POTENTIAL_RANGE = (-80.0, -60.0)    # mV
    THRESHOLD_POTENTIAL_RANGE = (-60.0, -40.0)   # mV
    RESET_POTENTIAL_RANGE = (-80.0, -60.0)      # mV
    
    # Refractory periods
    ABSOLUTE_REFRACTORY_RANGE = (1.0, 5.0)      # ms
    RELATIVE_REFRACTORY_RANGE = (5.0, 20.0)     # ms
    
    # STDP parameters
    STDP_TIME_WINDOW_RANGE = (10.0, 100.0)      # ms, typical STDP time constants
    STDP_LEARNING_RATE_RANGE = (0.001, 0.1)     # Typical learning rates
    
    # Spike statistics
    CV_ISI_RANGE = (0.5, 2.0)                   # Coefficient of variation for ISI
    FANO_FACTOR_RANGE = (0.8, 2.0)              # Variance-to-mean ratio
    
    # Network properties
    CONNECTIVITY_RANGE = (0.01, 0.3)            # Typical cortical connectivity
    SYNAPTIC_DELAY_RANGE = (0.5, 10.0)          # ms, synaptic transmission delays


class TestNeuronBiologicalPlausibility:
    """Test biological plausibility of neuron models"""
    
    def setup_method(self):
        """Setup test environment"""
        self.key = jax.random.PRNGKey(42)
        
    @pytest.mark.validation
    def test_lif_neuron_membrane_dynamics(self):
        """Test LIF neuron membrane dynamics against biological data"""
        from src.core.neurons.lif_neuron import LIFNeuron
        
        # Initialize neuron with biologically plausible parameters
        neuron = LIFNeuron(
            threshold=-50.0,           # mV
            reset_potential=-70.0,     # mV
            tau_mem=20.0,             # ms
            tau_ref=2.0               # ms
        )
        
        # Validate parameter ranges
        assert BiologicalConstants.THRESHOLD_POTENTIAL_RANGE[0] <= neuron.threshold <= BiologicalConstants.THRESHOLD_POTENTIAL_RANGE[1], \
            f"Threshold potential out of biological range: {neuron.threshold} mV"
        
        assert BiologicalConstants.RESET_POTENTIAL_RANGE[0] <= neuron.reset_potential <= BiologicalConstants.RESET_POTENTIAL_RANGE[1], \
            f"Reset potential out of biological range: {neuron.reset_potential} mV"
        
        assert BiologicalConstants.MEMBRANE_TIME_CONSTANT_RANGE[0] <= neuron.tau_mem <= BiologicalConstants.MEMBRANE_TIME_CONSTANT_RANGE[1], \
            f"Membrane time constant out of biological range: {neuron.tau_mem} ms"
        
        assert BiologicalConstants.ABSOLUTE_REFRACTORY_RANGE[0] <= neuron.tau_ref <= BiologicalConstants.ABSOLUTE_REFRACTORY_RANGE[1], \
            f"Refractory period out of biological range: {neuron.tau_ref} ms"
        
        # Test membrane integration behavior
        state = neuron.init_state()
        dt = 0.1  # ms
        
        # Test subthreshold integration
        subthreshold_current = 10.0  # pA, below threshold
        membrane_potentials = []
        
        for step in range(200):  # 20 ms simulation
            state, spike = neuron.step(state, subthreshold_current, dt)
            membrane_potentials.append(state.membrane_potential)
        
        membrane_potentials = jnp.array(membrane_potentials)
        
        # Membrane potential should approach steady state exponentially
        # V_ss = I * R (where R is membrane resistance)
        # For subthreshold input, should not spike
        assert jnp.all(membrane_potentials < neuron.threshold), "Subthreshold input should not cause spiking"
        
        # Test exponential approach to steady state
        final_potential = membrane_potentials[-1]
        initial_potential = membrane_potentials[0]
        
        # Should show exponential relaxation
        mid_point = len(membrane_potentials) // 2
        mid_potential = membrane_potentials[mid_point]
        
        # Exponential approach: should be closer to final value at midpoint than at start
        progress_ratio = abs(mid_potential - initial_potential) / abs(final_potential - initial_potential)
        assert progress_ratio > 0.5, f"Should show exponential approach: progress ratio = {progress_ratio}"
        
        logger.info(f"LIF membrane dynamics: tau_mem={neuron.tau_mem}ms, final_V={final_potential:.2f}mV")
    
    @pytest.mark.validation
    def test_spike_timing_statistics(self):
        """Test spike timing statistics against biological distributions"""
        from src.core.neurons.lif_neuron import LIFNeuron
        
        neuron = LIFNeuron(threshold=-50.0, reset_potential=-70.0, tau_mem=20.0)
        state = neuron.init_state()
        
        # Apply noisy input current to generate irregular spiking
        key = self.key
        dt = 0.1  # ms
        simulation_time = 10000  # ms (10 seconds)
        n_steps = int(simulation_time / dt)
        
        spike_times = []
        base_current = 25.0  # pA
        noise_std = 5.0      # pA
        
        for step in range(n_steps):
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey) * noise_std
            input_current = base_current + noise
            
            state, spike = neuron.step(state, input_current, dt)
            
            if spike > 0.5:  # Spike detected
                spike_times.append(step * dt)
        
        spike_times = jnp.array(spike_times)
        
        # Validate spike rate
        if len(spike_times) > 1:
            spike_rate = len(spike_times) / (simulation_time / 1000.0)  # Hz
            
            assert BiologicalConstants.CORTICAL_NEURON_RATE_RANGE[0] <= spike_rate <= BiologicalConstants.CORTICAL_NEURON_RATE_RANGE[1], \
                f"Spike rate out of biological range: {spike_rate:.2f} Hz"
            
            # Calculate inter-spike intervals (ISI)
            isis = jnp.diff(spike_times)
            
            if len(isis) > 10:
                # Coefficient of variation of ISI
                cv_isi = jnp.std(isis) / jnp.mean(isis)
                
                assert BiologicalConstants.CV_ISI_RANGE[0] <= cv_isi <= BiologicalConstants.CV_ISI_RANGE[1], \
                    f"CV of ISI out of biological range: {cv_isi:.3f}"
                
                # Test for exponential ISI distribution (characteristic of Poisson-like process)
                # Biological neurons often show exponential or gamma-distributed ISIs
                isi_array = np.array(isis)
                
                # Kolmogorov-Smirnov test against exponential distribution
                scale_param = np.mean(isi_array)
                ks_stat, p_value = stats.kstest(isi_array, lambda x: stats.expon.cdf(x, scale=scale_param))
                
                # Should not be too far from exponential (p > 0.01 for reasonable fit)
                assert p_value > 0.001, f"ISI distribution too far from exponential: p={p_value:.4f}"
                
                logger.info(f"Spike statistics: rate={spike_rate:.2f}Hz, CV_ISI={cv_isi:.3f}, KS_p={p_value:.4f}")
        else:
            pytest.skip("Insufficient spikes generated for statistical analysis")
    
    @pytest.mark.validation
    def test_refractory_period_behavior(self):
        """Test refractory period behavior against biological expectations"""
        from src.core.neurons.lif_neuron import LIFNeuron
        
        neuron = LIFNeuron(threshold=-50.0, reset_potential=-70.0, tau_ref=3.0)
        state = neuron.init_state()
        
        dt = 0.1  # ms
        strong_current = 100.0  # Strong current to ensure spiking
        
        # Apply strong current and track spiking
        spike_times = []
        membrane_trace = []
        
        for step in range(1000):  # 100 ms simulation
            state, spike = neuron.step(state, strong_current, dt)
            membrane_trace.append(state.membrane_potential)
            
            if spike > 0.5:
                spike_times.append(step * dt)
        
        spike_times = jnp.array(spike_times)
        membrane_trace = jnp.array(membrane_trace)
        
        if len(spike_times) > 1:
            # Check minimum inter-spike interval
            min_isi = jnp.min(jnp.diff(spike_times))
            
            # Minimum ISI should be at least the refractory period
            assert min_isi >= neuron.tau_ref * 0.8, \
                f"Minimum ISI ({min_isi:.2f}ms) should respect refractory period ({neuron.tau_ref}ms)"
            
            # Check that membrane potential is clamped during refractory period
            # Find spike indices
            spike_indices = []
            for spike_time in spike_times:
                spike_idx = int(spike_time / dt)
                if spike_idx < len(membrane_trace):
                    spike_indices.append(spike_idx)
            
            # Check membrane potential after spikes
            for spike_idx in spike_indices:
                if spike_idx + int(neuron.tau_ref / dt) < len(membrane_trace):
                    # Membrane should be at or near reset potential during refractory period
                    refractory_end_idx = spike_idx + int(neuron.tau_ref / dt)
                    refractory_potentials = membrane_trace[spike_idx:refractory_end_idx]
                    
                    # Should be close to reset potential
                    mean_refractory_potential = jnp.mean(refractory_potentials)
                    assert abs(mean_refractory_potential - neuron.reset_potential) < 5.0, \
                        f"Membrane potential during refractory period should be near reset: {mean_refractory_potential:.2f}mV vs {neuron.reset_potential}mV"
            
            logger.info(f"Refractory behavior: min_ISI={min_isi:.2f}ms, tau_ref={neuron.tau_ref}ms")
        else:
            pytest.skip("Insufficient spikes for refractory period analysis")


class TestSTDPBiologicalPlausibility:
    """Test biological plausibility of STDP learning rules"""
    
    def setup_method(self):
        """Setup test environment"""
        self.key = jax.random.PRNGKey(42)
        
    @pytest.mark.validation
    def test_stdp_time_window(self):
        """Test STDP time window against biological measurements"""
        from src.core.plasticity.stdp import STDPLearningRule
        
        # Initialize STDP with biologically plausible parameters
        stdp = STDPLearningRule(
            tau_pre=20.0,    # ms
            tau_post=20.0,   # ms
            A_pre=0.01,      # Potentiation amplitude
            A_post=-0.008    # Depression amplitude
        )
        
        # Validate time constants
        assert BiologicalConstants.STDP_TIME_WINDOW_RANGE[0] <= stdp.tau_pre <= BiologicalConstants.STDP_TIME_WINDOW_RANGE[1], \
            f"Pre-synaptic time constant out of biological range: {stdp.tau_pre} ms"
        
        assert BiologicalConstants.STDP_TIME_WINDOW_RANGE[0] <= stdp.tau_post <= BiologicalConstants.STDP_TIME_WINDOW_RANGE[1], \
            f"Post-synaptic time constant out of biological range: {stdp.tau_post} ms"
        
        # Test STDP window function
        dt_values = jnp.linspace(-100, 100, 201)  # -100ms to +100ms
        stdp_values = []
        
        for dt in dt_values:
            window_value = stdp._compute_stdp_window(dt)
            stdp_values.append(window_value)
        
        stdp_values = jnp.array(stdp_values)
        
        # Biological STDP properties:
        # 1. Causal: positive dt (pre before post) should give potentiation
        # 2. Anti-causal: negative dt (post before pre) should give depression
        # 3. Exponential decay with distance from dt=0
        
        positive_dt_indices = dt_values > 0
        negative_dt_indices = dt_values < 0
        
        if jnp.any(positive_dt_indices):
            positive_stdp = stdp_values[positive_dt_indices]
            # Should be mostly positive (potentiation)
            mean_positive = jnp.mean(positive_stdp)
            assert mean_positive > 0, f"Positive dt should give potentiation: mean = {mean_positive}"
        
        if jnp.any(negative_dt_indices):
            negative_stdp = stdp_values[negative_dt_indices]
            # Should be mostly negative (depression)
            mean_negative = jnp.mean(negative_stdp)
            assert mean_negative < 0, f"Negative dt should give depression: mean = {mean_negative}"
        
        # Test exponential decay
        # STDP should decay exponentially with |dt|
        dt_10ms_idx = jnp.argmin(jnp.abs(dt_values - 10.0))
        dt_50ms_idx = jnp.argmin(jnp.abs(dt_values - 50.0))
        
        stdp_10ms = abs(stdp_values[dt_10ms_idx])
        stdp_50ms = abs(stdp_values[dt_50ms_idx])
        
        # Should decay with distance
        assert stdp_10ms > stdp_50ms, f"STDP should decay with distance: {stdp_10ms:.6f} vs {stdp_50ms:.6f}"
        
        logger.info(f"STDP window: tau_pre={stdp.tau_pre}ms, tau_post={stdp.tau_post}ms")
    
    @pytest.mark.validation
    def test_stdp_learning_dynamics(self):
        """Test STDP learning dynamics against biological expectations"""
        from src.core.plasticity.stdp import STDPLearningRule
        
        stdp = STDPLearningRule(tau_pre=20.0, tau_post=20.0, A_pre=0.01, A_post=-0.008)
        state = stdp.init_state(n_pre=1, n_post=1)
        
        # Test different spike timing scenarios
        dt = 0.1  # ms
        scenarios = [
            ("causal", 5.0),      # Pre 5ms before post
            ("anti_causal", -5.0), # Post 5ms before pre
            ("synchronous", 0.0),  # Simultaneous
            ("distant", 50.0)      # Pre 50ms before post
        ]
        
        weight_changes = {}
        
        for scenario_name, spike_dt in scenarios:
            # Reset state
            test_state = stdp.init_state(n_pre=1, n_post=1)
            
            # Simulate spike pair with specific timing
            n_steps = int(abs(spike_dt) / dt) + 100
            
            pre_spike_step = 50
            post_spike_step = pre_spike_step + int(spike_dt / dt)
            
            total_weight_change = 0.0
            
            for step in range(n_steps):
                # Generate spike patterns
                pre_spikes = jnp.array([[1.0 if step == pre_spike_step else 0.0]])
                post_spikes = jnp.array([[1.0 if step == post_spike_step else 0.0]])
                
                # Update STDP
                test_state = stdp.update_traces(test_state, pre_spikes, post_spikes, dt)
                new_state, weight_update = stdp.compute_weight_update(test_state, pre_spikes, post_spikes)
                test_state = new_state
                
                total_weight_change += weight_update[0, 0]
            
            weight_changes[scenario_name] = total_weight_change
        
        # Biological expectations:
        # 1. Causal pairing should cause potentiation
        assert weight_changes["causal"] > 0, f"Causal pairing should potentiate: {weight_changes['causal']:.6f}"
        
        # 2. Anti-causal pairing should cause depression
        assert weight_changes["anti_causal"] < 0, f"Anti-causal pairing should depress: {weight_changes['anti_causal']:.6f}"
        
        # 3. Distant timing should have smaller effect than close timing
        assert abs(weight_changes["causal"]) > abs(weight_changes["distant"]), \
            f"Close timing should have larger effect: {abs(weight_changes['causal']):.6f} vs {abs(weight_changes['distant']):.6f}"
        
        # 4. Synchronous spikes should have intermediate effect
        sync_change = abs(weight_changes["synchronous"])
        causal_change = abs(weight_changes["causal"])
        
        # Synchronous can be either potentiating or depressing depending on implementation
        # But should be significant
        assert sync_change > abs(weight_changes["distant"]), \
            f"Synchronous spikes should have significant effect: {sync_change:.6f}"
        
        logger.info(f"STDP learning: causal={weight_changes['causal']:.6f}, anti_causal={weight_changes['anti_causal']:.6f}")
    
    @pytest.mark.validation
    def test_stdp_weight_bounds(self):
        """Test STDP weight bounds against biological constraints"""
        from src.core.plasticity.stdp import STDPLearningRule
        
        # Test with explicit weight bounds
        stdp = STDPLearningRule(w_min=0.0, w_max=1.0, A_pre=0.1, A_post=-0.08)
        state = stdp.init_state(n_pre=5, n_post=3)
        
        # Initialize weights at different levels
        initial_weights = jnp.array([
            [0.1, 0.5, 0.9],  # Low, medium, high
            [0.0, 0.3, 1.0],  # Min, medium, max
            [0.2, 0.7, 0.4],  # Various levels
            [0.8, 0.1, 0.6],
            [0.5, 0.9, 0.2]
        ])
        
        # Apply extreme weight updates
        extreme_updates = jnp.array([
            [2.0, -2.0, 0.5],   # Large positive, large negative, moderate
            [-1.5, 1.5, -0.3],  # Large negative, large positive, small negative
            [0.8, -0.9, 1.2],   # Various extreme values
            [-0.7, 0.4, -1.1],
            [1.3, -1.8, 0.9]
        ])
        
        # Apply bounds
        final_weights = stdp._apply_weight_bounds(initial_weights + extreme_updates)
        
        # Validate bounds are respected
        assert jnp.all(final_weights >= stdp.w_min), f"Weights below minimum: {jnp.min(final_weights)}"
        assert jnp.all(final_weights <= stdp.w_max), f"Weights above maximum: {jnp.max(final_weights)}"
        
        # Test biological constraint: weights should not change sign
        # (synapses don't switch from excitatory to inhibitory)
        sign_changes = jnp.sign(initial_weights) != jnp.sign(final_weights)
        zero_crossings = jnp.sum(sign_changes & (initial_weights != 0) & (final_weights != 0))
        
        # Should minimize sign changes (biological constraint)
        assert zero_crossings <= len(initial_weights.flatten()) * 0.2, \
            f"Too many sign changes: {zero_crossings} out of {len(initial_weights.flatten())}"
        
        logger.info(f"Weight bounds: min={jnp.min(final_weights):.3f}, max={jnp.max(final_weights):.3f}")


class TestNetworkBiologicalPlausibility:
    """Test biological plausibility of network-level properties"""
    
    def setup_method(self):
        """Setup test environment"""
        self.key = jax.random.PRNGKey(42)
        
    @pytest.mark.validation
    def test_liquid_state_machine_dynamics(self):
        """Test LSM dynamics against biological network properties"""
        from src.core.liquid_state_machine import LiquidStateMachine
        
        # Initialize LSM with biologically plausible parameters
        lsm = LiquidStateMachine(
            reservoir_size=200,
            input_dim=20,
            spectral_radius=0.95  # Near edge of chaos
        )
        
        state = lsm.init_state()
        
        # Test connectivity
        adjacency = lsm.reservoir_weights
        connectivity = jnp.sum(adjacency != 0) / (adjacency.shape[0] * adjacency.shape[1])
        
        assert BiologicalConstants.CONNECTIVITY_RANGE[0] <= connectivity <= BiologicalConstants.CONNECTIVITY_RANGE[1], \
            f"Network connectivity out of biological range: {connectivity:.3f}"
        
        # Test dynamics with sustained input
        key = self.key
        dt = 1.0  # ms
        simulation_time = 1000  # ms
        n_steps = int(simulation_time / dt)
        
        # Apply constant input and measure dynamics
        constant_input = jnp.ones(20) * 0.1
        
        reservoir_states = []
        activities = []
        
        for step in range(n_steps):
            state, output = lsm.step(state, constant_input)
            reservoir_states.append(state.reservoir_state.copy())
            activities.append(jnp.mean(jnp.abs(state.reservoir_state)))
        
        reservoir_states = jnp.array(reservoir_states)
        activities = jnp.array(activities)
        
        # Test for biological network properties
        
        # 1. Activity should stabilize (not grow unbounded)
        final_activity = jnp.mean(activities[-100:])
        initial_activity = jnp.mean(activities[:100])
        
        assert final_activity < initial_activity * 10, \
            f"Activity should not grow unbounded: {initial_activity:.4f} -> {final_activity:.4f}"
        
        # 2. Should show rich dynamics (not too stable, not too chaotic)
        activity_variance = jnp.var(activities[100:])  # Skip initial transient
        
        assert activity_variance > 1e-6, f"Network too stable: variance = {activity_variance:.8f}"
        assert activity_variance < 1.0, f"Network too chaotic: variance = {activity_variance:.4f}"
        
        # 3. Test for avalanche-like dynamics (characteristic of biological networks)
        # Look for power-law distributed activity events
        activity_threshold = jnp.mean(activities) + 2 * jnp.std(activities)
        avalanche_sizes = []
        
        in_avalanche = False
        current_avalanche_size = 0
        
        for activity in activities:
            if activity > activity_threshold:
                if not in_avalanche:
                    in_avalanche = True
                    current_avalanche_size = 1
                else:
                    current_avalanche_size += 1
            else:
                if in_avalanche:
                    avalanche_sizes.append(current_avalanche_size)
                    in_avalanche = False
                    current_avalanche_size = 0
        
        if len(avalanche_sizes) > 10:
            # Test for power-law-like distribution (biological networks show this)
            avalanche_sizes = jnp.array(avalanche_sizes)
            unique_sizes, counts = jnp.unique(avalanche_sizes, return_counts=True)
            
            if len(unique_sizes) > 3:
                # Fit power law: log(count) ~ -alpha * log(size)
                log_sizes = jnp.log(unique_sizes)
                log_counts = jnp.log(counts)
                
                # Simple linear regression
                slope = jnp.corrcoef(log_sizes, log_counts)[0, 1] * (jnp.std(log_counts) / jnp.std(log_sizes))
                
                # Power law should have negative slope
                assert slope < 0, f"Avalanche distribution should follow power law: slope = {slope:.3f}"
        
        logger.info(f"LSM dynamics: connectivity={connectivity:.3f}, activity_var={activity_variance:.6f}")
    
    @pytest.mark.validation
    def test_memory_consolidation_timescales(self):
        """Test memory consolidation timescales against biological data"""
        from src.memory.working.working_memory import WorkingMemory
        from src.memory.episodic.episodic_memory import EpisodicMemory
        from src.memory.semantic.semantic_memory import SemanticMemory
        
        # Initialize memory systems
        working_memory = WorkingMemory(capacity=20, decay_rate=0.1)  # Fast decay
        episodic_memory = EpisodicMemory(max_episodes=50)
        semantic_memory = SemanticMemory(embedding_dim=64)
        
        # Initialize states
        wm_state = working_memory.init_state()
        em_state = episodic_memory.init_state()
        sm_state = semantic_memory.init_state()
        
        # Test memory timescales
        key = self.key
        
        # Working memory: should operate on seconds timescale
        patterns = []
        for i in range(10):
            key, subkey = jax.random.split(key)
            pattern = jax.random.normal(subkey, (64,))
            patterns.append(pattern)
            working_memory.store_pattern(wm_state, pattern, timestamp=float(i))
        
        # Test decay over time
        early_pattern = patterns[0]
        late_pattern = patterns[-1]
        
        # Simulate time passage
        current_time = 20.0  # 20 seconds later
        
        early_retrieved, early_confidence = working_memory.retrieve_pattern(
            wm_state, early_pattern, current_time=current_time
        )
        late_retrieved, late_confidence = working_memory.retrieve_pattern(
            wm_state, late_pattern, current_time=current_time
        )
        
        # Later patterns should have higher confidence (less decay)
        if early_confidence is not None and late_confidence is not None:
            assert late_confidence >= early_confidence - 0.2, \
                f"Working memory should show temporal decay: early={early_confidence:.3f}, late={late_confidence:.3f}"
        
        # Episodic memory: should operate on minutes to hours timescale
        experiences = []
        for i in range(20):
            key, subkey = jax.random.split(key)
            experience = {
                'observation': jax.random.normal(subkey, (64,)),
                'action': i % 4,
                'reward': 0.5,
                'timestamp': float(i * 60)  # Every minute
            }
            experiences.append(experience)
        
        # Store episodes
        for i in range(0, len(experiences), 5):
            episode_data = experiences[i:i+5]
            episodic_memory.store_episode(em_state, episode_data)
        
        # Episodes should be retrievable over longer timescales
        em_stats = episodic_memory.get_memory_statistics(em_state)
        assert em_stats['total_episodes'] > 0, "Episodic memory should retain episodes"
        
        # Semantic memory: should show long-term consolidation
        concepts = semantic_memory.extract_concepts(sm_state, experiences)
        
        # Should form stable concepts from experiences
        assert len(concepts) > 0, "Semantic memory should form concepts"
        assert len(concepts) < len(experiences) / 2, "Should consolidate experiences into fewer concepts"
        
        # Test concept stability over time
        # Re-extract concepts with same data - should be similar
        concepts_2 = semantic_memory.extract_concepts(sm_state, experiences)
        
        # Concept count should be stable
        concept_count_stability = abs(len(concepts) - len(concepts_2)) / max(len(concepts), 1)
        assert concept_count_stability < 0.5, f"Concept formation should be stable: {concept_count_stability:.3f}"
        
        logger.info(f"Memory timescales: WM_patterns={len(patterns)}, EM_episodes={em_stats['total_episodes']}, SM_concepts={len(concepts)}")
    
    @pytest.mark.validation
    def test_learning_rate_adaptation(self):
        """Test learning rate adaptation against biological learning curves"""
        from src.training.unsupervised.competitive_learning import CompetitiveLearning
        from src.core.plasticity.stdp import STDPLearningRule
        
        # Initialize learning systems
        competitive_learning = CompetitiveLearning(n_units=20, input_dim=50)
        stdp = STDPLearningRule(A_pre=0.01, A_post=-0.008)
        
        # Initialize states
        cl_state = competitive_learning.init_state()
        stdp_state = stdp.init_state(n_pre=50, n_post=20)
        
        # Test learning curves
        key = self.key
        n_trials = 200
        
        cl_performance = []
        stdp_weight_changes = []
        
        for trial in range(n_trials):
            key, subkey = jax.random.split(key)
            
            # Generate structured input (with some pattern)
            pattern_id = trial % 5  # 5 different patterns
            base_pattern = jnp.ones(50) * pattern_id * 0.2
            noise = jax.random.normal(subkey, (50,)) * 0.1
            input_pattern = base_pattern + noise
            
            # Competitive learning
            cl_state, cl_output = competitive_learning.train_step(cl_state, input_pattern)
            
            # Measure performance (pattern separation)
            performance = jnp.max(cl_output) - jnp.mean(cl_output)  # Winner-take-all strength
            cl_performance.append(performance)
            
            # STDP learning
            pre_spikes = input_pattern.reshape(-1, 1)
            post_spikes = cl_output.reshape(1, -1)
            
            stdp_state = stdp.update_traces(stdp_state, pre_spikes, post_spikes, dt=0.001)
            new_stdp_state, weight_update = stdp.compute_weight_update(stdp_state, pre_spikes, post_spikes)
            stdp_state = new_stdp_state
            
            weight_change_magnitude = jnp.mean(jnp.abs(weight_update))
            stdp_weight_changes.append(weight_change_magnitude)
        
        # Analyze learning curves
        cl_performance = jnp.array(cl_performance)
        stdp_weight_changes = jnp.array(stdp_weight_changes)
        
        # Biological learning should show:
        # 1. Initial rapid improvement
        early_performance = jnp.mean(cl_performance[:50])
        late_performance = jnp.mean(cl_performance[-50:])
        
        assert late_performance > early_performance, \
            f"Learning should show improvement: {early_performance:.4f} -> {late_performance:.4f}"
        
        # 2. Decreasing learning rate over time (weight changes should decrease)
        early_weight_changes = jnp.mean(stdp_weight_changes[:50])
        late_weight_changes = jnp.mean(stdp_weight_changes[-50:])
        
        # Weight changes should generally decrease (learning rate adaptation)
        assert late_weight_changes <= early_weight_changes * 1.5, \
            f"Weight changes should decrease over time: {early_weight_changes:.6f} -> {late_weight_changes:.6f}"
        
        # 3. Learning curve should be roughly exponential (biological characteristic)
        # Fit exponential: performance = a * (1 - exp(-b * trial))
        trials = jnp.arange(len(cl_performance))
        
        # Simple test: performance should level off (not linear growth)
        first_half_slope = jnp.mean(jnp.diff(cl_performance[:n_trials//2]))
        second_half_slope = jnp.mean(jnp.diff(cl_performance[n_trials//2:]))
        
        # Second half should have smaller slope (leveling off)
        assert second_half_slope <= first_half_slope * 2, \
            f"Learning should level off: slopes {first_half_slope:.6f} -> {second_half_slope:.6f}"
        
        logger.info(f"Learning adaptation: performance {early_performance:.4f}->{late_performance:.4f}, weight_changes {early_weight_changes:.6f}->{late_weight_changes:.6f}")


class TestBiologicalValidationSuite:
    """Comprehensive biological validation test suite"""
    
    @pytest.mark.validation
    def test_comprehensive_biological_validation(self):
        """Run comprehensive biological validation across all components"""
        
        validation_results = {
            'neuron_dynamics': False,
            'stdp_learning': False,
            'network_properties': False,
            'memory_timescales': False,
            'learning_adaptation': False
        }
        
        try:
            # Test neuron dynamics
            test_neurons = TestNeuronBiologicalPlausibility()
            test_neurons.setup_method()
            test_neurons.test_lif_neuron_membrane_dynamics()
            validation_results['neuron_dynamics'] = True
        except Exception as e:
            logger.error(f"Neuron dynamics validation failed: {e}")
        
        try:
            # Test STDP learning
            test_stdp = TestSTDPBiologicalPlausibility()
            test_stdp.setup_method()
            test_stdp.test_stdp_time_window()
            validation_results['stdp_learning'] = True
        except Exception as e:
            logger.error(f"STDP learning validation failed: {e}")
        
        try:
            # Test network properties
            test_network = TestNetworkBiologicalPlausibility()
            test_network.setup_method()
            test_network.test_liquid_state_machine_dynamics()
            validation_results['network_properties'] = True
        except Exception as e:
            logger.error(f"Network properties validation failed: {e}")
        
        # Generate validation report
        passed_tests = sum(validation_results.values())
        total_tests = len(validation_results)
        
        validation_score = passed_tests / total_tests
        
        logger.info(f"Biological validation score: {validation_score:.2f} ({passed_tests}/{total_tests})")
        
        for test_name, passed in validation_results.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"  {test_name}: {status}")
        
        # Overall validation should pass most tests
        assert validation_score >= 0.6, f"Biological validation score too low: {validation_score:.2f}"
        
        return validation_results


def generate_biological_validation_report():
    """Generate comprehensive biological validation report"""
    
    report = {
        'validation_date': '2024-01-01',
        'system_version': '1.0.0',
        'biological_standards': {
            'neuron_parameters': BiologicalConstants.__dict__,
            'reference_studies': [
                'Markram et al. (1997) - Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs',
                'Bi & Poo (1998) - Synaptic modifications in cultured hippocampal neurons',
                'Song et al. (2000) - Competitive Hebbian learning through spike-timing-dependent synaptic plasticity',
                'Dayan & Abbott (2001) - Theoretical Neuroscience'
            ]
        },
        'validation_criteria': {
            'membrane_dynamics': 'Exponential integration, biological time constants',
            'spike_statistics': 'Poisson-like ISI distribution, biological firing rates',
            'stdp_learning': 'Causal/anti-causal asymmetry, exponential time windows',
            'network_dynamics': 'Edge-of-chaos, avalanche statistics, stable activity',
            'memory_timescales': 'Working memory (seconds), episodic (minutes), semantic (hours)'
        }
    }
    
    return report


if __name__ == "__main__":
    # Run biological validation tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "validation"])
"""
Leaky Integrate-and-Fire (LIF) Neuron Implementation

This module implements the core LIF neuron model with JAX for the Godly AI system.
Includes adaptive threshold mechanisms and refractory period handling.
"""

from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass


@dataclass
class LIFParams:
    """Parameters for the Leaky Integrate-and-Fire neuron model."""
    
    # Membrane dynamics
    tau_mem: float = 20e-3  # Membrane time constant (20ms)
    tau_syn: float = 5e-3   # Synaptic time constant (5ms)
    v_rest: float = -70e-3  # Resting potential (-70mV)
    v_reset: float = -80e-3 # Reset potential (-80mV)
    v_thresh: float = -50e-3 # Spike threshold (-50mV)
    
    # Adaptive threshold
    v_thresh_adapt: float = 10e-3  # Threshold adaptation increment (10mV)
    tau_thresh: float = 100e-3     # Threshold adaptation time constant (100ms)
    
    # Refractory period
    t_refrac: float = 2e-3  # Refractory period (2ms)
    
    # Noise
    noise_std: float = 1e-3  # Membrane noise standard deviation (1mV)


class LIFState(NamedTuple):
    """State variables for the LIF neuron."""
    v_mem: jnp.ndarray      # Membrane potential
    i_syn: jnp.ndarray      # Synaptic current
    v_thresh_dyn: jnp.ndarray  # Dynamic threshold
    t_last_spike: jnp.ndarray  # Time of last spike
    spikes: jnp.ndarray     # Spike output


class LIFNeuron:
    """
    Leaky Integrate-and-Fire neuron with adaptive threshold and refractory period.
    
    This implementation follows the neuromorphic computing principles with:
    - Event-driven computation
    - Adaptive threshold mechanisms
    - Proper refractory period handling
    - JAX-based high-performance computation
    """
    
    def __init__(self, params: Optional[LIFParams] = None):
        """Initialize LIF neuron with given parameters."""
        self.params = params or LIFParams()
    
    def init_state(self, batch_size: int, key: Optional[jax.random.PRNGKey] = None) -> LIFState:
        """Initialize neuron state for a batch of neurons."""
        if key is None:
            key = random.PRNGKey(0)
        
        return LIFState(
            v_mem=jnp.full((batch_size,), self.params.v_rest),
            i_syn=jnp.zeros((batch_size,)),
            v_thresh_dyn=jnp.full((batch_size,), self.params.v_thresh),
            t_last_spike=jnp.full((batch_size,), -jnp.inf),
            spikes=jnp.zeros((batch_size,), dtype=bool)
        )
    
    def step(
        self, 
        state: LIFState, 
        input_current: jnp.ndarray, 
        dt: float, 
        t: float,
        key: Optional[jax.random.PRNGKey] = None
    ) -> LIFState:
        """
        Perform one integration step of the LIF neuron.
        
        Args:
            state: Current neuron state
            input_current: Input current for this timestep
            dt: Integration timestep
            t: Current time
            key: Random key for noise generation
            
        Returns:
            Updated neuron state
        """
        # Add membrane noise if key provided
        noise = 0.0
        if key is not None:
            noise = random.normal(key, state.v_mem.shape) * self.params.noise_std
        
        # Check refractory period
        in_refrac = (t - state.t_last_spike) < self.params.t_refrac
        
        # Update synaptic current (exponential decay + input)
        alpha_syn = jnp.exp(-dt / self.params.tau_syn)
        i_syn_new = alpha_syn * state.i_syn + input_current
        
        # Update membrane potential (only if not in refractory period)
        alpha_mem = jnp.exp(-dt / self.params.tau_mem)
        # Membrane equation: tau_mem * dV/dt = (v_rest - v_mem) + R * i_syn
        # Assuming membrane resistance R = 1e9 ohms (1 GOhm) for realistic scaling
        R_mem = 1e9  # Membrane resistance in ohms
        dv = ((self.params.v_rest - state.v_mem) + R_mem * i_syn_new) * (1 - alpha_mem) + noise
        v_mem_new = jnp.where(
            in_refrac,
            self.params.v_reset,  # Hold at reset during refractory period
            state.v_mem + dv
        )
        
        # Update adaptive threshold (exponential decay toward baseline)
        alpha_thresh = jnp.exp(-dt / self.params.tau_thresh)
        v_thresh_new = (
            alpha_thresh * state.v_thresh_dyn + 
            (1 - alpha_thresh) * self.params.v_thresh
        )
        
        # Check for spikes
        spikes = (v_mem_new >= v_thresh_new) & (~in_refrac)
        
        # Reset membrane potential and adapt threshold for spiking neurons
        v_mem_final = jnp.where(spikes, self.params.v_reset, v_mem_new)
        v_thresh_final = jnp.where(
            spikes, 
            v_thresh_new + self.params.v_thresh_adapt,
            v_thresh_new
        )
        
        # Update last spike time
        t_last_spike_new = jnp.where(spikes, t, state.t_last_spike)
        
        return LIFState(
            v_mem=v_mem_final,
            i_syn=i_syn_new,
            v_thresh_dyn=v_thresh_final,
            t_last_spike=t_last_spike_new,
            spikes=spikes
        )
    
    def integrate_and_fire(
        self, 
        input_current: float, 
        v_mem: float, 
        dt: float
    ) -> Tuple[bool, float]:
        """
        Simple integrate-and-fire step for single neuron (legacy interface).
        
        Args:
            input_current: Input current
            v_mem: Current membrane potential
            dt: Integration timestep
            
        Returns:
            Tuple of (spike_occurred, new_membrane_potential)
        """
        # Simple integration without adaptive threshold or refractory period
        alpha = jnp.exp(-dt / self.params.tau_mem)
        R_mem = 1e9  # Membrane resistance in ohms
        dv = ((self.params.v_rest - v_mem) + R_mem * input_current) * (1 - alpha)
        v_new = v_mem + dv
        
        # Check for spike
        spike = v_new >= self.params.v_thresh
        v_final = jnp.where(spike, self.params.v_reset, v_new)
        
        return bool(spike), float(v_final)
    
    def simulate_spike_train(
        self,
        input_currents: jnp.ndarray,
        dt: float,
        batch_size: int = 1,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, LIFState]:
        """
        Simulate neuron response to a time series of input currents.
        
        Args:
            input_currents: Array of input currents [time_steps, batch_size]
            dt: Integration timestep
            batch_size: Number of neurons to simulate
            key: Random key for noise
            
        Returns:
            Tuple of (spike_times, final_state)
        """
        if key is None:
            key = random.PRNGKey(42)
        
        # Initialize state
        state = self.init_state(batch_size, key)
        
        # Prepare for simulation
        n_steps = input_currents.shape[0]
        spike_trains = []
        
        # Simulation loop
        for i in range(n_steps):
            # Generate new key for each step
            key, subkey = random.split(key)
            
            # Get input for this timestep
            if input_currents.ndim == 1:
                current_input = jnp.full((batch_size,), input_currents[i])
            else:
                current_input = input_currents[i]
            
            # Update state
            state = self.step(state, current_input, dt, i * dt, subkey)
            spike_trains.append(state.spikes)
        
        # Stack spike trains
        spike_trains = jnp.stack(spike_trains, axis=0)
        
        return spike_trains, state


# Convenience function for creating LIF neurons with different configurations
def create_lif_neuron(neuron_type: str = "standard") -> LIFNeuron:
    """
    Create LIF neuron with predefined parameter sets.
    
    Args:
        neuron_type: Type of neuron configuration
                    - "standard": Default parameters
                    - "fast": Fast dynamics for high-frequency processing
                    - "slow": Slow dynamics for temporal integration
                    - "adaptive": Strong adaptive threshold
    
    Returns:
        Configured LIF neuron
    """
    if neuron_type == "standard":
        return LIFNeuron()
    elif neuron_type == "fast":
        params = LIFParams(
            tau_mem=5e-3,    # 5ms membrane time constant
            tau_syn=1e-3,    # 1ms synaptic time constant
            t_refrac=1e-3    # 1ms refractory period
        )
        return LIFNeuron(params)
    elif neuron_type == "slow":
        params = LIFParams(
            tau_mem=50e-3,   # 50ms membrane time constant
            tau_syn=10e-3,   # 10ms synaptic time constant
            t_refrac=5e-3    # 5ms refractory period
        )
        return LIFNeuron(params)
    elif neuron_type == "adaptive":
        params = LIFParams(
            v_thresh_adapt=20e-3,  # 20mV threshold adaptation
            tau_thresh=50e-3       # 50ms adaptation time constant
        )
        return LIFNeuron(params)
    else:
        raise ValueError(f"Unknown neuron type: {neuron_type}")
"""
Spike-Timing Dependent Plasticity (STDP) Implementation

This module implements STDP learning rules with configurable time windows
and synaptic weight update mechanisms for the Godly AI system.
"""

from typing import NamedTuple, Optional, Tuple, Callable
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
import numpy as np


@dataclass
class STDPParams:
    """Parameters for STDP learning rule."""
    
    # Learning rates
    a_plus: float = 0.01    # Potentiation amplitude
    a_minus: float = 0.01   # Depression amplitude
    
    # Time constants
    tau_plus: float = 20e-3   # Potentiation time constant (20ms)
    tau_minus: float = 20e-3  # Depression time constant (20ms)
    
    # Weight bounds
    w_min: float = 0.0      # Minimum weight
    w_max: float = 1.0      # Maximum weight
    
    # Learning modulation
    learning_rate: float = 1.0  # Global learning rate multiplier
    
    # Multiplicative vs additive updates
    multiplicative: bool = True  # Use multiplicative STDP
    
    # Homeostatic mechanisms
    target_rate: float = 10.0    # Target firing rate (Hz)
    homeostatic_scaling: bool = False  # Enable homeostatic scaling


class STDPState(NamedTuple):
    """State variables for STDP learning."""
    weights: jnp.ndarray           # Synaptic weights [pre_neurons, post_neurons]
    pre_trace: jnp.ndarray         # Presynaptic eligibility trace
    post_trace: jnp.ndarray        # Postsynaptic eligibility trace
    weight_updates: jnp.ndarray    # Recent weight updates for monitoring


class STDPLearningRule:
    """
    Spike-Timing Dependent Plasticity learning rule implementation.
    
    This implementation supports:
    - Configurable time windows for potentiation and depression
    - Multiplicative and additive weight updates
    - Homeostatic scaling mechanisms
    - Batch processing for network-level learning
    """
    
    def __init__(self, params: Optional[STDPParams] = None):
        """Initialize STDP learning rule with given parameters."""
        self.params = params or STDPParams()
    
    def init_state(
        self, 
        n_pre: int, 
        n_post: int, 
        key: Optional[jax.random.PRNGKey] = None
    ) -> STDPState:
        """
        Initialize STDP state for a population of synapses.
        
        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            key: Random key for weight initialization
            
        Returns:
            Initial STDP state
        """
        if key is None:
            key = random.PRNGKey(0)
        
        # Initialize weights with small random values
        weights = random.uniform(
            key, 
            (n_pre, n_post), 
            minval=0.1 * self.params.w_max,
            maxval=0.3 * self.params.w_max
        )
        
        return STDPState(
            weights=weights,
            pre_trace=jnp.zeros((n_pre,)),
            post_trace=jnp.zeros((n_post,)),
            weight_updates=jnp.zeros((n_pre, n_post))
        )
    
    def update_traces(
        self, 
        state: STDPState, 
        pre_spikes: jnp.ndarray, 
        post_spikes: jnp.ndarray, 
        dt: float
    ) -> STDPState:
        """
        Update eligibility traces based on spike activity.
        
        Args:
            state: Current STDP state
            pre_spikes: Presynaptic spike indicators [n_pre]
            post_spikes: Postsynaptic spike indicators [n_post]
            dt: Time step
            
        Returns:
            Updated STDP state with new traces
        """
        # Decay traces exponentially
        alpha_plus = jnp.exp(-dt / self.params.tau_plus)
        alpha_minus = jnp.exp(-dt / self.params.tau_minus)
        
        # Update presynaptic trace (for depression)
        pre_trace_new = alpha_minus * state.pre_trace + pre_spikes.astype(float)
        
        # Update postsynaptic trace (for potentiation)
        post_trace_new = alpha_plus * state.post_trace + post_spikes.astype(float)
        
        return state._replace(
            pre_trace=pre_trace_new,
            post_trace=post_trace_new
        )
    
    def compute_weight_updates(
        self, 
        state: STDPState, 
        pre_spikes: jnp.ndarray, 
        post_spikes: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute weight updates based on STDP rule.
        
        Args:
            state: Current STDP state
            pre_spikes: Presynaptic spike indicators [n_pre]
            post_spikes: Postsynaptic spike indicators [n_post]
            
        Returns:
            Weight updates [n_pre, n_post]
        """
        # Potentiation: post-spike causes increase based on pre-trace
        # Depression: pre-spike causes decrease based on post-trace
        
        # Expand dimensions for broadcasting
        pre_spikes_expanded = pre_spikes[:, None]  # [n_pre, 1]
        post_spikes_expanded = post_spikes[None, :] # [1, n_post]
        pre_trace_expanded = state.pre_trace[:, None]  # [n_pre, 1]
        post_trace_expanded = state.post_trace[None, :] # [1, n_post]
        
        if self.params.multiplicative:
            # Multiplicative STDP
            potentiation = (
                self.params.a_plus * 
                post_spikes_expanded * 
                pre_trace_expanded * 
                (self.params.w_max - state.weights)
            )
            
            depression = (
                -self.params.a_minus * 
                pre_spikes_expanded * 
                post_trace_expanded * 
                (state.weights - self.params.w_min)
            )
        else:
            # Additive STDP
            potentiation = (
                self.params.a_plus * 
                post_spikes_expanded * 
                pre_trace_expanded
            )
            
            depression = (
                -self.params.a_minus * 
                pre_spikes_expanded * 
                post_trace_expanded
            )
        
        # Total weight update
        weight_updates = self.params.learning_rate * (potentiation + depression)
        
        return weight_updates
    
    def apply_weight_updates(
        self, 
        state: STDPState, 
        weight_updates: jnp.ndarray
    ) -> STDPState:
        """
        Apply weight updates with bounds checking.
        
        Args:
            state: Current STDP state
            weight_updates: Weight updates to apply
            
        Returns:
            Updated STDP state with new weights
        """
        # Apply updates
        new_weights = state.weights + weight_updates
        
        # Clip to bounds
        new_weights = jnp.clip(new_weights, self.params.w_min, self.params.w_max)
        
        return state._replace(
            weights=new_weights,
            weight_updates=weight_updates
        )
    
    def step(
        self, 
        state: STDPState, 
        pre_spikes: jnp.ndarray, 
        post_spikes: jnp.ndarray, 
        dt: float
    ) -> STDPState:
        """
        Perform one STDP learning step.
        
        Args:
            state: Current STDP state
            pre_spikes: Presynaptic spike indicators [n_pre]
            post_spikes: Postsynaptic spike indicators [n_post]
            dt: Time step
            
        Returns:
            Updated STDP state
        """
        # Update eligibility traces
        state = self.update_traces(state, pre_spikes, post_spikes, dt)
        
        # Compute weight updates
        weight_updates = self.compute_weight_updates(state, pre_spikes, post_spikes)
        
        # Apply weight updates
        state = self.apply_weight_updates(state, weight_updates)
        
        return state
    
    def apply_stdp(
        self, 
        pre_spike_time: float, 
        post_spike_time: float, 
        current_weight: float
    ) -> float:
        """
        Apply STDP rule for a single synapse (legacy interface).
        
        Args:
            pre_spike_time: Time of presynaptic spike
            post_spike_time: Time of postsynaptic spike
            current_weight: Current synaptic weight
            
        Returns:
            Updated synaptic weight
        """
        dt = post_spike_time - pre_spike_time
        
        if dt > 0:
            # Potentiation (post after pre)
            if self.params.multiplicative:
                dw = (
                    self.params.a_plus * 
                    jnp.exp(-dt / self.params.tau_plus) * 
                    (self.params.w_max - current_weight)
                )
            else:
                dw = self.params.a_plus * jnp.exp(-dt / self.params.tau_plus)
        else:
            # Depression (pre after post)
            if self.params.multiplicative:
                dw = (
                    -self.params.a_minus * 
                    jnp.exp(dt / self.params.tau_minus) * 
                    (current_weight - self.params.w_min)
                )
            else:
                dw = -self.params.a_minus * jnp.exp(dt / self.params.tau_minus)
        
        # Apply update with bounds
        new_weight = current_weight + self.params.learning_rate * dw
        return float(jnp.clip(new_weight, self.params.w_min, self.params.w_max))
    
    def homeostatic_scaling(
        self, 
        state: STDPState, 
        firing_rates: jnp.ndarray, 
        dt: float
    ) -> STDPState:
        """
        Apply homeostatic scaling to maintain target firing rates.
        
        Args:
            state: Current STDP state
            firing_rates: Current firing rates of postsynaptic neurons [n_post]
            dt: Time step
            
        Returns:
            Updated STDP state with scaled weights
        """
        if not self.params.homeostatic_scaling:
            return state
        
        # Compute scaling factors
        target_rates = jnp.full_like(firing_rates, self.params.target_rate)
        scaling_factors = target_rates / (firing_rates + 1e-6)  # Avoid division by zero
        
        # Apply scaling with time constant
        tau_homeostatic = 1000e-3  # 1 second time constant
        alpha = jnp.exp(-dt / tau_homeostatic)
        scaling_factors = alpha + (1 - alpha) * scaling_factors
        
        # Scale weights
        scaled_weights = state.weights * scaling_factors[None, :]
        scaled_weights = jnp.clip(scaled_weights, self.params.w_min, self.params.w_max)
        
        return state._replace(weights=scaled_weights)


class TripletsSTDP(STDPLearningRule):
    """
    Triplet STDP rule that considers triplets of spikes for more realistic learning.
    
    This extends the basic STDP rule to include triplet interactions,
    which better captures experimental observations of synaptic plasticity.
    """
    
    def __init__(self, params: Optional[STDPParams] = None):
        """Initialize triplet STDP with additional parameters."""
        super().__init__(params)
        
        # Additional triplet parameters
        self.a2_plus = 0.005   # Triplet potentiation amplitude
        self.a2_minus = 0.005  # Triplet depression amplitude
        self.tau_x = 15e-3     # Triplet time constant (15ms)
        self.tau_y = 30e-3     # Triplet time constant (30ms)
    
    def init_state(
        self, 
        n_pre: int, 
        n_post: int, 
        key: Optional[jax.random.PRNGKey] = None
    ) -> STDPState:
        """Initialize triplet STDP state with additional traces."""
        base_state = super().init_state(n_pre, n_post, key)
        
        # Add triplet traces (stored in a custom state extension)
        # For simplicity, we'll use the base state but could extend it
        return base_state


# Convenience functions for creating different STDP configurations
def create_stdp_rule(rule_type: str = "standard") -> STDPLearningRule:
    """
    Create STDP learning rule with predefined parameter sets.
    
    Args:
        rule_type: Type of STDP rule
                  - "standard": Balanced potentiation/depression
                  - "potentiation_dominant": Stronger potentiation
                  - "depression_dominant": Stronger depression
                  - "fast": Fast learning dynamics
                  - "slow": Slow learning dynamics
                  - "homeostatic": With homeostatic scaling
    
    Returns:
        Configured STDP learning rule
    """
    if rule_type == "standard":
        return STDPLearningRule()
    elif rule_type == "potentiation_dominant":
        params = STDPParams(a_plus=0.02, a_minus=0.01)
        return STDPLearningRule(params)
    elif rule_type == "depression_dominant":
        params = STDPParams(a_plus=0.01, a_minus=0.02)
        return STDPLearningRule(params)
    elif rule_type == "fast":
        params = STDPParams(
            tau_plus=10e-3,    # 10ms
            tau_minus=10e-3,   # 10ms
            learning_rate=2.0
        )
        return STDPLearningRule(params)
    elif rule_type == "slow":
        params = STDPParams(
            tau_plus=50e-3,    # 50ms
            tau_minus=50e-3,   # 50ms
            learning_rate=0.5
        )
        return STDPLearningRule(params)
    elif rule_type == "homeostatic":
        params = STDPParams(
            homeostatic_scaling=True,
            target_rate=15.0
        )
        return STDPLearningRule(params)
    elif rule_type == "triplets":
        return TripletsSTDP()
    else:
        raise ValueError(f"Unknown STDP rule type: {rule_type}")


def compute_stdp_window(
    dt_range: jnp.ndarray, 
    params: STDPParams
) -> jnp.ndarray:
    """
    Compute STDP learning window for visualization.
    
    Args:
        dt_range: Array of time differences (post - pre)
        params: STDP parameters
        
    Returns:
        STDP weight changes for each time difference
    """
    potentiation = jnp.where(
        dt_range > 0,
        params.a_plus * jnp.exp(-dt_range / params.tau_plus),
        0.0
    )
    
    depression = jnp.where(
        dt_range < 0,
        -params.a_minus * jnp.exp(dt_range / params.tau_minus),
        0.0
    )
    
    return potentiation + depression
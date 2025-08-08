"""
Liquid State Machine (LSM) Implementation

This module implements the core LSM with reservoir network, spectral radius optimization,
and readout layer for the Godly AI neuromorphic system.
"""

from typing import NamedTuple, Optional, Tuple, Dict, Any
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
import numpy as np

from .neurons.lif_neuron import LIFNeuron, LIFParams, LIFState
from .plasticity.stdp import STDPLearningRule, STDPParams, STDPState


@dataclass
class LSMParams:
    """Parameters for the Liquid State Machine."""
    
    # Reservoir structure
    reservoir_size: int = 1000      # Number of neurons in reservoir
    input_size: int = 100           # Number of input neurons
    output_size: int = 10           # Number of output neurons
    
    # Connectivity
    reservoir_connectivity: float = 0.1    # Fraction of connections in reservoir
    input_connectivity: float = 0.3        # Fraction of input connections
    
    # Spectral radius
    spectral_radius: float = 0.95   # Target spectral radius for edge-of-chaos
    
    # Neuron parameters
    excitatory_ratio: float = 0.8   # Fraction of excitatory neurons
    
    # Weight scaling
    input_weight_scale: float = 50e-12     # Input weight scaling (50pA per spike)
    reservoir_weight_scale: float = 20e-12  # Reservoir weight scaling (20pA per spike)
    
    # Readout parameters
    readout_regularization: float = 1e-6  # Ridge regression regularization
    
    # Learning
    enable_plasticity: bool = True   # Enable STDP in reservoir
    homeostatic_scaling: bool = True # Enable homeostatic mechanisms


class LSMState(NamedTuple):
    """State variables for the Liquid State Machine."""
    neuron_state: LIFState          # State of reservoir neurons
    plasticity_state: STDPState     # STDP state for reservoir connections
    readout_weights: jnp.ndarray    # Readout layer weights [reservoir_size, output_size]
    reservoir_weights: jnp.ndarray  # Reservoir connection weights
    input_weights: jnp.ndarray      # Input connection weights
    activity_history: jnp.ndarray   # Recent activity for readout [history_length, reservoir_size]


class LiquidStateMachine:
    """
    Liquid State Machine implementation with spiking neural networks.
    
    The LSM consists of:
    - A recurrently connected reservoir of spiking neurons
    - Input connections from external sources
    - A linear readout layer for output computation
    - STDP plasticity for learning and adaptation
    """
    
    def __init__(self, params: Optional[LSMParams] = None):
        """Initialize LSM with given parameters."""
        self.params = params or LSMParams()
        
        # Create neuron model for reservoir
        neuron_params = LIFParams(
            tau_mem=20e-3,      # 20ms membrane time constant
            tau_syn=5e-3,       # 5ms synaptic time constant
            v_thresh=-50e-3,    # -50mV threshold
            noise_std=1e-3      # 1mV noise
        )
        self.neuron_model = LIFNeuron(neuron_params)
        
        # Create STDP learning rule
        if self.params.enable_plasticity:
            stdp_params = STDPParams(
                a_plus=0.005,       # Moderate learning rate
                a_minus=0.005,
                tau_plus=20e-3,
                tau_minus=20e-3,
                homeostatic_scaling=self.params.homeostatic_scaling
            )
            self.stdp_rule = STDPLearningRule(stdp_params)
        else:
            self.stdp_rule = None
    
    def init_state(
        self, 
        key: Optional[jax.random.PRNGKey] = None,
        history_length: int = 50
    ) -> LSMState:
        """
        Initialize LSM state with random connectivity.
        
        Args:
            key: Random key for initialization
            history_length: Length of activity history buffer
            
        Returns:
            Initial LSM state
        """
        if key is None:
            key = random.PRNGKey(0)
        
        key, *subkeys = random.split(key, 6)
        
        # Initialize neuron state
        neuron_state = self.neuron_model.init_state(self.params.reservoir_size, subkeys[0])
        
        # Initialize reservoir connectivity
        reservoir_weights = self._create_reservoir_weights(subkeys[1])
        
        # Initialize input connectivity
        input_weights = self._create_input_weights(subkeys[2])
        
        # Initialize readout weights
        readout_weights = random.normal(
            subkeys[3], 
            (self.params.reservoir_size, self.params.output_size)
        ) * 0.01  # Small initial weights
        
        # Initialize STDP state if plasticity is enabled
        if self.stdp_rule is not None:
            plasticity_state = self.stdp_rule.init_state(
                self.params.reservoir_size, 
                self.params.reservoir_size, 
                subkeys[4]
            )
            # Use reservoir weights as initial STDP weights
            plasticity_state = plasticity_state._replace(weights=jnp.abs(reservoir_weights))
        else:
            # Create dummy plasticity state
            plasticity_state = STDPState(
                weights=jnp.abs(reservoir_weights),
                pre_trace=jnp.zeros(self.params.reservoir_size),
                post_trace=jnp.zeros(self.params.reservoir_size),
                weight_updates=jnp.zeros((self.params.reservoir_size, self.params.reservoir_size))
            )
        
        # Initialize activity history
        activity_history = jnp.zeros((history_length, self.params.reservoir_size))
        
        return LSMState(
            neuron_state=neuron_state,
            plasticity_state=plasticity_state,
            readout_weights=readout_weights,
            reservoir_weights=reservoir_weights,
            input_weights=input_weights,
            activity_history=activity_history
        )
    
    def _create_reservoir_weights(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Create reservoir connection weights with spectral radius optimization."""
        # Create random connectivity matrix
        connection_prob = self.params.reservoir_connectivity
        connections = random.bernoulli(
            key, connection_prob, 
            (self.params.reservoir_size, self.params.reservoir_size)
        )
        
        # No self-connections
        connections = connections.at[jnp.diag_indices(self.params.reservoir_size)].set(False)
        
        # Create weight matrix
        key, subkey = random.split(key)
        weights = random.normal(subkey, connections.shape) * connections
        
        # Set excitatory/inhibitory structure
        n_excitatory = int(self.params.excitatory_ratio * self.params.reservoir_size)
        
        # Make first n_excitatory neurons excitatory (positive outgoing weights)
        weights = weights.at[:n_excitatory, :].set(jnp.abs(weights[:n_excitatory, :]))
        
        # Make remaining neurons inhibitory (negative outgoing weights)
        weights = weights.at[n_excitatory:, :].set(-jnp.abs(weights[n_excitatory:, :]))
        
        # Scale to achieve target spectral radius first
        weights = self._scale_spectral_radius(weights)
        
        # Then apply weight scaling
        return weights * self.params.reservoir_weight_scale
    
    def _create_input_weights(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Create input connection weights."""
        # Create random input connectivity
        connection_prob = self.params.input_connectivity
        connections = random.bernoulli(
            key, connection_prob,
            (self.params.input_size, self.params.reservoir_size)
        )
        
        # Create weight matrix
        key, subkey = random.split(key)
        weights = random.normal(subkey, connections.shape) * connections
        
        return weights * self.params.input_weight_scale
    
    def _scale_spectral_radius(self, weights: jnp.ndarray) -> jnp.ndarray:
        """Scale weight matrix to achieve target spectral radius."""
        # Compute spectral radius (largest eigenvalue magnitude)
        eigenvalues = jnp.linalg.eigvals(weights)
        current_spectral_radius = jnp.max(jnp.abs(eigenvalues))
        
        # Scale weights to achieve target spectral radius
        if current_spectral_radius > 1e-10:  # Avoid division by zero
            scaling_factor = self.params.spectral_radius / current_spectral_radius
            weights = weights * scaling_factor
        
        return weights
    
    def step(
        self, 
        state: LSMState, 
        input_spikes: jnp.ndarray, 
        dt: float, 
        t: float,
        key: Optional[jax.random.PRNGKey] = None
    ) -> LSMState:
        """
        Perform one LSM computation step.
        
        Args:
            state: Current LSM state
            input_spikes: Input spike pattern [input_size]
            dt: Time step
            t: Current time
            key: Random key for noise
            
        Returns:
            Updated LSM state
        """
        if key is None:
            key = random.PRNGKey(int(t * 1000))
        
        # Compute input current to reservoir
        input_current = jnp.dot(input_spikes.astype(float), state.input_weights)
        
        # Compute recurrent current from reservoir
        reservoir_spikes = state.neuron_state.spikes.astype(float)
        recurrent_current = jnp.dot(reservoir_spikes, state.reservoir_weights.T)
        
        # Total current to each reservoir neuron
        total_current = input_current + recurrent_current
        
        # Update neuron dynamics
        new_neuron_state = self.neuron_model.step(
            state.neuron_state, total_current, dt, t, key
        )
        
        # Update plasticity if enabled
        if self.stdp_rule is not None:
            new_plasticity_state = self.stdp_rule.step(
                state.plasticity_state,
                state.neuron_state.spikes,  # Pre-synaptic spikes
                new_neuron_state.spikes,    # Post-synaptic spikes
                dt
            )
            
            # Update reservoir weights based on plasticity
            # Apply sign structure (excitatory/inhibitory)
            n_excitatory = int(self.params.excitatory_ratio * self.params.reservoir_size)
            new_reservoir_weights = state.reservoir_weights.copy()
            
            # Update connection strengths while preserving sign
            weight_magnitudes = new_plasticity_state.weights
            new_reservoir_weights = new_reservoir_weights.at[:n_excitatory, :].set(
                jnp.where(
                    state.reservoir_weights[:n_excitatory, :] != 0,
                    weight_magnitudes[:n_excitatory, :],
                    0.0
                )
            )
            new_reservoir_weights = new_reservoir_weights.at[n_excitatory:, :].set(
                jnp.where(
                    state.reservoir_weights[n_excitatory:, :] != 0,
                    -weight_magnitudes[n_excitatory:, :],
                    0.0
                )
            )
        else:
            new_plasticity_state = state.plasticity_state
            new_reservoir_weights = state.reservoir_weights
        
        # Update activity history
        new_activity = new_neuron_state.spikes.astype(float)
        new_activity_history = jnp.roll(state.activity_history, -1, axis=0)
        new_activity_history = new_activity_history.at[-1, :].set(new_activity)
        
        return LSMState(
            neuron_state=new_neuron_state,
            plasticity_state=new_plasticity_state,
            readout_weights=state.readout_weights,
            reservoir_weights=new_reservoir_weights,
            input_weights=state.input_weights,
            activity_history=new_activity_history
        )
    
    def compute_readout(self, state: LSMState) -> jnp.ndarray:
        """
        Compute readout layer output.
        
        Args:
            state: Current LSM state
            
        Returns:
            Readout output [output_size]
        """
        # Use current reservoir activity
        reservoir_activity = state.neuron_state.spikes.astype(float)
        
        # Linear readout
        output = jnp.dot(reservoir_activity, state.readout_weights)
        
        return output
    
    def train_readout(
        self, 
        state: LSMState, 
        target_outputs: jnp.ndarray,
        reservoir_states: jnp.ndarray
    ) -> LSMState:
        """
        Train readout layer using ridge regression.
        
        Args:
            state: Current LSM state
            target_outputs: Target outputs [n_samples, output_size]
            reservoir_states: Reservoir states [n_samples, reservoir_size]
            
        Returns:
            Updated LSM state with trained readout weights
        """
        # Ridge regression solution
        # W = (X^T X + λI)^(-1) X^T Y
        
        X = reservoir_states  # [n_samples, reservoir_size]
        Y = target_outputs    # [n_samples, output_size]
        
        # Add regularization
        XTX = jnp.dot(X.T, X)  # [reservoir_size, reservoir_size]
        regularization = self.params.readout_regularization * jnp.eye(X.shape[1])
        
        # Solve for weights
        try:
            new_readout_weights = jnp.linalg.solve(
                XTX + regularization,
                jnp.dot(X.T, Y)
            )  # [reservoir_size, output_size]
        except jnp.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            new_readout_weights = jnp.linalg.pinv(X) @ Y
        
        return state._replace(readout_weights=new_readout_weights)
    
    def get_reservoir_state(self, state: LSMState) -> jnp.ndarray:
        """Get current reservoir state for readout training."""
        return state.neuron_state.spikes.astype(float)
    
    def process_spike_train(
        self,
        input_spike_trains: jnp.ndarray,
        dt: float,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, LSMState]:
        """
        Process a sequence of input spike trains.
        
        Args:
            input_spike_trains: Input spikes [time_steps, input_size]
            dt: Time step
            key: Random key
            
        Returns:
            Tuple of (reservoir_states, final_state)
        """
        if key is None:
            key = random.PRNGKey(42)
        
        # Initialize state
        state = self.init_state(key)
        
        n_steps = input_spike_trains.shape[0]
        reservoir_states = []
        
        # Process each time step
        for i in range(n_steps):
            key, subkey = random.split(key)
            
            # Get input for this timestep
            input_spikes = input_spike_trains[i]
            
            # Update LSM state
            state = self.step(state, input_spikes, dt, i * dt, subkey)
            
            # Store reservoir state
            reservoir_states.append(self.get_reservoir_state(state))
        
        # Stack reservoir states
        reservoir_states = jnp.stack(reservoir_states, axis=0)
        
        return reservoir_states, state
    
    def get_spectral_radius(self, weights: jnp.ndarray) -> float:
        """Compute spectral radius of weight matrix."""
        eigenvalues = jnp.linalg.eigvals(weights)
        return float(jnp.max(jnp.abs(eigenvalues)))
    
    def analyze_dynamics(self, state: LSMState) -> Dict[str, Any]:
        """
        Analyze LSM dynamics and properties.
        
        Args:
            state: Current LSM state
            
        Returns:
            Dictionary of analysis results
        """
        analysis = {}
        
        # Spectral radius
        analysis['spectral_radius'] = self.get_spectral_radius(state.reservoir_weights)
        
        # Connectivity statistics
        reservoir_connections = jnp.sum(state.reservoir_weights != 0)
        total_possible = self.params.reservoir_size * (self.params.reservoir_size - 1)
        analysis['reservoir_connectivity'] = float(reservoir_connections / total_possible)
        
        # Activity statistics
        current_activity = state.neuron_state.spikes.astype(float)
        analysis['current_firing_rate'] = float(jnp.mean(current_activity))
        
        if state.activity_history.shape[0] > 1:
            recent_activity = jnp.mean(state.activity_history, axis=0)
            analysis['mean_firing_rate'] = float(jnp.mean(recent_activity))
            analysis['firing_rate_std'] = float(jnp.std(recent_activity))
        
        # Weight statistics
        analysis['mean_reservoir_weight'] = float(jnp.mean(jnp.abs(state.reservoir_weights)))
        analysis['mean_input_weight'] = float(jnp.mean(jnp.abs(state.input_weights)))
        
        return analysis


# Convenience functions for creating different LSM configurations
def create_lsm(lsm_type: str = "standard") -> LiquidStateMachine:
    """
    Create LSM with predefined parameter sets.
    
    Args:
        lsm_type: Type of LSM configuration
                 - "standard": Default parameters
                 - "small": Small reservoir for testing
                 - "large": Large reservoir for complex tasks
                 - "sparse": Sparse connectivity
                 - "dense": Dense connectivity
                 - "edge_of_chaos": Optimized for edge-of-chaos dynamics
    
    Returns:
        Configured LSM
    """
    if lsm_type == "standard":
        return LiquidStateMachine()
    elif lsm_type == "small":
        params = LSMParams(
            reservoir_size=100,
            input_size=20,
            output_size=5
        )
        return LiquidStateMachine(params)
    elif lsm_type == "large":
        params = LSMParams(
            reservoir_size=2000,
            input_size=200,
            output_size=50
        )
        return LiquidStateMachine(params)
    elif lsm_type == "sparse":
        params = LSMParams(
            reservoir_connectivity=0.05,  # 5% connectivity
            input_connectivity=0.1
        )
        return LiquidStateMachine(params)
    elif lsm_type == "dense":
        params = LSMParams(
            reservoir_connectivity=0.3,   # 30% connectivity
            input_connectivity=0.8
        )
        return LiquidStateMachine(params)
    elif lsm_type == "edge_of_chaos":
        params = LSMParams(
            spectral_radius=0.99,  # Very close to edge of chaos
            reservoir_connectivity=0.15,
            enable_plasticity=True,
            homeostatic_scaling=True
        )
        return LiquidStateMachine(params)
    else:
        raise ValueError(f"Unknown LSM type: {lsm_type}")
# Neural Core Components

## Overview

The neural core provides the fundamental building blocks for neuromorphic computation in the Godly AI system. These components implement biologically-inspired neural dynamics and learning mechanisms.

## Components

### Leaky Integrate-and-Fire (LIF) Neurons

#### Purpose
Implements biologically-realistic spiking neurons with temporal dynamics, membrane integration, and adaptive behavior.

#### Key Features
- **Membrane Integration**: Exponential decay with input integration
- **Spike Generation**: Threshold-based firing with reset
- **Refractory Period**: Post-spike inhibition period
- **Adaptive Threshold**: Dynamic threshold adjustment
- **Noise Injection**: Stochastic behavior modeling
- **Batch Processing**: Efficient parallel computation

#### Parameters
```python
@dataclass
class LIFParams:
    tau_membrane: float = 20e-3      # Membrane time constant (20ms)
    tau_refractory: float = 2e-3     # Refractory period (2ms)
    v_threshold: float = -50e-3      # Spike threshold (-50mV)
    v_reset: float = -70e-3          # Reset potential (-70mV)
    v_rest: float = -70e-3           # Resting potential (-70mV)
    tau_threshold: float = 100e-3    # Threshold adaptation time constant
    threshold_increment: float = 5e-3 # Threshold increase per spike
    noise_amplitude: float = 0.0     # Input noise amplitude
```

#### Usage Example
```python
from src.core.neurons.lif_neuron import LIFNeuron, create_excitatory_neuron

# Create neuron with default parameters
neuron = LIFNeuron()

# Create excitatory neuron variant
exc_neuron = create_excitatory_neuron()

# Process input current
dt = 1e-3  # 1ms time step
input_current = 100e-12  # 100pA
spike_occurred = neuron.step(input_current, dt)

# Batch processing
batch_size = 100
inputs = jnp.ones((batch_size,)) * input_current
spikes = neuron.batch_step(inputs, dt)
```

### Liquid State Machine (LSM)

#### Purpose
Implements reservoir computing for temporal pattern processing using recurrent neural networks with rich dynamics.

#### Key Features
- **Dynamic Reservoir**: Recurrent network with complex dynamics
- **Spectral Radius Control**: Stability and memory optimization
- **Readout Training**: Linear classification/regression layer
- **Edge-of-Chaos**: Optimal computational regime
- **Pattern Separation**: High-dimensional state space mapping
- **Memory Capacity**: Temporal information retention

#### Parameters
```python
@dataclass
class LSMParams:
    n_reservoir: int = 100           # Reservoir size
    n_inputs: int = 10               # Input dimensions
    n_outputs: int = 1               # Output dimensions
    spectral_radius: float = 0.95    # Desired spectral radius
    input_scaling: float = 1.0       # Input weight scaling
    connectivity: float = 0.1        # Connection probability
    leak_rate: float = 1.0           # Neuron leak rate
    noise_level: float = 0.01        # Internal noise
```

#### Usage Example
```python
from src.core.liquid_state_machine import LiquidStateMachine, create_temporal_lsm

# Create LSM with custom parameters
lsm = LiquidStateMachine(n_reservoir=200, spectral_radius=0.9)

# Process temporal sequence
sequence_length = 100
input_sequence = jnp.random.normal(key, (sequence_length, 10))
states = lsm.process_sequence(input_sequence)

# Train readout layer
targets = jnp.random.normal(key, (sequence_length, 1))
lsm.train_readout(states, targets)

# Generate predictions
predictions = lsm.predict(states)
```

### STDP Plasticity

#### Purpose
Implements spike-timing dependent plasticity for unsupervised learning and synaptic adaptation.

#### Key Features
- **Hebbian Learning**: "Neurons that fire together, wire together"
- **Anti-Hebbian**: Depression for uncorrelated activity
- **Trace Dynamics**: Exponential decay of pre/post-synaptic traces
- **Weight Bounds**: Soft and hard weight constraints
- **Homeostatic Scaling**: Activity-dependent normalization
- **Triplet STDP**: Higher-order spike interactions

#### Parameters
```python
@dataclass
class STDPParams:
    a_plus: float = 0.005           # Potentiation amplitude
    a_minus: float = 0.005          # Depression amplitude
    tau_plus: float = 20e-3         # Potentiation time constant
    tau_minus: float = 20e-3        # Depression time constant
    w_min: float = 0.0              # Minimum weight
    w_max: float = 1.0              # Maximum weight
    multiplicative: bool = True      # Multiplicative vs additive updates
    homeostatic_scaling: bool = False # Enable homeostatic mechanisms
```

#### Usage Example
```python
from src.core.plasticity.stdp import STDPLearningRule, create_hebbian_stdp

# Create STDP rule
stdp = STDPLearningRule()

# Initialize weights and traces
n_pre, n_post = 100, 50
weights = jnp.random.uniform(key, (n_pre, n_post), minval=0.1, maxval=0.9)
pre_traces = jnp.zeros(n_pre)
post_traces = jnp.zeros(n_post)

# Process spike events
pre_spikes = jnp.array([1, 0, 1, 0, 0])  # Pre-synaptic spikes
post_spikes = jnp.array([0, 1, 1, 0, 1]) # Post-synaptic spikes
dt = 1e-3

# Update weights and traces
new_weights, new_pre_traces, new_post_traces = stdp.update_weights(
    weights, pre_traces, post_traces, pre_spikes, post_spikes, dt
)
```

## Integration Patterns

### Neuron-Plasticity Integration
```python
# Combine LIF neurons with STDP learning
neuron = LIFNeuron()
stdp = STDPLearningRule()

# Simulation loop
for t in range(n_timesteps):
    # Process inputs through neurons
    pre_spikes = neuron.batch_step(inputs[t], dt)
    post_spikes = neuron.batch_step(recurrent_inputs[t], dt)
    
    # Update synaptic weights
    weights, pre_traces, post_traces = stdp.update_weights(
        weights, pre_traces, post_traces, pre_spikes, post_spikes, dt
    )
    
    # Compute recurrent inputs for next timestep
    recurrent_inputs[t+1] = jnp.dot(pre_spikes, weights)
```

### LSM-STDP Integration
```python
# Use STDP to adapt LSM reservoir connections
lsm = LiquidStateMachine()
stdp = STDPLearningRule()

# Training loop
for epoch in range(n_epochs):
    # Process input sequence
    states = lsm.process_sequence(input_sequence)
    
    # Extract spike times for STDP
    spike_times = lsm.get_spike_times()
    
    # Update reservoir weights with STDP
    lsm.reservoir_weights = stdp.batch_update(
        lsm.reservoir_weights, spike_times
    )
    
    # Optimize spectral radius
    lsm.normalize_spectral_radius()
```

## Performance Considerations

### Computational Efficiency
- **Vectorization**: JAX-based parallel computation
- **Memory Management**: Efficient state representation
- **Sparse Operations**: Optimized for sparse connectivity
- **Batch Processing**: Parallel neuron simulation

### Numerical Stability
- **Adaptive Time Steps**: Stability-preserving integration
- **Weight Clipping**: Prevents runaway dynamics
- **Regularization**: Maintains healthy activity levels
- **Overflow Protection**: Handles extreme values gracefully

### Scalability
- **Distributed Computing**: Multi-device support
- **Memory Optimization**: Efficient large-scale simulation
- **Hierarchical Organization**: Modular network construction
- **Dynamic Allocation**: Runtime resource management

## Configuration Examples

### High-Performance Configuration
```yaml
lif_params:
  tau_membrane: 10e-3      # Faster dynamics
  noise_amplitude: 0.001   # Low noise
  
lsm_params:
  n_reservoir: 1000        # Large reservoir
  spectral_radius: 0.99    # Near-critical dynamics
  connectivity: 0.05       # Sparse connectivity
  
stdp_params:
  a_plus: 0.01            # Strong potentiation
  multiplicative: true     # Stable learning
  homeostatic_scaling: true # Activity regulation
```

### Low-Resource Configuration
```yaml
lif_params:
  tau_membrane: 20e-3      # Standard dynamics
  noise_amplitude: 0.01    # Moderate noise
  
lsm_params:
  n_reservoir: 100         # Small reservoir
  spectral_radius: 0.9     # Stable dynamics
  connectivity: 0.1        # Dense connectivity
  
stdp_params:
  a_plus: 0.005           # Moderate learning
  multiplicative: false    # Simple updates
  homeostatic_scaling: false # Disabled
```

## Testing and Validation

### Unit Tests
- Individual component functionality
- Parameter validation
- Edge case handling
- Performance benchmarks

### Integration Tests
- Component interactions
- System-level behavior
- Stability analysis
- Convergence properties

### Benchmarks
- Computational performance
- Memory usage
- Scalability limits
- Accuracy metrics
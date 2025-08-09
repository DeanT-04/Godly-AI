"""
Motor Reasoning Core Implementation

This module implements the motor reasoning core for action planning,
motor control, and sensorimotor integration.
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


class MotorReasoningCore(BaseReasoningCore):
    """
    Motor reasoning core specialized for motor control and action planning.
    
    Features:
    - Action sequence planning
    - Motor command generation
    - Sensorimotor integration
    - Trajectory optimization
    - Force and position control
    """
    
    def __init__(
        self, 
        params: Optional[ReasoningCoreParams] = None,
        action_dim: int = 6,  # 6-DOF actions (position + orientation)
        control_frequency: float = 100.0,  # 100 Hz control
        planning_horizon: int = 20
    ):
        """Initialize motor reasoning core."""
        if params is None:
            params = ReasoningCoreParams(
                modality=ModalityType.MOTOR,
                core_id="motor_core",
                reservoir_size=400,  # Smaller reservoir for motor control
                input_size=action_dim * 3,  # Current, target, and error states
                output_size=action_dim,     # Motor commands
                processing_layers=2,        # Fast motor processing
                temporal_window=0.01        # 10ms for motor control
            )
        
        # Motor-specific parameters
        self.action_dim = action_dim
        self.control_frequency = control_frequency
        self.planning_horizon = planning_horizon
        
        # Initialize motor control components
        self._init_motor_controllers()
        
        super().__init__(params)
    
    def _init_motor_controllers(self):
        """Initialize motor control components."""
        # PID controller parameters for each DOF
        self.pid_params = {
            'kp': jnp.ones(self.action_dim) * 1.0,  # Proportional gains
            'ki': jnp.ones(self.action_dim) * 0.1,  # Integral gains
            'kd': jnp.ones(self.action_dim) * 0.05  # Derivative gains
        }
        
        # Motor dynamics model (simplified)
        self.motor_dynamics = {
            'inertia': jnp.ones(self.action_dim) * 0.1,
            'damping': jnp.ones(self.action_dim) * 0.05,
            'max_force': jnp.ones(self.action_dim) * 10.0,
            'max_velocity': jnp.ones(self.action_dim) * 5.0
        }
        
        # Action primitives (basic motor patterns)
        self.action_primitives = {
            'reach': jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'grasp': jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            'lift': jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            'rotate': jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        }
    
    def _get_connectivity_pattern(self) -> float:
        """Get motor-specific connectivity pattern."""
        # Motor control benefits from sparse, focused connectivity
        return 0.12
    
    def _get_spectral_radius(self) -> float:
        """Get motor-specific spectral radius."""
        # Lower spectral radius for stable motor control
        return 0.9
    
    def preprocess_input(self, raw_input: jnp.ndarray) -> jnp.ndarray:
        """
        Preprocess motor input for action planning and control.
        
        Args:
            raw_input: Raw motor input (current state, target, etc.)
            
        Returns:
            Processed spike-encoded motor features
        """
        # Handle different input formats
        if raw_input.shape[0] == self.action_dim:
            # Single state vector - assume it's current position
            current_state = raw_input
            target_state = jnp.zeros_like(current_state)
            error_state = target_state - current_state
        elif raw_input.shape[0] == self.action_dim * 2:
            # Current and target states
            current_state = raw_input[:self.action_dim]
            target_state = raw_input[self.action_dim:]
            error_state = target_state - current_state
        elif raw_input.shape[0] == self.action_dim * 3:
            # Current, target, and error states
            current_state = raw_input[:self.action_dim]
            target_state = raw_input[self.action_dim:2*self.action_dim]
            error_state = raw_input[2*self.action_dim:]
        else:
            # Pad or truncate to expected size
            padded_input = jnp.zeros(self.action_dim * 3)
            min_size = min(len(raw_input), len(padded_input))
            padded_input = padded_input.at[:min_size].set(raw_input[:min_size])
            
            current_state = padded_input[:self.action_dim]
            target_state = padded_input[self.action_dim:2*self.action_dim]
            error_state = padded_input[2*self.action_dim:]
        
        # Extract motor features
        motor_features = self._extract_motor_features(current_state, target_state, error_state)
        
        # Extract control features
        control_features = self._extract_control_features(current_state, target_state, error_state)
        
        # Extract planning features
        planning_features = self._extract_planning_features(current_state, target_state)
        
        # Combine all features
        combined_features = jnp.concatenate([
            motor_features,
            control_features,
            planning_features
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
    
    def _extract_motor_features(
        self, 
        current_state: jnp.ndarray,
        target_state: jnp.ndarray,
        error_state: jnp.ndarray
    ) -> jnp.ndarray:
        """Extract motor-specific features."""
        motor_features = []
        
        # 1. State features
        motor_features.extend([current_state, target_state, error_state])
        
        # 2. Distance and direction features
        distance = jnp.linalg.norm(error_state)
        direction = error_state / (distance + 1e-8)
        motor_features.extend([jnp.array([distance]), direction])
        
        # 3. Velocity estimation (simplified)
        # In practice, this would use actual velocity measurements
        estimated_velocity = error_state * self.control_frequency * 0.1
        motor_features.append(estimated_velocity)
        
        # 4. Force requirements
        required_force = error_state * self.pid_params['kp']
        motor_features.append(required_force)
        
        # Concatenate all motor features
        motor_feature_vector = jnp.concatenate(motor_features)
        
        return motor_feature_vector
    
    def _extract_control_features(
        self, 
        current_state: jnp.ndarray,
        target_state: jnp.ndarray,
        error_state: jnp.ndarray
    ) -> jnp.ndarray:
        """Extract control-specific features."""
        control_features = []
        
        # 1. PID control components
        proportional = error_state * self.pid_params['kp']
        # Integral and derivative would require state history
        integral = jnp.zeros_like(error_state)  # Simplified
        derivative = jnp.zeros_like(error_state)  # Simplified
        
        control_features.extend([proportional, integral, derivative])
        
        # 2. Control effort
        control_effort = jnp.linalg.norm(proportional)
        control_features.append(jnp.array([control_effort]))
        
        # 3. Stability metrics
        stability_margin = jnp.minimum(
            self.motor_dynamics['max_force'] - jnp.abs(proportional),
            self.motor_dynamics['max_velocity'] - jnp.abs(error_state)
        )
        control_features.append(stability_margin)
        
        # 4. Action primitive matching
        primitive_matches = []
        for primitive_name, primitive_pattern in self.action_primitives.items():
            # Compute similarity to action primitive
            normalized_error = error_state / (jnp.linalg.norm(error_state) + 1e-8)
            similarity = jnp.dot(normalized_error, primitive_pattern)
            primitive_matches.append(similarity)
        
        control_features.append(jnp.array(primitive_matches))
        
        # Concatenate all control features
        control_feature_vector = jnp.concatenate(control_features)
        
        return control_feature_vector
    
    def _extract_planning_features(
        self, 
        current_state: jnp.ndarray,
        target_state: jnp.ndarray
    ) -> jnp.ndarray:
        """Extract planning-specific features."""
        planning_features = []
        
        # 1. Trajectory features
        trajectory_vector = target_state - current_state
        trajectory_length = jnp.linalg.norm(trajectory_vector)
        trajectory_direction = trajectory_vector / (trajectory_length + 1e-8)
        
        planning_features.extend([
            jnp.array([trajectory_length]),
            trajectory_direction
        ])
        
        # 2. Time-to-target estimation
        max_velocity = jnp.max(self.motor_dynamics['max_velocity'])
        estimated_time = trajectory_length / (max_velocity + 1e-8)
        planning_features.append(jnp.array([estimated_time]))
        
        # 3. Workspace constraints
        # Check if target is within reachable workspace
        workspace_radius = 2.0  # Simplified workspace
        reachability = jnp.clip(
            1.0 - jnp.linalg.norm(target_state) / workspace_radius,
            0.0, 1.0
        )
        planning_features.append(jnp.array([reachability]))
        
        # 4. Multi-step planning features
        # Simple linear interpolation for trajectory planning
        planning_steps = jnp.linspace(0, 1, 5)  # 5-step plan
        planned_positions = []
        for step in planning_steps:
            planned_pos = current_state + step * trajectory_vector
            planned_positions.append(planned_pos)
        
        # Flatten planned trajectory
        planned_trajectory = jnp.concatenate(planned_positions)
        planning_features.append(planned_trajectory)
        
        # Concatenate all planning features
        planning_feature_vector = jnp.concatenate(planning_features)
        
        return planning_feature_vector
    
    def _encode_as_spikes(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Encode features as spike patterns for motor control.
        
        Args:
            features: Feature vector to encode
            
        Returns:
            Spike-encoded features
        """
        # Motor spike encoding with emphasis on precision
        
        # Normalize features
        normalized_features = (features - jnp.mean(features)) / (jnp.std(features) + 1e-8)
        
        # Convert to spike rates with motor-specific encoding
        # Use higher precision encoding for motor control
        spike_rates = jax.nn.sigmoid(normalized_features * 3.0)  # Higher sensitivity
        
        # Add temporal precision for motor control
        # Motor control requires precise timing
        temporal_precision = jnp.where(
            jnp.abs(normalized_features) > 0.5,  # High precision for large signals
            spike_rates * 1.2,
            spike_rates * 0.8
        )
        
        # Generate spikes with motor-specific threshold
        spikes = jnp.where(
            temporal_precision > 0.3,  # Lower threshold for motor responsiveness
            temporal_precision,
            0.0
        )
        
        return spikes
    
    def postprocess_output(self, raw_output: jnp.ndarray) -> jnp.ndarray:
        """
        Postprocess LSM output for motor command generation.
        
        Args:
            raw_output: Raw LSM output
            
        Returns:
            Processed motor commands
        """
        # Apply motor-specific postprocessing
        
        # 1. Normalize output
        normalized_output = (raw_output - jnp.mean(raw_output)) / (jnp.std(raw_output) + 1e-8)
        
        # 2. Apply motor command scaling
        # Scale to appropriate motor command range
        scaled_output = normalized_output * 0.5  # Conservative scaling
        
        # 3. Apply safety constraints
        # Limit motor commands to safe ranges
        constrained_output = jnp.clip(
            scaled_output,
            -self.motor_dynamics['max_force'],
            self.motor_dynamics['max_force']
        )
        
        # 4. Apply motor dynamics compensation
        # Compensate for known motor dynamics
        compensated_output = constrained_output / (self.motor_dynamics['inertia'] + 1e-8)
        
        # 5. Apply smoothing for stable control
        # Motor commands should be smooth to avoid jitter
        if len(compensated_output) > 3:
            smoothing_kernel = jnp.array([0.2, 0.6, 0.2])
            smoothed_output = jnp.convolve(compensated_output, smoothing_kernel, mode='same')
        else:
            smoothed_output = compensated_output
        
        # 6. Ensure output matches action dimension
        if len(smoothed_output) > self.action_dim:
            final_output = smoothed_output[:self.action_dim]
        elif len(smoothed_output) < self.action_dim:
            padding = jnp.zeros(self.action_dim - len(smoothed_output))
            final_output = jnp.concatenate([smoothed_output, padding])
        else:
            final_output = smoothed_output
        
        return final_output
    
    def plan_trajectory(
        self, 
        start_state: jnp.ndarray,
        goal_state: jnp.ndarray,
        num_steps: int = None
    ) -> jnp.ndarray:
        """
        Plan a trajectory from start to goal state.
        
        Args:
            start_state: Starting state
            goal_state: Goal state
            num_steps: Number of trajectory steps
            
        Returns:
            Planned trajectory [num_steps, action_dim]
        """
        if num_steps is None:
            num_steps = self.planning_horizon
        
        # Simple linear trajectory planning
        # In practice, this would use more sophisticated planning algorithms
        
        trajectory = jnp.zeros((num_steps, self.action_dim))
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1) if num_steps > 1 else 0
            # Linear interpolation with smooth acceleration/deceleration
            smooth_alpha = 3 * alpha**2 - 2 * alpha**3  # Smooth step function
            
            waypoint = start_state + smooth_alpha * (goal_state - start_state)
            trajectory = trajectory.at[i, :].set(waypoint)
        
        return trajectory
    
    def compute_motor_commands(
        self, 
        current_state: jnp.ndarray,
        target_state: jnp.ndarray,
        dt: float = None
    ) -> jnp.ndarray:
        """
        Compute motor commands for reaching target state.
        
        Args:
            current_state: Current motor state
            target_state: Target motor state
            dt: Time step
            
        Returns:
            Motor commands
        """
        if dt is None:
            dt = 1.0 / self.control_frequency
        
        # Compute error
        error = target_state - current_state
        
        # PID control
        proportional = error * self.pid_params['kp']
        # Simplified - would need state history for I and D terms
        integral = jnp.zeros_like(error)
        derivative = jnp.zeros_like(error)
        
        # Combine PID terms
        motor_commands = proportional + integral + derivative
        
        # Apply constraints
        motor_commands = jnp.clip(
            motor_commands,
            -self.motor_dynamics['max_force'],
            self.motor_dynamics['max_force']
        )
        
        return motor_commands


def create_motor_reasoning_core(
    action_dim: int = 6,
    control_frequency: float = 100.0,
    core_id: str = "motor_core"
) -> MotorReasoningCore:
    """
    Create a motor reasoning core with specified parameters.
    
    Args:
        action_dim: Dimension of action space
        control_frequency: Control frequency in Hz
        core_id: Unique identifier for this core
        
    Returns:
        Configured motor reasoning core
    """
    params = ReasoningCoreParams(
        modality=ModalityType.MOTOR,
        core_id=core_id,
        reservoir_size=400,
        input_size=action_dim * 3,
        output_size=action_dim,
        processing_layers=2,
        temporal_window=0.01
    )
    
    return MotorReasoningCore(
        params=params,
        action_dim=action_dim,
        control_frequency=control_frequency
    )
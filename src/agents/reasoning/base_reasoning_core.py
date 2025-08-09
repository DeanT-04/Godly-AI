"""
Base Reasoning Core Implementation

This module provides the base class for all specialized reasoning cores in the
multi-modal reasoning system. Each reasoning core uses a specialized reservoir
module for domain-specific processing.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
from enum import Enum

from ...core.liquid_state_machine import LiquidStateMachine, LSMParams, LSMState


class ModalityType(Enum):
    """Enumeration of supported modality types."""
    VISUAL = "visual"
    AUDIO = "audio"
    TEXT = "text"
    MOTOR = "motor"


@dataclass
class ReasoningCoreParams:
    """Parameters for reasoning core configuration."""
    
    # Core identification
    modality: ModalityType
    core_id: str
    
    # Reservoir configuration
    reservoir_size: int = 500
    input_size: int = 100
    output_size: int = 50
    
    # Modality-specific parameters
    processing_layers: int = 3
    temporal_window: float = 0.1  # 100ms temporal window
    
    # Competition and synchronization
    competition_strength: float = 0.5
    sync_threshold: float = 0.3
    
    # Resource allocation
    base_resource_allocation: float = 0.25  # Equal allocation by default
    max_resource_allocation: float = 0.8    # Maximum resources one core can use


class ReasoningCoreState:
    """State container for reasoning core."""
    
    def __init__(
        self,
        lsm_state: LSMState,
        activity_level: float = 0.0,
        resource_allocation: float = 0.25,
        sync_signals: Dict[str, jnp.ndarray] = None,
        processing_history: List[jnp.ndarray] = None
    ):
        self.lsm_state = lsm_state
        self.activity_level = activity_level
        self.resource_allocation = resource_allocation
        self.sync_signals = sync_signals or {}
        self.processing_history = processing_history or []


class BaseReasoningCore(ABC):
    """
    Base class for all specialized reasoning cores.
    
    Each reasoning core implements domain-specific processing using liquid state
    machines with modality-specific adaptations. Cores can synchronize with peers
    and compete for computational resources.
    """
    
    def __init__(self, params: ReasoningCoreParams):
        """Initialize the reasoning core."""
        self.params = params
        
        # Create specialized LSM for this modality
        lsm_params = self._create_lsm_params()
        self.lsm = LiquidStateMachine(lsm_params)
        
        # Initialize peer connections
        self.peer_cores: Dict[str, 'BaseReasoningCore'] = {}
        
        # Performance tracking
        self.performance_history: List[float] = []
        
    def _create_lsm_params(self) -> LSMParams:
        """Create LSM parameters specialized for this modality."""
        return LSMParams(
            reservoir_size=self.params.reservoir_size,
            input_size=self.params.input_size,
            output_size=self.params.output_size,
            reservoir_connectivity=self._get_connectivity_pattern(),
            spectral_radius=self._get_spectral_radius(),
            enable_plasticity=True,
            homeostatic_scaling=True
        )
    
    @abstractmethod
    def _get_connectivity_pattern(self) -> float:
        """Get modality-specific connectivity pattern."""
        pass
    
    @abstractmethod
    def _get_spectral_radius(self) -> float:
        """Get modality-specific spectral radius."""
        pass
    
    @abstractmethod
    def preprocess_input(self, raw_input: jnp.ndarray) -> jnp.ndarray:
        """Preprocess raw input for modality-specific encoding."""
        pass
    
    @abstractmethod
    def postprocess_output(self, raw_output: jnp.ndarray) -> jnp.ndarray:
        """Postprocess LSM output for modality-specific interpretation."""
        pass
    
    def init_state(self, key: Optional[jax.random.PRNGKey] = None) -> ReasoningCoreState:
        """Initialize reasoning core state."""
        if key is None:
            key = random.PRNGKey(42)
        
        # Initialize LSM state
        lsm_state = self.lsm.init_state(key)
        
        return ReasoningCoreState(
            lsm_state=lsm_state,
            activity_level=0.0,
            resource_allocation=self.params.base_resource_allocation
        )
    
    def process_input(
        self,
        state: ReasoningCoreState,
        input_data: jnp.ndarray,
        dt: float,
        t: float,
        key: Optional[jax.random.PRNGKey] = None
    ) -> Tuple[jnp.ndarray, ReasoningCoreState]:
        """
        Process input through the reasoning core.
        
        Args:
            state: Current reasoning core state
            input_data: Raw input data
            dt: Time step
            t: Current time
            key: Random key for stochastic processes
            
        Returns:
            Tuple of (processed_output, updated_state)
        """
        if key is None:
            key = random.PRNGKey(int(t * 1000))
        
        # Preprocess input for this modality
        processed_input = self.preprocess_input(input_data)
        
        # Apply resource allocation scaling
        scaled_input = processed_input * state.resource_allocation
        
        # Process through LSM
        new_lsm_state = self.lsm.step(
            state.lsm_state, scaled_input, dt, t, key
        )
        
        # Compute LSM output
        lsm_output = self.lsm.compute_readout(new_lsm_state)
        
        # Postprocess output
        final_output = self.postprocess_output(lsm_output)
        
        # Update activity level based on output magnitude
        activity_level = float(jnp.mean(jnp.abs(final_output)))
        
        # Update processing history
        new_history = state.processing_history[-9:] + [final_output]  # Keep last 10
        
        # Create updated state
        new_state = ReasoningCoreState(
            lsm_state=new_lsm_state,
            activity_level=activity_level,
            resource_allocation=state.resource_allocation,
            sync_signals=state.sync_signals.copy(),
            processing_history=new_history
        )
        
        return final_output, new_state
    
    def synchronize_with_peers(
        self,
        state: ReasoningCoreState,
        peer_states: Dict[str, ReasoningCoreState]
    ) -> ReasoningCoreState:
        """
        Synchronize with peer reasoning cores through spike synchronization.
        
        Args:
            state: Current reasoning core state
            peer_states: States of peer reasoning cores
            
        Returns:
            Updated reasoning core state with synchronization signals
        """
        sync_signals = {}
        
        for peer_id, peer_state in peer_states.items():
            if peer_id in self.peer_cores:
                # Compute synchronization signal based on spike correlation
                sync_signal = self._compute_sync_signal(state, peer_state)
                sync_signals[peer_id] = sync_signal
        
        # Update state with synchronization signals
        new_state = ReasoningCoreState(
            lsm_state=state.lsm_state,
            activity_level=state.activity_level,
            resource_allocation=state.resource_allocation,
            sync_signals=sync_signals,
            processing_history=state.processing_history
        )
        
        return new_state
    
    def _compute_sync_signal(
        self,
        own_state: ReasoningCoreState,
        peer_state: ReasoningCoreState
    ) -> jnp.ndarray:
        """Compute synchronization signal with a peer core."""
        # Get current spike patterns
        own_spikes = own_state.lsm_state.neuron_state.spikes.astype(float)
        peer_spikes = peer_state.lsm_state.neuron_state.spikes.astype(float)
        
        # Handle different reservoir sizes by using statistical measures
        # instead of direct dot product
        own_activity = jnp.mean(own_spikes)
        peer_activity = jnp.mean(peer_spikes)
        
        # Compute synchronization based on activity correlation
        # Use activity levels and patterns
        own_pattern = jnp.array([
            jnp.mean(own_spikes),
            jnp.std(own_spikes),
            jnp.sum(own_spikes > 0.5) / len(own_spikes)  # Spike rate
        ])
        
        peer_pattern = jnp.array([
            jnp.mean(peer_spikes),
            jnp.std(peer_spikes), 
            jnp.sum(peer_spikes > 0.5) / len(peer_spikes)  # Spike rate
        ])
        
        # Compute pattern similarity
        sync_strength = jnp.dot(own_pattern, peer_pattern) / (
            jnp.linalg.norm(own_pattern) * jnp.linalg.norm(peer_pattern) + 1e-8
        )
        
        # Create synchronization signal
        sync_signal = jnp.array([sync_strength])
        
        return sync_signal
    
    def compete_for_resources(
        self,
        state: ReasoningCoreState,
        peer_states: Dict[str, ReasoningCoreState],
        total_resources: float = 1.0
    ) -> float:
        """
        Compete for computational resources based on activity and performance.
        
        Args:
            state: Current reasoning core state
            peer_states: States of peer reasoning cores
            total_resources: Total available resources
            
        Returns:
            New resource allocation for this core
        """
        # Compute competition score based on activity level and recent performance
        activity_score = state.activity_level
        
        # Add performance bonus if we have performance history
        performance_score = 0.0
        if self.performance_history:
            recent_performance = jnp.mean(jnp.array(self.performance_history[-5:]))
            performance_score = float(recent_performance)
        
        # Combined competition score
        competition_score = (
            activity_score * 0.7 + 
            performance_score * 0.3
        ) * self.params.competition_strength
        
        # Collect all competition scores
        all_scores = {"self": competition_score}
        for peer_id, peer_state in peer_states.items():
            if peer_id in self.peer_cores:
                peer_core = self.peer_cores[peer_id]
                peer_activity = peer_state.activity_level
                peer_performance = 0.0
                if peer_core.performance_history:
                    peer_performance = float(jnp.mean(
                        jnp.array(peer_core.performance_history[-5:])
                    ))
                
                peer_score = (
                    peer_activity * 0.7 + 
                    peer_performance * 0.3
                ) * peer_core.params.competition_strength
                
                all_scores[peer_id] = peer_score
        
        # Compute softmax allocation
        scores_array = jnp.array(list(all_scores.values()))
        softmax_weights = jax.nn.softmax(scores_array)
        
        # Get this core's allocation
        self_index = list(all_scores.keys()).index("self")
        base_allocation = float(softmax_weights[self_index]) * total_resources
        
        # Apply constraints
        new_allocation = jnp.clip(
            base_allocation,
            0.1,  # Minimum allocation
            self.params.max_resource_allocation
        )
        
        return float(new_allocation)
    
    def register_peer(self, peer_id: str, peer_core: 'BaseReasoningCore'):
        """Register a peer reasoning core for synchronization and competition."""
        self.peer_cores[peer_id] = peer_core
    
    def update_performance(self, performance_score: float):
        """Update performance history for resource competition."""
        self.performance_history.append(performance_score)
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_modality_info(self) -> Dict[str, Any]:
        """Get information about this reasoning core's modality."""
        return {
            "modality": self.params.modality.value,
            "core_id": self.params.core_id,
            "reservoir_size": self.params.reservoir_size,
            "input_size": self.params.input_size,
            "output_size": self.params.output_size,
            "num_peers": len(self.peer_cores),
            "performance_history_length": len(self.performance_history)
        }
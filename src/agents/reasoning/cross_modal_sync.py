"""
Cross-Modal Synchronization System

This module implements the cross-modal synchronization system that enables
coordination between different reasoning cores through spike synchronization,
competitive resource allocation, and cross-modal information integration.
"""

from typing import Dict, List, Tuple, Optional, Any
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .base_reasoning_core import BaseReasoningCore, ReasoningCoreState, ModalityType


class SyncMode(Enum):
    """Synchronization modes for cross-modal coordination."""
    PHASE_LOCK = "phase_lock"      # Phase-locked synchronization
    COHERENCE = "coherence"        # Coherence-based synchronization
    COMPETITIVE = "competitive"    # Competitive synchronization
    COOPERATIVE = "cooperative"    # Cooperative synchronization


@dataclass
class SyncParams:
    """Parameters for cross-modal synchronization."""
    
    # Synchronization parameters
    sync_strength: float = 0.5          # Overall synchronization strength
    phase_coupling: float = 0.3         # Phase coupling strength
    coherence_threshold: float = 0.4    # Coherence threshold for sync
    
    # Competition parameters
    competition_strength: float = 0.7   # Resource competition strength
    resource_decay: float = 0.95        # Resource allocation decay
    min_allocation: float = 0.1         # Minimum resource allocation
    
    # Integration parameters
    integration_window: float = 0.1     # Integration time window (100ms)
    cross_modal_gain: float = 0.2       # Cross-modal influence gain
    attention_decay: float = 0.9        # Attention decay rate
    
    # Temporal parameters
    sync_time_constant: float = 0.05    # Synchronization time constant
    adaptation_rate: float = 0.01       # Adaptation rate for sync parameters


class CrossModalSynchronizer:
    """
    Cross-modal synchronization system for coordinating reasoning cores.
    
    This system manages synchronization between different modality reasoning cores,
    implements competitive resource allocation, and enables cross-modal information
    integration through spike-based communication.
    """
    
    def __init__(self, params: Optional[SyncParams] = None):
        """Initialize cross-modal synchronizer."""
        self.params = params or SyncParams()
        
        # Registered reasoning cores
        self.cores: Dict[str, BaseReasoningCore] = {}
        self.core_states: Dict[str, ReasoningCoreState] = {}
        
        # Synchronization state
        self.sync_history: List[Dict[str, float]] = []
        self.resource_history: List[Dict[str, float]] = []
        self.integration_history: List[Dict[str, jnp.ndarray]] = []
        
        # Cross-modal connections
        self.connection_weights: Dict[Tuple[str, str], float] = {}
        self.sync_phases: Dict[str, float] = {}
        
        # Performance tracking
        self.sync_performance: Dict[str, List[float]] = {}
        self.integration_performance: Dict[str, List[float]] = {}
        
    def register_core(self, core: BaseReasoningCore, initial_state: ReasoningCoreState):
        """Register a reasoning core for synchronization."""
        core_id = core.params.core_id
        self.cores[core_id] = core
        self.core_states[core_id] = initial_state
        
        # Initialize synchronization state
        self.sync_phases[core_id] = 0.0
        self.sync_performance[core_id] = []
        self.integration_performance[core_id] = []
        
        # Initialize connections with existing cores
        for existing_id in self.cores.keys():
            if existing_id != core_id:
                # Initialize bidirectional connections
                self.connection_weights[(core_id, existing_id)] = 0.5
                self.connection_weights[(existing_id, core_id)] = 0.5
    
    def synchronize_cores(
        self,
        dt: float,
        t: float,
        sync_mode: SyncMode = SyncMode.COHERENCE
    ) -> Dict[str, ReasoningCoreState]:
        """
        Synchronize all registered reasoning cores.
        
        Args:
            dt: Time step
            t: Current time
            sync_mode: Synchronization mode
            
        Returns:
            Updated core states with synchronization
        """
        if len(self.cores) < 2:
            return self.core_states.copy()
        
        # Compute synchronization signals
        sync_signals = self._compute_sync_signals(sync_mode, dt, t)
        
        # Apply synchronization to each core
        updated_states = {}
        for core_id, core_state in self.core_states.items():
            # Apply synchronization influence
            sync_influence = sync_signals.get(core_id, jnp.zeros(1))
            updated_state = self._apply_sync_influence(
                core_state, sync_influence, dt
            )
            updated_states[core_id] = updated_state
        
        # Update synchronization history
        sync_metrics = {
            core_id: float(jnp.mean(jnp.abs(sync_signals.get(core_id, jnp.zeros(1)))))
            for core_id in self.cores.keys()
        }
        self.sync_history.append(sync_metrics)
        
        # Keep history bounded
        if len(self.sync_history) > 1000:
            self.sync_history = self.sync_history[-1000:]
        
        self.core_states = updated_states
        return updated_states
    
    def _compute_sync_signals(
        self,
        sync_mode: SyncMode,
        dt: float,
        t: float
    ) -> Dict[str, jnp.ndarray]:
        """Compute synchronization signals for all cores."""
        sync_signals = {}
        
        if sync_mode == SyncMode.PHASE_LOCK:
            sync_signals = self._compute_phase_lock_signals(dt, t)
        elif sync_mode == SyncMode.COHERENCE:
            sync_signals = self._compute_coherence_signals(dt, t)
        elif sync_mode == SyncMode.COMPETITIVE:
            sync_signals = self._compute_competitive_signals(dt, t)
        else:  # COOPERATIVE
            sync_signals = self._compute_cooperative_signals(dt, t)
        
        return sync_signals
    
    def _compute_phase_lock_signals(
        self,
        dt: float,
        t: float
    ) -> Dict[str, jnp.ndarray]:
        """Compute phase-locking synchronization signals."""
        sync_signals = {}
        
        # Extract spike phases from each core
        core_phases = {}
        for core_id, state in self.core_states.items():
            spikes = state.lsm_state.neuron_state.spikes.astype(float)
            # Compute instantaneous phase using Hilbert transform approximation
            phase = self._compute_spike_phase(spikes)
            core_phases[core_id] = phase
        
        # Compute phase-locking signals
        for core_id in self.cores.keys():
            phase_diffs = []
            weights = []
            
            for other_id in self.cores.keys():
                if other_id != core_id:
                    # Phase difference
                    phase_diff = core_phases[other_id] - core_phases[core_id]
                    phase_diffs.append(phase_diff)
                    
                    # Connection weight
                    weight = self.connection_weights.get((core_id, other_id), 0.5)
                    weights.append(weight)
            
            if phase_diffs:
                # Weighted average of phase differences
                phase_diffs = jnp.array(phase_diffs)
                weights = jnp.array(weights)
                
                avg_phase_diff = jnp.sum(phase_diffs * weights) / jnp.sum(weights)
                
                # Phase-locking signal (Kuramoto-like)
                sync_signal = self.params.phase_coupling * jnp.sin(avg_phase_diff)
                sync_signals[core_id] = jnp.array([sync_signal])
            else:
                sync_signals[core_id] = jnp.zeros(1)
        
        return sync_signals
    
    def _compute_coherence_signals(
        self,
        dt: float,
        t: float
    ) -> Dict[str, jnp.ndarray]:
        """Compute coherence-based synchronization signals."""
        sync_signals = {}
        
        # Compute coherence between all pairs of cores
        coherence_matrix = self._compute_coherence_matrix()
        
        for core_id in self.cores.keys():
            coherence_values = []
            weights = []
            
            for other_id in self.cores.keys():
                if other_id != core_id:
                    coherence = coherence_matrix.get((core_id, other_id), 0.0)
                    coherence_values.append(coherence)
                    
                    weight = self.connection_weights.get((core_id, other_id), 0.5)
                    weights.append(weight)
            
            if coherence_values:
                coherence_values = jnp.array(coherence_values)
                weights = jnp.array(weights)
                
                # Weighted coherence signal
                avg_coherence = jnp.sum(coherence_values * weights) / jnp.sum(weights)
                
                # Synchronization signal based on coherence threshold
                if avg_coherence > self.params.coherence_threshold:
                    sync_strength = (avg_coherence - self.params.coherence_threshold) * 2.0
                    sync_signal = self.params.sync_strength * sync_strength
                else:
                    sync_signal = 0.0
                
                sync_signals[core_id] = jnp.array([sync_signal])
            else:
                sync_signals[core_id] = jnp.zeros(1)
        
        return sync_signals
    
    def _compute_competitive_signals(
        self,
        dt: float,
        t: float
    ) -> Dict[str, jnp.ndarray]:
        """Compute competitive synchronization signals."""
        sync_signals = {}
        
        # Get activity levels from all cores
        activities = {}
        for core_id, state in self.core_states.items():
            activities[core_id] = state.activity_level
        
        # Compute competitive signals
        total_activity = sum(activities.values())
        
        for core_id in self.cores.keys():
            if total_activity > 0:
                # Competitive advantage based on relative activity
                relative_activity = activities[core_id] / total_activity
                
                # Winner-take-more dynamics
                competitive_signal = (relative_activity - 0.25) * 4.0  # Center around equal share
                competitive_signal = jnp.tanh(competitive_signal)  # Bounded
                
                sync_signals[core_id] = jnp.array([competitive_signal])
            else:
                sync_signals[core_id] = jnp.zeros(1)
        
        return sync_signals
    
    def _compute_cooperative_signals(
        self,
        dt: float,
        t: float
    ) -> Dict[str, jnp.ndarray]:
        """Compute cooperative synchronization signals."""
        sync_signals = {}
        
        # Compute mutual information between cores
        mutual_info_matrix = self._compute_mutual_information_matrix()
        
        for core_id in self.cores.keys():
            cooperation_values = []
            weights = []
            
            for other_id in self.cores.keys():
                if other_id != core_id:
                    mutual_info = mutual_info_matrix.get((core_id, other_id), 0.0)
                    cooperation_values.append(mutual_info)
                    
                    weight = self.connection_weights.get((core_id, other_id), 0.5)
                    weights.append(weight)
            
            if cooperation_values:
                cooperation_values = jnp.array(cooperation_values)
                weights = jnp.array(weights)
                
                # Cooperative signal based on mutual information
                avg_cooperation = jnp.sum(cooperation_values * weights) / jnp.sum(weights)
                cooperative_signal = self.params.sync_strength * avg_cooperation
                
                sync_signals[core_id] = jnp.array([cooperative_signal])
            else:
                sync_signals[core_id] = jnp.zeros(1)
        
        return sync_signals
    
    def _compute_spike_phase(self, spikes: jnp.ndarray) -> float:
        """Compute instantaneous phase from spike pattern."""
        if jnp.sum(spikes) == 0:
            return 0.0
        
        # Simple phase estimation based on spike timing
        spike_indices = jnp.where(spikes > 0.5)[0]
        
        if len(spike_indices) == 0:
            return 0.0
        
        # Use center of mass of spikes as phase indicator
        phase = jnp.mean(spike_indices) / len(spikes) * 2 * jnp.pi
        return float(phase % (2 * jnp.pi))
    
    def _compute_coherence_matrix(self) -> Dict[Tuple[str, str], float]:
        """Compute coherence between all pairs of cores."""
        coherence_matrix = {}
        
        for core_id1 in self.cores.keys():
            for core_id2 in self.cores.keys():
                if core_id1 != core_id2:
                    coherence = self._compute_pairwise_coherence(core_id1, core_id2)
                    coherence_matrix[(core_id1, core_id2)] = coherence
        
        return coherence_matrix
    
    def _compute_pairwise_coherence(self, core_id1: str, core_id2: str) -> float:
        """Compute coherence between two cores."""
        state1 = self.core_states[core_id1]
        state2 = self.core_states[core_id2]
        
        spikes1 = state1.lsm_state.neuron_state.spikes.astype(float)
        spikes2 = state2.lsm_state.neuron_state.spikes.astype(float)
        
        # Compute cross-correlation
        if len(spikes1) == 0 or len(spikes2) == 0:
            return 0.0
        
        # Normalize spike patterns
        norm1 = jnp.linalg.norm(spikes1)
        norm2 = jnp.linalg.norm(spikes2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute coherence as normalized cross-correlation
        min_len = min(len(spikes1), len(spikes2))
        coherence = jnp.dot(spikes1[:min_len], spikes2[:min_len]) / (norm1 * norm2)
        
        return float(jnp.abs(coherence))
    
    def _compute_mutual_information_matrix(self) -> Dict[Tuple[str, str], float]:
        """Compute mutual information between all pairs of cores."""
        mutual_info_matrix = {}
        
        for core_id1 in self.cores.keys():
            for core_id2 in self.cores.keys():
                if core_id1 != core_id2:
                    mutual_info = self._compute_pairwise_mutual_information(core_id1, core_id2)
                    mutual_info_matrix[(core_id1, core_id2)] = mutual_info
        
        return mutual_info_matrix
    
    def _compute_pairwise_mutual_information(self, core_id1: str, core_id2: str) -> float:
        """Compute mutual information between two cores."""
        state1 = self.core_states[core_id1]
        state2 = self.core_states[core_id2]
        
        spikes1 = state1.lsm_state.neuron_state.spikes.astype(float)
        spikes2 = state2.lsm_state.neuron_state.spikes.astype(float)
        
        # Simplified mutual information estimation
        # In practice, would use more sophisticated methods
        
        if len(spikes1) == 0 or len(spikes2) == 0:
            return 0.0
        
        # Discretize spike patterns
        bins1 = jnp.where(spikes1 > 0.5, 1, 0)
        bins2 = jnp.where(spikes2 > 0.5, 1, 0)
        
        min_len = min(len(bins1), len(bins2))
        bins1 = bins1[:min_len]
        bins2 = bins2[:min_len]
        
        # Compute joint and marginal probabilities
        p_00 = jnp.mean((bins1 == 0) & (bins2 == 0))
        p_01 = jnp.mean((bins1 == 0) & (bins2 == 1))
        p_10 = jnp.mean((bins1 == 1) & (bins2 == 0))
        p_11 = jnp.mean((bins1 == 1) & (bins2 == 1))
        
        p_0 = jnp.mean(bins1 == 0)
        p_1 = jnp.mean(bins1 == 1)
        q_0 = jnp.mean(bins2 == 0)
        q_1 = jnp.mean(bins2 == 1)
        
        # Compute mutual information
        mutual_info = 0.0
        
        for p_xy, p_x, p_y in [(p_00, p_0, q_0), (p_01, p_0, q_1), 
                               (p_10, p_1, q_0), (p_11, p_1, q_1)]:
            if p_xy > 1e-8 and p_x > 1e-8 and p_y > 1e-8:
                mutual_info += p_xy * jnp.log(p_xy / (p_x * p_y))
        
        return float(jnp.maximum(mutual_info, 0.0))
    
    def _apply_sync_influence(
        self,
        state: ReasoningCoreState,
        sync_signal: jnp.ndarray,
        dt: float
    ) -> ReasoningCoreState:
        """Apply synchronization influence to a core state."""
        # Create updated state with synchronization influence
        new_sync_signals = state.sync_signals.copy()
        new_sync_signals['global_sync'] = sync_signal
        
        # Apply synchronization to LSM state (simplified)
        # In practice, this would modify the LSM dynamics
        new_lsm_state = state.lsm_state  # Keep LSM state unchanged for now
        
        updated_state = ReasoningCoreState(
            lsm_state=new_lsm_state,
            activity_level=state.activity_level,
            resource_allocation=state.resource_allocation,
            sync_signals=new_sync_signals,
            processing_history=state.processing_history
        )
        
        return updated_state
    
    def allocate_resources(
        self,
        total_resources: float = 1.0,
        allocation_mode: str = "competitive"
    ) -> Dict[str, float]:
        """
        Allocate computational resources among reasoning cores.
        
        Args:
            total_resources: Total available resources
            allocation_mode: Resource allocation mode ("competitive", "equal", "adaptive")
            
        Returns:
            Resource allocation for each core
        """
        if not self.cores:
            return {}
        
        if allocation_mode == "equal":
            # Equal allocation
            allocation_per_core = total_resources / len(self.cores)
            allocations = {core_id: allocation_per_core for core_id in self.cores.keys()}
        
        elif allocation_mode == "adaptive":
            # Adaptive allocation based on performance history
            allocations = self._compute_adaptive_allocation(total_resources)
        
        else:  # competitive
            # Competitive allocation based on activity and performance
            allocations = self._compute_competitive_allocation(total_resources)
        
        # Apply resource constraints
        for core_id in allocations:
            core = self.cores[core_id]
            allocations[core_id] = jnp.clip(
                allocations[core_id],
                self.params.min_allocation,
                core.params.max_resource_allocation
            )
        
        # Normalize to ensure total doesn't exceed available resources
        total_allocated = sum(allocations.values())
        if total_allocated > total_resources:
            scale_factor = total_resources / total_allocated
            allocations = {k: v * scale_factor for k, v in allocations.items()}
        
        # Update core states with new allocations
        for core_id, allocation in allocations.items():
            if core_id in self.core_states:
                state = self.core_states[core_id]
                self.core_states[core_id] = ReasoningCoreState(
                    lsm_state=state.lsm_state,
                    activity_level=state.activity_level,
                    resource_allocation=float(allocation),
                    sync_signals=state.sync_signals,
                    processing_history=state.processing_history
                )
        
        # Update resource history
        self.resource_history.append(allocations.copy())
        if len(self.resource_history) > 1000:
            self.resource_history = self.resource_history[-1000:]
        
        return allocations
    
    def _compute_competitive_allocation(self, total_resources: float) -> Dict[str, float]:
        """Compute competitive resource allocation."""
        # Get activity levels and performance scores
        scores = {}
        
        for core_id in self.cores.keys():
            state = self.core_states[core_id]
            core = self.cores[core_id]
            
            # Activity score
            activity_score = state.activity_level
            
            # Performance score
            performance_score = 0.0
            if core.performance_history:
                recent_performance = jnp.mean(jnp.array(core.performance_history[-5:]))
                performance_score = float(recent_performance)
            
            # Combined score
            combined_score = (
                activity_score * 0.6 + 
                performance_score * 0.4
            ) * self.params.competition_strength
            
            scores[core_id] = combined_score
        
        # Softmax allocation
        score_values = jnp.array(list(scores.values()))
        if jnp.sum(score_values) > 0:
            allocation_weights = jax.nn.softmax(score_values)
        else:
            allocation_weights = jnp.ones(len(scores)) / len(scores)
        
        # Convert to allocations
        allocations = {}
        for i, core_id in enumerate(scores.keys()):
            allocations[core_id] = float(allocation_weights[i] * total_resources)
        
        return allocations
    
    def _compute_adaptive_allocation(self, total_resources: float) -> Dict[str, float]:
        """Compute adaptive resource allocation based on learning progress."""
        allocations = {}
        
        # Base allocation
        base_allocation = total_resources / len(self.cores)
        
        for core_id in self.cores.keys():
            core = self.cores[core_id]
            
            # Compute learning progress
            learning_progress = 0.0
            if len(core.performance_history) >= 2:
                recent_performance = jnp.mean(jnp.array(core.performance_history[-3:]))
                older_performance = jnp.mean(jnp.array(core.performance_history[-6:-3]))
                learning_progress = recent_performance - older_performance
            
            # Adaptive bonus based on learning progress
            adaptive_bonus = learning_progress * 0.2 * base_allocation
            
            allocations[core_id] = base_allocation + adaptive_bonus
        
        return allocations
    
    def integrate_cross_modal_information(
        self,
        primary_core_id: str,
        integration_strength: float = None
    ) -> jnp.ndarray:
        """
        Integrate information across modalities for enhanced processing.
        
        Args:
            primary_core_id: ID of the primary core receiving integration
            integration_strength: Strength of cross-modal integration
            
        Returns:
            Integrated cross-modal features
        """
        if integration_strength is None:
            integration_strength = self.params.cross_modal_gain
        
        if primary_core_id not in self.cores:
            return jnp.zeros(1)
        
        primary_state = self.core_states[primary_core_id]
        primary_spikes = primary_state.lsm_state.neuron_state.spikes.astype(float)
        
        # Collect information from other modalities
        cross_modal_features = []
        
        for other_id in self.cores.keys():
            if other_id != primary_core_id:
                other_state = self.core_states[other_id]
                other_spikes = other_state.lsm_state.neuron_state.spikes.astype(float)
                
                # Compute cross-modal feature
                connection_weight = self.connection_weights.get(
                    (primary_core_id, other_id), 0.5
                )
                
                # Cross-modal feature based on spike correlation
                if len(primary_spikes) > 0 and len(other_spikes) > 0:
                    min_len = min(len(primary_spikes), len(other_spikes))
                    correlation = jnp.corrcoef(
                        primary_spikes[:min_len], 
                        other_spikes[:min_len]
                    )[0, 1]
                    
                    if jnp.isnan(correlation):
                        correlation = 0.0
                    
                    cross_modal_feature = connection_weight * correlation * integration_strength
                    cross_modal_features.append(cross_modal_feature)
        
        # Combine cross-modal features
        if cross_modal_features:
            integrated_features = jnp.array(cross_modal_features)
        else:
            integrated_features = jnp.zeros(1)
        
        # Update integration history
        integration_data = {
            'primary_core': primary_core_id,
            'features': integrated_features,
            'timestamp': len(self.integration_history)
        }
        self.integration_history.append(integration_data)
        
        if len(self.integration_history) > 1000:
            self.integration_history = self.integration_history[-1000:]
        
        return integrated_features
    
    def update_connection_weights(
        self,
        learning_rate: float = None,
        adaptation_mode: str = "hebbian"
    ):
        """
        Update cross-modal connection weights based on activity patterns.
        
        Args:
            learning_rate: Learning rate for weight updates
            adaptation_mode: Weight adaptation mode ("hebbian", "anti_hebbian", "homeostatic")
        """
        if learning_rate is None:
            learning_rate = self.params.adaptation_rate
        
        for (core_id1, core_id2), current_weight in self.connection_weights.items():
            if core_id1 in self.core_states and core_id2 in self.core_states:
                state1 = self.core_states[core_id1]
                state2 = self.core_states[core_id2]
                
                activity1 = state1.activity_level
                activity2 = state2.activity_level
                
                if adaptation_mode == "hebbian":
                    # Hebbian learning: strengthen connections between co-active cores
                    weight_update = learning_rate * activity1 * activity2
                elif adaptation_mode == "anti_hebbian":
                    # Anti-Hebbian: weaken connections between co-active cores
                    weight_update = -learning_rate * activity1 * activity2
                else:  # homeostatic
                    # Homeostatic: maintain balanced connectivity
                    target_activity = 0.5
                    weight_update = learning_rate * (
                        (target_activity - activity1) * activity2 +
                        (target_activity - activity2) * activity1
                    ) * 0.5
                
                # Update weight with bounds
                new_weight = current_weight + weight_update
                self.connection_weights[(core_id1, core_id2)] = float(
                    jnp.clip(new_weight, 0.0, 1.0)
                )
    
    def get_synchronization_metrics(self) -> Dict[str, Any]:
        """Get comprehensive synchronization metrics."""
        if not self.sync_history:
            return {}
        
        metrics = {}
        
        # Recent synchronization strength
        recent_sync = self.sync_history[-10:] if len(self.sync_history) >= 10 else self.sync_history
        
        for core_id in self.cores.keys():
            core_sync_values = [entry.get(core_id, 0.0) for entry in recent_sync]
            
            metrics[f"{core_id}_sync_mean"] = float(jnp.mean(jnp.array(core_sync_values)))
            metrics[f"{core_id}_sync_std"] = float(jnp.std(jnp.array(core_sync_values)))
        
        # Global synchronization metrics
        if len(self.cores) > 1:
            # Compute pairwise synchronization
            pairwise_sync = []
            for i, core_id1 in enumerate(self.cores.keys()):
                for j, core_id2 in enumerate(self.cores.keys()):
                    if i < j:  # Avoid duplicates
                        coherence = self._compute_pairwise_coherence(core_id1, core_id2)
                        pairwise_sync.append(coherence)
            
            if pairwise_sync:
                metrics["global_coherence"] = float(jnp.mean(jnp.array(pairwise_sync)))
                metrics["coherence_std"] = float(jnp.std(jnp.array(pairwise_sync)))
        
        # Resource allocation metrics
        if self.resource_history:
            recent_resources = self.resource_history[-10:]
            for core_id in self.cores.keys():
                resource_values = [entry.get(core_id, 0.0) for entry in recent_resources]
                metrics[f"{core_id}_resource_mean"] = float(jnp.mean(jnp.array(resource_values)))
        
        return metrics
    
    def reset_synchronization(self):
        """Reset synchronization state."""
        self.sync_history.clear()
        self.resource_history.clear()
        self.integration_history.clear()
        
        # Reset sync phases
        for core_id in self.cores.keys():
            self.sync_phases[core_id] = 0.0
        
        # Reset connection weights to default
        for connection in self.connection_weights:
            self.connection_weights[connection] = 0.5


def create_cross_modal_synchronizer(
    sync_strength: float = 0.5,
    competition_strength: float = 0.7,
    integration_gain: float = 0.2
) -> CrossModalSynchronizer:
    """
    Create a cross-modal synchronizer with specified parameters.
    
    Args:
        sync_strength: Overall synchronization strength
        competition_strength: Resource competition strength
        integration_gain: Cross-modal integration gain
        
    Returns:
        Configured cross-modal synchronizer
    """
    params = SyncParams(
        sync_strength=sync_strength,
        competition_strength=competition_strength,
        cross_modal_gain=integration_gain
    )
    
    return CrossModalSynchronizer(params)
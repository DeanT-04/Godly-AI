"""
Cross-Modal Synchronization System Demo

This demo shows how the cross-modal synchronization system coordinates
different reasoning cores through spike synchronization, competitive
resource allocation, and cross-modal information integration.
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.reasoning import (
    create_visual_reasoning_core,
    create_audio_reasoning_core,
    create_text_reasoning_core,
    create_motor_reasoning_core,
    create_cross_modal_synchronizer,
    SyncMode
)


def create_sample_inputs():
    """Create sample inputs for different modalities."""
    key = random.PRNGKey(42)
    keys = random.split(key, 4)
    
    # Visual input: 32x32 grayscale image
    visual_input = random.normal(keys[0], (32, 32, 1)) * 0.5 + 0.5
    visual_input = jnp.clip(visual_input, 0, 1)
    
    # Audio input: mel-spectrogram features
    audio_input = random.normal(keys[1], (64,)) * 0.3 + 0.2
    audio_input = jnp.abs(audio_input)
    
    # Text input: character sequence (as indices)
    text_input = random.randint(keys[2], (128,), 0, 256)
    
    # Motor input: 6-DOF state (current, target, error)
    motor_input = random.normal(keys[3], (18,)) * 0.1
    
    return visual_input, audio_input, text_input, motor_input


def run_cross_modal_sync_demo():
    """Run the cross-modal synchronization demonstration."""
    print("Cross-Modal Synchronization System Demo")
    print("=" * 50)
    
    # Initialize random key
    key = random.PRNGKey(123)
    keys = random.split(key, 5)
    
    # Create reasoning cores
    print("\n1. Creating reasoning cores...")
    visual_core = create_visual_reasoning_core(core_id="visual_demo")
    audio_core = create_audio_reasoning_core(core_id="audio_demo")
    text_core = create_text_reasoning_core(core_id="text_demo")
    motor_core = create_motor_reasoning_core(core_id="motor_demo")
    
    # Initialize core states
    visual_state = visual_core.init_state(keys[0])
    audio_state = audio_core.init_state(keys[1])
    text_state = text_core.init_state(keys[2])
    motor_state = motor_core.init_state(keys[3])
    
    print(f"  - Visual core: {visual_core.params.reservoir_size} neurons")
    print(f"  - Audio core: {audio_core.params.reservoir_size} neurons")
    print(f"  - Text core: {text_core.params.reservoir_size} neurons")
    print(f"  - Motor core: {motor_core.params.reservoir_size} neurons")
    
    # Create cross-modal synchronizer
    print("\n2. Creating cross-modal synchronizer...")
    synchronizer = create_cross_modal_synchronizer(
        sync_strength=0.6,
        competition_strength=0.8,
        integration_gain=0.3
    )
    
    # Register cores with synchronizer
    synchronizer.register_core(visual_core, visual_state)
    synchronizer.register_core(audio_core, audio_state)
    synchronizer.register_core(text_core, text_state)
    synchronizer.register_core(motor_core, motor_state)
    
    print(f"  - Registered {len(synchronizer.cores)} cores")
    print(f"  - Created {len(synchronizer.connection_weights)} connections")
    
    # Create sample inputs
    print("\n3. Creating sample inputs...")
    visual_input, audio_input, text_input, motor_input = create_sample_inputs()
    
    print(f"  - Visual input shape: {visual_input.shape}")
    print(f"  - Audio input shape: {audio_input.shape}")
    print(f"  - Text input shape: {text_input.shape}")
    print(f"  - Motor input shape: {motor_input.shape}")
    
    # Simulation parameters
    dt = 0.01  # 10ms time step
    total_time = 1.0  # 1 second simulation
    num_steps = int(total_time / dt)
    
    # Storage for results
    sync_metrics_history = []
    resource_history = []
    activity_history = {core_id: [] for core_id in synchronizer.cores.keys()}
    
    print(f"\n4. Running simulation ({num_steps} steps, {total_time}s)...")
    
    # Run simulation
    for step in range(num_steps):
        t = step * dt
        step_key = random.PRNGKey(step)
        step_keys = random.split(step_key, 4)
        
        # Process inputs through each core
        visual_output, new_visual_state = visual_core.process_input(
            synchronizer.core_states["visual_demo"],
            visual_input.flatten(),
            dt, t, step_keys[0]
        )
        
        audio_output, new_audio_state = audio_core.process_input(
            synchronizer.core_states["audio_demo"],
            audio_input,
            dt, t, step_keys[1]
        )
        
        text_output, new_text_state = text_core.process_input(
            synchronizer.core_states["text_demo"],
            text_input.astype(float),
            dt, t, step_keys[2]
        )
        
        motor_output, new_motor_state = motor_core.process_input(
            synchronizer.core_states["motor_demo"],
            motor_input,
            dt, t, step_keys[3]
        )
        
        # Update synchronizer with new states
        synchronizer.core_states["visual_demo"] = new_visual_state
        synchronizer.core_states["audio_demo"] = new_audio_state
        synchronizer.core_states["text_demo"] = new_text_state
        synchronizer.core_states["motor_demo"] = new_motor_state
        
        # Test different synchronization modes
        if step < num_steps // 4:
            sync_mode = SyncMode.COHERENCE
        elif step < num_steps // 2:
            sync_mode = SyncMode.PHASE_LOCK
        elif step < 3 * num_steps // 4:
            sync_mode = SyncMode.COMPETITIVE
        else:
            sync_mode = SyncMode.COOPERATIVE
        
        # Synchronize cores
        synchronized_states = synchronizer.synchronize_cores(dt, t, sync_mode)
        
        # Allocate resources
        if step % 10 == 0:  # Update resources every 100ms
            allocations = synchronizer.allocate_resources(
                total_resources=1.0,
                allocation_mode="competitive"
            )
            resource_history.append(allocations.copy())
        
        # Update connection weights
        if step % 50 == 0:  # Update weights every 500ms
            synchronizer.update_connection_weights(
                learning_rate=0.01,
                adaptation_mode="hebbian"
            )
        
        # Cross-modal integration
        if step % 20 == 0:  # Integration every 200ms
            for core_id in synchronizer.cores.keys():
                integrated_features = synchronizer.integrate_cross_modal_information(
                    primary_core_id=core_id,
                    integration_strength=0.2
                )
        
        # Record metrics
        if step % 10 == 0:
            metrics = synchronizer.get_synchronization_metrics()
            sync_metrics_history.append(metrics)
            
            # Record activity levels
            for core_id in synchronizer.cores.keys():
                activity = synchronizer.core_states[core_id].activity_level
                activity_history[core_id].append(activity)
        
        # Update core performance (simulate learning)
        if step % 100 == 0:
            for core_id, core in synchronizer.cores.items():
                # Simulate performance improvement
                performance = 0.5 + 0.3 * np.sin(step * 0.01) + np.random.normal(0, 0.1)
                performance = np.clip(performance, 0, 1)
                core.update_performance(performance)
    
    print("  - Simulation completed!")
    
    # Analyze results
    print("\n5. Analyzing results...")
    
    # Final synchronization metrics
    final_metrics = synchronizer.get_synchronization_metrics()
    print("\nFinal Synchronization Metrics:")
    for key, value in final_metrics.items():
        print(f"  - {key}: {value:.4f}")
    
    # Resource allocation analysis
    if resource_history:
        print("\nResource Allocation Analysis:")
        final_allocation = resource_history[-1]
        for core_id, allocation in final_allocation.items():
            print(f"  - {core_id}: {allocation:.3f}")
    
    # Activity analysis
    print("\nActivity Level Analysis:")
    for core_id, activities in activity_history.items():
        if activities:
            mean_activity = np.mean(activities)
            std_activity = np.std(activities)
            print(f"  - {core_id}: {mean_activity:.4f} ± {std_activity:.4f}")
    
    # Cross-modal integration analysis
    integration_count = len(synchronizer.integration_history)
    print(f"\nCross-Modal Integration Events: {integration_count}")
    
    # Connection weight analysis
    print("\nConnection Weight Analysis:")
    for (core1, core2), weight in synchronizer.connection_weights.items():
        print(f"  - {core1} -> {core2}: {weight:.4f}")
    
    # Visualization
    print("\n6. Creating visualizations...")
    create_visualizations(
        sync_metrics_history,
        resource_history,
        activity_history,
        synchronizer
    )
    
    print("\nDemo completed successfully!")
    return synchronizer, sync_metrics_history, resource_history, activity_history


def create_visualizations(
    sync_metrics_history: List[Dict],
    resource_history: List[Dict],
    activity_history: Dict[str, List[float]],
    synchronizer
):
    """Create visualizations of the synchronization results."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cross-Modal Synchronization System Analysis', fontsize=16)
    
    # Plot 1: Activity levels over time
    ax1 = axes[0, 0]
    for core_id, activities in activity_history.items():
        if activities:
            time_points = np.arange(len(activities)) * 0.1  # Every 100ms
            ax1.plot(time_points, activities, label=core_id, linewidth=2)
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Activity Level')
    ax1.set_title('Core Activity Levels Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Resource allocation over time
    ax2 = axes[0, 1]
    if resource_history:
        core_ids = list(resource_history[0].keys())
        time_points = np.arange(len(resource_history)) * 0.1  # Every 100ms
        
        for core_id in core_ids:
            allocations = [entry[core_id] for entry in resource_history]
            ax2.plot(time_points, allocations, label=core_id, linewidth=2)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Resource Allocation')
    ax2.set_title('Resource Allocation Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Synchronization metrics
    ax3 = axes[1, 0]
    if sync_metrics_history:
        # Extract global coherence if available
        coherence_values = []
        time_points = []
        
        for i, metrics in enumerate(sync_metrics_history):
            if 'global_coherence' in metrics:
                coherence_values.append(metrics['global_coherence'])
                time_points.append(i * 0.1)
        
        if coherence_values:
            ax3.plot(time_points, coherence_values, 'b-', linewidth=2, label='Global Coherence')
            ax3.set_ylabel('Coherence')
        
        # Add sync strength for individual cores
        core_ids = list(synchronizer.cores.keys())
        for core_id in core_ids:
            sync_key = f"{core_id}_sync_mean"
            if sync_key in sync_metrics_history[-1]:
                sync_values = []
                for metrics in sync_metrics_history:
                    if sync_key in metrics:
                        sync_values.append(metrics[sync_key])
                
                if sync_values:
                    ax3.plot(time_points[:len(sync_values)], sync_values, 
                            '--', alpha=0.7, label=f'{core_id} sync')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Synchronization Metrics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Connection weight matrix
    ax4 = axes[1, 1]
    core_ids = list(synchronizer.cores.keys())
    n_cores = len(core_ids)
    
    # Create connection matrix
    connection_matrix = np.zeros((n_cores, n_cores))
    for i, core1 in enumerate(core_ids):
        for j, core2 in enumerate(core_ids):
            if i != j:
                weight = synchronizer.connection_weights.get((core1, core2), 0.0)
                connection_matrix[i, j] = weight
    
    im = ax4.imshow(connection_matrix, cmap='viridis', vmin=0, vmax=1)
    ax4.set_xticks(range(n_cores))
    ax4.set_yticks(range(n_cores))
    ax4.set_xticklabels([cid.replace('_demo', '') for cid in core_ids], rotation=45)
    ax4.set_yticklabels([cid.replace('_demo', '') for cid in core_ids])
    ax4.set_title('Connection Weight Matrix')
    
    # Add colorbar
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    # Add weight values to matrix
    for i in range(n_cores):
        for j in range(n_cores):
            if i != j:
                text = ax4.text(j, i, f'{connection_matrix[i, j]:.2f}',
                               ha="center", va="center", color="white", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('cross_modal_sync_analysis.png', dpi=300, bbox_inches='tight')
    print("  - Saved visualization to 'cross_modal_sync_analysis.png'")
    
    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        print("  - Plot display not available in current environment")


def demonstrate_sync_modes():
    """Demonstrate different synchronization modes."""
    print("\n" + "=" * 50)
    print("Synchronization Modes Demonstration")
    print("=" * 50)
    
    # Create simple setup
    key = random.PRNGKey(456)
    keys = random.split(key, 3)
    
    visual_core = create_visual_reasoning_core(core_id="visual_sync")
    audio_core = create_audio_reasoning_core(core_id="audio_sync")
    
    visual_state = visual_core.init_state(keys[0])
    audio_state = audio_core.init_state(keys[1])
    
    synchronizer = create_cross_modal_synchronizer()
    synchronizer.register_core(visual_core, visual_state)
    synchronizer.register_core(audio_core, audio_state)
    
    # Test each synchronization mode
    modes = [SyncMode.COHERENCE, SyncMode.PHASE_LOCK, SyncMode.COMPETITIVE, SyncMode.COOPERATIVE]
    
    for mode in modes:
        print(f"\nTesting {mode.value} synchronization:")
        
        # Run synchronization
        updated_states = synchronizer.synchronize_cores(0.01, 0.0, mode)
        
        # Get sync signals
        for core_id, state in updated_states.items():
            if 'global_sync' in state.sync_signals:
                sync_signal = state.sync_signals['global_sync']
                signal_strength = float(jnp.mean(jnp.abs(sync_signal)))
                print(f"  - {core_id}: sync strength = {signal_strength:.4f}")
        
        # Get synchronization metrics
        metrics = synchronizer.get_synchronization_metrics()
        if 'global_coherence' in metrics:
            print(f"  - Global coherence: {metrics['global_coherence']:.4f}")


if __name__ == "__main__":
    # Run the main demo
    synchronizer, sync_history, resource_history, activity_history = run_cross_modal_sync_demo()
    
    # Demonstrate different sync modes
    demonstrate_sync_modes()
    
    print("\n" + "=" * 50)
    print("Cross-Modal Synchronization Demo Complete!")
    print("=" * 50)
#!/usr/bin/env python3
"""
Meta-Memory System Demo

This script demonstrates the meta-memory system's capabilities including:
- Learning strategy storage and retrieval
- Meta-parameter adaptation
- Learning-to-learn capability tracking
- Strategy template evolution
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.meta import (
    MetaMemory,
    MetaMemoryParams,
    LearningStrategy,
    create_meta_memory
)


def simulate_learning_task(
    task_name: str,
    strategy: LearningStrategy,
    meta_params: Dict[str, float],
    difficulty: float = 0.5
) -> Dict[str, Any]:
    """
    Simulate a learning task with given strategy and parameters.
    
    Args:
        task_name: Name of the task
        strategy: Learning strategy to use
        meta_params: Meta-parameters for the strategy
        difficulty: Task difficulty (0-1)
        
    Returns:
        Dictionary with learning results
    """
    # Simulate learning time based on strategy and parameters
    base_time = 60.0  # Base learning time in seconds
    
    # Different strategies have different characteristics
    if strategy == LearningStrategy.GRADIENT_DESCENT:
        learning_rate = meta_params.get('learning_rate', 0.01)
        time_factor = 1.0 / (learning_rate * 10)  # Higher LR = faster learning
        performance_factor = min(1.0, learning_rate * 50)  # But too high hurts performance
    elif strategy == LearningStrategy.EVOLUTIONARY:
        population_size = meta_params.get('population_size', 50)
        time_factor = population_size / 50.0  # Larger population = more time
        performance_factor = min(1.0, population_size / 100.0)  # But better performance
    elif strategy == LearningStrategy.REINFORCEMENT:
        exploration_rate = meta_params.get('exploration_rate', 0.1)
        time_factor = 1.0 + exploration_rate  # More exploration = more time
        performance_factor = 0.8 + exploration_rate * 0.5  # But better final performance
    else:
        time_factor = 1.0
        performance_factor = 0.7
    
    # Adjust for task difficulty
    time_factor *= (1.0 + difficulty)
    performance_factor *= (1.0 - difficulty * 0.3)
    
    # Add some randomness
    time_noise = np.random.normal(0, 0.1)
    performance_noise = np.random.normal(0, 0.05)
    
    learning_time = base_time * time_factor * (1 + time_noise)
    learning_time = max(10.0, learning_time)  # Minimum 10 seconds
    
    # Calculate performance improvement
    initial_performance = 0.2 + np.random.normal(0, 0.05)
    initial_performance = np.clip(initial_performance, 0.1, 0.4)
    
    performance_improvement = performance_factor * 0.6 + performance_noise
    performance_improvement = np.clip(performance_improvement, 0.0, 0.8)
    
    final_performance = initial_performance + performance_improvement
    final_performance = np.clip(final_performance, 0.0, 1.0)
    
    return {
        'task_name': task_name,
        'strategy': strategy,
        'meta_parameters': meta_params,
        'initial_performance': float(initial_performance),
        'final_performance': float(final_performance),
        'learning_time': float(learning_time),
        'difficulty': difficulty,
        'success': performance_improvement > 0.05
    }


def run_meta_memory_demo():
    """Run the meta-memory system demonstration."""
    print("🧠 Meta-Memory System Demo")
    print("=" * 50)
    
    # Create meta-memory system
    print("\n1. Initializing Meta-Memory System...")
    meta_memory = create_meta_memory("standard")
    key = jax.random.PRNGKey(42)
    state = meta_memory.init_state(key)
    
    print(f"   ✓ Initialized with {len(state.strategy_templates)} default strategies")
    
    # Define some learning tasks
    tasks = [
        ("image_classification", 0.6),
        ("text_sentiment", 0.4),
        ("robot_navigation", 0.8),
        ("speech_recognition", 0.7),
        ("game_playing", 0.5),
        ("anomaly_detection", 0.6),
        ("recommendation", 0.3),
        ("time_series_prediction", 0.7)
    ]
    
    print(f"\n2. Simulating Learning on {len(tasks)} Different Tasks...")
    print("-" * 50)
    
    # Phase 1: Initial learning experiences
    for i, (task_name, difficulty) in enumerate(tasks):
        print(f"\n   Task {i+1}: {task_name} (difficulty: {difficulty:.1f})")
        
        # Retrieve best strategy for this task
        task_similarity = 0.3 + i * 0.1  # Gradually increasing similarity
        strategy_id, meta_params, confidence = meta_memory.retrieve_learning_strategy(
            state=state,
            task_similarity=task_similarity,
            task_type=task_name.split('_')[0],  # Use first part as task type
            task_context={'difficulty': difficulty}
        )
        
        print(f"   → Selected strategy: {strategy_id} (confidence: {confidence:.3f})")
        print(f"   → Meta-parameters: {meta_params}")
        
        # Convert strategy_id to enum
        try:
            strategy_enum = LearningStrategy(strategy_id)
        except ValueError:
            strategy_enum = LearningStrategy.GRADIENT_DESCENT
        
        # Simulate learning
        result = simulate_learning_task(task_name, strategy_enum, meta_params, difficulty)
        
        # Store learning experience
        state, exp_id = meta_memory.store_learning_experience(
            state=state,
            task=task_name,
            performance=result['final_performance'],
            strategy=strategy_enum,
            meta_parameters=meta_params,
            task_context={'difficulty': difficulty},
            learning_time=result['learning_time'],
            initial_performance=result['initial_performance']
        )
        
        # Update meta-parameters based on performance
        performance_feedback = (result['final_performance'] - result['initial_performance']) * 2 - 1  # Scale to [-1, 1]
        state = meta_memory.update_meta_parameters(
            state=state,
            performance_feedback=performance_feedback,
            strategy_id=strategy_id,
            task_type=task_name.split('_')[0]
        )
        
        print(f"   → Performance: {result['initial_performance']:.3f} → {result['final_performance']:.3f}")
        print(f"   → Learning time: {result['learning_time']:.1f}s")
        print(f"   → Success: {'✓' if result['success'] else '✗'}")
    
    # Show learning statistics
    print(f"\n3. Learning Statistics After Initial Phase...")
    print("-" * 50)
    stats = meta_memory.get_learning_statistics(state)
    
    print(f"   Total experiences: {stats['total_experiences']}")
    print(f"   Success rate: {stats['success_rate']:.3f}")
    print(f"   Average improvement: {stats['average_improvement']:.3f}")
    print(f"   Learning efficiency: {stats['learning_efficiency']:.6f}")
    print(f"   Strategy diversity: {stats['strategy_diversity']}")
    print(f"   Task diversity: {stats['task_diversity']}")
    print(f"   Meta-learning progress: {stats['meta_learning_progress']:.6f}")
    
    # Show strategy performance breakdown
    print(f"\n   Strategy Performance Breakdown:")
    for strategy_name, perf in stats['strategy_performance'].items():
        print(f"   → {strategy_name}: {perf['success_rate']:.3f} success, "
              f"{perf['average_improvement']:.3f} avg improvement, "
              f"{perf['usage_count']} uses")
    
    # Phase 2: Demonstrate meta-learning by repeating some tasks
    print(f"\n4. Demonstrating Meta-Learning (Repeating Tasks)...")
    print("-" * 50)
    
    repeated_tasks = [
        ("image_classification", 0.6),
        ("text_sentiment", 0.4),
        ("robot_navigation", 0.8)
    ]
    
    initial_efficiency = []
    final_efficiency = []
    
    for task_name, difficulty in repeated_tasks:
        print(f"\n   Repeating: {task_name}")
        
        # Retrieve strategy (should be better adapted now)
        strategy_id, meta_params, confidence = meta_memory.retrieve_learning_strategy(
            state=state,
            task_similarity=0.9,  # High similarity to previous experience
            task_type=task_name.split('_')[0],
            task_context={'difficulty': difficulty}
        )
        
        print(f"   → Strategy: {strategy_id} (confidence: {confidence:.3f})")
        
        # Convert strategy_id to enum
        try:
            strategy_enum = LearningStrategy(strategy_id)
        except ValueError:
            strategy_enum = LearningStrategy.GRADIENT_DESCENT
        
        # Simulate learning (should be more efficient)
        result = simulate_learning_task(task_name, strategy_enum, meta_params, difficulty)
        
        # Calculate efficiency
        efficiency = (result['final_performance'] - result['initial_performance']) / result['learning_time']
        final_efficiency.append(efficiency)
        
        # Store experience
        state, exp_id = meta_memory.store_learning_experience(
            state=state,
            task=task_name,
            performance=result['final_performance'],
            strategy=strategy_enum,
            meta_parameters=meta_params,
            task_context={'difficulty': difficulty},
            learning_time=result['learning_time'],
            initial_performance=result['initial_performance']
        )
        
        # Update meta-parameters
        performance_feedback = (result['final_performance'] - result['initial_performance']) * 2 - 1
        state = meta_memory.update_meta_parameters(
            state=state,
            performance_feedback=performance_feedback,
            strategy_id=strategy_id,
            task_type=task_name.split('_')[0]
        )
        
        print(f"   → Performance: {result['initial_performance']:.3f} → {result['final_performance']:.3f}")
        print(f"   → Learning efficiency: {efficiency:.6f}")
    
    # Final statistics
    print(f"\n5. Final Learning Statistics...")
    print("-" * 50)
    final_stats = meta_memory.get_learning_statistics(state)
    
    print(f"   Total experiences: {final_stats['total_experiences']}")
    print(f"   Success rate: {final_stats['success_rate']:.3f}")
    print(f"   Average improvement: {final_stats['average_improvement']:.3f}")
    print(f"   Learning efficiency: {final_stats['learning_efficiency']:.6f}")
    print(f"   Meta-learning progress: {final_stats['meta_learning_progress']:.6f}")
    
    # Show improvement in meta-learning
    if final_stats['meta_learning_progress'] > 0:
        print(f"\n   🎉 Meta-learning improvement detected!")
        print(f"   → The system is learning to learn more efficiently")
        print(f"   → Progress: +{final_stats['meta_learning_progress']:.6f} efficiency units")
    else:
        print(f"\n   📊 Meta-learning in progress...")
        print(f"   → More experiences needed to see clear meta-learning")
    
    # Show strategy evolution
    print(f"\n6. Strategy Template Evolution...")
    print("-" * 50)
    
    for strategy_id, template in state.strategy_templates.items():
        if template.usage_count > 0:
            print(f"   {strategy_id}:")
            print(f"   → Success rate: {template.success_rate:.3f}")
            print(f"   → Average efficiency: {template.average_efficiency:.3f}")
            print(f"   → Usage count: {template.usage_count}")
            print(f"   → Applicable tasks: {template.applicable_tasks}")
            print()
    
    print("🎯 Meta-Memory Demo Complete!")
    print("\nKey Capabilities Demonstrated:")
    print("✓ Learning strategy storage and retrieval")
    print("✓ Meta-parameter adaptation based on performance")
    print("✓ Learning-to-learn capability tracking")
    print("✓ Strategy template evolution over time")
    print("✓ Task-specific strategy selection")
    print("✓ Performance-based meta-parameter tuning")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        run_meta_memory_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
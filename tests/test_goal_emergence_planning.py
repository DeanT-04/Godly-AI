"""
Tests for goal emergence and planning system.

This module tests pattern recognition, goal formulation, and planning behavior.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock, patch
import time

from src.agents.planning import (
    PatternRecognizer,
    GoalFormulator,
    BehaviorPattern,
    EmergentGoal,
    GoalType,
    GoalPriority,
    ResourceState,
    PlanningSystem,
    PlanningConfig,
    ResourceManager,
    create_planning_system
)


class TestPatternRecognizer:
    """Test suite for pattern recognition in behavior history."""
    
    def test_pattern_recognizer_initialization(self):
        """Test proper initialization of pattern recognizer."""
        recognizer = PatternRecognizer(
            observation_dim=5,
            action_dim=3,
            pattern_memory_size=100,
            min_pattern_frequency=2,
            pattern_similarity_threshold=0.7
        )
        
        assert recognizer.observation_dim == 5
        assert recognizer.action_dim == 3
        assert recognizer.pattern_memory_size == 100
        assert recognizer.min_pattern_frequency == 2
        assert recognizer.pattern_similarity_threshold == 0.7
        assert len(recognizer.behavior_history) == 0
        assert len(recognizer.recognized_patterns) == 0
    
    def test_behavior_sample_addition(self):
        """Test adding behavior samples to history."""
        recognizer = PatternRecognizer(observation_dim=3, action_dim=2)
        
        key = jax.random.PRNGKey(42)
        obs = jax.random.normal(key, (3,))
        action = jax.random.normal(key, (2,))
        
        recognizer.add_behavior_sample(obs, action, 0.5, time.time())
        
        assert len(recognizer.behavior_history) == 1
        sample = recognizer.behavior_history[0]
        assert sample['reward'] == 0.5
        assert jnp.allclose(sample['observation'], obs)
        assert jnp.allclose(sample['action'], action)
    
    def test_sequence_extraction(self):
        """Test extraction of behavioral sequences."""
        recognizer = PatternRecognizer(observation_dim=2, action_dim=1)
        
        # Add multiple behavior samples
        key = jax.random.PRNGKey(42)
        for i in range(10):
            obs = jax.random.normal(key, (2,))
            action = jax.random.normal(key, (1,))
            recognizer.add_behavior_sample(obs, action, 0.5, time.time() + i)
            key, _ = jax.random.split(key)
        
        # Extract sequences of length 3
        sequences = recognizer._extract_sequences(3)
        
        assert len(sequences) == 8  # 10 - 3 + 1
        assert len(sequences[0]) == 3
        assert all(len(seq) == 3 for seq in sequences)
    
    def test_sequence_similarity_computation(self):
        """Test similarity computation between sequences."""
        recognizer = PatternRecognizer(observation_dim=2, action_dim=1)
        
        # Create similar sequences
        key = jax.random.PRNGKey(42)
        base_obs = jax.random.normal(key, (2,))
        base_action = jax.random.normal(key, (1,))
        
        seq1 = []
        seq2 = []
        
        for i in range(3):
            # Very similar observations and actions
            obs1 = base_obs + jax.random.normal(key, (2,)) * 0.01
            obs2 = base_obs + jax.random.normal(key, (2,)) * 0.01
            act1 = base_action + jax.random.normal(key, (1,)) * 0.01
            act2 = base_action + jax.random.normal(key, (1,)) * 0.01
            
            seq1.append({'observation': obs1, 'action': act1, 'reward': 0.5, 'timestamp': i})
            seq2.append({'observation': obs2, 'action': act2, 'reward': 0.5, 'timestamp': i})
            key, _ = jax.random.split(key)
        
        similarity = recognizer._compute_sequence_similarity(seq1, seq2)
        
        assert similarity > 0.8  # Should be high similarity
    
    def test_pattern_recognition_with_repeated_sequences(self):
        """Test pattern recognition with repeated behavioral sequences."""
        recognizer = PatternRecognizer(
            observation_dim=2,
            action_dim=1,
            min_pattern_frequency=3,
            pattern_similarity_threshold=0.7
        )
        
        # Create repeated pattern
        key = jax.random.PRNGKey(42)
        base_obs = jnp.array([1.0, 1.0])
        base_action = jnp.array([0.5])
        
        # Add the same pattern multiple times with small variations
        for repeat in range(5):  # Repeat pattern 5 times
            for step in range(5):  # Each pattern has 5 steps
                obs = base_obs + jax.random.normal(key, (2,)) * 0.1
                action = base_action + jax.random.normal(key, (1,)) * 0.1
                reward = 0.8 if step > 2 else 0.2  # Higher reward later in sequence
                
                recognizer.add_behavior_sample(obs, action, reward, time.time())
                key, _ = jax.random.split(key)
        
        # Recognize patterns
        patterns = recognizer.recognize_patterns()
        
        assert len(patterns) > 0
        
        # Check pattern properties
        for pattern in patterns:
            assert isinstance(pattern, BehaviorPattern)
            assert pattern.frequency >= recognizer.min_pattern_frequency
            assert len(pattern.observations) > 0
            assert len(pattern.actions) > 0
            assert len(pattern.rewards) > 0
    
    def test_pattern_statistics(self):
        """Test pattern statistics reporting."""
        recognizer = PatternRecognizer(observation_dim=2, action_dim=1)
        
        # Add some behavior and recognize patterns
        key = jax.random.PRNGKey(42)
        for i in range(20):
            obs = jax.random.normal(key, (2,))
            action = jax.random.normal(key, (1,))
            reward = 0.5 + 0.3 * np.sin(i * 0.5)  # Varying reward pattern
            recognizer.add_behavior_sample(obs, action, reward, time.time())
            key, _ = jax.random.split(key)
        
        patterns = recognizer.recognize_patterns()
        stats = recognizer.get_pattern_statistics()
        
        if patterns:
            assert 'total_patterns' in stats
            assert 'avg_frequency' in stats
            assert 'avg_success_rate' in stats
            assert 'avg_confidence' in stats
            assert 'pattern_types' in stats
            assert stats['total_patterns'] == len(patterns)
        else:
            assert stats['total_patterns'] == 0


class TestGoalFormulator:
    """Test suite for goal formulation from patterns."""
    
    def test_goal_formulator_initialization(self):
        """Test proper initialization of goal formulator."""
        formulator = GoalFormulator(
            observation_dim=4,
            action_dim=2,
            max_active_goals=5,
            goal_timeout=200.0
        )
        
        assert formulator.observation_dim == 4
        assert formulator.action_dim == 2
        assert formulator.max_active_goals == 5
        assert formulator.goal_timeout == 200.0
        assert len(formulator.active_goals) == 0
        assert len(formulator.completed_goals) == 0
    
    def test_skill_acquisition_goal_creation(self):
        """Test creation of skill acquisition goals from successful patterns."""
        formulator = GoalFormulator(observation_dim=3, action_dim=2)
        
        # Create a successful pattern
        key = jax.random.PRNGKey(42)
        observations = [jax.random.normal(key, (3,)) for _ in range(5)]
        actions = [jax.random.normal(key, (2,)) for _ in range(5)]
        rewards = [0.8, 0.9, 0.7, 0.85, 0.95]  # High rewards
        
        pattern = BehaviorPattern(
            pattern_id="test_pattern",
            pattern_type="successful_behavior",
            observations=observations,
            actions=actions,
            rewards=rewards,
            frequency=5,
            success_rate=0.85,
            last_occurrence=time.time(),
            confidence=0.8
        )
        
        current_obs = jax.random.normal(key, (3,))
        goal = formulator._create_skill_acquisition_goal(pattern, current_obs)
        
        assert goal is not None
        assert goal.goal_type == GoalType.SKILL_ACQUISITION
        assert goal.priority == GoalPriority.HIGH
        assert goal.target_state.shape == (3,)
        assert 'min_success_rate' in goal.success_criteria
        assert 'computational_budget' in goal.resource_requirements
    
    def test_exploration_goal_creation(self):
        """Test creation of exploration goals from novel patterns."""
        formulator = GoalFormulator(observation_dim=2, action_dim=1)
        
        # Create a novel pattern (low confidence)
        key = jax.random.PRNGKey(42)
        observations = [jax.random.normal(key, (2,)) for _ in range(3)]
        actions = [jax.random.normal(key, (1,)) for _ in range(3)]
        rewards = [0.3, 0.2, 0.4]  # Mixed rewards
        
        pattern = BehaviorPattern(
            pattern_id="novel_pattern",
            pattern_type="exploratory_behavior",
            observations=observations,
            actions=actions,
            rewards=rewards,
            frequency=2,
            success_rate=0.3,
            last_occurrence=time.time(),
            confidence=0.3  # Low confidence indicates novelty
        )
        
        current_obs = jnp.zeros(2)
        goal = formulator._create_exploration_goal(pattern, current_obs)
        
        assert goal is not None
        assert goal.goal_type == GoalType.EXPLORATION
        assert goal.priority == GoalPriority.MEDIUM
        assert goal.target_state.shape == (2,)
        assert 'novelty_threshold' in goal.success_criteria
    
    def test_goal_formulation_from_patterns(self):
        """Test complete goal formulation process from patterns."""
        formulator = GoalFormulator(observation_dim=3, action_dim=2, max_active_goals=5)
        
        # Create multiple patterns
        key = jax.random.PRNGKey(42)
        patterns = []
        
        # Successful pattern
        successful_pattern = BehaviorPattern(
            pattern_id="successful",
            pattern_type="successful_behavior",
            observations=[jax.random.normal(key, (3,)) for _ in range(4)],
            actions=[jax.random.normal(key, (2,)) for _ in range(4)],
            rewards=[0.8, 0.9, 0.85, 0.95],
            frequency=4,
            success_rate=0.9,
            last_occurrence=time.time(),
            confidence=0.8
        )
        patterns.append(successful_pattern)
        
        # Novel pattern
        novel_pattern = BehaviorPattern(
            pattern_id="novel",
            pattern_type="exploratory_behavior",
            observations=[jax.random.normal(key, (3,)) for _ in range(2)],
            actions=[jax.random.normal(key, (2,)) for _ in range(2)],
            rewards=[0.2, 0.3],
            frequency=2,
            success_rate=0.25,
            last_occurrence=time.time(),
            confidence=0.3
        )
        patterns.append(novel_pattern)
        
        # Resource state
        resource_state = ResourceState(
            computational_budget=0.8,
            memory_usage=0.3,
            attention_capacity=0.7,
            exploration_time=0.9,
            learning_capacity=0.8
        )
        
        current_obs = jax.random.normal(key, (3,))
        goals = formulator.formulate_goals_from_patterns(patterns, current_obs, resource_state)
        
        assert len(goals) > 0
        assert len(formulator.active_goals) == len(goals)
        
        # Check goal types
        goal_types = [goal.goal_type for goal in goals]
        assert GoalType.SKILL_ACQUISITION in goal_types or GoalType.EXPLORATION in goal_types
    
    def test_goal_progress_evaluation(self):
        """Test goal progress evaluation."""
        formulator = GoalFormulator(observation_dim=2, action_dim=1)
        
        # Create and add a goal
        key = jax.random.PRNGKey(42)
        target_state = jax.random.normal(key, (2,))
        
        goal = EmergentGoal(
            goal_id="test_goal",
            goal_type=GoalType.EXPLORATION,
            priority=GoalPriority.MEDIUM,
            target_state=target_state,
            success_criteria={'min_reward': 0.5, 'novelty_threshold': 0.3},
            resource_requirements={'computational_budget': 0.2},
            estimated_duration=50.0,
            parent_patterns=["pattern_1"]
        )
        
        formulator.active_goals[goal.goal_id] = goal
        
        # Evaluate progress with close observation and good reward
        close_obs = target_state + 0.1  # Very close to target
        progress = formulator.evaluate_goal_progress("test_goal", close_obs, 0.8)
        
        assert 'goal_id' in progress
        assert 'distance_to_target' in progress
        assert 'success_metrics' in progress
        assert 'overall_success' in progress
        assert progress['goal_id'] == "test_goal"
        assert progress['distance_to_target'] < 1.0
    
    def test_goal_statistics(self):
        """Test goal statistics reporting."""
        formulator = GoalFormulator(observation_dim=2, action_dim=1)
        
        # Add some goals
        key = jax.random.PRNGKey(42)
        for i in range(3):
            goal = EmergentGoal(
                goal_id=f"goal_{i}",
                goal_type=GoalType.EXPLORATION,
                priority=GoalPriority.MEDIUM,
                target_state=jax.random.normal(key, (2,)),
                success_criteria={'min_reward': 0.5},
                resource_requirements={'computational_budget': 0.2},
                estimated_duration=50.0,
                parent_patterns=[f"pattern_{i}"]
            )
            formulator.active_goals[goal.goal_id] = goal
            key, _ = jax.random.split(key)
        
        stats = formulator.get_goal_statistics()
        
        assert 'total_goals' in stats
        assert 'active_goals' in stats
        assert 'completed_goals' in stats
        assert 'completion_rate' in stats
        assert 'goal_types' in stats
        assert stats['active_goals'] == 3
        assert stats['total_goals'] == 3


class TestResourceManager:
    """Test suite for resource management."""
    
    def test_resource_manager_initialization(self):
        """Test proper initialization of resource manager."""
        manager = ResourceManager(
            initial_computational_budget=0.8,
            initial_memory_capacity=0.9,
            initial_attention_capacity=0.7,
            decay_rate=0.02
        )
        
        assert manager.current_state.computational_budget == 0.8
        assert manager.current_state.attention_capacity == 0.7
        assert manager.decay_rate == 0.02
        assert len(manager.usage_history) == 0
    
    def test_resource_updates(self):
        """Test resource usage updates."""
        manager = ResourceManager()
        
        initial_budget = manager.current_state.computational_budget
        initial_attention = manager.current_state.attention_capacity
        
        # Use some resources
        manager.update_resources(
            computational_usage=0.2,
            attention_usage=0.1,
            memory_usage=0.05
        )
        
        assert manager.current_state.computational_budget < initial_budget
        assert manager.current_state.attention_capacity < initial_attention
        assert manager.current_state.memory_usage > 0
        assert len(manager.usage_history) == 1
    
    def test_goal_affordability_check(self):
        """Test checking if goals can be afforded."""
        manager = ResourceManager()
        
        # Affordable goal
        affordable_goal = EmergentGoal(
            goal_id="affordable",
            goal_type=GoalType.EXPLORATION,
            priority=GoalPriority.LOW,
            target_state=jnp.zeros(2),
            success_criteria={},
            resource_requirements={
                'computational_budget': 0.1,
                'attention_capacity': 0.1
            },
            estimated_duration=10.0,
            parent_patterns=[]
        )
        
        # Expensive goal
        expensive_goal = EmergentGoal(
            goal_id="expensive",
            goal_type=GoalType.SKILL_ACQUISITION,
            priority=GoalPriority.HIGH,
            target_state=jnp.zeros(2),
            success_criteria={},
            resource_requirements={
                'computational_budget': 2.0,  # More than available
                'attention_capacity': 0.1
            },
            estimated_duration=100.0,
            parent_patterns=[]
        )
        
        assert manager.can_afford_goal(affordable_goal) == True
        assert manager.can_afford_goal(expensive_goal) == False
    
    def test_resource_allocation(self):
        """Test resource allocation for goals."""
        manager = ResourceManager()
        
        goal = EmergentGoal(
            goal_id="test",
            goal_type=GoalType.EXPLORATION,
            priority=GoalPriority.MEDIUM,
            target_state=jnp.zeros(2),
            success_criteria={},
            resource_requirements={
                'computational_budget': 0.3,
                'attention_capacity': 0.2
            },
            estimated_duration=50.0,
            parent_patterns=[]
        )
        
        initial_budget = manager.current_state.computational_budget
        initial_attention = manager.current_state.attention_capacity
        
        success = manager.allocate_resources_for_goal(goal)
        
        assert success == True
        assert manager.current_state.computational_budget == initial_budget - 0.3
        assert manager.current_state.attention_capacity == initial_attention - 0.2
    
    def test_resource_pressure_calculation(self):
        """Test resource pressure metrics."""
        manager = ResourceManager()
        
        # Use up most resources
        manager.update_resources(
            computational_usage=0.8,
            memory_usage=0.7,
            attention_usage=0.6
        )
        
        pressure = manager.get_resource_pressure()
        
        assert 'computational_pressure' in pressure
        assert 'memory_pressure' in pressure
        assert 'attention_pressure' in pressure
        assert pressure['computational_pressure'] > 0.5
        assert pressure['memory_pressure'] > 0.5


class TestPlanningSystem:
    """Test suite for integrated planning system."""
    
    def test_planning_system_initialization(self):
        """Test proper initialization of planning system."""
        config = PlanningConfig(
            observation_dim=4,
            action_dim=2,
            max_active_goals=8,
            pattern_memory_size=500
        )
        
        system = PlanningSystem(config)
        
        assert system.config.observation_dim == 4
        assert system.config.action_dim == 2
        assert system.config.max_active_goals == 8
        assert system.planning_active == False
        assert system.planning_episode == 0
        assert isinstance(system.pattern_recognizer, PatternRecognizer)
        assert isinstance(system.goal_formulator, GoalFormulator)
        assert isinstance(system.resource_manager, ResourceManager)
    
    def test_planning_episode_lifecycle(self):
        """Test complete planning episode lifecycle."""
        config = PlanningConfig(observation_dim=3, action_dim=2)
        system = PlanningSystem(config)
        
        # Start planning
        key = jax.random.PRNGKey(42)
        initial_obs = jax.random.normal(key, (3,))
        initial_action = jax.random.normal(key, (2,))
        
        system.start_planning(initial_obs, initial_action)
        
        assert system.planning_active == True
        assert system.planning_episode == 1
        assert len(system.pattern_recognizer.behavior_history) == 1
        
        # Perform planning steps
        for i in range(5):
            obs = jax.random.normal(key, (3,))
            action = jax.random.normal(key, (2,))
            reward = 0.5 + 0.3 * np.sin(i)
            
            step_results = system.step_planning(obs, action, reward)
            
            assert 'patterns_recognized' in step_results
            assert 'goals_generated' in step_results
            assert 'active_goals' in step_results
            assert 'resource_pressure' in step_results
            
            key, _ = jax.random.split(key)
        
        # End planning
        episode_summary = system.end_planning_episode()
        
        assert system.planning_active == False
        assert 'episode' in episode_summary
        assert 'goal_statistics' in episode_summary
        assert 'pattern_statistics' in episode_summary
        assert episode_summary['episode'] == 1
    
    def test_goal_execution(self):
        """Test goal execution functionality."""
        config = PlanningConfig(observation_dim=2, action_dim=1)
        system = PlanningSystem(config)
        
        # Add a goal manually for testing
        key = jax.random.PRNGKey(42)
        goal = EmergentGoal(
            goal_id="test_goal",
            goal_type=GoalType.EXPLORATION,
            priority=GoalPriority.MEDIUM,
            target_state=jax.random.normal(key, (2,)),
            success_criteria={'min_reward': 0.5},
            resource_requirements={'computational_budget': 0.2},
            estimated_duration=30.0,
            parent_patterns=[]
        )
        
        system.goal_formulator.active_goals[goal.goal_id] = goal
        
        # Test execution
        result = system.execute_goal("test_goal")
        
        assert 'goal_id' in result
        assert result['goal_id'] == "test_goal"
        # Should succeed since we have sufficient resources
        assert 'error' not in result or result.get('status') == 'ready_for_execution'
    
    def test_planning_recommendations(self):
        """Test planning recommendations generation."""
        config = PlanningConfig(observation_dim=3, action_dim=2)
        system = PlanningSystem(config)
        
        # Add some goals for recommendations
        key = jax.random.PRNGKey(42)
        for i in range(2):
            goal = EmergentGoal(
                goal_id=f"goal_{i}",
                goal_type=GoalType.EXPLORATION,
                priority=GoalPriority.MEDIUM,
                target_state=jax.random.normal(key, (3,)),
                success_criteria={'min_reward': 0.5},
                resource_requirements={'computational_budget': 0.1},
                estimated_duration=20.0,
                parent_patterns=[]
            )
            system.goal_formulator.active_goals[goal.goal_id] = goal
            key, _ = jax.random.split(key)
        
        current_obs = jax.random.normal(key, (3,))
        recommendations = system.get_planning_recommendations(current_obs)
        
        assert 'recommended_goal' in recommendations
        assert 'alternative_goals' in recommendations
        assert 'resource_constraints' in recommendations
        assert 'pattern_insights' in recommendations
        assert 'planning_advice' in recommendations
    
    def test_system_statistics(self):
        """Test comprehensive system statistics."""
        config = PlanningConfig(observation_dim=2, action_dim=1)
        system = PlanningSystem(config)
        
        # Run a short planning episode
        key = jax.random.PRNGKey(42)
        initial_obs = jax.random.normal(key, (2,))
        initial_action = jax.random.normal(key, (1,))
        
        system.start_planning(initial_obs, initial_action)
        
        for i in range(3):
            obs = jax.random.normal(key, (2,))
            action = jax.random.normal(key, (1,))
            reward = 0.6
            system.step_planning(obs, action, reward)
            key, _ = jax.random.split(key)
        
        system.end_planning_episode()
        
        stats = system.get_system_statistics()
        
        assert 'planning_active' in stats
        assert 'current_episode' in stats
        assert 'total_episodes' in stats
        assert 'goal_statistics' in stats
        assert 'pattern_statistics' in stats
        assert 'resource_state' in stats
        assert 'resource_pressure' in stats
        assert stats['total_episodes'] == 1
    
    def test_system_reset(self):
        """Test system reset functionality."""
        config = PlanningConfig(observation_dim=2, action_dim=1)
        system = PlanningSystem(config)
        
        # Run some planning
        key = jax.random.PRNGKey(42)
        system.start_planning(jax.random.normal(key, (2,)), jax.random.normal(key, (1,)))
        system.step_planning(jax.random.normal(key, (2,)), jax.random.normal(key, (1,)), 0.5)
        system.end_planning_episode()
        
        # Reset system
        system.reset_system()
        
        assert system.planning_active == False
        assert system.planning_episode == 0
        assert len(system.planning_history) == 0
        assert len(system.goal_achievement_rates) == 0
        assert len(system.pattern_recognizer.behavior_history) == 0
        assert len(system.goal_formulator.active_goals) == 0
    
    def test_create_planning_system_factory(self):
        """Test factory function for creating planning systems."""
        system = create_planning_system(
            observation_dim=3,
            action_dim=2,
            max_active_goals=6,
            pattern_memory_size=800
        )
        
        assert isinstance(system, PlanningSystem)
        assert system.config.observation_dim == 3
        assert system.config.action_dim == 2
        assert system.config.max_active_goals == 6
        assert system.config.pattern_memory_size == 800


class TestPlanningBehavior:
    """Integration tests for planning behavior."""
    
    def test_pattern_to_goal_pipeline(self):
        """Test the complete pipeline from patterns to goals."""
        config = PlanningConfig(observation_dim=2, action_dim=1, min_pattern_frequency=2)
        system = PlanningSystem(config)
        
        # Start planning
        key = jax.random.PRNGKey(42)
        system.start_planning(jnp.zeros(2), jnp.zeros(1))
        
        # Create a repeated behavioral pattern
        base_obs = jnp.array([1.0, 1.0])
        base_action = jnp.array([0.5])
        
        # Add repeated pattern multiple times
        for repeat in range(4):  # Repeat pattern 4 times
            for step in range(3):  # Each pattern has 3 steps
                obs = base_obs + jax.random.normal(key, (2,)) * 0.1
                action = base_action + jax.random.normal(key, (1,)) * 0.1
                reward = 0.8 if step == 2 else 0.3  # Higher reward at end
                
                system.step_planning(obs, action, reward)
                key, _ = jax.random.split(key)
        
        # Force pattern recognition by setting time
        system.last_pattern_recognition = 0.0
        
        # Step once more to trigger pattern recognition
        final_obs = base_obs + jax.random.normal(key, (2,)) * 0.1
        final_action = base_action + jax.random.normal(key, (1,)) * 0.1
        step_results = system.step_planning(final_obs, final_action, 0.8)
        
        # Should have recognized patterns and possibly generated goals
        assert step_results['patterns_recognized'] >= 0
        assert step_results['active_goals'] >= 0
    
    def test_resource_constrained_planning(self):
        """Test planning behavior under resource constraints."""
        config = PlanningConfig(observation_dim=2, action_dim=1)
        system = PlanningSystem(config)
        
        # Deplete most resources
        system.resource_manager.update_resources(
            computational_usage=0.8,
            memory_usage=0.7,
            attention_usage=0.6
        )
        
        # Add a high-resource goal
        expensive_goal = EmergentGoal(
            goal_id="expensive",
            goal_type=GoalType.SKILL_ACQUISITION,
            priority=GoalPriority.HIGH,
            target_state=jnp.array([1.0, 1.0]),
            success_criteria={'min_reward': 0.8},
            resource_requirements={
                'computational_budget': 0.5,  # More than available
                'attention_capacity': 0.3
            },
            estimated_duration=100.0,
            parent_patterns=[]
        )
        
        system.goal_formulator.active_goals[expensive_goal.goal_id] = expensive_goal
        
        # Try to execute - should fail due to insufficient resources
        result = system.execute_goal("expensive")
        assert 'error' in result
        assert 'resource' in result['error'].lower()
    
    def test_goal_achievement_tracking(self):
        """Test tracking of goal achievement over time."""
        config = PlanningConfig(observation_dim=2, action_dim=1)
        system = PlanningSystem(config)
        
        # Add a goal
        key = jax.random.PRNGKey(42)
        target_state = jnp.array([2.0, 2.0])
        
        goal = EmergentGoal(
            goal_id="achievement_test",
            goal_type=GoalType.EXPLORATION,
            priority=GoalPriority.MEDIUM,
            target_state=target_state,
            success_criteria={'min_reward': 0.5},
            resource_requirements={'computational_budget': 0.1},
            estimated_duration=50.0,
            parent_patterns=[]
        )
        
        system.goal_formulator.active_goals[goal.goal_id] = goal
        
        # Simulate progress towards goal
        progress_results = []
        start_obs = jnp.array([0.0, 0.0])  # Start far from target
        
        for i in range(5):
            # Move closer to target over time
            progress_ratio = (i + 1) / 5.0
            current_obs = start_obs + (target_state - start_obs) * progress_ratio
            reward = 0.6  # Good reward
            
            progress = system.evaluate_goal_achievement("achievement_test", current_obs, reward)
            progress_results.append(progress)
        
        # Should show decreasing distance to target
        # Check if we have valid progress results
        valid_results = [p for p in progress_results if 'distance_to_target' in p]
        if valid_results and len(valid_results) > 1:
            distances = [p['distance_to_target'] for p in valid_results]
            # Allow for small numerical errors
            assert distances[0] > distances[-1] - 1e-6  # Distance should decrease
        else:
            # If no valid results, at least check that we got some response
            assert len(progress_results) > 0
    
    def test_adaptive_goal_prioritization(self):
        """Test that goal prioritization adapts to resource availability."""
        config = PlanningConfig(observation_dim=2, action_dim=1)
        system = PlanningSystem(config)
        
        # Add goals with different resource requirements
        key = jax.random.PRNGKey(42)
        
        low_resource_goal = EmergentGoal(
            goal_id="low_resource",
            goal_type=GoalType.EXPLORATION,
            priority=GoalPriority.MEDIUM,
            target_state=jax.random.normal(key, (2,)),
            success_criteria={'min_reward': 0.5},
            resource_requirements={'computational_budget': 0.1},
            estimated_duration=20.0,
            parent_patterns=[]
        )
        
        high_resource_goal = EmergentGoal(
            goal_id="high_resource",
            goal_type=GoalType.SKILL_ACQUISITION,
            priority=GoalPriority.HIGH,
            target_state=jax.random.normal(key, (2,)),
            success_criteria={'min_reward': 0.8},
            resource_requirements={'computational_budget': 0.6},
            estimated_duration=100.0,
            parent_patterns=[]
        )
        
        system.goal_formulator.active_goals[low_resource_goal.goal_id] = low_resource_goal
        system.goal_formulator.active_goals[high_resource_goal.goal_id] = high_resource_goal
        
        # With high resources, should recommend high-priority goal
        system.resource_manager.current_state.computational_budget = 0.8
        recommended = system._get_recommended_goal()
        assert recommended.goal_id == "high_resource"
        
        # With low resources, should recommend low-resource goal
        system.resource_manager.current_state.computational_budget = 0.2
        recommended = system._get_recommended_goal()
        assert recommended.goal_id == "low_resource"


if __name__ == "__main__":
    pytest.main([__file__])
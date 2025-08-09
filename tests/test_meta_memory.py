"""
Tests for Meta-Memory System

This module tests the meta-memory implementation including learning strategy
storage and retrieval, meta-parameter adaptation, and learning-to-learn capabilities.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import time
from unittest.mock import patch

from src.memory.meta import (
    MetaMemory,
    MetaMemoryParams,
    MetaMemoryState,
    LearningExperience,
    StrategyTemplate,
    LearningStrategy,
    create_meta_memory
)


class TestMetaMemoryParams:
    """Test MetaMemoryParams configuration."""
    
    def test_default_params(self):
        """Test default parameter values."""
        params = MetaMemoryParams()
        
        assert params.learning_history_size == 1000
        assert params.strategy_cache_size == 50
        assert params.hour_timescale == 3600.0
        assert params.consolidation_interval == 1800.0
        assert params.strategy_similarity_threshold == 0.7
        assert params.performance_improvement_threshold == 0.05
        assert params.adaptation_rate == 0.1
        assert params.exploration_rate == 0.2
        assert params.consolidation_strength == 0.8
        assert params.forgetting_threshold == 0.1
        assert params.performance_window == 10
        assert params.success_threshold == 0.8
    
    def test_custom_params(self):
        """Test custom parameter configuration."""
        params = MetaMemoryParams(
            learning_history_size=500,
            adaptation_rate=0.2,
            exploration_rate=0.3
        )
        
        assert params.learning_history_size == 500
        assert params.adaptation_rate == 0.2
        assert params.exploration_rate == 0.3
        # Other params should remain default
        assert params.strategy_cache_size == 50


class TestLearningExperience:
    """Test LearningExperience data structure."""
    
    def test_learning_experience_creation(self):
        """Test creating a learning experience."""
        experience = LearningExperience(
            experience_id="test_exp_001",
            task_type="classification",
            task_context={"dataset": "mnist", "size": 1000},
            strategy_used=LearningStrategy.GRADIENT_DESCENT,
            meta_parameters={"learning_rate": 0.01, "batch_size": 32},
            initial_performance=0.3,
            final_performance=0.8,
            learning_time=120.0,
            timestamp=time.time(),
            success=True,
            difficulty=0.6,
            transfer_source=None
        )
        
        assert experience.experience_id == "test_exp_001"
        assert experience.task_type == "classification"
        assert experience.strategy_used == LearningStrategy.GRADIENT_DESCENT
        assert experience.success is True
        assert experience.difficulty == 0.6
    
    def test_performance_improvement_property(self):
        """Test performance improvement calculation."""
        experience = LearningExperience(
            experience_id="test_exp_002",
            task_type="regression",
            task_context={},
            strategy_used=LearningStrategy.GRADIENT_DESCENT,
            meta_parameters={},
            initial_performance=0.2,
            final_performance=0.7,
            learning_time=60.0,
            timestamp=time.time(),
            success=True,
            difficulty=0.5,
            transfer_source=None
        )
        
        assert abs(experience.performance_improvement - 0.5) < 1e-10
    
    def test_learning_efficiency_property(self):
        """Test learning efficiency calculation."""
        experience = LearningExperience(
            experience_id="test_exp_003",
            task_type="optimization",
            task_context={},
            strategy_used=LearningStrategy.EVOLUTIONARY,
            meta_parameters={},
            initial_performance=0.1,
            final_performance=0.6,
            learning_time=100.0,
            timestamp=time.time(),
            success=True,
            difficulty=0.7,
            transfer_source=None
        )
        
        assert experience.learning_efficiency == 0.005  # 0.5 improvement / 100 time
    
    def test_zero_learning_time_efficiency(self):
        """Test learning efficiency with zero learning time."""
        experience = LearningExperience(
            experience_id="test_exp_004",
            task_type="test",
            task_context={},
            strategy_used=LearningStrategy.GRADIENT_DESCENT,
            meta_parameters={},
            initial_performance=0.3,
            final_performance=0.8,
            learning_time=0.0,
            timestamp=time.time(),
            success=True,
            difficulty=0.5,
            transfer_source=None
        )
        
        assert experience.learning_efficiency == 0.0


class TestStrategyTemplate:
    """Test StrategyTemplate functionality."""
    
    def test_strategy_template_creation(self):
        """Test creating a strategy template."""
        template = StrategyTemplate(
            strategy_id="test_gradient_descent",
            strategy_type=LearningStrategy.GRADIENT_DESCENT,
            meta_parameters={"learning_rate": 0.01, "momentum": 0.9},
            applicable_tasks=["classification", "regression"],
            success_rate=0.75,
            average_efficiency=0.6,
            usage_count=10,
            last_updated=time.time(),
            creation_time=time.time()
        )
        
        assert template.strategy_id == "test_gradient_descent"
        assert template.strategy_type == LearningStrategy.GRADIENT_DESCENT
        assert template.success_rate == 0.75
        assert template.usage_count == 10
    
    def test_is_applicable_matching_task(self):
        """Test applicability for matching task type."""
        template = StrategyTemplate(
            strategy_id="test_strategy",
            strategy_type=LearningStrategy.GRADIENT_DESCENT,
            meta_parameters={},
            applicable_tasks=["classification", "regression"],
            success_rate=0.8,
            average_efficiency=0.7,
            usage_count=5,
            last_updated=time.time(),
            creation_time=time.time()
        )
        
        applicability = template.is_applicable("classification", {})
        assert applicability >= 0.8  # Base score + performance bonus
    
    def test_is_applicable_non_matching_task(self):
        """Test applicability for non-matching task type."""
        template = StrategyTemplate(
            strategy_id="test_strategy",
            strategy_type=LearningStrategy.GRADIENT_DESCENT,
            meta_parameters={},
            applicable_tasks=["classification"],
            success_rate=0.6,
            average_efficiency=0.5,
            usage_count=3,
            last_updated=time.time(),
            creation_time=time.time()
        )
        
        applicability = template.is_applicable("optimization", {})
        assert applicability >= 0.2  # Base score for non-matching
        assert applicability <= 0.4  # Should be lower than matching


class TestMetaMemory:
    """Test MetaMemory core functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = MetaMemoryParams(
            learning_history_size=10,  # Small for testing
            strategy_cache_size=5,
            consolidation_interval=60.0  # 1 minute for testing
        )
        self.meta_memory = MetaMemory(self.params)
        self.key = jax.random.PRNGKey(42)
    
    def test_initialization(self):
        """Test meta-memory initialization."""
        assert self.meta_memory.params.learning_history_size == 10
        assert len(self.meta_memory.default_strategies) > 0
        
        # Check that default strategies are created
        assert 'gradient_descent' in self.meta_memory.default_strategies
        assert 'evolutionary' in self.meta_memory.default_strategies
        assert 'reinforcement' in self.meta_memory.default_strategies
    
    def test_init_state(self):
        """Test state initialization."""
        state = self.meta_memory.init_state(self.key)
        
        assert isinstance(state, MetaMemoryState)
        assert len(state.learning_experiences) == 0
        assert len(state.strategy_templates) > 0  # Should have default strategies
        assert len(state.task_performance_history) == 0
        assert len(state.meta_parameter_history) == 0
        assert state.total_experiences == 0
        assert 'consolidation_count' in state.consolidation_state
    
    def test_store_learning_experience(self):
        """Test storing a learning experience."""
        state = self.meta_memory.init_state(self.key)
        
        new_state, exp_id = self.meta_memory.store_learning_experience(
            state=state,
            task="classification",
            performance=0.85,
            strategy=LearningStrategy.GRADIENT_DESCENT,
            meta_parameters={"learning_rate": 0.01},
            task_context={"dataset": "test"},
            learning_time=60.0,
            initial_performance=0.3
        )
        
        assert len(new_state.learning_experiences) == 1
        assert exp_id in new_state.learning_experiences
        assert new_state.total_experiences == 1
        
        experience = new_state.learning_experiences[exp_id]
        assert experience.task_type == "classification"
        assert experience.final_performance == 0.85
        assert experience.strategy_used == LearningStrategy.GRADIENT_DESCENT
        assert experience.success is True  # Performance improvement > threshold
        
        # Check task performance history
        assert "classification" in new_state.task_performance_history
        assert new_state.task_performance_history["classification"] == [0.85]
    
    def test_store_multiple_experiences(self):
        """Test storing multiple learning experiences."""
        state = self.meta_memory.init_state(self.key)
        
        # Store first experience
        state, exp_id1 = self.meta_memory.store_learning_experience(
            state=state,
            task="classification",
            performance=0.8,
            strategy=LearningStrategy.GRADIENT_DESCENT
        )
        
        # Store second experience
        state, exp_id2 = self.meta_memory.store_learning_experience(
            state=state,
            task="regression",
            performance=0.7,
            strategy=LearningStrategy.EVOLUTIONARY
        )
        
        assert len(state.learning_experiences) == 2
        assert exp_id1 != exp_id2
        assert state.total_experiences == 2
        
        # Check both tasks in history
        assert "classification" in state.task_performance_history
        assert "regression" in state.task_performance_history
    
    def test_capacity_management(self):
        """Test that memory capacity is managed correctly."""
        state = self.meta_memory.init_state(self.key)
        
        # Store more experiences than capacity
        for i in range(15):  # More than capacity of 10
            state, _ = self.meta_memory.store_learning_experience(
                state=state,
                task=f"task_{i}",
                performance=0.5 + i * 0.03,
                strategy=LearningStrategy.GRADIENT_DESCENT
            )
        
        # Should not exceed capacity
        assert len(state.learning_experiences) <= self.params.learning_history_size
        assert state.total_experiences == 15  # But total count should be accurate
    
    def test_retrieve_learning_strategy(self):
        """Test retrieving learning strategy."""
        state = self.meta_memory.init_state(self.key)
        
        # Store some experiences to build history
        state, _ = self.meta_memory.store_learning_experience(
            state=state,
            task="classification",
            performance=0.9,
            strategy=LearningStrategy.GRADIENT_DESCENT,
            meta_parameters={"learning_rate": 0.01}
        )
        
        # Retrieve strategy for similar task
        strategy_id, meta_params, confidence = self.meta_memory.retrieve_learning_strategy(
            state=state,
            task_similarity=0.8,
            task_type="classification",
            task_context={"dataset": "similar"}
        )
        
        assert isinstance(strategy_id, str)
        assert isinstance(meta_params, dict)
        assert 0.0 <= confidence <= 1.0
        assert len(meta_params) > 0
    
    def test_retrieve_strategy_no_task_type(self):
        """Test retrieving strategy without task type."""
        state = self.meta_memory.init_state(self.key)
        
        strategy_id, meta_params, confidence = self.meta_memory.retrieve_learning_strategy(
            state=state,
            task_similarity=0.5
        )
        
        # Should still return a strategy (fallback)
        assert isinstance(strategy_id, str)
        assert isinstance(meta_params, dict)
        assert confidence >= 0.0
    
    def test_update_meta_parameters(self):
        """Test updating meta-parameters based on feedback."""
        state = self.meta_memory.init_state(self.key)
        
        # Get initial strategy state
        initial_strategy = state.strategy_templates['gradient_descent']
        initial_success_rate = initial_strategy.success_rate
        
        # Update with positive feedback
        new_state = self.meta_memory.update_meta_parameters(
            state=state,
            performance_feedback=0.3,  # Positive feedback
            strategy_id='gradient_descent',
            task_type='classification'
        )
        
        # Check that strategy was updated
        updated_strategy = new_state.strategy_templates['gradient_descent']
        assert updated_strategy.success_rate >= initial_success_rate
        assert updated_strategy.usage_count == initial_strategy.usage_count + 1
        
        # Check global performance history
        assert 'global_performance' in new_state.meta_parameter_history
        assert new_state.meta_parameter_history['global_performance'] == [0.3]
    
    def test_update_meta_parameters_negative_feedback(self):
        """Test updating meta-parameters with negative feedback."""
        state = self.meta_memory.init_state(self.key)
        
        initial_strategy = state.strategy_templates['gradient_descent']
        initial_success_rate = initial_strategy.success_rate
        
        # Update with negative feedback
        new_state = self.meta_memory.update_meta_parameters(
            state=state,
            performance_feedback=-0.2,  # Negative feedback
            strategy_id='gradient_descent'
        )
        
        # Success rate should decrease
        updated_strategy = new_state.strategy_templates['gradient_descent']
        assert updated_strategy.success_rate < initial_success_rate
    
    def test_get_learning_statistics_empty(self):
        """Test getting statistics with no experiences."""
        state = self.meta_memory.init_state(self.key)
        
        stats = self.meta_memory.get_learning_statistics(state)
        
        assert stats['total_experiences'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_improvement'] == 0.0
        assert stats['learning_efficiency'] == 0.0
        assert stats['strategy_diversity'] == 0
        assert stats['task_diversity'] == 0
        assert stats['meta_learning_progress'] == 0.0
    
    def test_get_learning_statistics_with_data(self):
        """Test getting statistics with learning experiences."""
        state = self.meta_memory.init_state(self.key)
        
        # Add several experiences
        experiences_data = [
            ("classification", 0.8, LearningStrategy.GRADIENT_DESCENT, True),
            ("regression", 0.6, LearningStrategy.EVOLUTIONARY, False),
            ("optimization", 0.9, LearningStrategy.REINFORCEMENT, True),
            ("classification", 0.85, LearningStrategy.GRADIENT_DESCENT, True),
        ]
        
        for task, perf, strategy, success in experiences_data:
            initial_perf = 0.3 if success else 0.7  # Ensure success matches expectation
            state, _ = self.meta_memory.store_learning_experience(
                state=state,
                task=task,
                performance=perf,
                strategy=strategy,
                initial_performance=initial_perf,
                learning_time=60.0
            )
        
        stats = self.meta_memory.get_learning_statistics(state)
        
        assert stats['total_experiences'] == 4
        assert 0.0 <= stats['success_rate'] <= 1.0
        assert stats['strategy_diversity'] >= 2  # At least 2 different strategies
        assert stats['task_diversity'] >= 2  # At least 2 different tasks
        assert 'strategy_performance' in stats
        
        # Check strategy performance breakdown
        strategy_perf = stats['strategy_performance']
        assert LearningStrategy.GRADIENT_DESCENT.value in strategy_perf
    
    @patch('time.time')
    def test_consolidation_trigger(self, mock_time):
        """Test that consolidation is triggered after interval."""
        # Mock time to control consolidation timing
        mock_time.return_value = 1000.0
        
        state = self.meta_memory.init_state(self.key)
        
        # Store an experience
        state, _ = self.meta_memory.store_learning_experience(
            state=state,
            task="test_task",
            performance=0.8,
            strategy=LearningStrategy.GRADIENT_DESCENT
        )
        
        # Advance time past consolidation interval
        mock_time.return_value = 1000.0 + self.params.consolidation_interval + 1
        
        # Store another experience (should trigger consolidation)
        new_state, _ = self.meta_memory.store_learning_experience(
            state=state,
            task="test_task_2",
            performance=0.7,
            strategy=LearningStrategy.EVOLUTIONARY
        )
        
        # Check that consolidation occurred
        assert new_state.consolidation_state['consolidation_count'] > 0
        assert len(new_state.consolidation_state['pending_experiences']) == 1  # Only new experience
    
    def test_strategy_adaptation(self):
        """Test that strategies adapt based on task context."""
        state = self.meta_memory.init_state(self.key)
        
        # Store experience with high difficulty
        state, _ = self.meta_memory.store_learning_experience(
            state=state,
            task="difficult_task",
            performance=0.6,
            strategy=LearningStrategy.GRADIENT_DESCENT,
            task_context={"difficulty": 0.9},
            learning_time=120.0
        )
        
        # Retrieve strategy for high difficulty task
        strategy_id, adapted_params, confidence = self.meta_memory.retrieve_learning_strategy(
            state=state,
            task_similarity=0.8,
            task_type="difficult_task",
            task_context={"difficulty": 0.9}
        )
        
        # Get original parameters for comparison
        original_strategy = state.strategy_templates[strategy_id]
        original_params = original_strategy.meta_parameters
        
        # Adapted parameters should be more conservative for difficult tasks
        if 'learning_rate' in adapted_params and 'learning_rate' in original_params:
            assert adapted_params['learning_rate'] <= original_params['learning_rate']


class TestMetaMemoryConvenienceFunctions:
    """Test convenience functions for creating meta-memory instances."""
    
    def test_create_standard_meta_memory(self):
        """Test creating standard meta-memory."""
        meta_memory = create_meta_memory("standard")
        
        assert isinstance(meta_memory, MetaMemory)
        assert meta_memory.params.learning_history_size == 1000
        assert meta_memory.params.adaptation_rate == 0.1
    
    def test_create_fast_adaptation_meta_memory(self):
        """Test creating fast adaptation meta-memory."""
        meta_memory = create_meta_memory("fast_adaptation")
        
        assert isinstance(meta_memory, MetaMemory)
        assert meta_memory.params.adaptation_rate == 0.3
        assert meta_memory.params.exploration_rate == 0.4
        assert meta_memory.params.consolidation_interval == 900.0
    
    def test_create_conservative_meta_memory(self):
        """Test creating conservative meta-memory."""
        meta_memory = create_meta_memory("conservative")
        
        assert isinstance(meta_memory, MetaMemory)
        assert meta_memory.params.adaptation_rate == 0.05
        assert meta_memory.params.exploration_rate == 0.1
        assert meta_memory.params.consolidation_interval == 3600.0
    
    def test_create_exploratory_meta_memory(self):
        """Test creating exploratory meta-memory."""
        meta_memory = create_meta_memory("exploratory")
        
        assert isinstance(meta_memory, MetaMemory)
        assert meta_memory.params.exploration_rate == 0.5
        assert meta_memory.params.strategy_similarity_threshold == 0.5
        assert meta_memory.params.performance_improvement_threshold == 0.02
    
    def test_create_large_capacity_meta_memory(self):
        """Test creating large capacity meta-memory."""
        meta_memory = create_meta_memory("large_capacity")
        
        assert isinstance(meta_memory, MetaMemory)
        assert meta_memory.params.learning_history_size == 5000
        assert meta_memory.params.strategy_cache_size == 200
        assert meta_memory.params.performance_window == 50
    
    def test_create_invalid_meta_memory_type(self):
        """Test creating meta-memory with invalid type."""
        with pytest.raises(ValueError, match="Unknown memory type"):
            create_meta_memory("invalid_type")


class TestMetaMemoryIntegration:
    """Integration tests for meta-memory system."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.meta_memory = create_meta_memory("standard")
        self.key = jax.random.PRNGKey(123)
    
    def test_full_learning_cycle(self):
        """Test a complete learning cycle with meta-memory."""
        state = self.meta_memory.init_state(self.key)
        
        # Simulate learning on multiple tasks
        tasks = ["classification", "regression", "optimization"]
        strategies = [LearningStrategy.GRADIENT_DESCENT, LearningStrategy.EVOLUTIONARY, LearningStrategy.REINFORCEMENT]
        
        for i, (task, strategy) in enumerate(zip(tasks, strategies)):
            # Store learning experience
            performance = 0.6 + i * 0.1  # Improving performance
            state, exp_id = self.meta_memory.store_learning_experience(
                state=state,
                task=task,
                performance=performance,
                strategy=strategy,
                meta_parameters={"learning_rate": 0.01 - i * 0.002},
                learning_time=60.0 + i * 10,
                initial_performance=0.3
            )
            
            # Update meta-parameters based on performance
            feedback = (performance - 0.3) / 0.7  # Normalize to [-1, 1]
            state = self.meta_memory.update_meta_parameters(
                state=state,
                performance_feedback=feedback,
                strategy_id=strategy.value,
                task_type=task
            )
        
        # Test strategy retrieval for new task
        strategy_id, meta_params, confidence = self.meta_memory.retrieve_learning_strategy(
            state=state,
            task_similarity=0.7,
            task_type="classification",  # Similar to previous task
            task_context={"difficulty": 0.5}
        )
        
        # Verify results
        assert len(state.learning_experiences) == 3
        assert len(state.task_performance_history) == 3
        assert confidence > 0.5  # Should have reasonable confidence
        
        # Get final statistics
        stats = self.meta_memory.get_learning_statistics(state)
        assert stats['total_experiences'] == 3
        assert stats['strategy_diversity'] == 3
        assert stats['task_diversity'] == 3
        assert stats['success_rate'] > 0.5  # Most experiences should be successful
    
    def test_meta_learning_progression(self):
        """Test that meta-learning shows progression over time."""
        state = self.meta_memory.init_state(self.key)
        
        # Simulate learning progression with improving efficiency
        base_efficiency = 0.01
        for i in range(20):
            # Gradually improving learning efficiency
            learning_time = max(30.0, 120.0 - i * 4)  # Decreasing learning time
            performance_improvement = min(0.8, 0.3 + i * 0.02)  # Increasing improvement
            
            state, _ = self.meta_memory.store_learning_experience(
                state=state,
                task=f"task_type_{i % 3}",  # Cycle through 3 task types
                performance=0.3 + performance_improvement,
                strategy=LearningStrategy.GRADIENT_DESCENT,
                learning_time=learning_time,
                initial_performance=0.3
            )
        
        # Check meta-learning progression
        stats = self.meta_memory.get_learning_statistics(state)
        
        # Should show positive meta-learning progress
        assert stats['meta_learning_progress'] >= 0.0
        assert stats['total_experiences'] == 20
        
        # Recent experiences should be more efficient
        experiences = list(state.learning_experiences.values())
        recent_experiences = sorted(experiences, key=lambda x: x.timestamp)[-5:]
        early_experiences = sorted(experiences, key=lambda x: x.timestamp)[:5]
        
        recent_avg_efficiency = np.mean([exp.learning_efficiency for exp in recent_experiences])
        early_avg_efficiency = np.mean([exp.learning_efficiency for exp in early_experiences])
        
        # Recent should be better than early (meta-learning)
        assert recent_avg_efficiency >= early_avg_efficiency


if __name__ == "__main__":
    pytest.main([__file__])
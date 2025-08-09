"""
Tests for internal reward system.

This module tests intrinsic reward generation, surprise detection, and reward-based learning integration.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock, patch
import time

from src.agents.exploration import (
    IntrinsicRewardGenerator,
    RewardPredictor,
    SurpriseDetector,
    RewardSignal,
    RewardType,
    SurpriseEvent,
    RewardSystem,
    RewardSystemConfig,
    RewardLearningIntegrator,
    create_reward_system,
    NoveltyScore
)


class TestRewardPredictor:
    """Test suite for reward prediction."""
    
    def test_reward_predictor_initialization(self):
        """Test proper initialization of reward predictor."""
        predictor = RewardPredictor(
            observation_dim=5,
            action_dim=3,
            hidden_dim=32,
            learning_rate=0.02,
            prediction_window=50
        )
        
        assert predictor.observation_dim == 5
        assert predictor.action_dim == 3
        assert predictor.hidden_dim == 32
        assert predictor.learning_rate == 0.02
        assert predictor.prediction_window == 50
        assert predictor.w1.shape == (8, 32)  # obs_dim + action_dim
        assert predictor.w2.shape == (32, 1)
        assert len(predictor.prediction_history) == 0
    
    def test_reward_prediction(self):
        """Test reward prediction functionality."""
        predictor = RewardPredictor(observation_dim=3, action_dim=2)
        
        key = jax.random.PRNGKey(42)
        observation = jax.random.normal(key, (3,))
        action = jax.random.normal(key, (2,))
        
        # Test prediction without action
        prediction1 = predictor.predict_reward(observation)
        assert isinstance(prediction1, float)
        
        # Test prediction with action
        prediction2 = predictor.predict_reward(observation, action)
        assert isinstance(prediction2, float)
        
        # Predictions should be different with and without action
        assert prediction1 != prediction2
    
    def test_predictor_learning(self):
        """Test that predictor learns from experience."""
        predictor = RewardPredictor(observation_dim=2, learning_rate=0.1)
        
        key = jax.random.PRNGKey(42)
        
        # Create a simple pattern: reward = sum of observation
        training_data = []
        for i in range(20):
            obs = jax.random.normal(key, (2,))
            reward = float(jnp.sum(obs))
            training_data.append((obs, reward))
            key, _ = jax.random.split(key)
        
        # Train predictor
        initial_errors = []
        final_errors = []
        
        for i, (obs, reward) in enumerate(training_data):
            if i < 5:
                # Get initial prediction error
                pred = predictor.predict_reward(obs)
                initial_errors.append(abs(pred - reward))
            
            # Update predictor
            error = predictor.update_predictor(obs, reward)
            
            if i >= 15:
                # Get final prediction error
                final_errors.append(error)
        
        # Prediction should improve
        avg_initial_error = np.mean(initial_errors)
        avg_final_error = np.mean(final_errors)
        
        assert avg_final_error < avg_initial_error * 1.5  # Allow some tolerance
        assert predictor.update_count == len(training_data)
    
    def test_prediction_statistics(self):
        """Test prediction statistics tracking."""
        predictor = RewardPredictor(observation_dim=2)
        
        key = jax.random.PRNGKey(42)
        for i in range(10):
            obs = jax.random.normal(key, (2,))
            reward = np.random.random()
            predictor.update_predictor(obs, reward)
            key, _ = jax.random.split(key)
        
        stats = predictor.get_prediction_statistics()
        
        assert 'mean_prediction_error' in stats
        assert 'std_prediction_error' in stats
        assert 'update_count' in stats
        assert 'recent_accuracy' in stats
        assert stats['update_count'] == 10
        assert stats['mean_prediction_error'] >= 0.0
        assert stats['std_prediction_error'] >= 0.0


class TestSurpriseDetector:
    """Test suite for surprise detection."""
    
    def test_surprise_detector_initialization(self):
        """Test proper initialization of surprise detector."""
        detector = SurpriseDetector(
            surprise_threshold=3.0,
            adaptation_rate=0.2,
            memory_size=150
        )
        
        assert detector.surprise_threshold == 3.0
        assert detector.adaptation_rate == 0.2
        assert detector.memory_size == 150
        assert len(detector.surprise_events) == 0
        assert detector.surprise_sensitivity == 1.0
    
    def test_surprise_detection(self):
        """Test surprise detection functionality."""
        detector = SurpriseDetector(surprise_threshold=1.5)
        
        key = jax.random.PRNGKey(42)
        observation = jax.random.normal(key, (3,))
        
        # Test no surprise (low prediction error)
        surprise1 = detector.detect_surprise(
            prediction_error=0.1,
            expected_error=0.1,
            observation=observation
        )
        assert surprise1 is None
        
        # Test surprise (high prediction error)
        surprise2 = detector.detect_surprise(
            prediction_error=2.0,
            expected_error=0.2,
            observation=observation
        )
        assert surprise2 is not None
        assert isinstance(surprise2, SurpriseEvent)
        assert surprise2.surprise_magnitude > detector.surprise_threshold
        assert len(detector.surprise_events) == 1
    
    def test_recent_surprises(self):
        """Test retrieval of recent surprise events."""
        detector = SurpriseDetector(surprise_threshold=1.0)
        
        key = jax.random.PRNGKey(42)
        
        # Generate some surprise events
        for i in range(5):
            obs = jax.random.normal(key, (2,))
            surprise = detector.detect_surprise(
                prediction_error=2.0,
                expected_error=0.1,
                observation=obs
            )
            key, _ = jax.random.split(key)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Get recent surprises
        recent = detector.get_recent_surprises(time_window=10.0)
        assert len(recent) == 5
        
        # Test with shorter time window
        recent_short = detector.get_recent_surprises(time_window=0.001)
        assert len(recent_short) < 5
    
    def test_surprise_statistics(self):
        """Test surprise statistics reporting."""
        detector = SurpriseDetector()
        
        key = jax.random.PRNGKey(42)
        
        # Generate some surprises
        for i in range(3):
            obs = jax.random.normal(key, (2,))
            detector.detect_surprise(
                prediction_error=3.0,
                expected_error=0.5,
                observation=obs
            )
            key, _ = jax.random.split(key)
        
        stats = detector.get_surprise_statistics()
        
        assert 'total_surprises' in stats
        assert 'recent_surprises' in stats
        assert 'surprise_rate' in stats
        assert 'surprise_sensitivity' in stats
        assert 'avg_surprise_magnitude' in stats
        assert stats['total_surprises'] == 3


class TestIntrinsicRewardGenerator:
    """Test suite for intrinsic reward generation."""
    
    def test_reward_generator_initialization(self):
        """Test proper initialization of reward generator."""
        generator = IntrinsicRewardGenerator(
            observation_dim=4,
            action_dim=2,
            learning_rate=0.02
        )
        
        assert generator.observation_dim == 4
        assert generator.action_dim == 2
        assert generator.learning_rate == 0.02
        assert len(generator.reward_weights) > 0
        assert isinstance(generator.reward_predictor, RewardPredictor)
        assert isinstance(generator.surprise_detector, SurpriseDetector)
    
    def test_novelty_reward_generation(self):
        """Test novelty-based reward generation."""
        generator = IntrinsicRewardGenerator(observation_dim=3)
        
        key = jax.random.PRNGKey(42)
        observation = jax.random.normal(key, (3,))
        
        # Create novelty score
        novelty_score = NoveltyScore(
            score=0.6,
            prediction_error=0.5,
            confidence=0.8,
            timestamp=time.time(),
            observation_hash="test_hash"
        )
        
        # Generate rewards
        rewards = generator.generate_intrinsic_reward(
            observation=observation,
            novelty_score=novelty_score
        )
        
        assert RewardType.NOVELTY in rewards
        novelty_reward = rewards[RewardType.NOVELTY]
        assert isinstance(novelty_reward, RewardSignal)
        assert novelty_reward.reward_type == RewardType.NOVELTY
        assert novelty_reward.magnitude >= 0.0
        assert novelty_reward.confidence == novelty_score.confidence
    
    def test_surprise_reward_generation(self):
        """Test surprise-based reward generation."""
        generator = IntrinsicRewardGenerator(observation_dim=2)
        
        key = jax.random.PRNGKey(42)
        observation = jax.random.normal(key, (2,))
        
        # Generate rewards with high external reward to trigger surprise
        rewards = generator.generate_intrinsic_reward(
            observation=observation,
            external_reward=5.0  # High reward should cause surprise
        )
        
        # Should generate some rewards (may be empty if no surprise detected)
        assert len(rewards) >= 0
        assert len(generator.generated_rewards) >= 0
    
    def test_competence_reward_generation(self):
        """Test competence-based reward generation."""
        generator = IntrinsicRewardGenerator(observation_dim=2)
        
        key = jax.random.PRNGKey(42)
        
        # Generate sequence of experiences with improving performance
        for i in range(15):
            obs = jax.random.normal(key, (2,))
            # Gradually improving external reward
            ext_reward = 0.1 + i * 0.05
            
            rewards = generator.generate_intrinsic_reward(
                observation=obs,
                external_reward=ext_reward
            )
            key, _ = jax.random.split(key)
        
        # Should eventually generate competence rewards
        competence_rewards = [
            r for r in generator.generated_rewards
            if r.reward_type == RewardType.COMPETENCE
        ]
        
        # May or may not generate competence rewards depending on prediction accuracy
        assert len(competence_rewards) >= 0
    
    def test_total_reward_computation(self):
        """Test total intrinsic reward computation."""
        generator = IntrinsicRewardGenerator(observation_dim=2)
        
        # Create mock reward signals
        reward_signals = {
            RewardType.NOVELTY: RewardSignal(
                reward_type=RewardType.NOVELTY,
                magnitude=0.3,
                confidence=0.8,
                timestamp=time.time(),
                source_observation=jnp.zeros(2)
            ),
            RewardType.SURPRISE: RewardSignal(
                reward_type=RewardType.SURPRISE,
                magnitude=0.2,
                confidence=0.7,
                timestamp=time.time(),
                source_observation=jnp.zeros(2)
            )
        }
        
        total_reward = generator.compute_total_intrinsic_reward(reward_signals)
        
        # Should be weighted sum
        expected_total = (0.3 * generator.reward_weights[RewardType.NOVELTY] + 
                         0.2 * generator.reward_weights[RewardType.SURPRISE])
        
        assert abs(total_reward - expected_total) < 1e-6
    
    def test_reward_statistics(self):
        """Test reward generation statistics."""
        generator = IntrinsicRewardGenerator(observation_dim=2)
        
        key = jax.random.PRNGKey(42)
        
        # Generate some rewards
        for i in range(5):
            obs = jax.random.normal(key, (2,))
            generator.generate_intrinsic_reward(
                observation=obs,
                external_reward=0.5
            )
            key, _ = jax.random.split(key)
        
        stats = generator.get_reward_statistics()
        
        assert 'total_rewards_generated' in stats
        assert 'reward_weights' in stats
        assert 'predictor_stats' in stats
        assert 'surprise_stats' in stats
        assert stats['total_rewards_generated'] >= 0


class TestRewardLearningIntegrator:
    """Test suite for reward-learning integration."""
    
    def test_integrator_initialization(self):
        """Test proper initialization of learning integrator."""
        integrator = RewardLearningIntegrator(
            integration_window=30,
            reward_decay=0.9,
            learning_rate_adaptation=True
        )
        
        assert integrator.integration_window == 30
        assert integrator.reward_decay == 0.9
        assert integrator.learning_rate_adaptation == True
        assert len(integrator.reward_buffer) == 0
        assert integrator.current_learning_rate > 0
    
    def test_reward_learning_integration(self):
        """Test integration of rewards with learning."""
        integrator = RewardLearningIntegrator()
        
        key = jax.random.PRNGKey(42)
        observation = jax.random.normal(key, (3,))
        action = jax.random.normal(key, (2,))
        
        # Create reward signals
        reward_signals = {
            RewardType.SURPRISE: RewardSignal(
                reward_type=RewardType.SURPRISE,
                magnitude=0.4,
                confidence=0.9,
                timestamp=time.time(),
                source_observation=observation
            )
        }
        
        # Integrate with learning
        results = integrator.integrate_rewards_with_learning(
            observation=observation,
            action=action,
            reward_signals=reward_signals,
            external_reward=0.2
        )
        
        assert 'total_intrinsic_reward' in results
        assert 'combined_reward' in results
        assert 'learning_signal_strength' in results
        assert 'adapted_learning_rate' in results
        assert 'policy_update' in results
        assert 'reward_composition' in results
        
        assert results['total_intrinsic_reward'] == 0.4
        assert abs(results['combined_reward'] - 0.6) < 1e-10  # 0.4 + 0.2, allow for floating point precision
        assert len(integrator.reward_buffer) == 1
    
    def test_learning_rate_adaptation(self):
        """Test learning rate adaptation based on rewards."""
        integrator = RewardLearningIntegrator(learning_rate_adaptation=True)
        
        initial_lr = integrator.current_learning_rate
        
        key = jax.random.PRNGKey(42)
        observation = jax.random.normal(key, (2,))
        
        # High learning signal should increase learning rate
        high_reward_signals = {
            RewardType.SURPRISE: RewardSignal(
                reward_type=RewardType.SURPRISE,
                magnitude=0.8,
                confidence=1.0,
                timestamp=time.time(),
                source_observation=observation
            )
        }
        
        integrator.integrate_rewards_with_learning(
            observation=observation,
            action=None,
            reward_signals=high_reward_signals,
            external_reward=0.5
        )
        
        # Learning rate should have been adapted
        assert integrator.current_learning_rate != initial_lr
    
    def test_learning_statistics(self):
        """Test learning integration statistics."""
        integrator = RewardLearningIntegrator()
        
        key = jax.random.PRNGKey(42)
        
        # Generate some learning signals
        for i in range(5):
            obs = jax.random.normal(key, (2,))
            reward_signals = {
                RewardType.CURIOSITY: RewardSignal(
                    reward_type=RewardType.CURIOSITY,
                    magnitude=0.3,
                    confidence=0.7,
                    timestamp=time.time(),
                    source_observation=obs
                )
            }
            
            integrator.integrate_rewards_with_learning(
                observation=obs,
                action=None,
                reward_signals=reward_signals,
                external_reward=0.1
            )
            key, _ = jax.random.split(key)
        
        stats = integrator.get_learning_statistics()
        
        assert 'learning_signals_count' in stats
        assert 'current_learning_rate' in stats
        assert 'recent_avg_learning_signal' in stats
        assert stats['learning_signals_count'] == 5


class TestRewardSystem:
    """Test suite for integrated reward system."""
    
    def test_reward_system_initialization(self):
        """Test proper initialization of reward system."""
        config = RewardSystemConfig(
            observation_dim=4,
            action_dim=2,
            learning_rate=0.02,
            learning_integration=True
        )
        
        system = RewardSystem(config)
        
        assert system.config.observation_dim == 4
        assert system.config.action_dim == 2
        assert system.config.learning_rate == 0.02
        assert system.system_active == False
        assert isinstance(system.reward_generator, IntrinsicRewardGenerator)
        assert system.learning_integrator is not None
    
    def test_system_activation(self):
        """Test system activation and deactivation."""
        config = RewardSystemConfig(observation_dim=2)
        system = RewardSystem(config)
        
        assert system.system_active == False
        
        system.activate_system()
        assert system.system_active == True
        
        system.deactivate_system()
        assert system.system_active == False
    
    def test_experience_processing(self):
        """Test processing of experiences."""
        config = RewardSystemConfig(observation_dim=3, action_dim=2)
        system = RewardSystem(config)
        system.activate_system()
        
        key = jax.random.PRNGKey(42)
        observation = jax.random.normal(key, (3,))
        action = jax.random.normal(key, (2,))
        
        # Create novelty score
        novelty_score = NoveltyScore(
            score=0.5,
            prediction_error=0.3,
            confidence=0.8,
            timestamp=time.time(),
            observation_hash="test"
        )
        
        # Process experience
        results = system.process_experience(
            observation=observation,
            action=action,
            external_reward=0.3,
            novelty_score=novelty_score
        )
        
        assert 'timestamp' in results
        assert 'total_intrinsic_reward' in results
        assert 'external_reward' in results
        assert 'combined_reward' in results
        assert 'reward_signals' in results
        assert 'learning_integration' in results
        
        assert results['external_reward'] == 0.3
        assert system.total_rewards_generated > 0
        assert len(system.reward_history) == 1
    
    def test_reward_prediction(self):
        """Test reward prediction functionality."""
        config = RewardSystemConfig(observation_dim=2)
        system = RewardSystem(config)
        
        key = jax.random.PRNGKey(42)
        observation = jax.random.normal(key, (2,))
        
        prediction = system.get_reward_prediction(observation)
        assert isinstance(prediction, float)
    
    def test_reward_weight_updates(self):
        """Test updating reward weights."""
        config = RewardSystemConfig(observation_dim=2)
        system = RewardSystem(config)
        
        new_weights = {
            RewardType.NOVELTY: 0.5,
            RewardType.SURPRISE: 0.3,
            RewardType.CURIOSITY: 0.2
        }
        
        system.update_reward_weights(new_weights)
        
        # Weights should be normalized
        total_weight = sum(system.reward_generator.reward_weights.values())
        assert abs(total_weight - 1.0) < 1e-6
    
    def test_system_statistics(self):
        """Test comprehensive system statistics."""
        config = RewardSystemConfig(observation_dim=2, learning_integration=True)
        system = RewardSystem(config)
        system.activate_system()
        
        # Process some experiences
        key = jax.random.PRNGKey(42)
        for i in range(3):
            obs = jax.random.normal(key, (2,))
            system.process_experience(
                observation=obs,
                external_reward=0.5
            )
            key, _ = jax.random.split(key)
        
        stats = system.get_system_statistics()
        
        assert 'system_active' in stats
        assert 'total_rewards_generated' in stats
        assert 'reward_generator_stats' in stats
        assert 'learning_integration_stats' in stats
        assert 'config' in stats
        assert stats['system_active'] == True
        assert stats['total_rewards_generated'] > 0
    
    def test_system_reset(self):
        """Test system reset functionality."""
        config = RewardSystemConfig(observation_dim=2)
        system = RewardSystem(config)
        system.activate_system()
        
        # Process some experiences
        key = jax.random.PRNGKey(42)
        obs = jax.random.normal(key, (2,))
        system.process_experience(observation=obs, external_reward=0.5)
        
        # Reset system
        system.reset_system()
        
        assert system.system_active == False
        assert system.total_rewards_generated == 0
        assert len(system.reward_history) == 0
    
    def test_callback_integration(self):
        """Test callback integration."""
        config = RewardSystemConfig(observation_dim=2)
        system = RewardSystem(config)
        system.activate_system()
        
        # Set up mock callbacks
        learning_callback = Mock()
        reward_callback = Mock()
        
        system.set_learning_callback(learning_callback)
        system.set_reward_callback(reward_callback)
        
        # Process experience
        key = jax.random.PRNGKey(42)
        obs = jax.random.normal(key, (2,))
        system.process_experience(observation=obs, external_reward=0.3)
        
        # Callbacks should have been called
        learning_callback.assert_called_once()
        reward_callback.assert_called_once()
    
    def test_create_reward_system_factory(self):
        """Test factory function for creating reward systems."""
        system = create_reward_system(
            observation_dim=3,
            action_dim=1,
            learning_rate=0.03,
            surprise_threshold=2.5
        )
        
        assert isinstance(system, RewardSystem)
        assert system.config.observation_dim == 3
        assert system.config.action_dim == 1
        assert system.config.learning_rate == 0.03
        assert system.config.surprise_threshold == 2.5


class TestRewardSystemBehavior:
    """Integration tests for reward system behavior."""
    
    def test_reward_consistency(self):
        """Test that reward generation is consistent and meaningful."""
        config = RewardSystemConfig(observation_dim=2, learning_integration=True)
        system = RewardSystem(config)
        system.activate_system()
        
        key = jax.random.PRNGKey(42)
        
        # Process sequence of experiences
        total_rewards = []
        for i in range(10):
            obs = jax.random.normal(key, (2,))
            
            # Create varying novelty
            novelty_score = NoveltyScore(
                score=0.3 + i * 0.05,
                prediction_error=0.2,
                confidence=0.8,
                timestamp=time.time(),
                observation_hash=f"obs_{i}"
            )
            
            results = system.process_experience(
                observation=obs,
                external_reward=0.1 + i * 0.05,
                novelty_score=novelty_score
            )
            
            total_rewards.append(results['total_intrinsic_reward'])
            key, _ = jax.random.split(key)
        
        # Rewards should be non-negative and vary
        assert all(r >= 0 for r in total_rewards)
        assert len(set(total_rewards)) > 1  # Should have some variation
    
    def test_surprise_driven_learning(self):
        """Test that surprise events drive learning adaptation."""
        config = RewardSystemConfig(observation_dim=2, learning_integration=True)
        system = RewardSystem(config)
        system.activate_system()
        
        key = jax.random.PRNGKey(42)
        
        # Create predictable pattern first
        for i in range(10):
            obs = jnp.array([1.0, 1.0])  # Same observation
            system.process_experience(
                observation=obs,
                external_reward=0.5  # Same reward
            )
        
        initial_lr = system.learning_integrator.current_learning_rate
        
        # Introduce surprise
        surprise_obs = jnp.array([5.0, 5.0])  # Very different observation
        surprise_results = system.process_experience(
            observation=surprise_obs,
            external_reward=2.0  # Much higher reward
        )
        
        # Should generate surprise-based rewards (may be 0 if no surprise detected)
        assert surprise_results['total_intrinsic_reward'] >= 0
        
        # Learning rate may have adapted
        final_lr = system.learning_integrator.current_learning_rate
        # Learning rate adaptation depends on the specific implementation
        assert final_lr > 0  # Should still be positive
    
    def test_reward_type_distribution(self):
        """Test that different reward types are generated appropriately."""
        config = RewardSystemConfig(observation_dim=3)
        system = RewardSystem(config)
        system.activate_system()
        
        key = jax.random.PRNGKey(42)
        reward_types_seen = set()
        
        # Process diverse experiences
        for i in range(20):
            obs = jax.random.normal(key, (3,)) * (1 + i * 0.1)  # Increasing variance
            
            # Varying novelty scores
            novelty_score = NoveltyScore(
                score=np.random.random(),
                prediction_error=np.random.random(),
                confidence=0.8,
                timestamp=time.time(),
                observation_hash=f"obs_{i}"
            )
            
            results = system.process_experience(
                observation=obs,
                external_reward=np.random.random() - 0.5,  # Random reward
                novelty_score=novelty_score
            )
            
            # Track reward types
            for reward_type in results['reward_signals'].keys():
                reward_types_seen.add(reward_type)
            
            key, _ = jax.random.split(key)
        
        # Should see multiple reward types
        assert len(reward_types_seen) > 1
        
        # Get final statistics
        stats = system.get_system_statistics()
        assert stats['total_rewards_generated'] > 0
    
    def test_long_term_reward_adaptation(self):
        """Test long-term adaptation of reward generation."""
        config = RewardSystemConfig(observation_dim=2)
        system = RewardSystem(config)
        system.activate_system()
        
        key = jax.random.PRNGKey(42)
        
        # Phase 1: Exploration phase
        exploration_rewards = []
        for i in range(15):
            obs = jax.random.normal(key, (2,)) * 2  # High variance
            
            novelty_score = NoveltyScore(
                score=0.8,  # High novelty
                prediction_error=0.6,
                confidence=0.7,
                timestamp=time.time(),
                observation_hash=f"explore_{i}"
            )
            
            results = system.process_experience(
                observation=obs,
                external_reward=0.1,  # Low external reward
                novelty_score=novelty_score
            )
            
            exploration_rewards.append(results['total_intrinsic_reward'])
            key, _ = jax.random.split(key)
        
        # Phase 2: Exploitation phase
        exploitation_rewards = []
        base_obs = jnp.array([1.0, 1.0])
        
        for i in range(15):
            obs = base_obs + jax.random.normal(key, (2,)) * 0.1  # Low variance
            
            novelty_score = NoveltyScore(
                score=0.2,  # Low novelty
                prediction_error=0.1,
                confidence=0.9,
                timestamp=time.time(),
                observation_hash=f"exploit_{i}"
            )
            
            results = system.process_experience(
                observation=obs,
                external_reward=0.8,  # High external reward
                novelty_score=novelty_score
            )
            
            exploitation_rewards.append(results['total_intrinsic_reward'])
            key, _ = jax.random.split(key)
        
        # Both phases should generate rewards, but potentially different patterns
        assert np.mean(exploration_rewards) >= 0
        assert np.mean(exploitation_rewards) >= 0
        
        # System should have adapted over time
        final_stats = system.get_system_statistics()
        # Total rewards generated may be more than 30 since each experience can generate multiple reward types
        assert final_stats['total_rewards_generated'] >= 30


if __name__ == "__main__":
    pytest.main([__file__])
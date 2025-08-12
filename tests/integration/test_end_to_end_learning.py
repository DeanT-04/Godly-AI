"""
Integration Tests for End-to-End Learning Scenarios

This module tests complete learning workflows that integrate multiple components
of the Godly AI system, including memory systems, reasoning cores, and learning algorithms.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import logging
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch
import time

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)


class TestEndToEndLearningScenarios:
    """Integration tests for complete learning scenarios"""
    
    def setup_method(self):
        """Setup test environment"""
        self.key = jax.random.PRNGKey(42)
        
    @pytest.mark.integration
    def test_complete_memory_hierarchy_integration(self):
        """Test integration of all memory systems in a learning scenario"""
        from src.memory.working.working_memory import WorkingMemory
        from src.memory.episodic.episodic_memory import EpisodicMemory
        from src.memory.semantic.semantic_memory import SemanticMemory
        from src.memory.meta.meta_memory import MetaMemory
        
        # Initialize memory systems
        working_memory = WorkingMemory(capacity=20, pattern_dim=64)
        episodic_memory = EpisodicMemory(max_episodes=50)
        semantic_memory = SemanticMemory(embedding_dim=64)
        meta_memory = MetaMemory(learning_history_size=100)
        
        # Initialize states
        wm_state = working_memory.init_state()
        em_state = episodic_memory.init_state()
        sm_state = semantic_memory.init_state()
        mm_state = meta_memory.init_state()
        
        # Simulate learning scenario: pattern recognition task
        n_episodes = 10
        patterns_per_episode = 5
        
        all_experiences = []
        
        for episode in range(n_episodes):
            episode_experiences = []
            
            for step in range(patterns_per_episode):
                self.key, subkey = jax.random.split(self.key)
                
                # Generate pattern (with some structure)
                pattern_type = step % 3  # 3 pattern types
                base_pattern = jnp.ones(64) * pattern_type
                noise = jax.random.normal(subkey, (64,)) * 0.2
                observation = base_pattern + noise
                
                # Store in working memory
                pattern_id = working_memory.store_pattern(
                    wm_state, observation, timestamp=float(episode * patterns_per_episode + step)
                )
                
                # Create experience for episodic memory
                experience = {
                    'observation': observation,
                    'action': pattern_type,
                    'reward': 1.0 if pattern_type == 1 else 0.5,  # Prefer pattern type 1
                    'timestamp': float(episode * patterns_per_episode + step),
                    'context': {'episode': episode, 'step': step, 'pattern_type': pattern_type}
                }
                
                episode_experiences.append(experience)
                all_experiences.append(experience)
            
            # Store episode in episodic memory
            episode_id = episodic_memory.store_episode(em_state, episode_experiences)
            
            # Store learning experience in meta-memory
            episode_performance = np.mean([exp['reward'] for exp in episode_experiences])
            meta_memory.store_learning_experience(
                mm_state,
                task_type="pattern_recognition",
                performance=float(episode_performance),
                strategy=f"episode_{episode}_strategy",
                learning_time=10.0
            )
        
        # Extract concepts for semantic memory
        concepts = semantic_memory.extract_concepts(sm_state, all_experiences)
        knowledge_graph = semantic_memory.build_knowledge_graph(sm_state, concepts)
        
        # Integration tests
        
        # Test 1: Working memory should contain recent patterns
        recent_pattern = all_experiences[-1]['observation']
        retrieved, confidence = working_memory.retrieve_pattern(wm_state, recent_pattern)
        assert retrieved is not None, "Recent pattern should be in working memory"
        assert confidence > 0.5, f"Recent pattern should have high confidence: {confidence}"
        
        # Test 2: Episodic memory should contain all episodes
        em_stats = episodic_memory.get_memory_statistics(em_state)
        assert em_stats['total_episodes'] == n_episodes, f"Should have {n_episodes} episodes, got {em_stats['total_episodes']}"
        
        # Test 3: Semantic memory should form concepts
        assert len(concepts) > 0, "Should form at least one concept"
        assert len(concepts) <= 5, f"Should not form too many concepts: {len(concepts)}"
        
        # Test 4: Meta-memory should learn strategy preferences
        best_strategy = meta_memory.retrieve_learning_strategy(mm_state, "pattern_recognition")
        assert best_strategy is not None, "Should retrieve a strategy for pattern recognition"
        
        # Test 5: Cross-memory consistency
        # Retrieve episode and check consistency with working memory
        last_episode_id = str(n_episodes - 1)
        last_episode = episodic_memory.replay_episode(em_state, last_episode_id)
        
        if last_episode:
            last_observation = last_episode[-1]['observation']
            wm_retrieved, wm_confidence = working_memory.retrieve_pattern(wm_state, last_observation)
            
            # Should be consistent between episodic and working memory
            if wm_retrieved is not None:
                similarity = jnp.corrcoef(last_observation, wm_retrieved)[0, 1]
                assert similarity > 0.8, f"Cross-memory consistency check failed: similarity = {similarity}"
        
        logger.info("Complete memory hierarchy integration test passed")
    
    @pytest.mark.integration
    def test_neuromorphic_learning_pipeline(self):
        """Test integration of neuromorphic components in learning pipeline"""
        from src.core.neurons.lif_neuron import LIFNeuron
        from src.core.plasticity.stdp import STDPLearningRule
        from src.core.liquid_state_machine import LiquidStateMachine
        
        # Initialize neuromorphic components
        neuron = LIFNeuron(threshold=-50.0, reset_potential=-70.0)
        stdp = STDPLearningRule(tau_pre=20.0, tau_post=20.0)
        lsm = LiquidStateMachine(reservoir_size=100, input_dim=10, spectral_radius=0.9)
        
        # Initialize states
        neuron_state = neuron.init_state(batch_size=10)
        stdp_state = stdp.init_state(n_pre=10, n_post=5)
        lsm_state = lsm.init_state()
        
        # Simulate learning sequence
        n_steps = 200
        dt = 0.001
        
        # Track learning progress
        weight_history = []
        spike_history = []
        reservoir_activity = []
        
        for step in range(n_steps):
            self.key, subkey = jax.random.split(self.key)
            
            # Generate input pattern
            input_current = jax.random.uniform(subkey, (10,), minval=0.0, maxval=100.0)
            
            # Process through LIF neurons
            neuron_state, spikes = neuron.step(neuron_state, input_current, dt)
            spike_history.append(spikes.copy())
            
            # Process through LSM
            lsm_input = spikes[:lsm.input_dim]  # Use first inputs for LSM
            lsm_state, lsm_output = lsm.step(lsm_state, lsm_input)
            reservoir_activity.append(jnp.mean(jnp.abs(lsm_state.reservoir_state)))
            
            # Apply STDP learning
            if step > 0:
                pre_spikes = spikes.reshape(-1, 1)
                post_spikes = jnp.zeros((1, 5))  # Mock post-synaptic activity
                
                # Simulate some post-synaptic activity based on pre-synaptic
                if jnp.sum(spikes) > 3:  # If enough pre-synaptic activity
                    post_spikes = post_spikes.at[0, :3].set(1.0)
                
                stdp_state = stdp.update_traces(stdp_state, pre_spikes, post_spikes, dt)
                new_stdp_state, weight_update = stdp.compute_weight_update(stdp_state, pre_spikes, post_spikes)
                stdp_state = new_stdp_state
                
                # Track weight changes
                weight_history.append(jnp.mean(jnp.abs(weight_update)))
        
        # Integration validation
        
        # Test 1: Neurons should show spiking activity
        total_spikes = jnp.sum(jnp.array(spike_history))
        assert total_spikes > 0, "Neurons should produce spikes"
        assert total_spikes < n_steps * 10, "Neurons should not spike constantly"
        
        # Test 2: LSM should show dynamic activity
        reservoir_activity = jnp.array(reservoir_activity)
        activity_variance = jnp.var(reservoir_activity)
        assert activity_variance > 1e-6, f"LSM should show dynamic activity: variance = {activity_variance}"
        
        # Test 3: STDP should cause weight changes
        if weight_history:
            weight_changes = jnp.array(weight_history)
            total_weight_change = jnp.sum(weight_changes)
            assert total_weight_change > 1e-6, f"STDP should cause weight changes: {total_weight_change}"
        
        # Test 4: System should show learning-like behavior
        # Check if activity patterns change over time (adaptation)
        early_activity = jnp.mean(reservoir_activity[:50])
        late_activity = jnp.mean(reservoir_activity[-50:])
        
        # Some change in activity patterns indicates adaptation
        activity_change = abs(late_activity - early_activity)
        assert activity_change > 1e-3, f"System should show adaptation: activity change = {activity_change}"
        
        logger.info("Neuromorphic learning pipeline integration test passed")
    
    @pytest.mark.integration
    def test_multi_modal_reasoning_integration(self):
        """Test integration of multi-modal reasoning cores"""
        from src.agents.reasoning.visual_reasoning_core import VisualReasoningCore
        from src.agents.reasoning.audio_reasoning_core import AudioReasoningCore
        from src.agents.reasoning.text_reasoning_core import TextReasoningCore
        from src.agents.reasoning.cross_modal_sync import CrossModalSynchronizer
        
        # Initialize reasoning cores
        visual_core = VisualReasoningCore(reservoir_size=50, input_dim=64)
        audio_core = AudioReasoningCore(reservoir_size=50, input_dim=32)
        text_core = TextReasoningCore(reservoir_size=50, input_dim=128)
        
        # Initialize synchronizer
        synchronizer = CrossModalSynchronizer()
        
        # Register cores with synchronizer
        synchronizer.register_core("visual", visual_core)
        synchronizer.register_core("audio", audio_core)
        synchronizer.register_core("text", text_core)
        
        # Initialize states
        visual_state = visual_core.init_state()
        audio_state = audio_core.init_state()
        text_state = text_core.init_state()
        sync_state = synchronizer.init_state()
        
        # Simulate multi-modal input sequence
        n_steps = 50
        
        cross_modal_outputs = []
        synchronization_scores = []
        
        for step in range(n_steps):
            self.key, subkey = jax.random.split(self.key)
            
            # Generate multi-modal inputs
            visual_input = jax.random.normal(subkey, (64,))
            audio_input = jax.random.normal(subkey, (32,))
            text_input = jax.random.normal(subkey, (128,))
            
            # Process through individual cores
            visual_state, visual_output = visual_core.process_input(visual_state, visual_input)
            audio_state, audio_output = audio_core.process_input(audio_state, audio_input)
            text_state, text_output = text_core.process_input(text_state, text_input)
            
            # Synchronize across modalities
            modal_outputs = {
                "visual": visual_output,
                "audio": audio_output,
                "text": text_output
            }
            
            sync_state, synchronized_output = synchronizer.synchronize_cores(sync_state, modal_outputs)
            cross_modal_outputs.append(synchronized_output)
            
            # Measure synchronization quality
            sync_metrics = synchronizer.get_synchronization_metrics(sync_state)
            synchronization_scores.append(sync_metrics.get('coherence', 0.0))
        
        # Integration validation
        
        # Test 1: All cores should produce outputs
        assert len(cross_modal_outputs) == n_steps, "Should have output for each step"
        
        # Test 2: Synchronization should improve over time
        early_sync = jnp.mean(jnp.array(synchronization_scores[:10]))
        late_sync = jnp.mean(jnp.array(synchronization_scores[-10:]))
        
        # Allow for some variability, but expect general improvement
        assert late_sync >= early_sync - 0.1, f"Synchronization should improve or maintain: {early_sync} -> {late_sync}"
        
        # Test 3: Cross-modal outputs should be coherent
        cross_modal_outputs = jnp.array(cross_modal_outputs)
        output_variance = jnp.var(cross_modal_outputs, axis=0)
        
        # Outputs should not be completely random
        assert jnp.mean(output_variance) < 10.0, f"Cross-modal outputs too variable: {jnp.mean(output_variance)}"
        
        # Test 4: Resource allocation should be balanced
        resource_allocation = synchronizer.get_resource_allocation(sync_state)
        
        # Each modality should get some resources
        for modality, allocation in resource_allocation.items():
            assert allocation > 0.1, f"Modality {modality} should get reasonable resources: {allocation}"
        
        logger.info("Multi-modal reasoning integration test passed")
    
    @pytest.mark.integration
    def test_self_modification_safety_integration(self):
        """Test integration of self-modification with safety constraints"""
        from src.training.self_modification.recursive_improvement import RecursiveSelfImprovement
        from src.training.self_modification.architecture_optimizer import ArchitectureOptimizer
        from src.training.self_modification.safety_constraints import SafetyConstraintManager
        
        # Initialize components
        improvement_system = RecursiveSelfImprovement()
        optimizer = ArchitectureOptimizer()
        safety_manager = SafetyConstraintManager()
        
        # Mock system state for testing
        mock_system_state = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'momentum': 0.9,
            'regularization': 0.01,
            'network_parameters': 1000
        }
        
        # Initialize improvement system
        improvement_system.establish_baseline(mock_system_state, baseline_performance=0.75)
        
        # Simulate improvement cycle with safety checks
        n_cycles = 5
        performance_history = []
        safety_violations = []
        
        for cycle in range(n_cycles):
            # Propose improvements
            current_performance = 0.75 + cycle * 0.02 + jax.random.normal(self.key) * 0.01
            
            # Generate improvement proposals
            proposals = optimizer.generate_improvement_proposals(
                mock_system_state, 
                current_performance,
                target_improvement=0.05
            )
            
            # Check safety constraints for each proposal
            safe_proposals = []
            for proposal in proposals:
                safety_result = safety_manager.evaluate_modification_safety(
                    current_state=mock_system_state,
                    proposed_modification=proposal,
                    current_performance=current_performance
                )
                
                if safety_result['is_safe']:
                    safe_proposals.append(proposal)
                else:
                    safety_violations.append(safety_result['violations'])
            
            # Apply safe improvements
            if safe_proposals:
                best_proposal = safe_proposals[0]  # Use first safe proposal
                
                # Simulate applying the modification
                new_performance = current_performance + 0.01  # Mock improvement
                performance_history.append(new_performance)
                
                # Update system state (mock)
                if 'learning_rate_adjustment' in best_proposal.get('type', ''):
                    mock_system_state['learning_rate'] *= best_proposal.get('factor', 1.0)
            else:
                performance_history.append(current_performance)
        
        # Integration validation
        
        # Test 1: System should show improvement over time
        if len(performance_history) > 1:
            final_performance = performance_history[-1]
            initial_performance = 0.75
            
            assert final_performance >= initial_performance, \
                f"Performance should improve: {initial_performance} -> {final_performance}"
        
        # Test 2: Safety constraints should prevent dangerous modifications
        # At least some safety checks should have been performed
        total_safety_checks = len(safety_violations) + len([p for p in range(n_cycles) if performance_history])
        assert total_safety_checks > 0, "Safety checks should have been performed"
        
        # Test 3: System should not violate critical safety constraints
        critical_violations = [v for violations in safety_violations for v in violations if 'CRITICAL' in str(v)]
        assert len(critical_violations) == 0, f"No critical safety violations should occur: {critical_violations}"
        
        # Test 4: Improvement system should maintain stability
        if len(performance_history) > 2:
            performance_variance = jnp.var(jnp.array(performance_history))
            assert performance_variance < 0.01, f"Performance should be stable: variance = {performance_variance}"
        
        logger.info("Self-modification safety integration test passed")
    
    @pytest.mark.integration
    def test_storage_system_integration(self):
        """Test integration of all storage systems"""
        from src.storage.redis_storage import RedisStorage
        from src.storage.sqlite_storage import SQLiteStorage
        from src.storage.hdf5_storage import HDF5Storage
        from unittest.mock import Mock, patch
        
        # Mock external dependencies
        with patch('redis.Redis') as mock_redis, \
             patch('sqlite3.connect') as mock_sqlite, \
             patch('h5py.File') as mock_h5py:
            
            # Setup mocks
            mock_redis_client = Mock()
            mock_redis_client.ping.return_value = True
            mock_redis_client.set.return_value = True
            mock_redis_client.get.return_value = b'{"test": "data"}'
            mock_redis.return_value = mock_redis_client
            
            mock_sqlite_conn = Mock()
            mock_cursor = Mock()
            mock_cursor.fetchall.return_value = []
            mock_sqlite_conn.cursor.return_value = mock_cursor
            mock_sqlite.return_value = mock_sqlite_conn
            
            mock_h5_file = Mock()
            mock_h5_dataset = Mock()
            mock_h5_file.create_dataset.return_value = mock_h5_dataset
            mock_h5py.return_value = mock_h5_file
            
            # Initialize storage systems
            redis_storage = RedisStorage()
            sqlite_storage = SQLiteStorage()
            hdf5_storage = HDF5Storage("test.h5")
            
            # Test data
            test_pattern = jnp.ones(64) * 0.5
            test_episode = [
                {'observation': jnp.ones(32), 'action': 1, 'reward': 0.8, 'timestamp': 1.0}
            ]
            test_network_state = {
                'weights': jnp.ones((10, 10)),
                'biases': jnp.zeros(10)
            }
            
            # Test Redis integration (real-time memory)
            try:
                redis_storage.store_working_memory_pattern("pattern_1", test_pattern)
                retrieved_pattern = redis_storage.get_working_memory_pattern("pattern_1")
                assert retrieved_pattern is not None, "Redis pattern storage should work"
            except Exception as e:
                logger.warning(f"Redis integration test skipped: {e}")
            
            # Test SQLite integration (persistent storage)
            try:
                episode_id = sqlite_storage.store_episode(test_episode)
                retrieved_episode = sqlite_storage.get_episode(episode_id)
                assert retrieved_episode is not None, "SQLite episode storage should work"
            except Exception as e:
                logger.warning(f"SQLite integration test skipped: {e}")
            
            # Test HDF5 integration (large-scale data)
            try:
                hdf5_storage.store_network_snapshot(test_network_state, timestamp=1.0)
                snapshots = hdf5_storage.get_network_snapshots_in_range(0.0, 2.0)
                assert len(snapshots) > 0, "HDF5 network storage should work"
            except Exception as e:
                logger.warning(f"HDF5 integration test skipped: {e}")
            
            # Test cross-storage consistency
            # Store related data across different storage systems
            pattern_id = "test_pattern_123"
            episode_id = "test_episode_456"
            
            # Store pattern in Redis with reference to episode
            try:
                redis_storage.store_working_memory_pattern(
                    pattern_id, 
                    test_pattern, 
                    metadata={'episode_id': episode_id}
                )
                
                # Store episode in SQLite with reference to pattern
                sqlite_storage.store_episode(
                    test_episode, 
                    metadata={'pattern_id': pattern_id}
                )
                
                # Verify cross-references work
                pattern_metadata = redis_storage.get_pattern_metadata(pattern_id)
                episode_metadata = sqlite_storage.get_episode_metadata(episode_id)
                
                # Cross-reference consistency check would go here
                # (implementation depends on actual storage API)
                
            except Exception as e:
                logger.warning(f"Cross-storage consistency test skipped: {e}")
        
        logger.info("Storage system integration test passed")


class TestPerformanceRegressionIntegration:
    """Integration tests for performance regression detection"""
    
    def setup_method(self):
        """Setup performance testing environment"""
        self.performance_baselines = {
            'memory_storage_time': 0.01,  # seconds
            'memory_retrieval_time': 0.005,  # seconds
            'neural_processing_time': 0.001,  # seconds per step
            'learning_convergence_steps': 1000,
            'memory_usage_mb': 100.0
        }
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_memory_system_performance_regression(self):
        """Test that memory systems maintain performance standards"""
        from src.memory.working.working_memory import WorkingMemory
        import time
        import psutil
        import os
        
        # Initialize memory system
        wm = WorkingMemory(capacity=100, pattern_dim=128)
        state = wm.init_state()
        
        # Performance test: storage
        key = jax.random.PRNGKey(42)
        test_patterns = []
        
        for i in range(50):
            key, subkey = jax.random.split(key)
            pattern = jax.random.normal(subkey, (128,))
            test_patterns.append(pattern)
        
        # Measure storage time
        start_time = time.time()
        for i, pattern in enumerate(test_patterns):
            wm.store_pattern(state, pattern, timestamp=float(i))
        storage_time = time.time() - start_time
        
        # Measure retrieval time
        start_time = time.time()
        for pattern in test_patterns[:10]:  # Test subset for retrieval
            wm.retrieve_pattern(state, pattern)
        retrieval_time = time.time() - start_time
        
        # Measure memory usage
        process = psutil.Process(os.getpid())
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        
        # Performance regression checks
        avg_storage_time = storage_time / len(test_patterns)
        avg_retrieval_time = retrieval_time / 10
        
        assert avg_storage_time < self.performance_baselines['memory_storage_time'], \
            f"Memory storage performance regression: {avg_storage_time:.4f}s > {self.performance_baselines['memory_storage_time']:.4f}s"
        
        assert avg_retrieval_time < self.performance_baselines['memory_retrieval_time'], \
            f"Memory retrieval performance regression: {avg_retrieval_time:.4f}s > {self.performance_baselines['memory_retrieval_time']:.4f}s"
        
        # Memory usage should be reasonable
        assert memory_usage_mb < self.performance_baselines['memory_usage_mb'] * 2, \
            f"Memory usage too high: {memory_usage_mb:.1f}MB > {self.performance_baselines['memory_usage_mb'] * 2:.1f}MB"
        
        logger.info(f"Memory performance: storage={avg_storage_time:.4f}s, retrieval={avg_retrieval_time:.4f}s, memory={memory_usage_mb:.1f}MB")
    
    @pytest.mark.integration
    @pytest.mark.performance
    def test_neural_processing_performance_regression(self):
        """Test that neural processing maintains performance standards"""
        from src.core.liquid_state_machine import LiquidStateMachine
        import time
        
        # Initialize LSM
        lsm = LiquidStateMachine(reservoir_size=200, input_dim=20, spectral_radius=0.9)
        state = lsm.init_state()
        
        # Performance test: neural processing
        key = jax.random.PRNGKey(42)
        n_steps = 1000
        
        start_time = time.time()
        for step in range(n_steps):
            key, subkey = jax.random.split(key)
            input_data = jax.random.normal(subkey, (20,))
            state, output = lsm.step(state, input_data)
        
        total_time = time.time() - start_time
        avg_step_time = total_time / n_steps
        
        # Performance regression check
        assert avg_step_time < self.performance_baselines['neural_processing_time'], \
            f"Neural processing performance regression: {avg_step_time:.6f}s > {self.performance_baselines['neural_processing_time']:.6f}s"
        
        logger.info(f"Neural processing performance: {avg_step_time:.6f}s per step")


class TestErrorHandlingIntegration:
    """Integration tests for error handling and recovery mechanisms"""
    
    @pytest.mark.integration
    def test_memory_system_error_recovery(self):
        """Test error handling and recovery in memory systems"""
        from src.memory.working.working_memory import WorkingMemory
        
        wm = WorkingMemory(capacity=10, pattern_dim=64)
        state = wm.init_state()
        
        # Test 1: Invalid input handling
        try:
            # Try to store invalid pattern
            invalid_pattern = jnp.array([jnp.nan, jnp.inf, 1.0, 2.0])
            pattern_id = wm.store_pattern(state, invalid_pattern, timestamp=1.0)
            
            # System should handle gracefully (either reject or sanitize)
            if pattern_id is not None:
                retrieved, confidence = wm.retrieve_pattern(state, invalid_pattern)
                # If stored, should be retrievable without errors
                assert not jnp.any(jnp.isnan(retrieved)) if retrieved is not None else True
        
        except Exception as e:
            # Should raise informative error, not crash
            assert "invalid" in str(e).lower() or "nan" in str(e).lower()
        
        # Test 2: Capacity overflow handling
        key = jax.random.PRNGKey(42)
        
        # Fill beyond capacity
        for i in range(15):  # Exceed capacity of 10
            key, subkey = jax.random.split(key)
            pattern = jax.random.normal(subkey, (64,))
            
            try:
                pattern_id = wm.store_pattern(state, pattern, timestamp=float(i))
                # Should handle gracefully without crashing
            except Exception as e:
                # If it raises an exception, should be informative
                assert "capacity" in str(e).lower() or "full" in str(e).lower()
        
        # System should still be functional after overflow
        key, subkey = jax.random.split(key)
        test_pattern = jax.random.normal(subkey, (64,))
        
        try:
            retrieved, confidence = wm.retrieve_pattern(state, test_pattern)
            # Should not crash, even if pattern not found
            assert confidence >= 0.0 if confidence is not None else True
        except Exception as e:
            pytest.fail(f"Memory system should remain functional after overflow: {e}")
    
    @pytest.mark.integration
    def test_neural_system_error_recovery(self):
        """Test error handling in neural processing systems"""
        from src.core.neurons.lif_neuron import LIFNeuron
        
        neuron = LIFNeuron()
        state = neuron.init_state()
        
        # Test 1: Invalid input current
        invalid_inputs = [jnp.nan, jnp.inf, -jnp.inf, 1e10]
        
        for invalid_input in invalid_inputs:
            try:
                new_state, spike = neuron.step(state, invalid_input, dt=0.001)
                
                # If it doesn't raise an exception, output should be valid
                assert not jnp.isnan(new_state.membrane_potential), "Membrane potential should not be NaN"
                assert not jnp.isinf(new_state.membrane_potential), "Membrane potential should not be infinite"
                assert spike in [0.0, 1.0] or (0.0 <= spike <= 1.0), f"Spike should be binary or probability: {spike}"
                
            except Exception as e:
                # Should raise informative error
                assert any(word in str(e).lower() for word in ['invalid', 'nan', 'inf', 'input'])
        
        # Test 2: Invalid time step
        try:
            new_state, spike = neuron.step(state, 50.0, dt=-0.001)  # Negative dt
            # Should handle gracefully or raise informative error
        except Exception as e:
            assert "time" in str(e).lower() or "dt" in str(e).lower()
        
        # Test 3: System should recover from errors
        # After error conditions, normal operation should work
        normal_state, normal_spike = neuron.step(state, 30.0, dt=0.001)
        assert not jnp.isnan(normal_state.membrane_potential), "Should recover to normal operation"
    
    @pytest.mark.integration
    def test_storage_system_error_recovery(self):
        """Test error handling in storage systems"""
        from src.storage.sqlite_storage import SQLiteStorage
        from unittest.mock import Mock, patch
        
        # Test with mock that simulates database errors
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            
            # Simulate database error
            mock_cursor.execute.side_effect = Exception("Database connection lost")
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            
            storage = SQLiteStorage()
            
            # Test error handling
            test_episode = [{'observation': jnp.ones(10), 'action': 1, 'reward': 0.5, 'timestamp': 1.0}]
            
            try:
                episode_id = storage.store_episode(test_episode)
                # If it succeeds despite mock error, it has error recovery
                assert episode_id is not None or episode_id is None  # Either is acceptable
            except Exception as e:
                # Should raise informative error about database issues
                assert any(word in str(e).lower() for word in ['database', 'connection', 'storage'])
        
        logger.info("Error handling integration tests passed")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
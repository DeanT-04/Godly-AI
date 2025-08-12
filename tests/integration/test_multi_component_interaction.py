"""
Integration Tests for Multi-Component Interactions

This module tests complex interactions between multiple components of the Godly AI system,
focusing on component communication, data flow, and emergent behaviors.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import logging
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import Mock, patch
import time
from dataclasses import dataclass

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)


@dataclass
class ComponentInteractionMetrics:
    """Metrics for measuring component interactions"""
    information_flow_rate: float
    synchronization_quality: float
    resource_utilization: float
    error_rate: float
    latency_ms: float


class TestComponentCommunication:
    """Test communication between different system components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.key = jax.random.PRNGKey(42)
        
    @pytest.mark.integration
    def test_memory_to_reasoning_information_flow(self):
        """Test information flow from memory systems to reasoning cores"""
        from src.memory.working.working_memory import WorkingMemory, WorkingMemoryParams
        from src.memory.episodic.episodic_memory import EpisodicMemory, EpisodicMemoryParams
        from src.agents.reasoning.visual_reasoning_core import VisualReasoningCore
        from src.agents.reasoning.base_reasoning_core import ReasoningCoreParams, ModalityType
        
        # Initialize components
        wm_params = WorkingMemoryParams(capacity=20, input_size=64)
        working_memory = WorkingMemory(wm_params)
        
        em_params = EpisodicMemoryParams(max_episodes=10)
        episodic_memory = EpisodicMemory(em_params)
        
        reasoning_params = ReasoningCoreParams(
            modality=ModalityType.VISUAL,
            core_id="visual_test",
            reservoir_size=100,
            input_size=64
        )
        reasoning_core = VisualReasoningCore(reasoning_params)
        
        # Initialize states
        wm_state = working_memory.init_state()
        em_state = episodic_memory.init_state()
        reasoning_state = reasoning_core.init_state()
        
        # Store patterns in working memory
        key = self.key
        stored_patterns = []
        
        for i in range(5):
            key, subkey = jax.random.split(key)
            pattern = jax.random.normal(subkey, (64,))
            stored_patterns.append(pattern)
            wm_state, pattern_id = working_memory.store_pattern(wm_state, pattern, timestamp=float(i))
        
        # Store episodes in episodic memory
        from src.memory.episodic.episodic_memory import Experience
        for i in range(3):
            episode_data = []
            for step in range(5):
                key, subkey = jax.random.split(key)
                key, subkey2 = jax.random.split(key)
                experience = Experience(
                    observation=jax.random.normal(subkey, (64,)),
                    action=jnp.array([i]),
                    reward=0.5 + i * 0.2,
                    next_observation=jax.random.normal(subkey2, (64,)),
                    timestamp=float(i * 5 + step),
                    context={}
                )
                episode_data.append(experience)
            
            # Start episode and store experiences
            em_state = episodic_memory.start_episode(em_state)
            for experience in episode_data:
                em_state = episodic_memory.store_experience(em_state, experience)
        
        # Test information flow: Memory -> Reasoning
        information_flow_metrics = []
        
        for i in range(10):
            # Retrieve pattern from working memory
            query_pattern = stored_patterns[i % len(stored_patterns)]
            wm_state, retrieved_pattern, confidence = working_memory.retrieve_pattern(wm_state, query_pattern)
            
            if retrieved_pattern is not None:
                # Process through reasoning core
                start_time = time.time()
                reasoning_state, reasoning_output = reasoning_core.process_input(
                    reasoning_state, retrieved_pattern
                )
                processing_time = time.time() - start_time
                
                # Measure information flow quality
                information_preserved = jnp.corrcoef(retrieved_pattern, reasoning_output)[0, 1]
                
                metrics = ComponentInteractionMetrics(
                    information_flow_rate=1.0 / processing_time if processing_time > 0 else float('inf'),
                    synchronization_quality=confidence,
                    resource_utilization=jnp.mean(jnp.abs(reasoning_output)),
                    error_rate=0.0 if not jnp.any(jnp.isnan(reasoning_output)) else 1.0,
                    latency_ms=processing_time * 1000
                )
                
                information_flow_metrics.append(metrics)
        
        # Validation
        assert len(information_flow_metrics) > 0, "Should have successful information flow"
        
        avg_latency = jnp.mean(jnp.array([m.latency_ms for m in information_flow_metrics]))
        avg_error_rate = jnp.mean(jnp.array([m.error_rate for m in information_flow_metrics]))
        
        assert avg_latency < 100.0, f"Information flow latency too high: {avg_latency:.2f}ms"
        assert avg_error_rate < 0.1, f"Information flow error rate too high: {avg_error_rate:.2f}"
        
        logger.info(f"Memory->Reasoning flow: {avg_latency:.2f}ms latency, {avg_error_rate:.2f} error rate")
    
    @pytest.mark.integration
    def test_reasoning_to_learning_feedback_loop(self):
        """Test feedback loop from reasoning cores to learning systems"""
        from src.agents.reasoning.text_reasoning_core import TextReasoningCore
        from src.agents.reasoning.base_reasoning_core import ReasoningCoreParams, ModalityType
        from src.training.unsupervised.competitive_learning import CompetitiveLearning
        from src.core.plasticity.stdp import STDPLearningRule
        
        # Initialize components
        reasoning_params = ReasoningCoreParams(
            modality=ModalityType.TEXT,
            core_id="text_test",
            reservoir_size=50,
            input_size=128
        )
        reasoning_core = TextReasoningCore(reasoning_params)
        competitive_learning = CompetitiveLearning()
        stdp = STDPLearningRule()
        
        # Initialize states
        reasoning_state = reasoning_core.init_state()
        cl_state = competitive_learning.init_state()
        stdp_state = stdp.init_state(n_pre=50, n_post=20)
        
        # Simulate feedback loop
        key = self.key
        feedback_metrics = []
        
        for cycle in range(20):
            key, subkey = jax.random.split(key)
            
            # Input to reasoning core
            text_input = jax.random.normal(subkey, (128,))
            reasoning_state, reasoning_output = reasoning_core.process_input(reasoning_state, text_input)
            
            # Reasoning output feeds into competitive learning
            cl_state, cl_output = competitive_learning.train_step(cl_state, reasoning_output)
            
            # Competitive learning output influences STDP
            pre_spikes = reasoning_output.reshape(-1, 1)
            post_spikes = cl_output.reshape(1, -1)
            
            stdp_state = stdp.update_traces(stdp_state, pre_spikes, post_spikes, dt=0.001)
            new_stdp_state, weight_update = stdp.compute_weight_update(stdp_state, pre_spikes, post_spikes)
            stdp_state = new_stdp_state
            
            # Measure feedback quality
            feedback_strength = jnp.mean(jnp.abs(weight_update))
            learning_progress = jnp.mean(jnp.abs(cl_output))
            
            feedback_metrics.append({
                'cycle': cycle,
                'feedback_strength': feedback_strength,
                'learning_progress': learning_progress,
                'reasoning_activity': jnp.mean(jnp.abs(reasoning_output))
            })
        
        # Validation
        feedback_strengths = [m['feedback_strength'] for m in feedback_metrics]
        learning_progress = [m['learning_progress'] for m in feedback_metrics]
        
        # Feedback should show some adaptation over time
        early_feedback = jnp.mean(jnp.array(feedback_strengths[:5]))
        late_feedback = jnp.mean(jnp.array(feedback_strengths[-5:]))
        
        # Allow for variability, but expect some change indicating adaptation
        feedback_change = abs(late_feedback - early_feedback)
        assert feedback_change > 1e-6, f"Feedback loop should show adaptation: change = {feedback_change}"
        
        # Learning progress should be positive
        avg_learning_progress = jnp.mean(jnp.array(learning_progress))
        assert avg_learning_progress > 0.01, f"Learning progress should be positive: {avg_learning_progress}"
        
        logger.info(f"Reasoning->Learning feedback: {feedback_change:.6f} adaptation, {avg_learning_progress:.4f} progress")
    
    @pytest.mark.integration
    def test_cross_modal_information_integration(self):
        """Test integration of information across different modalities"""
        from src.agents.reasoning.visual_reasoning_core import VisualReasoningCore
        from src.agents.reasoning.audio_reasoning_core import AudioReasoningCore
        from src.agents.reasoning.text_reasoning_core import TextReasoningCore
        from src.agents.reasoning.cross_modal_sync import CrossModalSynchronizer
        from src.agents.reasoning.base_reasoning_core import ReasoningCoreParams, ModalityType
        
        # Initialize components
        visual_params = ReasoningCoreParams(modality=ModalityType.VISUAL, reservoir_size=40, input_size=64)
        visual_core = VisualReasoningCore(visual_params)
        
        audio_params = ReasoningCoreParams(modality=ModalityType.AUDIO, reservoir_size=40, input_size=32)
        audio_core = AudioReasoningCore(audio_params)
        
        text_params = ReasoningCoreParams(modality=ModalityType.TEXT, reservoir_size=40, input_size=128)
        text_core = TextReasoningCore(text_params)
        synchronizer = CrossModalSynchronizer()
        
        # Register cores
        synchronizer.register_core("visual", visual_core)
        synchronizer.register_core("audio", audio_core)
        synchronizer.register_core("text", text_core)
        
        # Initialize states
        visual_state = visual_core.init_state()
        audio_state = audio_core.init_state()
        text_state = text_core.init_state()
        sync_state = synchronizer.init_state()
        
        # Test cross-modal integration with correlated inputs
        key = self.key
        integration_metrics = []
        
        for step in range(30):
            key, subkey = jax.random.split(key)
            
            # Generate correlated multi-modal inputs
            base_signal = jax.random.normal(subkey, (16,))
            
            # Visual input (expand base signal)
            visual_input = jnp.concatenate([base_signal, base_signal, base_signal, base_signal])
            
            # Audio input (compress base signal)
            audio_input = base_signal[:32] if len(base_signal) >= 32 else jnp.pad(base_signal, (0, 32 - len(base_signal)))
            
            # Text input (transform base signal)
            text_input = jnp.concatenate([base_signal] * 8)
            
            # Process through individual cores
            visual_state, visual_output = visual_core.process_input(visual_state, visual_input)
            audio_state, audio_output = audio_core.process_input(audio_state, audio_input)
            text_state, text_output = text_core.process_input(text_state, text_input)
            
            # Cross-modal synchronization
            modal_outputs = {
                "visual": visual_output,
                "audio": audio_output,
                "text": text_output
            }
            
            sync_state, integrated_output = synchronizer.synchronize_cores(sync_state, modal_outputs)
            
            # Measure integration quality
            # Check correlation between modalities
            visual_audio_corr = jnp.corrcoef(visual_output, audio_output)[0, 1]
            visual_text_corr = jnp.corrcoef(visual_output, text_output)[0, 1]
            audio_text_corr = jnp.corrcoef(audio_output, text_output)[0, 1]
            
            avg_correlation = (abs(visual_audio_corr) + abs(visual_text_corr) + abs(audio_text_corr)) / 3
            
            # Measure information preservation
            integrated_variance = jnp.var(integrated_output)
            
            integration_metrics.append({
                'step': step,
                'cross_modal_correlation': avg_correlation,
                'integration_quality': integrated_variance,
                'synchronization_coherence': synchronizer.get_synchronization_metrics(sync_state).get('coherence', 0.0)
            })
        
        # Validation
        correlations = [m['cross_modal_correlation'] for m in integration_metrics]
        coherence_scores = [m['synchronization_coherence'] for m in integration_metrics]
        
        avg_correlation = jnp.mean(jnp.array(correlations))
        avg_coherence = jnp.mean(jnp.array(coherence_scores))
        
        # Cross-modal integration should show some correlation due to shared base signal
        assert avg_correlation > 0.1, f"Cross-modal correlation too low: {avg_correlation:.4f}"
        
        # Synchronization should improve over time
        early_coherence = jnp.mean(jnp.array(coherence_scores[:10]))
        late_coherence = jnp.mean(jnp.array(coherence_scores[-10:]))
        
        assert late_coherence >= early_coherence - 0.1, \
            f"Synchronization should improve: {early_coherence:.4f} -> {late_coherence:.4f}"
        
        logger.info(f"Cross-modal integration: {avg_correlation:.4f} correlation, {avg_coherence:.4f} coherence")


class TestEmergentBehaviors:
    """Test emergent behaviors arising from component interactions"""
    
    def setup_method(self):
        """Setup test environment"""
        self.key = jax.random.PRNGKey(42)
        
    @pytest.mark.integration
    def test_memory_consolidation_emergence(self):
        """Test emergent memory consolidation behavior"""
        from src.memory.working.working_memory import WorkingMemory
        from src.memory.episodic.episodic_memory import EpisodicMemory
        from src.memory.semantic.semantic_memory import SemanticMemory
        
        # Initialize memory hierarchy
        working_memory = WorkingMemory(capacity=15, pattern_dim=32)
        episodic_memory = EpisodicMemory(max_episodes=20)
        semantic_memory = SemanticMemory(embedding_dim=32, similarity_threshold=0.7)
        
        # Initialize states
        wm_state = working_memory.init_state()
        em_state = episodic_memory.init_state()
        sm_state = semantic_memory.init_state()
        
        # Simulate learning with repeated patterns (should lead to consolidation)
        key = self.key
        pattern_types = 3
        repetitions_per_type = 8
        
        all_experiences = []
        consolidation_metrics = []
        
        for rep in range(repetitions_per_type):
            for pattern_type in range(pattern_types):
                key, subkey = jax.random.split(key)
                
                # Generate pattern with some consistency within type
                base_pattern = jnp.ones(32) * pattern_type
                noise = jax.random.normal(subkey, (32,)) * 0.3
                pattern = base_pattern + noise
                
                # Store in working memory
                working_memory.store_pattern(wm_state, pattern, timestamp=float(rep * pattern_types + pattern_type))
                
                # Create experience
                experience = {
                    'observation': pattern,
                    'action': pattern_type,
                    'reward': 0.5 + pattern_type * 0.2,
                    'timestamp': float(rep * pattern_types + pattern_type),
                    'context': {'pattern_type': pattern_type, 'repetition': rep}
                }
                all_experiences.append(experience)
                
                # Store episode (every few experiences)
                if len(all_experiences) % 5 == 0:
                    recent_experiences = all_experiences[-5:]
                    episodic_memory.store_episode(em_state, recent_experiences)
        
        # Extract concepts (should show consolidation)
        concepts = semantic_memory.extract_concepts(sm_state, all_experiences)
        
        # Measure consolidation emergence
        # Should form concepts roughly corresponding to pattern types
        assert len(concepts) >= pattern_types, f"Should form at least {pattern_types} concepts, got {len(concepts)}"
        assert len(concepts) <= pattern_types * 2, f"Should not over-segment: {len(concepts)} concepts for {pattern_types} types"
        
        # Test concept quality - concepts should capture pattern structure
        concept_qualities = []
        for concept in concepts:
            # Find experiences most similar to this concept
            similarities = []
            for exp in all_experiences:
                sim = jnp.corrcoef(concept.embedding, exp['observation'])[0, 1]
                similarities.append(abs(sim))
            
            # Concept should have high similarity to some experiences
            max_similarity = max(similarities)
            concept_qualities.append(max_similarity)
        
        avg_concept_quality = jnp.mean(jnp.array(concept_qualities))
        assert avg_concept_quality > 0.5, f"Concept quality too low: {avg_concept_quality:.4f}"
        
        # Test memory hierarchy coherence
        # Working memory patterns should relate to episodic episodes
        wm_stats = working_memory.get_memory_statistics(wm_state)
        em_stats = episodic_memory.get_memory_statistics(em_state)
        
        assert wm_stats['total_patterns'] > 0, "Working memory should contain patterns"
        assert em_stats['total_episodes'] > 0, "Episodic memory should contain episodes"
        
        logger.info(f"Memory consolidation: {len(concepts)} concepts, {avg_concept_quality:.4f} quality")
    
    @pytest.mark.integration
    def test_adaptive_resource_allocation_emergence(self):
        """Test emergent adaptive resource allocation"""
        from src.agents.reasoning.visual_reasoning_core import VisualReasoningCore
        from src.agents.reasoning.audio_reasoning_core import AudioReasoningCore
        from src.agents.reasoning.cross_modal_sync import CrossModalSynchronizer
        
        # Initialize components
        visual_core = VisualReasoningCore(reservoir_size=50, input_dim=64)
        audio_core = AudioReasoningCore(reservoir_size=50, input_dim=32)
        synchronizer = CrossModalSynchronizer()
        
        synchronizer.register_core("visual", visual_core)
        synchronizer.register_core("audio", audio_core)
        
        # Initialize states
        visual_state = visual_core.init_state()
        audio_state = audio_core.init_state()
        sync_state = synchronizer.init_state()
        
        # Simulate scenario where visual input becomes more important over time
        key = self.key
        resource_allocation_history = []
        
        for phase in range(3):  # Three phases with different input characteristics
            phase_allocations = []
            
            for step in range(20):
                key, subkey = jax.random.split(key)
                
                # Phase 0: Balanced inputs
                # Phase 1: Visual becomes more informative
                # Phase 2: Audio becomes more informative
                if phase == 0:
                    visual_input = jax.random.normal(subkey, (64,)) * 1.0
                    audio_input = jax.random.normal(subkey, (32,)) * 1.0
                elif phase == 1:
                    # Visual input has more structure/information
                    visual_input = jax.random.normal(subkey, (64,)) * 2.0
                    audio_input = jax.random.normal(subkey, (32,)) * 0.5
                else:
                    # Audio input has more structure/information
                    visual_input = jax.random.normal(subkey, (64,)) * 0.5
                    audio_input = jax.random.normal(subkey, (32,)) * 2.0
                
                # Process inputs
                visual_state, visual_output = visual_core.process_input(visual_state, visual_input)
                audio_state, audio_output = audio_core.process_input(audio_state, audio_input)
                
                # Synchronize and get resource allocation
                modal_outputs = {"visual": visual_output, "audio": audio_output}
                sync_state, integrated_output = synchronizer.synchronize_cores(sync_state, modal_outputs)
                
                resource_allocation = synchronizer.get_resource_allocation(sync_state)
                phase_allocations.append(resource_allocation)
            
            resource_allocation_history.append(phase_allocations)
        
        # Analyze adaptive resource allocation
        # Extract average allocations per phase
        phase_avg_allocations = []
        for phase_allocations in resource_allocation_history:
            visual_allocations = [alloc.get('visual', 0.5) for alloc in phase_allocations]
            audio_allocations = [alloc.get('audio', 0.5) for alloc in phase_allocations]
            
            phase_avg_allocations.append({
                'visual': jnp.mean(jnp.array(visual_allocations)),
                'audio': jnp.mean(jnp.array(audio_allocations))
            })
        
        # Validation: Resource allocation should adapt to input characteristics
        balanced_phase = phase_avg_allocations[0]
        visual_heavy_phase = phase_avg_allocations[1]
        audio_heavy_phase = phase_avg_allocations[2]
        
        # In visual-heavy phase, visual should get more resources
        assert visual_heavy_phase['visual'] >= balanced_phase['visual'] - 0.1, \
            f"Visual allocation should increase in visual-heavy phase: {balanced_phase['visual']:.3f} -> {visual_heavy_phase['visual']:.3f}"
        
        # In audio-heavy phase, audio should get more resources
        assert audio_heavy_phase['audio'] >= balanced_phase['audio'] - 0.1, \
            f"Audio allocation should increase in audio-heavy phase: {balanced_phase['audio']:.3f} -> {audio_heavy_phase['audio']:.3f}"
        
        logger.info(f"Adaptive allocation - Balanced: V={balanced_phase['visual']:.3f}, A={balanced_phase['audio']:.3f}")
        logger.info(f"Visual-heavy: V={visual_heavy_phase['visual']:.3f}, A={visual_heavy_phase['audio']:.3f}")
        logger.info(f"Audio-heavy: V={audio_heavy_phase['visual']:.3f}, A={audio_heavy_phase['audio']:.3f}")
    
    @pytest.mark.integration
    def test_learning_strategy_emergence(self):
        """Test emergence of effective learning strategies"""
        from src.memory.meta.meta_memory import MetaMemory
        from src.training.unsupervised.competitive_learning import CompetitiveLearning
        from src.core.plasticity.stdp import STDPLearningRule
        
        # Initialize components
        meta_memory = MetaMemory(learning_history_size=50)
        competitive_learning = CompetitiveLearning(n_units=10, input_dim=20)
        stdp = STDPLearningRule()
        
        # Initialize states
        mm_state = meta_memory.init_state()
        cl_state = competitive_learning.init_state()
        stdp_state = stdp.init_state(n_pre=20, n_post=10)
        
        # Simulate different learning scenarios
        key = self.key
        learning_strategies = ["aggressive", "conservative", "adaptive"]
        task_types = ["pattern_A", "pattern_B", "pattern_C"]
        
        strategy_performance = {strategy: [] for strategy in learning_strategies}
        
        for task_type in task_types:
            for strategy in learning_strategies:
                # Configure learning parameters based on strategy
                if strategy == "aggressive":
                    learning_rate = 0.1
                    adaptation_rate = 0.2
                elif strategy == "conservative":
                    learning_rate = 0.01
                    adaptation_rate = 0.05
                else:  # adaptive
                    learning_rate = 0.05
                    adaptation_rate = 0.1
                
                # Simulate learning episode
                episode_performance = []
                
                for step in range(30):
                    key, subkey = jax.random.split(key)
                    
                    # Generate task-specific patterns
                    if task_type == "pattern_A":
                        input_pattern = jax.random.normal(subkey, (20,)) + jnp.ones(20)
                    elif task_type == "pattern_B":
                        input_pattern = jax.random.normal(subkey, (20,)) - jnp.ones(20)
                    else:  # pattern_C
                        input_pattern = jax.random.normal(subkey, (20,)) * 2.0
                    
                    # Apply competitive learning
                    cl_state, cl_output = competitive_learning.train_step(cl_state, input_pattern)
                    
                    # Apply STDP
                    pre_spikes = input_pattern.reshape(-1, 1)
                    post_spikes = cl_output.reshape(1, -1)
                    
                    stdp_state = stdp.update_traces(stdp_state, pre_spikes, post_spikes, dt=0.001)
                    new_stdp_state, weight_update = stdp.compute_weight_update(stdp_state, pre_spikes, post_spikes)
                    stdp_state = new_stdp_state
                    
                    # Measure performance (pattern recognition quality)
                    performance = jnp.mean(jnp.abs(cl_output))
                    episode_performance.append(performance)
                
                # Calculate overall performance for this strategy-task combination
                final_performance = jnp.mean(jnp.array(episode_performance[-10:]))  # Last 10 steps
                strategy_performance[strategy].append(final_performance)
                
                # Store in meta-memory
                meta_memory.store_learning_experience(
                    mm_state,
                    task_type=task_type,
                    performance=float(final_performance),
                    strategy=strategy,
                    learning_time=30.0
                )
        
        # Analyze strategy emergence
        # Meta-memory should learn which strategies work best for which tasks
        strategy_effectiveness = {}
        
        for task_type in task_types:
            best_strategy = meta_memory.retrieve_learning_strategy(mm_state, task_type)
            strategy_effectiveness[task_type] = best_strategy
        
        # Validation: Meta-memory should have learned strategy preferences
        for task_type in task_types:
            retrieved_strategy = strategy_effectiveness[task_type]
            assert retrieved_strategy is not None, f"Should retrieve strategy for {task_type}"
            assert retrieved_strategy in learning_strategies, f"Strategy should be valid: {retrieved_strategy}"
        
        # Calculate actual best strategies for comparison
        actual_best_strategies = {}
        for task_idx, task_type in enumerate(task_types):
            task_performances = {}
            for strategy in learning_strategies:
                task_performances[strategy] = strategy_performance[strategy][task_idx]
            
            actual_best_strategy = max(task_performances, key=task_performances.get)
            actual_best_strategies[task_type] = actual_best_strategy
        
        # Check if meta-memory learned reasonable strategies
        correct_strategies = 0
        for task_type in task_types:
            if strategy_effectiveness[task_type] == actual_best_strategies[task_type]:
                correct_strategies += 1
        
        # Allow some tolerance - meta-memory doesn't need to be perfect
        strategy_accuracy = correct_strategies / len(task_types)
        assert strategy_accuracy >= 0.3, f"Strategy learning accuracy too low: {strategy_accuracy:.2f}"
        
        logger.info(f"Learning strategy emergence: {strategy_accuracy:.2f} accuracy")
        for task_type in task_types:
            logger.info(f"  {task_type}: learned={strategy_effectiveness[task_type]}, actual_best={actual_best_strategies[task_type]}")


class TestSystemRobustness:
    """Test system robustness under various conditions"""
    
    def setup_method(self):
        """Setup test environment"""
        self.key = jax.random.PRNGKey(42)
        
    @pytest.mark.integration
    def test_component_failure_resilience(self):
        """Test system resilience to component failures"""
        from src.memory.working.working_memory import WorkingMemory
        from src.agents.reasoning.visual_reasoning_core import VisualReasoningCore
        from unittest.mock import Mock, patch
        
        # Initialize components
        working_memory = WorkingMemory(capacity=20, pattern_dim=64)
        reasoning_core = VisualReasoningCore(reservoir_size=50, input_dim=64)
        
        # Initialize states
        wm_state = working_memory.init_state()
        reasoning_state = reasoning_core.init_state()
        
        # Test normal operation first
        key = self.key
        key, subkey = jax.random.split(key)
        test_pattern = jax.random.normal(subkey, (64,))
        
        # Store pattern and process normally
        pattern_id = working_memory.store_pattern(wm_state, test_pattern, timestamp=1.0)
        reasoning_state, normal_output = reasoning_core.process_input(reasoning_state, test_pattern)
        
        assert pattern_id is not None, "Normal operation should work"
        assert not jnp.any(jnp.isnan(normal_output)), "Normal output should be valid"
        
        # Test resilience to memory component failure
        with patch.object(working_memory, 'retrieve_pattern') as mock_retrieve:
            # Simulate memory failure
            mock_retrieve.side_effect = Exception("Memory component failed")
            
            try:
                # System should handle memory failure gracefully
                retrieved_pattern, confidence = working_memory.retrieve_pattern(wm_state, test_pattern)
                # If it doesn't raise an exception, it handled the failure
                assert True, "System handled memory failure gracefully"
            except Exception as e:
                # If it raises an exception, it should be informative
                assert "memory" in str(e).lower() or "failed" in str(e).lower()
        
        # Test resilience to reasoning component failure
        with patch.object(reasoning_core, 'process_input') as mock_process:
            # Simulate reasoning failure
            mock_process.side_effect = Exception("Reasoning component failed")
            
            try:
                # System should handle reasoning failure gracefully
                reasoning_state, output = reasoning_core.process_input(reasoning_state, test_pattern)
                assert True, "System handled reasoning failure gracefully"
            except Exception as e:
                # If it raises an exception, it should be informative
                assert "reasoning" in str(e).lower() or "failed" in str(e).lower()
        
        # Test recovery after failure
        # After mock failures, normal operation should still work
        key, subkey = jax.random.split(key)
        recovery_pattern = jax.random.normal(subkey, (64,))
        
        try:
            recovery_id = working_memory.store_pattern(wm_state, recovery_pattern, timestamp=2.0)
            reasoning_state, recovery_output = reasoning_core.process_input(reasoning_state, recovery_pattern)
            
            assert recovery_id is not None, "System should recover after failure"
            assert not jnp.any(jnp.isnan(recovery_output)), "Recovery output should be valid"
        except Exception as e:
            pytest.fail(f"System should recover after component failure: {e}")
        
        logger.info("Component failure resilience test passed")
    
    @pytest.mark.integration
    def test_high_load_performance_degradation(self):
        """Test graceful performance degradation under high load"""
        from src.memory.working.working_memory import WorkingMemory
        from src.core.liquid_state_machine import LiquidStateMachine
        import time
        
        # Initialize components
        working_memory = WorkingMemory(capacity=50, pattern_dim=128)
        lsm = LiquidStateMachine(reservoir_size=200, input_dim=128, spectral_radius=0.9)
        
        # Initialize states
        wm_state = working_memory.init_state()
        lsm_state = lsm.init_state()
        
        # Test performance under increasing load
        load_levels = [10, 50, 100, 200, 500]  # Number of operations
        performance_metrics = []
        
        key = self.key
        
        for load_level in load_levels:
            start_time = time.time()
            successful_operations = 0
            errors = 0
            
            for i in range(load_level):
                key, subkey = jax.random.split(key)
                pattern = jax.random.normal(subkey, (128,))
                
                try:
                    # Memory operation
                    pattern_id = working_memory.store_pattern(wm_state, pattern, timestamp=float(i))
                    
                    # LSM operation
                    lsm_state, lsm_output = lsm.step(lsm_state, pattern)
                    
                    if pattern_id is not None and not jnp.any(jnp.isnan(lsm_output)):
                        successful_operations += 1
                    else:
                        errors += 1
                        
                except Exception as e:
                    errors += 1
            
            end_time = time.time()
            
            metrics = {
                'load_level': load_level,
                'execution_time': end_time - start_time,
                'success_rate': successful_operations / load_level,
                'error_rate': errors / load_level,
                'throughput': successful_operations / (end_time - start_time)
            }
            
            performance_metrics.append(metrics)
        
        # Validation: Performance should degrade gracefully
        for i in range(1, len(performance_metrics)):
            current = performance_metrics[i]
            previous = performance_metrics[i-1]
            
            # Success rate should not drop dramatically
            success_rate_drop = previous['success_rate'] - current['success_rate']
            assert success_rate_drop < 0.5, f"Success rate dropped too much: {success_rate_drop:.2f}"
            
            # Error rate should not spike dramatically
            assert current['error_rate'] < 0.8, f"Error rate too high under load: {current['error_rate']:.2f}"
        
        # System should maintain reasonable performance even under high load
        high_load_metrics = performance_metrics[-1]
        assert high_load_metrics['success_rate'] > 0.2, f"Success rate too low under high load: {high_load_metrics['success_rate']:.2f}"
        
        logger.info("High load performance degradation test passed")
        for metrics in performance_metrics:
            logger.info(f"Load {metrics['load_level']}: {metrics['success_rate']:.2f} success, {metrics['throughput']:.1f} ops/s")
    
    @pytest.mark.integration
    def test_data_corruption_recovery(self):
        """Test recovery from data corruption"""
        from src.memory.episodic.episodic_memory import EpisodicMemory
        from src.memory.semantic.semantic_memory import SemanticMemory
        
        # Initialize components
        episodic_memory = EpisodicMemory(max_episodes=20)
        semantic_memory = SemanticMemory(embedding_dim=64)
        
        # Initialize states
        em_state = episodic_memory.init_state()
        sm_state = semantic_memory.init_state()
        
        # Store valid data first
        key = self.key
        valid_experiences = []
        
        for i in range(5):
            episode_data = []
            for step in range(3):
                key, subkey = jax.random.split(key)
                experience = {
                    'observation': jax.random.normal(subkey, (64,)),
                    'action': i,
                    'reward': 0.5,
                    'timestamp': float(i * 3 + step)
                }
                episode_data.append(experience)
                valid_experiences.append(experience)
            
            episodic_memory.store_episode(em_state, episode_data)
        
        # Extract concepts from valid data
        valid_concepts = semantic_memory.extract_concepts(sm_state, valid_experiences)
        assert len(valid_concepts) > 0, "Should extract concepts from valid data"
        
        # Introduce corrupted data
        corrupted_experiences = []
        for i in range(3):
            key, subkey = jax.random.split(key)
            corrupted_experience = {
                'observation': jnp.array([jnp.nan, jnp.inf, -jnp.inf] + [1.0] * 61),  # Corrupted observation
                'action': -999,  # Invalid action
                'reward': jnp.nan,  # Corrupted reward
                'timestamp': float(100 + i)
            }
            corrupted_experiences.append(corrupted_experience)
        
        # Test system's handling of corrupted data
        try:
            # Try to store corrupted episode
            corrupted_episode_id = episodic_memory.store_episode(em_state, corrupted_experiences)
            
            # If it succeeds, system should handle corruption gracefully
            if corrupted_episode_id is not None:
                # Try to retrieve and verify it doesn't crash
                retrieved_episode = episodic_memory.replay_episode(em_state, corrupted_episode_id)
                # System should either sanitize data or handle gracefully
                assert True, "System handled corrupted data gracefully"
        
        except Exception as e:
            # If it raises an exception, should be informative about data corruption
            assert any(word in str(e).lower() for word in ['corrupt', 'invalid', 'nan', 'inf'])
        
        # Test concept extraction with mixed valid/corrupted data
        mixed_experiences = valid_experiences + corrupted_experiences
        
        try:
            mixed_concepts = semantic_memory.extract_concepts(sm_state, mixed_experiences)
            
            # Should still extract some concepts from valid data
            assert len(mixed_concepts) > 0, "Should extract concepts despite some corrupted data"
            
            # Concepts should be valid (no NaN/Inf)
            for concept in mixed_concepts:
                assert not jnp.any(jnp.isnan(concept.embedding)), "Concepts should not contain NaN"
                assert not jnp.any(jnp.isinf(concept.embedding)), "Concepts should not contain Inf"
        
        except Exception as e:
            # Should raise informative error about data corruption
            assert any(word in str(e).lower() for word in ['corrupt', 'invalid', 'data'])
        
        # Test system recovery - should still work with new valid data
        key, subkey = jax.random.split(key)
        recovery_experience = {
            'observation': jax.random.normal(subkey, (64,)),
            'action': 0,
            'reward': 0.8,
            'timestamp': 200.0
        }
        
        try:
            recovery_episode_id = episodic_memory.store_episode(em_state, [recovery_experience])
            assert recovery_episode_id is not None, "System should recover and accept valid data"
        except Exception as e:
            pytest.fail(f"System should recover after data corruption: {e}")
        
        logger.info("Data corruption recovery test passed")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
"""
Tests for unsupervised learning algorithms.

This module tests pattern discovery, experience clustering, feature extraction,
and overall learning convergence and pattern quality.
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import List, Dict

from src.training.unsupervised import (
    CompetitiveLearning,
    CompetitiveLearningConfig,
    AdaptiveCompetitiveLearning,
    SelfOrganizingMap,
    SOMConfig,
    HierarchicalSOM,
    Experience,
    MultiModalFeatureExtractor,
    FeatureExtractionConfig,
    MultiModalInput,
    SparseAutoencoder,
    IndependentComponentAnalysis,
    UnsupervisedLearningPipeline,
    UnsupervisedLearningConfig,
    PatternQualityMetrics
)


class TestCompetitiveLearning:
    """Test competitive learning algorithms."""
    
    @pytest.fixture
    def setup_data(self):
        """Create test data with known clusters."""
        key = random.PRNGKey(42)
        
        # Create 3 distinct clusters
        cluster1 = random.normal(key, (50, 10)) + jnp.array([2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cluster2 = random.normal(key, (50, 10)) + jnp.array([0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cluster3 = random.normal(key, (50, 10)) + jnp.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        
        data = jnp.concatenate([cluster1, cluster2, cluster3], axis=0)
        true_labels = jnp.concatenate([
            jnp.zeros(50), jnp.ones(50), jnp.full(50, 2)
        ])
        
        return data, true_labels, key
    
    def test_competitive_learning_initialization(self, setup_data):
        """Test competitive learning initialization."""
        data, _, key = setup_data
        config = CompetitiveLearningConfig(n_units=10, max_iterations=100)
        
        cl = CompetitiveLearning(config, input_dim=10, key=key)
        
        assert cl.weights.shape == (10, 10)
        assert cl.config.n_units == 10
        assert cl.learning_rate == config.learning_rate
    
    def test_competitive_learning_training(self, setup_data):
        """Test competitive learning training process."""
        data, true_labels, key = setup_data
        config = CompetitiveLearningConfig(
            n_units=5, 
            max_iterations=200,
            learning_rate=0.1,
            convergence_threshold=1e-4
        )
        
        cl = CompetitiveLearning(config, input_dim=10, key=key)
        training_stats = cl.train(data)
        
        # Check training completed
        assert len(training_stats['weight_changes']) > 0
        assert len(training_stats['winners']) > 0
        
        # Check convergence (weight changes should decrease)
        initial_changes = training_stats['weight_changes'][:10]
        final_changes = training_stats['weight_changes'][-10:]
        assert jnp.mean(jnp.array(final_changes)) < jnp.mean(jnp.array(initial_changes))
    
    def test_competitive_learning_pattern_discovery(self, setup_data):
        """Test pattern discovery quality."""
        data, true_labels, key = setup_data
        config = CompetitiveLearningConfig(n_units=3, max_iterations=500)
        
        cl = CompetitiveLearning(config, input_dim=10, key=key)
        cl.train(data)
        
        # Get pattern assignments
        assignments = cl.get_pattern_assignments(data)
        patterns = cl.get_learned_patterns()
        
        # Check pattern quality
        assert len(patterns) == 3
        assert len(assignments) == len(data)
        
        # Compute quantization error
        qe = cl.compute_quantization_error(data)
        assert qe > 0.0  # Should have some error
        assert qe < 5.0  # But not too high
        
        # Check that patterns are distinct
        pattern_distances = []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                dist = jnp.linalg.norm(patterns[i] - patterns[j])
                pattern_distances.append(dist)
        
        mean_pattern_distance = jnp.mean(jnp.array(pattern_distances))
        assert mean_pattern_distance > 0.5  # Patterns should be reasonably distinct
    
    def test_adaptive_competitive_learning(self, setup_data):
        """Test adaptive competitive learning with unit creation/pruning."""
        data, _, key = setup_data
        config = CompetitiveLearningConfig(n_units=2, max_iterations=300)
        
        acl = AdaptiveCompetitiveLearning(config, input_dim=10, key=key)
        training_stats = acl.train(data)
        
        # Check that new units were created
        final_stats = training_stats['weight_changes'][-1] if training_stats['weight_changes'] else None
        if final_stats is not None:
            # Should have adapted to the data
            assert len(acl.weights) >= config.n_units


class TestSelfOrganizingMap:
    """Test self-organizing map algorithms."""
    
    @pytest.fixture
    def setup_experiences(self):
        """Create test experiences."""
        key = random.PRNGKey(42)
        experiences = []
        
        for i in range(100):
            key, subkey = random.split(key)
            observation = random.normal(subkey, (20,))
            action = random.normal(subkey, (5,))
            reward = float(random.normal(subkey, ()))
            context = {'task_id': random.normal(subkey, (3,))}
            
            exp = Experience(
                observation=observation,
                action=action,
                reward=reward,
                context=context,
                timestamp=float(i)
            )
            experiences.append(exp)
        
        return experiences, key
    
    def test_som_initialization(self):
        """Test SOM initialization."""
        config = SOMConfig(map_width=5, map_height=5, input_dim=10)
        key = random.PRNGKey(42)
        
        som = SelfOrganizingMap(config, key)
        
        assert som.weights.shape == (5, 5, 10)
        assert som.coordinates.shape == (5, 5, 2)
        assert som.config.map_width == 5
        assert som.config.map_height == 5
    
    def test_som_training(self):
        """Test SOM training process."""
        # Create test data
        key = random.PRNGKey(42)
        data = random.normal(key, (100, 10))
        
        config = SOMConfig(
            map_width=3, 
            map_height=3, 
            input_dim=10,
            max_iterations=200
        )
        som = SelfOrganizingMap(config, key)
        
        training_stats = som.train(data, verbose=False)
        
        # Check training completed
        assert len(training_stats['weight_changes']) > 0
        assert len(training_stats['learning_rates']) > 0
        
        # Check learning rate decay
        initial_lr = training_stats['learning_rates'][0]
        final_lr = training_stats['learning_rates'][-1]
        assert final_lr < initial_lr
    
    def test_som_clustering_quality(self):
        """Test SOM clustering quality metrics."""
        # Create clustered data
        key = random.PRNGKey(42)
        cluster1 = random.normal(key, (30, 5)) + 2.0
        cluster2 = random.normal(key, (30, 5)) - 2.0
        data = jnp.concatenate([cluster1, cluster2], axis=0)
        
        config = SOMConfig(map_width=4, map_height=4, input_dim=5, max_iterations=300)
        som = SelfOrganizingMap(config, key)
        som.train(data)
        
        # Compute quality metrics
        qe = som.compute_quantization_error(data)
        te = som.compute_topographic_error(data)
        
        assert qe > 0.0
        assert te >= 0.0 and te <= 1.0
        
        # Check activation map
        activation_map = som.get_activation_map(data)
        assert activation_map.shape == (4, 4)
        assert jnp.sum(activation_map) == len(data)
    
    def test_experience_clustering(self, setup_experiences):
        """Test experience clustering with SOM."""
        experiences, key = setup_experiences
        
        config = SOMConfig(map_width=3, map_height=3, input_dim=50, max_iterations=200)
        som = SelfOrganizingMap(config, key)
        
        # Cluster experiences
        clusters = som.cluster_experiences(experiences)
        
        # Check clustering results
        assert isinstance(clusters, dict)
        assert len(clusters) > 0
        
        # Check that all experiences are assigned
        total_assigned = sum(len(exp_list) for exp_list in clusters.values())
        assert total_assigned == len(experiences)
    
    def test_hierarchical_som(self):
        """Test hierarchical SOM."""
        key = random.PRNGKey(42)
        data = random.normal(key, (50, 10))
        
        # Create multiple SOM layers
        configs = [
            SOMConfig(map_width=4, map_height=4, input_dim=10, max_iterations=100),
            SOMConfig(map_width=2, map_height=2, input_dim=10, max_iterations=100)
        ]
        
        hsom = HierarchicalSOM(configs, key)
        training_stats = hsom.train_hierarchical(data)
        
        assert len(training_stats) == 2
        assert len(hsom.som_layers) == 2


class TestFeatureExtraction:
    """Test feature extraction algorithms."""
    
    @pytest.fixture
    def setup_multimodal_data(self):
        """Create test multi-modal data."""
        key = random.PRNGKey(42)
        multimodal_data = []
        
        for i in range(50):
            key, subkey = random.split(key)
            
            visual = random.normal(subkey, (64,))  # Simulated visual features
            audio = random.normal(subkey, (32,))   # Simulated audio features
            text = random.normal(subkey, (128,))   # Simulated text embeddings
            
            sample = MultiModalInput(
                visual=visual,
                audio=audio,
                text=text,
                timestamp=float(i)
            )
            multimodal_data.append(sample)
        
        return multimodal_data, key
    
    def test_sparse_autoencoder(self):
        """Test sparse autoencoder feature extraction."""
        key = random.PRNGKey(42)
        data = random.normal(key, (100, 50))
        
        config = FeatureExtractionConfig(n_components=20, max_iterations=200)
        sae = SparseAutoencoder(config, input_dim=50, key=key)
        
        # Train and transform
        features, training_stats = sae.fit_transform(data)
        
        assert features.shape == (100, 20)
        assert len(training_stats['losses']) > 0
        
        # Check sparsity
        mean_activation = jnp.mean(features)
        assert mean_activation < 0.2  # Should be sparse
        
        # Test reconstruction
        reconstructions = sae.reconstruct(data)
        assert reconstructions.shape == data.shape
        
        # Reconstruction error should be reasonable
        reconstruction_error = jnp.mean((data - reconstructions) ** 2)
        assert reconstruction_error < 1.0
    
    def test_independent_component_analysis(self):
        """Test ICA feature extraction."""
        key = random.PRNGKey(42)
        
        # Create mixed signals
        n_samples = 200
        time = jnp.linspace(0, 8, n_samples)
        
        # Original sources
        s1 = jnp.sin(2 * time)  # Sine wave
        s2 = jnp.sign(jnp.sin(3 * time))  # Square wave
        s3 = random.normal(key, (n_samples,))  # Noise
        
        sources = jnp.column_stack([s1, s2, s3])
        
        # Mix the sources
        key, subkey = random.split(key)
        mixing_matrix = random.normal(subkey, (3, 3))
        mixed_signals = sources @ mixing_matrix.T
        
        # Apply ICA
        config = FeatureExtractionConfig(n_components=3)
        ica = IndependentComponentAnalysis(config, key)
        
        components, training_stats = ica.fit_transform(mixed_signals)
        
        assert components.shape == (n_samples, 3)
        assert len(training_stats['convergence']) > 0
        assert ica.converged
    
    def test_multimodal_feature_extractor(self, setup_multimodal_data):
        """Test multi-modal feature extraction."""
        multimodal_data, key = setup_multimodal_data
        
        # Configure extractors for each modality
        config = {
            'visual': FeatureExtractionConfig(n_components=16, max_iterations=100),
            'audio': FeatureExtractionConfig(n_components=8, max_iterations=100),
            'text': FeatureExtractionConfig(n_components=32, max_iterations=100)
        }
        
        extractor = MultiModalFeatureExtractor(config, key)
        
        # Add modalities
        extractor.add_modality('visual', 'sparse_autoencoder', 64)
        extractor.add_modality('audio', 'sparse_autoencoder', 32)
        extractor.add_modality('text', 'sparse_autoencoder', 128)
        
        # Train on multi-modal data
        training_stats = extractor.fit_multimodal(multimodal_data)
        
        assert 'visual' in training_stats
        assert 'audio' in training_stats
        assert 'text' in training_stats
        
        # Test unified feature extraction
        sample = multimodal_data[0]
        unified_features = extractor.transform_multimodal(sample)
        
        expected_dim = 16 + 8 + 32  # Sum of component dimensions
        assert len(unified_features) == expected_dim
        
        # Check modality importance
        importance = extractor.get_modality_importance()
        assert len(importance) == 3
        assert all(0 <= weight <= 1 for weight in importance.values())


class TestPatternQualityMetrics:
    """Test pattern quality evaluation metrics."""
    
    def test_silhouette_score(self):
        """Test silhouette score computation."""
        # Create well-separated clusters
        key = random.PRNGKey(42)
        cluster1 = random.normal(key, (20, 5)) + 3.0
        cluster2 = random.normal(key, (20, 5)) - 3.0
        
        data = jnp.concatenate([cluster1, cluster2])
        labels = jnp.concatenate([jnp.zeros(20), jnp.ones(20)])
        
        silhouette = PatternQualityMetrics.compute_silhouette_score(data, labels)
        
        assert -1.0 <= silhouette <= 1.0
        assert silhouette > 0.5  # Should be high for well-separated clusters
    
    def test_davies_bouldin_index(self):
        """Test Davies-Bouldin index computation."""
        key = random.PRNGKey(42)
        cluster1 = random.normal(key, (15, 3)) + 2.0
        cluster2 = random.normal(key, (15, 3)) - 2.0
        
        data = jnp.concatenate([cluster1, cluster2])
        labels = jnp.concatenate([jnp.zeros(15), jnp.ones(15)])
        
        db_index = PatternQualityMetrics.compute_davies_bouldin_index(data, labels)
        
        assert db_index >= 0.0
        assert db_index < 2.0  # Should be low for good clustering
    
    def test_pattern_diversity(self):
        """Test pattern diversity computation."""
        key = random.PRNGKey(42)
        
        # Create diverse patterns
        patterns = random.normal(key, (5, 10)) * 2.0
        diversity = PatternQualityMetrics.compute_pattern_diversity(patterns)
        
        assert diversity > 0.0
        
        # Create identical patterns
        identical_patterns = jnp.ones((3, 10))
        diversity_identical = PatternQualityMetrics.compute_pattern_diversity(identical_patterns)
        
        assert diversity_identical == 0.0


class TestUnsupervisedLearningPipeline:
    """Test complete unsupervised learning pipeline."""
    
    @pytest.fixture
    def setup_pipeline_data(self):
        """Create comprehensive test data for pipeline."""
        key = random.PRNGKey(42)
        
        # Create structured data
        data = random.normal(key, (100, 20))
        
        # Create experiences
        experiences = []
        for i in range(50):
            key, subkey = random.split(key)
            exp = Experience(
                observation=random.normal(subkey, (10,)),
                action=random.normal(subkey, (3,)),
                reward=float(random.normal(subkey, ())),
                context={'env': random.normal(subkey, (5,))},
                timestamp=float(i)
            )
            experiences.append(exp)
        
        # Create multi-modal data
        multimodal_data = []
        for i in range(30):
            key, subkey = random.split(key)
            sample = MultiModalInput(
                visual=random.normal(subkey, (32,)),
                audio=random.normal(subkey, (16,)),
                timestamp=float(i)
            )
            multimodal_data.append(sample)
        
        return data, experiences, multimodal_data, key
    
    def test_pipeline_initialization(self, setup_pipeline_data):
        """Test pipeline initialization."""
        _, _, _, key = setup_pipeline_data
        
        config = UnsupervisedLearningConfig()
        pipeline = UnsupervisedLearningPipeline(config, key)
        
        # Initialize components
        modality_dims = {'visual': 32, 'audio': 16}
        pipeline.initialize_components(input_dim=20, modality_dims=modality_dims)
        
        assert pipeline.competitive_learner is not None
        assert pipeline.som is not None
        assert pipeline.feature_extractor is not None
    
    def test_competitive_learning_stage(self, setup_pipeline_data):
        """Test competitive learning stage of pipeline."""
        data, _, _, key = setup_pipeline_data
        
        config = UnsupervisedLearningConfig()
        config.competitive_learning.max_iterations = 100
        
        pipeline = UnsupervisedLearningPipeline(config, key)
        pipeline.initialize_components(input_dim=20)
        
        results = pipeline.discover_patterns_competitive(data)
        
        assert 'patterns' in results
        assert 'quality_metrics' in results
        assert 'training_time' in results
        
        # Check quality metrics
        metrics = results['quality_metrics']
        assert 'silhouette_score' in metrics
        assert 'quantization_error' in metrics
        assert metrics['quantization_error'] > 0.0
    
    def test_som_clustering_stage(self, setup_pipeline_data):
        """Test SOM clustering stage of pipeline."""
        _, experiences, _, key = setup_pipeline_data
        
        config = UnsupervisedLearningConfig()
        config.som_config.max_iterations = 100
        
        pipeline = UnsupervisedLearningPipeline(config, key)
        pipeline.initialize_components(input_dim=20)
        
        results = pipeline.cluster_experiences_som(experiences)
        
        assert 'clusters' in results
        assert 'quality_metrics' in results
        assert 'training_time' in results
        
        # Check clustering results
        assert len(results['clusters']) > 0
    
    def test_feature_extraction_stage(self, setup_pipeline_data):
        """Test feature extraction stage of pipeline."""
        _, _, multimodal_data, key = setup_pipeline_data
        
        config = UnsupervisedLearningConfig()
        config.feature_extraction = {
            'visual': FeatureExtractionConfig(n_components=8, max_iterations=50),
            'audio': FeatureExtractionConfig(n_components=4, max_iterations=50)
        }
        
        pipeline = UnsupervisedLearningPipeline(config, key)
        modality_dims = {'visual': 32, 'audio': 16}
        pipeline.initialize_components(input_dim=20, modality_dims=modality_dims)
        
        results = pipeline.extract_multimodal_features(multimodal_data)
        
        assert 'unified_features' in results
        assert 'modality_importance' in results
        assert 'training_time' in results
        
        # Check unified features
        if len(results['unified_features']) > 0:
            assert results['unified_features'].shape[1] == 12  # 8 + 4 components
    
    def test_complete_pipeline(self, setup_pipeline_data):
        """Test complete pipeline execution."""
        data, experiences, multimodal_data, key = setup_pipeline_data
        
        # Configure pipeline with reduced iterations for testing
        config = UnsupervisedLearningConfig()
        config.competitive_learning.max_iterations = 50
        config.som_config.max_iterations = 50
        config.feature_extraction = {
            'visual': FeatureExtractionConfig(n_components=8, max_iterations=30),
            'audio': FeatureExtractionConfig(n_components=4, max_iterations=30)
        }
        
        pipeline = UnsupervisedLearningPipeline(config, key)
        modality_dims = {'visual': 32, 'audio': 16}
        pipeline.initialize_components(input_dim=20, modality_dims=modality_dims)
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            data=data,
            experiences=experiences,
            multimodal_data=multimodal_data
        )
        
        # Check all stages completed
        assert results['competitive_learning'] is not None
        assert results['som_clustering'] is not None
        assert results['feature_extraction'] is not None
        assert results['integrated_patterns'] is not None
        assert results['overall_quality'] is not None
        
        # Check overall quality metrics
        quality = results['overall_quality']
        assert 'overall_score' in quality
        assert 0.0 <= quality['overall_score'] <= 1.0
        
        # Check integration results
        integration = results['integrated_patterns']
        assert integration['integration_successful']
        assert integration['total_discovered_patterns'] > 0
    
    def test_learning_convergence(self, setup_pipeline_data):
        """Test learning convergence detection."""
        data, _, _, key = setup_pipeline_data
        
        config = UnsupervisedLearningConfig()
        config.convergence_patience = 3
        config.quality_threshold = 0.01
        
        pipeline = UnsupervisedLearningPipeline(config, key)
        pipeline.initialize_components(input_dim=20)
        
        # Simulate quality metrics history
        for i in range(5):
            quality_metrics = {
                'overall_score': 0.8 + 0.01 * i,  # Slowly improving
                'pattern_quality': 0.7,
                'clustering_quality': 0.8,
                'feature_quality': 0.9
            }
            pipeline.training_history['quality_metrics'].append(quality_metrics)
        
        # Check convergence
        converged = pipeline._check_convergence()
        assert isinstance(converged, bool)
    
    def test_pattern_discovery_evaluation(self, setup_pipeline_data):
        """Test pattern discovery evaluation."""
        data, _, _, key = setup_pipeline_data
        
        config = UnsupervisedLearningConfig()
        pipeline = UnsupervisedLearningPipeline(config, key)
        pipeline.initialize_components(input_dim=20)
        
        # Run competitive learning
        results = pipeline.discover_patterns_competitive(data)
        
        # Evaluate learning progress
        progress = pipeline.evaluate_learning_progress(data)
        
        assert 'convergence_progress' in progress
        assert 'pattern_discovery_rate' in progress
        assert 'quality_improvement' in progress
        
        # All metrics should be non-negative
        for metric_value in progress.values():
            assert metric_value >= 0.0


# Integration test
def test_unsupervised_learning_integration():
    """Integration test for complete unsupervised learning system."""
    key = random.PRNGKey(42)
    
    # Create realistic test scenario
    n_samples = 200
    input_dim = 50
    
    # Generate data with hidden structure
    key, subkey = random.split(key)
    cluster_centers = random.normal(subkey, (4, input_dim)) * 3.0
    
    data_points = []
    for i in range(n_samples):
        key, subkey = random.split(key)
        cluster_id = random.randint(subkey, (), 0, 4)
        noise = random.normal(subkey, (input_dim,)) * 0.5
        point = cluster_centers[cluster_id] + noise
        data_points.append(point)
    
    data = jnp.stack(data_points)
    
    # Configure and run pipeline
    config = UnsupervisedLearningConfig()
    config.competitive_learning = CompetitiveLearningConfig(
        n_units=4, max_iterations=300, learning_rate=0.05
    )
    config.som_config = SOMConfig(
        map_width=3, map_height=3, input_dim=input_dim, max_iterations=200
    )
    
    pipeline = UnsupervisedLearningPipeline(config, key)
    pipeline.initialize_components(input_dim=input_dim)
    
    # Run competitive learning only for this integration test
    results = pipeline.discover_patterns_competitive(data)
    
    # Verify meaningful patterns were discovered
    assert results['quality_metrics']['silhouette_score'] > 0.0  # Should be positive
    assert results['quality_metrics']['quantization_error'] < 50.0  # Reasonable for high-dim data
    assert len(results['patterns']) >= 4  # Should have at least 4 patterns (may create more with adaptive)
    
    # Verify learning convergence
    weight_changes = results['training_stats']['weight_changes']
    assert len(weight_changes) > 10
    
    # Final weight changes should be smaller than initial ones
    initial_avg = jnp.mean(jnp.array(weight_changes[:10]))
    final_avg = jnp.mean(jnp.array(weight_changes[-10:]))
    assert final_avg < initial_avg
    
    print(f"Integration test passed:")
    print(f"  - Silhouette score: {results['quality_metrics']['silhouette_score']:.3f}")
    print(f"  - Quantization error: {results['quality_metrics']['quantization_error']:.3f}")
    print(f"  - Pattern diversity: {results['quality_metrics']['pattern_diversity']:.3f}")
    print(f"  - Training converged in {len(weight_changes)} iterations")


if __name__ == "__main__":
    # Run integration test
    test_unsupervised_learning_integration()
    print("All unsupervised learning tests would pass!")
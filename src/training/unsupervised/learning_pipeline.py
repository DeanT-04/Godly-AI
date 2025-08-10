"""
Autonomous unsupervised learning pipeline.

This module integrates competitive learning, self-organizing maps, and
feature extraction into a unified pipeline for autonomous pattern discovery.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass, field
import time

from .competitive_learning import CompetitiveLearning, CompetitiveLearningConfig, AdaptiveCompetitiveLearning
from .self_organizing_map import SelfOrganizingMap, SOMConfig, HierarchicalSOM, Experience
from .feature_extraction import (
    MultiModalFeatureExtractor, 
    FeatureExtractionConfig, 
    MultiModalInput,
    SparseAutoencoder,
    IndependentComponentAnalysis
)


@dataclass
class UnsupervisedLearningConfig:
    """Configuration for the complete unsupervised learning pipeline."""
    # Competitive learning settings
    competitive_learning: CompetitiveLearningConfig = field(default_factory=CompetitiveLearningConfig)
    use_adaptive_competitive: bool = True
    
    # SOM settings
    som_config: SOMConfig = field(default_factory=SOMConfig)
    use_hierarchical_som: bool = True
    som_layers: List[SOMConfig] = field(default_factory=list)
    
    # Feature extraction settings
    feature_extraction: Dict[str, FeatureExtractionConfig] = field(default_factory=dict)
    
    # Pipeline settings
    integration_mode: str = "sequential"  # "sequential" or "parallel"
    convergence_patience: int = 10
    quality_threshold: float = 0.1
    
    # Evaluation settings
    evaluation_frequency: int = 100
    save_intermediate_results: bool = True


class PatternQualityMetrics:
    """Metrics for evaluating pattern discovery quality."""
    
    @staticmethod
    def compute_silhouette_score(data: jnp.ndarray, labels: jnp.ndarray) -> float:
        """Compute silhouette score for clustering quality."""
        n_samples = len(data)
        if len(jnp.unique(labels)) < 2:
            return 0.0
        
        silhouette_scores = []
        
        for i in range(n_samples):
            # Same cluster distances
            same_cluster = data[labels == labels[i]]
            if len(same_cluster) > 1:
                a_i = jnp.mean(jnp.linalg.norm(same_cluster - data[i], axis=1))
            else:
                a_i = 0.0
            
            # Different cluster distances
            b_i = float('inf')
            for cluster_id in jnp.unique(labels):
                if cluster_id != labels[i]:
                    other_cluster = data[labels == cluster_id]
                    if len(other_cluster) > 0:
                        dist = jnp.mean(jnp.linalg.norm(other_cluster - data[i], axis=1))
                        b_i = min(b_i, dist)
            
            # Silhouette score for sample i
            if max(a_i, b_i) > 0:
                s_i = (b_i - a_i) / max(a_i, b_i)
            else:
                s_i = 0.0
            
            silhouette_scores.append(s_i)
        
        return float(jnp.mean(jnp.array(silhouette_scores)))
    
    @staticmethod
    def compute_davies_bouldin_index(data: jnp.ndarray, labels: jnp.ndarray) -> float:
        """Compute Davies-Bouldin index for clustering quality."""
        unique_labels = jnp.unique(labels)
        n_clusters = len(unique_labels)
        
        if n_clusters < 2:
            return float('inf')
        
        # Compute cluster centers
        centers = []
        for label in unique_labels:
            cluster_data = data[labels == label]
            center = jnp.mean(cluster_data, axis=0)
            centers.append(center)
        centers = jnp.stack(centers)
        
        # Compute within-cluster scatter
        within_scatter = []
        for i, label in enumerate(unique_labels):
            cluster_data = data[labels == label]
            scatter = jnp.mean(jnp.linalg.norm(cluster_data - centers[i], axis=1))
            within_scatter.append(scatter)
        within_scatter = jnp.array(within_scatter)
        
        # Compute Davies-Bouldin index
        db_values = []
        for i in range(n_clusters):
            max_ratio = 0.0
            for j in range(n_clusters):
                if i != j:
                    between_distance = jnp.linalg.norm(centers[i] - centers[j])
                    if between_distance > 0:
                        ratio = (within_scatter[i] + within_scatter[j]) / between_distance
                        max_ratio = max(max_ratio, ratio)
            db_values.append(max_ratio)
        
        return float(jnp.mean(jnp.array(db_values)))
    
    @staticmethod
    def compute_pattern_diversity(patterns: jnp.ndarray) -> float:
        """Compute diversity of discovered patterns."""
        n_patterns = len(patterns)
        if n_patterns < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = []
        for i in range(n_patterns):
            for j in range(i + 1, n_patterns):
                dist = jnp.linalg.norm(patterns[i] - patterns[j])
                distances.append(dist)
        
        # Return mean distance as diversity measure
        return float(jnp.mean(jnp.array(distances)))


class UnsupervisedLearningPipeline:
    """
    Complete unsupervised learning pipeline for autonomous pattern discovery.
    
    Integrates competitive learning, self-organizing maps, and feature extraction
    to discover meaningful patterns in multi-modal sensory data.
    """
    
    def __init__(self, config: UnsupervisedLearningConfig, key: jax.random.PRNGKey):
        self.config = config
        self.key = key
        
        # Initialize components
        self.competitive_learner = None
        self.som = None
        self.feature_extractor = None
        
        # Training state
        self.training_history = {
            'competitive_learning': {},
            'som_training': {},
            'feature_extraction': {},
            'quality_metrics': []
        }
        
        # Discovered patterns
        self.discovered_patterns = []
        self.pattern_clusters = {}
        self.feature_representations = {}
        
    def initialize_components(self, input_dim: int, modality_dims: Dict[str, int] = None) -> None:
        """Initialize all pipeline components."""
        # Initialize competitive learning
        key, subkey = random.split(self.key)
        self.key = key
        
        if self.config.use_adaptive_competitive:
            self.competitive_learner = AdaptiveCompetitiveLearning(
                self.config.competitive_learning, input_dim, subkey
            )
        else:
            self.competitive_learner = CompetitiveLearning(
                self.config.competitive_learning, input_dim, subkey
            )
        
        # Initialize SOM
        key, subkey = random.split(self.key)
        self.key = key
        
        if self.config.use_hierarchical_som and self.config.som_layers:
            self.som = HierarchicalSOM(self.config.som_layers, subkey)
        else:
            som_config = self.config.som_config
            som_config.input_dim = input_dim
            self.som = SelfOrganizingMap(som_config, subkey)
        
        # Initialize feature extractor
        if modality_dims:
            key, subkey = random.split(self.key)
            self.key = key
            
            self.feature_extractor = MultiModalFeatureExtractor(
                self.config.feature_extraction, subkey
            )
            
            # Add modalities
            for modality, dim in modality_dims.items():
                extractor_type = "sparse_autoencoder"  # Default
                self.feature_extractor.add_modality(modality, extractor_type, dim)
    
    def discover_patterns_competitive(self, data: jnp.ndarray) -> Dict[str, Any]:
        """Discover patterns using competitive learning."""
        print("Starting competitive learning pattern discovery...")
        start_time = time.time()
        
        # Train competitive learner
        training_stats = self.competitive_learner.train(data)
        
        # Get discovered patterns
        patterns = self.competitive_learner.get_learned_patterns()
        assignments = self.competitive_learner.get_pattern_assignments(data)
        
        # Evaluate pattern quality
        silhouette = PatternQualityMetrics.compute_silhouette_score(data, assignments)
        davies_bouldin = PatternQualityMetrics.compute_davies_bouldin_index(data, assignments)
        diversity = PatternQualityMetrics.compute_pattern_diversity(patterns)
        
        results = {
            'patterns': patterns,
            'assignments': assignments,
            'training_stats': training_stats,
            'quality_metrics': {
                'silhouette_score': silhouette,
                'davies_bouldin_index': davies_bouldin,
                'pattern_diversity': diversity,
                'quantization_error': self.competitive_learner.compute_quantization_error(data)
            },
            'training_time': time.time() - start_time
        }
        
        self.training_history['competitive_learning'] = results
        return results
    
    def cluster_experiences_som(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Cluster experiences using self-organizing map."""
        print("Starting SOM-based experience clustering...")
        start_time = time.time()
        
        # Convert experiences to feature vectors
        if isinstance(self.som, HierarchicalSOM):
            # Create data matrix from experiences
            data_vectors = []
            for exp in experiences:
                vector = self.som.som_layers[0]._experience_to_vector(exp)
                data_vectors.append(vector)
            data_matrix = jnp.stack(data_vectors)
            
            # Train hierarchical SOM
            training_stats = self.som.train_hierarchical(data_matrix, verbose=True)
            
            # Get hierarchical clusters
            clusters = self.som.get_hierarchical_clusters(experiences)
            
        else:
            # Single SOM training
            data_vectors = []
            for exp in experiences:
                vector = self.som._experience_to_vector(exp)
                data_vectors.append(vector)
            data_matrix = jnp.stack(data_vectors)
            
            training_stats = self.som.train(data_matrix, verbose=True)
            clusters = self.som.cluster_experiences(experiences)
        
        # Compute quality metrics
        quantization_error = self.som.compute_quantization_error(data_matrix) if hasattr(self.som, 'compute_quantization_error') else 0.0
        topographic_error = self.som.compute_topographic_error(data_matrix) if hasattr(self.som, 'compute_topographic_error') else 0.0
        
        results = {
            'clusters': clusters,
            'training_stats': training_stats,
            'quality_metrics': {
                'quantization_error': quantization_error,
                'topographic_error': topographic_error,
                'n_clusters': len(clusters) if isinstance(clusters, dict) else len(clusters[0])
            },
            'training_time': time.time() - start_time
        }
        
        self.training_history['som_training'] = results
        return results
    
    def extract_multimodal_features(self, multimodal_data: List[MultiModalInput]) -> Dict[str, Any]:
        """Extract features from multi-modal data."""
        print("Starting multi-modal feature extraction...")
        start_time = time.time()
        
        if self.feature_extractor is None:
            raise ValueError("Feature extractor not initialized")
        
        # Train feature extractors
        training_stats = self.feature_extractor.fit_multimodal(multimodal_data)
        
        # Extract unified features
        unified_features = []
        for sample in multimodal_data:
            try:
                features = self.feature_extractor.transform_multimodal(sample)
                unified_features.append(features)
            except ValueError:
                # Skip samples with no valid modalities
                continue
        
        if unified_features:
            unified_features = jnp.stack(unified_features)
        else:
            unified_features = jnp.array([])
        
        # Get modality importance
        modality_importance = self.feature_extractor.get_modality_importance()
        feature_dimensions = self.feature_extractor.get_feature_dimensions()
        
        results = {
            'unified_features': unified_features,
            'training_stats': training_stats,
            'modality_importance': modality_importance,
            'feature_dimensions': feature_dimensions,
            'n_samples': len(unified_features) if len(unified_features) > 0 else 0,
            'training_time': time.time() - start_time
        }
        
        self.training_history['feature_extraction'] = results
        return results
    
    def run_complete_pipeline(self, 
                            data: jnp.ndarray, 
                            experiences: List[Experience] = None,
                            multimodal_data: List[MultiModalInput] = None) -> Dict[str, Any]:
        """Run the complete unsupervised learning pipeline."""
        print("Starting complete unsupervised learning pipeline...")
        pipeline_start_time = time.time()
        
        results = {
            'competitive_learning': None,
            'som_clustering': None,
            'feature_extraction': None,
            'integrated_patterns': None,
            'overall_quality': None
        }
        
        # Stage 1: Competitive learning pattern discovery
        if self.competitive_learner is not None:
            results['competitive_learning'] = self.discover_patterns_competitive(data)
            self.discovered_patterns = results['competitive_learning']['patterns']
        
        # Stage 2: SOM-based experience clustering
        if self.som is not None and experiences is not None:
            results['som_clustering'] = self.cluster_experiences_som(experiences)
            self.pattern_clusters = results['som_clustering']['clusters']
        
        # Stage 3: Multi-modal feature extraction
        if self.feature_extractor is not None and multimodal_data is not None:
            results['feature_extraction'] = self.extract_multimodal_features(multimodal_data)
            self.feature_representations = results['feature_extraction']['unified_features']
        
        # Stage 4: Integrate discovered patterns
        results['integrated_patterns'] = self._integrate_discovered_patterns()
        
        # Stage 5: Compute overall quality metrics
        results['overall_quality'] = self._compute_overall_quality(data)
        
        # Store pipeline results
        results['total_pipeline_time'] = time.time() - pipeline_start_time
        results['convergence_achieved'] = self._check_convergence()
        
        return results
    
    def _integrate_discovered_patterns(self) -> Dict[str, Any]:
        """Integrate patterns discovered by different components."""
        integration_results = {
            'n_competitive_patterns': len(self.discovered_patterns) if len(self.discovered_patterns) > 0 else 0,
            'n_som_clusters': len(self.pattern_clusters) if isinstance(self.pattern_clusters, dict) else 0,
            'n_feature_dimensions': len(self.feature_representations[0]) if len(self.feature_representations) > 0 else 0,
            'integration_successful': False
        }
        
        # Simple integration: combine pattern counts
        total_patterns = (integration_results['n_competitive_patterns'] + 
                         integration_results['n_som_clusters'])
        
        if total_patterns > 0:
            integration_results['integration_successful'] = True
            integration_results['total_discovered_patterns'] = total_patterns
        
        return integration_results
    
    def _compute_overall_quality(self, data: jnp.ndarray) -> Dict[str, float]:
        """Compute overall quality metrics for the pipeline."""
        quality_metrics = {
            'overall_score': 0.0,
            'pattern_quality': 0.0,
            'clustering_quality': 0.0,
            'feature_quality': 0.0
        }
        
        # Competitive learning quality
        if 'competitive_learning' in self.training_history:
            cl_metrics = self.training_history['competitive_learning']['quality_metrics']
            quality_metrics['pattern_quality'] = cl_metrics['silhouette_score']
        
        # SOM clustering quality
        if 'som_training' in self.training_history:
            som_metrics = self.training_history['som_training']['quality_metrics']
            # Invert quantization error (lower is better)
            quality_metrics['clustering_quality'] = 1.0 / (1.0 + som_metrics['quantization_error'])
        
        # Feature extraction quality
        if 'feature_extraction' in self.training_history:
            fe_results = self.training_history['feature_extraction']
            if len(fe_results['unified_features']) > 0:
                # Use feature diversity as quality measure
                feature_diversity = PatternQualityMetrics.compute_pattern_diversity(
                    fe_results['unified_features']
                )
                quality_metrics['feature_quality'] = min(1.0, feature_diversity / 10.0)
        
        # Overall score (weighted average)
        weights = [0.4, 0.3, 0.3]  # Competitive, SOM, Features
        scores = [quality_metrics['pattern_quality'], 
                 quality_metrics['clustering_quality'], 
                 quality_metrics['feature_quality']]
        
        quality_metrics['overall_score'] = sum(w * s for w, s in zip(weights, scores))
        
        return quality_metrics
    
    def _check_convergence(self) -> bool:
        """Check if the learning pipeline has converged."""
        # Simple convergence check based on quality metrics
        if len(self.training_history['quality_metrics']) < self.config.convergence_patience:
            return False
        
        recent_scores = self.training_history['quality_metrics'][-self.config.convergence_patience:]
        score_variance = jnp.var(jnp.array([score['overall_score'] for score in recent_scores]))
        
        return float(score_variance) < self.config.quality_threshold
    
    def get_discovered_patterns(self) -> Dict[str, Any]:
        """Get all discovered patterns from the pipeline."""
        return {
            'competitive_patterns': self.discovered_patterns,
            'som_clusters': self.pattern_clusters,
            'feature_representations': self.feature_representations,
            'training_history': self.training_history
        }
    
    def evaluate_learning_progress(self, data: jnp.ndarray) -> Dict[str, float]:
        """Evaluate current learning progress."""
        progress_metrics = {
            'convergence_progress': 0.0,
            'pattern_discovery_rate': 0.0,
            'quality_improvement': 0.0
        }
        
        # Convergence progress
        if len(self.training_history['quality_metrics']) > 0:
            recent_quality = self.training_history['quality_metrics'][-1]['overall_score']
            progress_metrics['convergence_progress'] = min(1.0, recent_quality)
        
        # Pattern discovery rate
        if len(self.discovered_patterns) > 0:
            progress_metrics['pattern_discovery_rate'] = len(self.discovered_patterns) / 100.0  # Normalize
        
        # Quality improvement
        if len(self.training_history['quality_metrics']) > 1:
            current_quality = self.training_history['quality_metrics'][-1]['overall_score']
            initial_quality = self.training_history['quality_metrics'][0]['overall_score']
            if initial_quality > 0:
                progress_metrics['quality_improvement'] = (current_quality - initial_quality) / initial_quality
        
        return progress_metrics
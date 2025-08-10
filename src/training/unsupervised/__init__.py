"""
Unsupervised learning algorithms for the Godly AI system.

This module handles autonomous learning without supervision, including:
- Competitive learning for pattern discovery
- Self-organizing maps for experience clustering
- Multi-modal feature extraction
- Integrated learning pipeline
"""

from .competitive_learning import (
    CompetitiveLearning,
    CompetitiveLearningConfig,
    AdaptiveCompetitiveLearning
)

from .self_organizing_map import (
    SelfOrganizingMap,
    SOMConfig,
    HierarchicalSOM,
    Experience
)

from .feature_extraction import (
    MultiModalFeatureExtractor,
    FeatureExtractionConfig,
    MultiModalInput,
    SparseAutoencoder,
    IndependentComponentAnalysis,
    FeatureExtractor
)

from .learning_pipeline import (
    UnsupervisedLearningPipeline,
    UnsupervisedLearningConfig,
    PatternQualityMetrics
)

__all__ = [
    # Competitive Learning
    'CompetitiveLearning',
    'CompetitiveLearningConfig', 
    'AdaptiveCompetitiveLearning',
    
    # Self-Organizing Maps
    'SelfOrganizingMap',
    'SOMConfig',
    'HierarchicalSOM',
    'Experience',
    
    # Feature Extraction
    'MultiModalFeatureExtractor',
    'FeatureExtractionConfig',
    'MultiModalInput',
    'SparseAutoencoder',
    'IndependentComponentAnalysis',
    'FeatureExtractor',
    
    # Learning Pipeline
    'UnsupervisedLearningPipeline',
    'UnsupervisedLearningConfig',
    'PatternQualityMetrics'
]
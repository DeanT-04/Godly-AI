"""
Exploration and curiosity engines for the Godly AI agents.

This module handles curiosity-driven exploration and novelty detection.
"""

from .novelty_detection import (
    NoveltyDetector,
    NoveltyScore,
    PredictionErrorNoveltyDetector,
    EnsembleNoveltyDetector,
    PredictionModel
)

from .curiosity_engine import (
    CuriosityEngine,
    ExplorationGoal,
    ExplorationStrategy,
    InterestModel,
    InterestRegion
)

from .exploration_system import (
    ExplorationSystem,
    ExplorationConfig,
    create_exploration_system
)

from .internal_reward import (
    IntrinsicRewardGenerator,
    RewardPredictor,
    SurpriseDetector,
    RewardSignal,
    RewardType,
    SurpriseEvent
)

from .reward_system import (
    RewardSystem,
    RewardSystemConfig,
    RewardLearningIntegrator,
    create_reward_system
)

__all__ = [
    # Novelty Detection
    'NoveltyDetector',
    'NoveltyScore', 
    'PredictionErrorNoveltyDetector',
    'EnsembleNoveltyDetector',
    'PredictionModel',
    
    # Curiosity Engine
    'CuriosityEngine',
    'ExplorationGoal',
    'ExplorationStrategy',
    'InterestModel',
    'InterestRegion',
    
    # Internal Reward System
    'IntrinsicRewardGenerator',
    'RewardPredictor',
    'SurpriseDetector',
    'RewardSignal',
    'RewardType',
    'SurpriseEvent',
    
    # Reward System
    'RewardSystem',
    'RewardSystemConfig',
    'RewardLearningIntegrator',
    'create_reward_system',
    
    # Main System
    'ExplorationSystem',
    'ExplorationConfig',
    'create_exploration_system'
]
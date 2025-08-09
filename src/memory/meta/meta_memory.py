"""
Meta-Memory Implementation for Learning-to-Learn

This module implements the meta-memory system that stores learning strategies,
meta-parameters, and learning-to-learn capabilities operating on hour+ timescales.
"""

from typing import NamedTuple, Optional, Tuple, Dict, List, Any, Union
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
import numpy as np
import hashlib
import time
import json
from enum import Enum

from ...core.liquid_state_machine import LiquidStateMachine, LSMParams, LSMState


class LearningStrategy(Enum):
    """Types of learning strategies."""
    GRADIENT_DESCENT = "gradient_descent"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    UNSUPERVISED = "unsupervised"
    TRANSFER = "transfer"
    META_LEARNING = "meta_learning"
    CURRICULUM = "curriculum"
    SELF_SUPERVISED = "self_supervised"


@dataclass
class MetaMemoryParams:
    """Parameters for the Meta-Memory system."""
    
    # Memory capacity
    learning_history_size: int = 1000    # Maximum learning experiences to store
    strategy_cache_size: int = 50        # Maximum strategies to cache
    
    # Time scales
    hour_timescale: float = 3600.0       # Hour+ timescale in seconds
    consolidation_interval: float = 1800.0  # 30 minutes between consolidations
    
    # Learning strategy parameters
    strategy_similarity_threshold: float = 0.7  # Threshold for strategy matching
    performance_improvement_threshold: float = 0.05  # Min improvement to store
    
    # Meta-parameter adaptation
    adaptation_rate: float = 0.1         # Rate of meta-parameter updates
    exploration_rate: float = 0.2        # Rate of strategy exploration
    
    # Memory consolidation
    consolidation_strength: float = 0.8   # Strength of memory consolidation
    forgetting_threshold: float = 0.1     # Threshold for forgetting strategies
    
    # Performance tracking
    performance_window: int = 10          # Window for performance averaging
    success_threshold: float = 0.8        # Threshold for successful learning


@dataclass
class LearningExperience:
    """A single learning experience with context and outcomes."""
    
    experience_id: str                   # Unique experience identifier
    task_type: str                       # Type of task learned
    task_context: Dict[str, Any]         # Context information about the task
    strategy_used: LearningStrategy      # Learning strategy employed
    meta_parameters: Dict[str, float]    # Meta-parameters used
    initial_performance: float           # Performance before learning
    final_performance: float             # Performance after learning
    learning_time: float                 # Time taken to learn
    timestamp: float                     # When the experience occurred
    success: bool                        # Whether learning was successful
    difficulty: float                    # Estimated task difficulty
    transfer_source: Optional[str]       # Source task if transfer learning
    
    @property
    def performance_improvement(self) -> float:
        """Calculate performance improvement."""
        return self.final_performance - self.initial_performance
    
    @property
    def learning_efficiency(self) -> float:
        """Calculate learning efficiency (improvement per unit time)."""
        if self.learning_time <= 0:
            return 0.0
        return self.performance_improvement / self.learning_time


@dataclass
class StrategyTemplate:
    """Template for a learning strategy with its parameters."""
    
    strategy_id: str                     # Unique strategy identifier
    strategy_type: LearningStrategy      # Type of strategy
    meta_parameters: Dict[str, float]    # Default meta-parameters
    applicable_tasks: List[str]          # Task types this strategy works for
    success_rate: float                  # Historical success rate
    average_efficiency: float            # Average learning efficiency
    usage_count: int                     # Number of times used
    last_updated: float                  # Last time parameters were updated
    creation_time: float                 # When strategy was created
    
    def is_applicable(self, task_type: str, task_context: Dict[str, Any]) -> float:
        """
        Determine if strategy is applicable to a task.
        
        Args:
            task_type: Type of task
            task_context: Context information about the task
            
        Returns:
            Applicability score (0-1)
        """
        if task_type in self.applicable_tasks:
            base_score = 0.8
        else:
            base_score = 0.2
        
        # Adjust based on success rate and efficiency
        performance_bonus = (self.success_rate + self.average_efficiency) * 0.1
        
        return min(1.0, base_score + performance_bonus)


class MetaMemoryState(NamedTuple):
    """State variables for the Meta-Memory system."""
    
    learning_experiences: Dict[str, LearningExperience]  # Stored experiences
    strategy_templates: Dict[str, StrategyTemplate]      # Available strategies
    task_performance_history: Dict[str, List[float]]     # Performance per task type
    meta_parameter_history: Dict[str, List[float]]       # Meta-parameter evolution
    consolidation_state: Dict[str, Any]                  # Consolidation tracking
    global_time: float                                   # Current time
    total_experiences: int                               # Total experiences stored
    last_consolidation: float                            # Last consolidation time


class MetaMemory:
    """
    Meta-Memory system for learning-to-learn capabilities.
    
    Implements:
    - Learning strategy storage and retrieval
    - Meta-parameter adaptation mechanisms
    - Learning-to-learn capability tracking
    - Strategy template evolution
    - Performance-based strategy selection
    """
    
    def __init__(self, params: Optional[MetaMemoryParams] = None):
        """Initialize Meta-Memory system."""
        self.params = params or MetaMemoryParams()
        
        # Initialize default learning strategies
        self.default_strategies = self._create_default_strategies()
    
    def init_state(self, key: Optional[jax.random.PRNGKey] = None) -> MetaMemoryState:
        """
        Initialize Meta-Memory state.
        
        Args:
            key: Random key for initialization
            
        Returns:
            Initial meta-memory state
        """
        current_time = time.time()
        
        return MetaMemoryState(
            learning_experiences={},
            strategy_templates=self.default_strategies.copy(),
            task_performance_history={},
            meta_parameter_history={},
            consolidation_state={
                'last_consolidation': current_time,
                'consolidation_count': 0,
                'pending_experiences': []
            },
            global_time=current_time,
            total_experiences=0,
            last_consolidation=current_time
        )
    
    def store_learning_experience(
        self,
        state: MetaMemoryState,
        task: str,
        performance: float,
        strategy: Union[str, LearningStrategy],
        meta_parameters: Optional[Dict[str, float]] = None,
        task_context: Optional[Dict[str, Any]] = None,
        learning_time: Optional[float] = None,
        initial_performance: Optional[float] = None
    ) -> Tuple[MetaMemoryState, str]:
        """
        Store a learning experience in meta-memory.
        
        Args:
            state: Current meta-memory state
            task: Task identifier or type
            performance: Final performance achieved
            strategy: Learning strategy used
            meta_parameters: Meta-parameters used during learning
            task_context: Additional context about the task
            learning_time: Time taken for learning
            initial_performance: Performance before learning
            
        Returns:
            Tuple of (updated_state, experience_id)
        """
        current_time = time.time()
        
        # Convert strategy to enum if string
        if isinstance(strategy, str):
            try:
                strategy_enum = LearningStrategy(strategy)
            except ValueError:
                strategy_enum = LearningStrategy.GRADIENT_DESCENT  # Default
        else:
            strategy_enum = strategy
        
        # Set defaults
        if meta_parameters is None:
            meta_parameters = {}
        if task_context is None:
            task_context = {}
        if learning_time is None:
            learning_time = 1.0  # Default learning time
        if initial_performance is None:
            initial_performance = max(0.0, performance - 0.1)  # Estimate
        
        # Generate experience ID
        experience_id = self._generate_experience_id(task, current_time)
        
        # Determine success and difficulty
        performance_improvement = performance - initial_performance
        success = performance_improvement >= self.params.performance_improvement_threshold
        difficulty = self._estimate_task_difficulty(task, task_context, performance_improvement)
        
        # Create learning experience
        experience = LearningExperience(
            experience_id=experience_id,
            task_type=task,
            task_context=task_context,
            strategy_used=strategy_enum,
            meta_parameters=meta_parameters,
            initial_performance=initial_performance,
            final_performance=performance,
            learning_time=learning_time,
            timestamp=current_time,
            success=success,
            difficulty=difficulty,
            transfer_source=task_context.get('transfer_source')
        )
        
        # Check if we need to make space
        new_experiences = state.learning_experiences.copy()
        if len(new_experiences) >= self.params.learning_history_size:
            new_experiences = self._make_space_for_experience(new_experiences)
        
        new_experiences[experience_id] = experience
        
        # Update task performance history
        new_task_history = state.task_performance_history.copy()
        if task not in new_task_history:
            new_task_history[task] = []
        new_task_history[task].append(performance)
        
        # Keep only recent performance history
        if len(new_task_history[task]) > self.params.performance_window:
            new_task_history[task] = new_task_history[task][-self.params.performance_window:]
        
        # Update meta-parameter history
        new_meta_history = state.meta_parameter_history.copy()
        for param_name, param_value in meta_parameters.items():
            if param_name not in new_meta_history:
                new_meta_history[param_name] = []
            new_meta_history[param_name].append(param_value)
            
            # Keep only recent history
            if len(new_meta_history[param_name]) > self.params.performance_window:
                new_meta_history[param_name] = new_meta_history[param_name][-self.params.performance_window:]
        
        # Update consolidation state
        new_consolidation_state = state.consolidation_state.copy()
        new_consolidation_state['pending_experiences'].append(experience_id)
        
        new_state = MetaMemoryState(
            learning_experiences=new_experiences,
            strategy_templates=state.strategy_templates,
            task_performance_history=new_task_history,
            meta_parameter_history=new_meta_history,
            consolidation_state=new_consolidation_state,
            global_time=current_time,
            total_experiences=state.total_experiences + 1,
            last_consolidation=state.last_consolidation
        )
        
        # Check if consolidation is needed
        if (current_time - state.last_consolidation) >= self.params.consolidation_interval:
            new_state = self._consolidate_memories(new_state)
            # Add current experience to pending list after consolidation
            updated_consolidation_state = new_state.consolidation_state.copy()
            updated_consolidation_state['pending_experiences'] = [experience_id]
            new_state = new_state._replace(consolidation_state=updated_consolidation_state)
        
        return new_state, experience_id
    
    def retrieve_learning_strategy(
        self,
        state: MetaMemoryState,
        task_similarity: float,
        task_type: Optional[str] = None,
        task_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, float], float]:
        """
        Retrieve the best learning strategy for a given task.
        
        Args:
            state: Current meta-memory state
            task_similarity: Similarity to previous tasks (0-1)
            task_type: Type of task (optional)
            task_context: Context about the task (optional)
            
        Returns:
            Tuple of (strategy_name, meta_parameters, confidence)
        """
        if task_context is None:
            task_context = {}
        
        # Score all available strategies
        strategy_scores = {}
        
        for strategy_id, strategy_template in state.strategy_templates.items():
            # Base applicability score
            if task_type:
                applicability = strategy_template.is_applicable(task_type, task_context)
            else:
                applicability = 0.5  # Neutral if no task type
            
            # Adjust based on task similarity
            similarity_bonus = task_similarity * 0.3
            
            # Performance-based scoring
            performance_score = (
                strategy_template.success_rate * 0.4 +
                strategy_template.average_efficiency * 0.3
            )
            
            # Recency bonus (prefer recently successful strategies)
            recency_bonus = self._compute_recency_bonus(
                strategy_template.last_updated, time.time()
            )
            
            total_score = (
                applicability * 0.4 +
                similarity_bonus +
                performance_score +
                recency_bonus * 0.1
            )
            
            # Clamp score to [0, 1] range
            strategy_scores[strategy_id] = float(np.clip(total_score, 0.0, 1.0))
        
        # Select best strategy
        if not strategy_scores:
            # Fallback to default
            return "gradient_descent", {}, 0.5
        
        best_strategy_id = max(strategy_scores.keys(), key=lambda k: strategy_scores[k])
        best_strategy = state.strategy_templates[best_strategy_id]
        confidence = strategy_scores[best_strategy_id]
        
        # Adapt meta-parameters based on task context and history
        adapted_params = self._adapt_meta_parameters(
            state, best_strategy, task_type, task_context
        )
        
        return best_strategy_id, adapted_params, confidence
    
    def update_meta_parameters(
        self,
        state: MetaMemoryState,
        performance_feedback: float,
        strategy_id: Optional[str] = None,
        task_type: Optional[str] = None
    ) -> MetaMemoryState:
        """
        Update meta-parameters based on performance feedback.
        
        Args:
            state: Current meta-memory state
            performance_feedback: Performance improvement (-1 to 1)
            strategy_id: Strategy that was used (optional)
            task_type: Type of task (optional)
            
        Returns:
            Updated meta-memory state
        """
        current_time = time.time()
        
        # Update strategy templates if strategy is specified
        new_strategy_templates = state.strategy_templates.copy()
        
        if strategy_id and strategy_id in new_strategy_templates:
            strategy = new_strategy_templates[strategy_id]
            
            # Update success rate using exponential moving average
            alpha = self.params.adaptation_rate
            if performance_feedback > 0:
                new_success_rate = (1 - alpha) * strategy.success_rate + alpha * 1.0
            else:
                new_success_rate = (1 - alpha) * strategy.success_rate + alpha * 0.0
            
            # Update efficiency estimate
            efficiency_estimate = abs(performance_feedback)  # Simplified
            new_efficiency = (1 - alpha) * strategy.average_efficiency + alpha * efficiency_estimate
            
            # Create updated strategy
            updated_strategy = StrategyTemplate(
                strategy_id=strategy.strategy_id,
                strategy_type=strategy.strategy_type,
                meta_parameters=strategy.meta_parameters.copy(),
                applicable_tasks=strategy.applicable_tasks.copy(),
                success_rate=new_success_rate,
                average_efficiency=new_efficiency,
                usage_count=strategy.usage_count + 1,
                last_updated=current_time,
                creation_time=strategy.creation_time
            )
            
            # Adapt meta-parameters based on feedback
            if performance_feedback > 0:
                # Positive feedback: slightly increase exploration
                for param_name in updated_strategy.meta_parameters:
                    if 'rate' in param_name.lower():
                        updated_strategy.meta_parameters[param_name] *= 1.05
            else:
                # Negative feedback: be more conservative
                for param_name in updated_strategy.meta_parameters:
                    if 'rate' in param_name.lower():
                        updated_strategy.meta_parameters[param_name] *= 0.95
            
            new_strategy_templates[strategy_id] = updated_strategy
        
        # Update global meta-parameter trends
        new_meta_history = state.meta_parameter_history.copy()
        new_meta_history['global_performance'] = new_meta_history.get('global_performance', [])
        new_meta_history['global_performance'].append(performance_feedback)
        
        if len(new_meta_history['global_performance']) > self.params.performance_window:
            new_meta_history['global_performance'] = new_meta_history['global_performance'][-self.params.performance_window:]
        
        return MetaMemoryState(
            learning_experiences=state.learning_experiences,
            strategy_templates=new_strategy_templates,
            task_performance_history=state.task_performance_history,
            meta_parameter_history=new_meta_history,
            consolidation_state=state.consolidation_state,
            global_time=current_time,
            total_experiences=state.total_experiences,
            last_consolidation=state.last_consolidation
        )
    
    def get_learning_statistics(self, state: MetaMemoryState) -> Dict[str, Any]:
        """
        Get comprehensive learning statistics.
        
        Args:
            state: Current meta-memory state
            
        Returns:
            Dictionary of learning statistics
        """
        if not state.learning_experiences:
            return {
                'total_experiences': 0,
                'success_rate': 0.0,
                'average_improvement': 0.0,
                'learning_efficiency': 0.0,
                'strategy_diversity': 0,
                'task_diversity': 0,
                'meta_learning_progress': 0.0
            }
        
        experiences = list(state.learning_experiences.values())
        
        # Basic statistics
        total_experiences = len(experiences)
        successful_experiences = sum(1 for exp in experiences if exp.success)
        success_rate = successful_experiences / total_experiences
        
        # Performance statistics
        improvements = [exp.performance_improvement for exp in experiences]
        average_improvement = float(np.mean(improvements))
        
        # Efficiency statistics
        efficiencies = [exp.learning_efficiency for exp in experiences if exp.learning_efficiency > 0]
        learning_efficiency = float(np.mean(efficiencies)) if efficiencies else 0.0
        
        # Diversity statistics
        strategies_used = set(exp.strategy_used for exp in experiences)
        strategy_diversity = len(strategies_used)
        
        tasks_learned = set(exp.task_type for exp in experiences)
        task_diversity = len(tasks_learned)
        
        # Meta-learning progress (improvement in learning efficiency over time)
        if len(experiences) >= 10:
            recent_efficiencies = [exp.learning_efficiency for exp in experiences[-10:]]
            early_efficiencies = [exp.learning_efficiency for exp in experiences[:10]]
            
            recent_avg = np.mean(recent_efficiencies) if recent_efficiencies else 0.0
            early_avg = np.mean(early_efficiencies) if early_efficiencies else 0.0
            
            meta_learning_progress = recent_avg - early_avg
        else:
            meta_learning_progress = 0.0
        
        # Strategy performance breakdown
        strategy_performance = {}
        for strategy in LearningStrategy:
            strategy_experiences = [exp for exp in experiences if exp.strategy_used == strategy]
            if strategy_experiences:
                strategy_success_rate = sum(1 for exp in strategy_experiences if exp.success) / len(strategy_experiences)
                strategy_avg_improvement = np.mean([exp.performance_improvement for exp in strategy_experiences])
                strategy_performance[strategy.value] = {
                    'success_rate': strategy_success_rate,
                    'average_improvement': float(strategy_avg_improvement),
                    'usage_count': len(strategy_experiences)
                }
        
        return {
            'total_experiences': total_experiences,
            'success_rate': success_rate,
            'average_improvement': average_improvement,
            'learning_efficiency': learning_efficiency,
            'strategy_diversity': strategy_diversity,
            'task_diversity': task_diversity,
            'meta_learning_progress': meta_learning_progress,
            'strategy_performance': strategy_performance,
            'consolidation_count': state.consolidation_state.get('consolidation_count', 0),
            'time_since_last_consolidation': time.time() - state.last_consolidation
        }
    
    def _create_default_strategies(self) -> Dict[str, StrategyTemplate]:
        """Create default learning strategy templates."""
        current_time = time.time()
        strategies = {}
        
        # Gradient Descent Strategy
        strategies['gradient_descent'] = StrategyTemplate(
            strategy_id='gradient_descent',
            strategy_type=LearningStrategy.GRADIENT_DESCENT,
            meta_parameters={
                'learning_rate': 0.01,
                'momentum': 0.9,
                'weight_decay': 1e-4,
                'batch_size': 32
            },
            applicable_tasks=['classification', 'regression', 'optimization'],
            success_rate=0.7,
            average_efficiency=0.6,
            usage_count=0,
            last_updated=current_time,
            creation_time=current_time
        )
        
        # Evolutionary Strategy
        strategies['evolutionary'] = StrategyTemplate(
            strategy_id='evolutionary',
            strategy_type=LearningStrategy.EVOLUTIONARY,
            meta_parameters={
                'population_size': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8,
                'selection_pressure': 0.3
            },
            applicable_tasks=['optimization', 'architecture_search', 'hyperparameter_tuning'],
            success_rate=0.6,
            average_efficiency=0.4,
            usage_count=0,
            last_updated=current_time,
            creation_time=current_time
        )
        
        # Reinforcement Learning Strategy
        strategies['reinforcement'] = StrategyTemplate(
            strategy_id='reinforcement',
            strategy_type=LearningStrategy.REINFORCEMENT,
            meta_parameters={
                'exploration_rate': 0.1,
                'discount_factor': 0.99,
                'learning_rate': 0.001,
                'replay_buffer_size': 10000
            },
            applicable_tasks=['control', 'decision_making', 'sequential'],
            success_rate=0.65,
            average_efficiency=0.5,
            usage_count=0,
            last_updated=current_time,
            creation_time=current_time
        )
        
        # Unsupervised Learning Strategy
        strategies['unsupervised'] = StrategyTemplate(
            strategy_id='unsupervised',
            strategy_type=LearningStrategy.UNSUPERVISED,
            meta_parameters={
                'clustering_threshold': 0.5,
                'dimensionality_reduction': 0.8,
                'anomaly_threshold': 0.1,
                'compression_ratio': 0.5
            },
            applicable_tasks=['clustering', 'dimensionality_reduction', 'anomaly_detection'],
            success_rate=0.55,
            average_efficiency=0.7,
            usage_count=0,
            last_updated=current_time,
            creation_time=current_time
        )
        
        # Transfer Learning Strategy
        strategies['transfer'] = StrategyTemplate(
            strategy_id='transfer',
            strategy_type=LearningStrategy.TRANSFER,
            meta_parameters={
                'transfer_ratio': 0.8,
                'fine_tuning_rate': 0.001,
                'layer_freeze_ratio': 0.5,
                'domain_adaptation_strength': 0.3
            },
            applicable_tasks=['few_shot', 'domain_adaptation', 'fine_tuning'],
            success_rate=0.75,
            average_efficiency=0.8,
            usage_count=0,
            last_updated=current_time,
            creation_time=current_time
        )
        
        return strategies
    
    def _generate_experience_id(self, task: str, timestamp: float) -> str:
        """Generate unique experience ID."""
        experience_hash = hashlib.md5(
            f"{task}_{timestamp}".encode()
        ).hexdigest()
        return f"exp_{experience_hash[:8]}"
    
    def _estimate_task_difficulty(
        self, 
        task: str, 
        task_context: Dict[str, Any], 
        performance_improvement: float
    ) -> float:
        """Estimate task difficulty based on context and performance."""
        # Base difficulty from context
        base_difficulty = task_context.get('difficulty', 0.5)
        
        # Adjust based on performance improvement
        # Lower improvement suggests higher difficulty
        if performance_improvement > 0:
            difficulty_adjustment = max(0.0, 0.5 - performance_improvement)
        else:
            difficulty_adjustment = min(0.5, abs(performance_improvement))
        
        estimated_difficulty = base_difficulty + difficulty_adjustment
        return float(np.clip(estimated_difficulty, 0.0, 1.0))
    
    def _make_space_for_experience(
        self, 
        experiences: Dict[str, LearningExperience]
    ) -> Dict[str, LearningExperience]:
        """Remove least important experiences to make space."""
        if len(experiences) < self.params.learning_history_size:
            return experiences
        
        # Score experiences for removal (lower is more likely to be removed)
        removal_scores = {}
        current_time = time.time()
        
        for exp_id, experience in experiences.items():
            # Combine recency, success, and learning efficiency
            recency_score = 1.0 / (current_time - experience.timestamp + 1.0)
            success_score = 1.0 if experience.success else 0.2
            efficiency_score = experience.learning_efficiency
            
            removal_scores[exp_id] = (
                0.3 * recency_score + 
                0.4 * success_score + 
                0.3 * efficiency_score
            )
        
        # Remove experiences with lowest scores
        experiences_to_remove = sorted(
            removal_scores.keys(), 
            key=lambda k: removal_scores[k]
        )[:len(experiences) - self.params.learning_history_size + 1]
        
        new_experiences = experiences.copy()
        for exp_id in experiences_to_remove:
            del new_experiences[exp_id]
        
        return new_experiences
    
    def _consolidate_memories(self, state: MetaMemoryState) -> MetaMemoryState:
        """Consolidate memories and update strategy templates."""
        current_time = time.time()
        
        # Get pending experiences for consolidation
        pending_exp_ids = state.consolidation_state.get('pending_experiences', [])
        
        if not pending_exp_ids:
            return state
        
        # Analyze pending experiences for patterns
        pending_experiences = [
            state.learning_experiences[exp_id] 
            for exp_id in pending_exp_ids 
            if exp_id in state.learning_experiences
        ]
        
        # Update strategy templates based on recent experiences
        new_strategy_templates = state.strategy_templates.copy()
        
        for experience in pending_experiences:
            strategy_id = experience.strategy_used.value
            
            if strategy_id in new_strategy_templates:
                strategy = new_strategy_templates[strategy_id]
                
                # Update strategy statistics
                alpha = self.params.consolidation_strength
                
                success_update = 1.0 if experience.success else 0.0
                new_success_rate = (1 - alpha) * strategy.success_rate + alpha * success_update
                
                efficiency_update = experience.learning_efficiency
                new_efficiency = (1 - alpha) * strategy.average_efficiency + alpha * efficiency_update
                
                # Update applicable tasks
                new_applicable_tasks = strategy.applicable_tasks.copy()
                if experience.task_type not in new_applicable_tasks and experience.success:
                    new_applicable_tasks.append(experience.task_type)
                
                # Create updated strategy
                new_strategy_templates[strategy_id] = StrategyTemplate(
                    strategy_id=strategy.strategy_id,
                    strategy_type=strategy.strategy_type,
                    meta_parameters=strategy.meta_parameters.copy(),
                    applicable_tasks=new_applicable_tasks,
                    success_rate=new_success_rate,
                    average_efficiency=new_efficiency,
                    usage_count=strategy.usage_count,
                    last_updated=current_time,
                    creation_time=strategy.creation_time
                )
        
        # Update consolidation state
        new_consolidation_state = state.consolidation_state.copy()
        new_consolidation_state['pending_experiences'] = []
        new_consolidation_state['consolidation_count'] = new_consolidation_state.get('consolidation_count', 0) + 1
        new_consolidation_state['last_consolidation'] = current_time
        
        return MetaMemoryState(
            learning_experiences=state.learning_experiences,
            strategy_templates=new_strategy_templates,
            task_performance_history=state.task_performance_history,
            meta_parameter_history=state.meta_parameter_history,
            consolidation_state=new_consolidation_state,
            global_time=current_time,
            total_experiences=state.total_experiences,
            last_consolidation=current_time
        )
    
    def _compute_recency_bonus(self, last_updated: float, current_time: float) -> float:
        """Compute recency bonus for strategy selection."""
        time_diff = current_time - last_updated
        # Exponential decay with half-life of 1 day
        half_life = 24 * 3600  # 24 hours
        return np.exp(-time_diff / half_life)
    
    def _adapt_meta_parameters(
        self,
        state: MetaMemoryState,
        strategy: StrategyTemplate,
        task_type: Optional[str],
        task_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Adapt meta-parameters based on task context and history."""
        adapted_params = strategy.meta_parameters.copy()
        
        # Get historical performance for this task type
        if task_type and task_type in state.task_performance_history:
            recent_performance = state.task_performance_history[task_type]
            avg_performance = np.mean(recent_performance)
            
            # Adapt learning rate based on recent performance
            if 'learning_rate' in adapted_params:
                if avg_performance < 0.5:
                    # Poor performance: increase learning rate
                    adapted_params['learning_rate'] *= 1.2
                elif avg_performance > 0.8:
                    # Good performance: decrease learning rate for stability
                    adapted_params['learning_rate'] *= 0.9
        
        # Adapt based on task difficulty
        difficulty = task_context.get('difficulty', 0.5)
        if difficulty > 0.7:  # High difficulty
            # More conservative parameters
            for param_name in adapted_params:
                if 'rate' in param_name.lower():
                    adapted_params[param_name] *= 0.8
        elif difficulty < 0.3:  # Low difficulty
            # More aggressive parameters
            for param_name in adapted_params:
                if 'rate' in param_name.lower():
                    adapted_params[param_name] *= 1.3
        
        return adapted_params


# Convenience functions
def create_meta_memory(memory_type: str = "standard") -> MetaMemory:
    """
    Create meta-memory with predefined parameter sets.
    
    Args:
        memory_type: Type of memory configuration
                    - "standard": Default parameters
                    - "fast_adaptation": Quick meta-parameter adaptation
                    - "conservative": Slow, stable adaptation
                    - "exploratory": High exploration rate
                    - "large_capacity": Large memory capacity
    
    Returns:
        Configured MetaMemory
    """
    if memory_type == "standard":
        return MetaMemory()
    elif memory_type == "fast_adaptation":
        params = MetaMemoryParams(
            adaptation_rate=0.3,
            exploration_rate=0.4,
            consolidation_interval=900.0  # 15 minutes
        )
        return MetaMemory(params)
    elif memory_type == "conservative":
        params = MetaMemoryParams(
            adaptation_rate=0.05,
            exploration_rate=0.1,
            consolidation_interval=3600.0  # 1 hour
        )
        return MetaMemory(params)
    elif memory_type == "exploratory":
        params = MetaMemoryParams(
            exploration_rate=0.5,
            strategy_similarity_threshold=0.5,
            performance_improvement_threshold=0.02
        )
        return MetaMemory(params)
    elif memory_type == "large_capacity":
        params = MetaMemoryParams(
            learning_history_size=5000,
            strategy_cache_size=200,
            performance_window=50
        )
        return MetaMemory(params)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
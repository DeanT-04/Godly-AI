"""
Learning progress tracking and visualization for the Godly AI System.

This module implements comprehensive learning progress monitoring including:
- Learning curve tracking and analysis
- Performance trend visualization
- Goal achievement monitoring
- Adaptive learning rate tracking
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime
import json
import threading


@dataclass
class LearningEvent:
    """Container for individual learning events."""
    timestamp: float
    task_type: str
    performance_score: float
    learning_rate: float
    episode_count: int
    convergence_metric: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'task_type': self.task_type,
            'performance_score': self.performance_score,
            'learning_rate': self.learning_rate,
            'episode_count': self.episode_count,
            'convergence_metric': self.convergence_metric,
            'metadata': self.metadata
        }


@dataclass
class GoalProgress:
    """Container for goal achievement tracking."""
    goal_id: str
    goal_description: str
    creation_time: float
    target_performance: float
    current_performance: float
    completion_percentage: float
    estimated_completion_time: Optional[float] = None
    is_completed: bool = False
    completion_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'goal_id': self.goal_id,
            'goal_description': self.goal_description,
            'creation_time': self.creation_time,
            'target_performance': self.target_performance,
            'current_performance': self.current_performance,
            'completion_percentage': self.completion_percentage,
            'estimated_completion_time': self.estimated_completion_time,
            'is_completed': self.is_completed,
            'completion_time': self.completion_time
        }


class LearningProgressTracker:
    """Tracks and analyzes learning progress across different tasks and modalities."""
    
    def __init__(self, max_history: int = 50000):
        self.max_history = max_history
        self.learning_events: deque = deque(maxlen=max_history)
        self.goals: Dict[str, GoalProgress] = {}
        self.task_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        
        # Learning analytics
        self.learning_curves: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.convergence_history: List[Tuple[float, str, float]] = []
        
    def record_learning_event(self, 
                            task_type: str,
                            performance_score: float,
                            learning_rate: float,
                            episode_count: int,
                            convergence_metric: float,
                            **metadata) -> None:
        """Record a learning event."""
        event = LearningEvent(
            timestamp=time.time(),
            task_type=task_type,
            performance_score=performance_score,
            learning_rate=learning_rate,
            episode_count=episode_count,
            convergence_metric=convergence_metric,
            metadata=metadata
        )
        
        with self._lock:
            self.learning_events.append(event)
            self.task_performance[task_type].append((event.timestamp, performance_score))
            self.learning_curves[task_type].append((event.timestamp, performance_score))
            
            # Track convergence
            if convergence_metric < 0.01:  # Convergence threshold
                self.convergence_history.append((event.timestamp, task_type, convergence_metric))
                
    def create_goal(self, 
                   goal_id: str,
                   description: str,
                   target_performance: float) -> None:
        """Create a new learning goal."""
        goal = GoalProgress(
            goal_id=goal_id,
            goal_description=description,
            creation_time=time.time(),
            target_performance=target_performance,
            current_performance=0.0,
            completion_percentage=0.0
        )
        
        with self._lock:
            self.goals[goal_id] = goal
            
    def update_goal_progress(self, 
                           goal_id: str,
                           current_performance: float) -> None:
        """Update progress on a learning goal."""
        with self._lock:
            if goal_id not in self.goals:
                return
                
            goal = self.goals[goal_id]
            goal.current_performance = current_performance
            goal.completion_percentage = min(100.0, (current_performance / goal.target_performance) * 100)
            
            # Check for completion
            if current_performance >= goal.target_performance and not goal.is_completed:
                goal.is_completed = True
                goal.completion_time = time.time()
                
    def get_learning_curve(self, 
                          task_type: str,
                          window_size: int = 100) -> Tuple[List[float], List[float]]:
        """Get learning curve for a specific task type."""
        with self._lock:
            if task_type not in self.learning_curves:
                return [], []
                
            curve_data = self.learning_curves[task_type][-window_size:]
            timestamps = [point[0] for point in curve_data]
            scores = [point[1] for point in curve_data]
            
            return timestamps, scores
            
    def get_performance_trend(self, 
                            task_type: str,
                            window_size: int = 50) -> Dict[str, float]:
        """Analyze performance trend for a task type."""
        timestamps, scores = self.get_learning_curve(task_type, window_size)
        
        if len(scores) < 10:
            return {'trend': 0.0, 'confidence': 0.0, 'recent_avg': 0.0}
            
        # Calculate trend using linear regression
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # Linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Calculate R-squared for confidence
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'trend': float(slope),
            'confidence': float(r_squared),
            'recent_avg': float(np.mean(scores[-10:])),
            'improvement_rate': float(slope * len(scores)) if len(scores) > 0 else 0.0
        }
        
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """Analyze convergence patterns across tasks."""
        with self._lock:
            if not self.convergence_history:
                return {'total_convergences': 0, 'convergence_rate': 0.0}
                
            # Group by task type
            task_convergences = defaultdict(list)
            for timestamp, task_type, metric in self.convergence_history:
                task_convergences[task_type].append((timestamp, metric))
                
            # Calculate convergence statistics
            total_convergences = len(self.convergence_history)
            time_span = time.time() - self.convergence_history[0][0] if self.convergence_history else 1
            convergence_rate = total_convergences / (time_span / 3600)  # per hour
            
            task_stats = {}
            for task_type, convergences in task_convergences.items():
                task_stats[task_type] = {
                    'count': len(convergences),
                    'avg_convergence_metric': np.mean([c[1] for c in convergences]),
                    'last_convergence': max(c[0] for c in convergences)
                }
                
            return {
                'total_convergences': total_convergences,
                'convergence_rate': convergence_rate,
                'task_statistics': task_stats,
                'recent_convergences': self.convergence_history[-10:]
            }
            
    def get_goal_summary(self) -> Dict[str, Any]:
        """Get summary of all learning goals."""
        with self._lock:
            completed_goals = [g for g in self.goals.values() if g.is_completed]
            active_goals = [g for g in self.goals.values() if not g.is_completed]
            
            return {
                'total_goals': len(self.goals),
                'completed_goals': len(completed_goals),
                'active_goals': len(active_goals),
                'completion_rate': len(completed_goals) / len(self.goals) if self.goals else 0.0,
                'avg_completion_time': np.mean([
                    g.completion_time - g.creation_time 
                    for g in completed_goals 
                    if g.completion_time
                ]) if completed_goals else 0.0,
                'goals': [g.to_dict() for g in self.goals.values()]
            }
            
    def generate_learning_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning progress report."""
        with self._lock:
            # Overall statistics
            total_events = len(self.learning_events)
            if total_events == 0:
                return {'error': 'No learning events recorded'}
                
            recent_events = list(self.learning_events)[-100:]
            
            # Performance statistics
            all_scores = [event.performance_score for event in recent_events]
            avg_performance = np.mean(all_scores)
            performance_std = np.std(all_scores)
            
            # Learning rate statistics
            all_rates = [event.learning_rate for event in recent_events]
            avg_learning_rate = np.mean(all_rates)
            
            # Task type analysis
            task_analysis = {}
            for task_type in set(event.task_type for event in recent_events):
                task_events = [e for e in recent_events if e.task_type == task_type]
                task_scores = [e.performance_score for e in task_events]
                
                task_analysis[task_type] = {
                    'event_count': len(task_events),
                    'avg_performance': np.mean(task_scores),
                    'performance_trend': self.get_performance_trend(task_type),
                    'latest_score': task_scores[-1] if task_scores else 0.0
                }
                
            return {
                'report_timestamp': time.time(),
                'total_learning_events': total_events,
                'recent_events_analyzed': len(recent_events),
                'overall_performance': {
                    'average': avg_performance,
                    'std_deviation': performance_std,
                    'min': min(all_scores),
                    'max': max(all_scores)
                },
                'learning_dynamics': {
                    'average_learning_rate': avg_learning_rate,
                    'convergence_analysis': self.get_convergence_analysis()
                },
                'task_analysis': task_analysis,
                'goal_summary': self.get_goal_summary(),
                'time_span_hours': (recent_events[-1].timestamp - recent_events[0].timestamp) / 3600 if len(recent_events) > 1 else 0
            }
            
    def visualize_learning_progress(self, 
                                  task_types: Optional[List[str]] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of learning progress."""
        if task_types is None:
            task_types = list(set(event.task_type for event in self.learning_events))
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Learning Progress Analysis', fontsize=16)
        
        # Learning curves
        ax1 = axes[0, 0]
        for task_type in task_types[:5]:  # Limit to 5 task types for readability
            timestamps, scores = self.get_learning_curve(task_type)
            if timestamps:
                dates = [datetime.fromtimestamp(ts) for ts in timestamps]
                ax1.plot(dates, scores, label=task_type, alpha=0.7)
                
        ax1.set_title('Learning Curves by Task Type')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Performance Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance distribution
        ax2 = axes[0, 1]
        all_scores = [event.performance_score for event in self.learning_events]
        if all_scores:
            ax2.hist(all_scores, bins=30, alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(all_scores), color='red', linestyle='--', label=f'Mean: {np.mean(all_scores):.3f}')
            ax2.set_title('Performance Score Distribution')
            ax2.set_xlabel('Performance Score')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        # Learning rate over time
        ax3 = axes[1, 0]
        timestamps = [event.timestamp for event in self.learning_events]
        learning_rates = [event.learning_rate for event in self.learning_events]
        if timestamps:
            dates = [datetime.fromtimestamp(ts) for ts in timestamps]
            ax3.plot(dates, learning_rates, alpha=0.7, color='green')
            ax3.set_title('Learning Rate Over Time')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Learning Rate')
            ax3.grid(True, alpha=0.3)
            
        # Goal progress
        ax4 = axes[1, 1]
        with self._lock:
            goal_names = list(self.goals.keys())[:10]  # Limit to 10 goals
            completion_percentages = [self.goals[name].completion_percentage for name in goal_names]
            
            if goal_names:
                bars = ax4.barh(goal_names, completion_percentages)
                ax4.set_title('Goal Completion Progress')
                ax4.set_xlabel('Completion Percentage')
                ax4.set_xlim(0, 100)
                
                # Color bars based on completion
                for i, (bar, percentage) in enumerate(zip(bars, completion_percentages)):
                    if percentage >= 100:
                        bar.set_color('green')
                    elif percentage >= 50:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
                        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def export_learning_data(self, filepath: str) -> None:
        """Export learning data to JSON file."""
        with self._lock:
            data = {
                'export_timestamp': time.time(),
                'learning_events': [event.to_dict() for event in self.learning_events],
                'goals': [goal.to_dict() for goal in self.goals.values()],
                'convergence_history': self.convergence_history,
                'learning_report': self.generate_learning_report()
            }
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time learning metrics for monitoring dashboard."""
        with self._lock:
            if not self.learning_events:
                return {}
                
            recent_events = list(self.learning_events)[-10:]
            
            return {
                'current_performance': recent_events[-1].performance_score if recent_events else 0.0,
                'current_learning_rate': recent_events[-1].learning_rate if recent_events else 0.0,
                'recent_avg_performance': np.mean([e.performance_score for e in recent_events]),
                'performance_trend': 'improving' if len(recent_events) >= 2 and recent_events[-1].performance_score > recent_events[-2].performance_score else 'stable',
                'active_goals_count': len([g for g in self.goals.values() if not g.is_completed]),
                'total_learning_events': len(self.learning_events),
                'last_update': recent_events[-1].timestamp if recent_events else 0.0
            }
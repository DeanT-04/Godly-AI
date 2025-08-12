"""
Tests for the learning progress tracking functionality.

This module tests:
- Learning event recording and analysis
- Goal progress tracking
- Learning curve generation
- Performance trend analysis
"""

import pytest
import time
import tempfile
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock, patch

from src.monitoring.learning_tracker import (
    LearningProgressTracker, LearningEvent, GoalProgress
)


class TestLearningEvent:
    """Test LearningEvent data structure."""
    
    def test_learning_event_creation(self):
        """Test LearningEvent creation and serialization."""
        event = LearningEvent(
            timestamp=time.time(),
            task_type="pattern_recognition",
            performance_score=0.85,
            learning_rate=0.01,
            episode_count=100,
            convergence_metric=0.05,
            metadata={"batch_size": 32, "optimizer": "adam"}
        )
        
        # Test serialization
        data = event.to_dict()
        assert isinstance(data, dict)
        assert data['task_type'] == "pattern_recognition"
        assert data['performance_score'] == 0.85
        assert data['metadata']['batch_size'] == 32
        
    def test_learning_event_default_metadata(self):
        """Test LearningEvent with default metadata."""
        event = LearningEvent(
            timestamp=time.time(),
            task_type="test_task",
            performance_score=0.5,
            learning_rate=0.001,
            episode_count=50,
            convergence_metric=0.1
        )
        
        assert event.metadata == {}


class TestGoalProgress:
    """Test GoalProgress data structure."""
    
    def test_goal_progress_creation(self):
        """Test GoalProgress creation and serialization."""
        goal = GoalProgress(
            goal_id="goal_001",
            goal_description="Achieve 90% accuracy on task A",
            creation_time=time.time(),
            target_performance=0.9,
            current_performance=0.7,
            completion_percentage=77.8
        )
        
        data = goal.to_dict()
        assert isinstance(data, dict)
        assert data['goal_id'] == "goal_001"
        assert data['target_performance'] == 0.9
        assert data['completion_percentage'] == 77.8
        assert data['is_completed'] is False
        
    def test_goal_completion_tracking(self):
        """Test goal completion status tracking."""
        goal = GoalProgress(
            goal_id="test_goal",
            goal_description="Test goal",
            creation_time=time.time(),
            target_performance=1.0,
            current_performance=1.0,
            completion_percentage=100.0,
            is_completed=True,
            completion_time=time.time()
        )
        
        assert goal.is_completed is True
        assert goal.completion_time is not None


class TestLearningProgressTracker:
    """Test LearningProgressTracker functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tracker = LearningProgressTracker(max_history=1000)
        
    def test_tracker_initialization(self):
        """Test LearningProgressTracker initialization."""
        assert self.tracker.max_history == 1000
        assert len(self.tracker.learning_events) == 0
        assert len(self.tracker.goals) == 0
        
    def test_record_learning_event(self):
        """Test learning event recording."""
        self.tracker.record_learning_event(
            task_type="classification",
            performance_score=0.8,
            learning_rate=0.01,
            episode_count=50,
            convergence_metric=0.02,
            batch_size=32
        )
        
        assert len(self.tracker.learning_events) == 1
        event = self.tracker.learning_events[0]
        assert event.task_type == "classification"
        assert event.performance_score == 0.8
        assert event.metadata['batch_size'] == 32
        
    def test_create_goal(self):
        """Test goal creation."""
        self.tracker.create_goal(
            goal_id="test_goal",
            description="Test goal description",
            target_performance=0.95
        )
        
        assert "test_goal" in self.tracker.goals
        goal = self.tracker.goals["test_goal"]
        assert goal.goal_description == "Test goal description"
        assert goal.target_performance == 0.95
        assert goal.current_performance == 0.0
        
    def test_update_goal_progress(self):
        """Test goal progress updates."""
        # Create a goal
        self.tracker.create_goal("test_goal", "Test", 1.0)
        
        # Update progress
        self.tracker.update_goal_progress("test_goal", 0.7)
        
        goal = self.tracker.goals["test_goal"]
        assert goal.current_performance == 0.7
        assert goal.completion_percentage == 70.0
        assert goal.is_completed is False
        
        # Complete the goal
        self.tracker.update_goal_progress("test_goal", 1.0)
        
        goal = self.tracker.goals["test_goal"]
        assert goal.current_performance == 1.0
        assert goal.completion_percentage == 100.0
        assert goal.is_completed is True
        assert goal.completion_time is not None
        
    def test_get_learning_curve(self):
        """Test learning curve generation."""
        # Add multiple learning events
        base_time = time.time()
        for i in range(10):
            self.tracker.record_learning_event(
                task_type="test_task",
                performance_score=0.5 + i * 0.05,  # Improving performance
                learning_rate=0.01,
                episode_count=i * 10,
                convergence_metric=0.1 - i * 0.01
            )
            time.sleep(0.001)  # Small delay to ensure different timestamps
            
        timestamps, scores = self.tracker.get_learning_curve("test_task")
        
        assert len(timestamps) == 10
        assert len(scores) == 10
        assert scores[0] == 0.5
        assert scores[-1] == 0.95
        
    def test_get_performance_trend(self):
        """Test performance trend analysis."""
        # Add learning events with improving trend
        for i in range(20):
            self.tracker.record_learning_event(
                task_type="improving_task",
                performance_score=0.3 + i * 0.03,  # Linear improvement
                learning_rate=0.01,
                episode_count=i * 5,
                convergence_metric=0.1
            )
            
        trend = self.tracker.get_performance_trend("improving_task")
        
        assert 'trend' in trend
        assert 'confidence' in trend
        assert 'recent_avg' in trend
        assert trend['trend'] > 0  # Should be positive (improving)
        assert trend['confidence'] > 0.9  # Should be high confidence for linear trend
        
    def test_convergence_analysis(self):
        """Test convergence pattern analysis."""
        # Add events with some convergences
        for i in range(15):
            convergence_metric = 0.1 if i < 10 else 0.005  # Converge after 10 episodes
            self.tracker.record_learning_event(
                task_type="converging_task",
                performance_score=0.8,
                learning_rate=0.01,
                episode_count=i,
                convergence_metric=convergence_metric
            )
            
        analysis = self.tracker.get_convergence_analysis()
        
        assert 'total_convergences' in analysis
        assert 'convergence_rate' in analysis
        assert 'task_statistics' in analysis
        assert analysis['total_convergences'] == 5  # Last 5 events converged
        
    def test_goal_summary(self):
        """Test goal summary generation."""
        # Create multiple goals with different states
        self.tracker.create_goal("goal1", "Completed goal", 1.0)
        self.tracker.create_goal("goal2", "Active goal", 0.8)
        self.tracker.create_goal("goal3", "Another active goal", 0.9)
        
        # Complete one goal
        self.tracker.update_goal_progress("goal1", 1.0)
        self.tracker.update_goal_progress("goal2", 0.5)
        self.tracker.update_goal_progress("goal3", 0.7)
        
        summary = self.tracker.get_goal_summary()
        
        assert summary['total_goals'] == 3
        assert summary['completed_goals'] == 1
        assert summary['active_goals'] == 2
        assert summary['completion_rate'] == 1/3
        assert len(summary['goals']) == 3
        
    def test_generate_learning_report(self):
        """Test comprehensive learning report generation."""
        # Add diverse learning events
        task_types = ["classification", "regression", "reinforcement"]
        
        for task_type in task_types:
            for i in range(10):
                self.tracker.record_learning_event(
                    task_type=task_type,
                    performance_score=0.6 + i * 0.03,
                    learning_rate=0.01,
                    episode_count=i * 5,
                    convergence_metric=0.05
                )
                
        # Create some goals
        self.tracker.create_goal("goal1", "Test goal 1", 0.9)
        self.tracker.create_goal("goal2", "Test goal 2", 0.95)
        
        report = self.tracker.generate_learning_report()
        
        assert 'report_timestamp' in report
        assert 'total_learning_events' in report
        assert 'overall_performance' in report
        assert 'learning_dynamics' in report
        assert 'task_analysis' in report
        assert 'goal_summary' in report
        
        # Check task analysis
        assert len(report['task_analysis']) == 3
        for task_type in task_types:
            assert task_type in report['task_analysis']
            
    def test_visualize_learning_progress(self):
        """Test learning progress visualization."""
        # Add test data
        for i in range(20):
            self.tracker.record_learning_event(
                task_type="visual_test",
                performance_score=0.4 + i * 0.02,
                learning_rate=0.01 - i * 0.0005,
                episode_count=i * 10,
                convergence_metric=0.1
            )
            
        # Create a goal
        self.tracker.create_goal("visual_goal", "Test visualization", 0.9)
        self.tracker.update_goal_progress("visual_goal", 0.7)
        
        # Generate visualization
        fig = self.tracker.visualize_learning_progress(
            task_types=["visual_test"]
        )
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4  # Should have 4 subplots
        
        plt.close(fig)  # Clean up
        
    def test_export_learning_data(self):
        """Test learning data export."""
        temp_dir = tempfile.mkdtemp()
        export_file = Path(temp_dir) / "learning_export.json"
        
        # Add test data
        self.tracker.record_learning_event(
            task_type="export_test",
            performance_score=0.8,
            learning_rate=0.01,
            episode_count=100,
            convergence_metric=0.02
        )
        
        self.tracker.create_goal("export_goal", "Export test goal", 0.9)
        
        # Export data
        self.tracker.export_learning_data(str(export_file))
        
        # Verify export
        assert export_file.exists()
        
        with open(export_file, 'r') as f:
            data = json.load(f)
            
        assert 'export_timestamp' in data
        assert 'learning_events' in data
        assert 'goals' in data
        assert 'learning_report' in data
        assert len(data['learning_events']) == 1
        assert len(data['goals']) == 1
        
    def test_get_real_time_metrics(self):
        """Test real-time metrics generation."""
        # Initially empty
        metrics = self.tracker.get_real_time_metrics()
        assert metrics == {}
        
        # Add some events
        for i in range(5):
            self.tracker.record_learning_event(
                task_type="realtime_test",
                performance_score=0.6 + i * 0.1,
                learning_rate=0.01,
                episode_count=i * 10,
                convergence_metric=0.05
            )
            
        self.tracker.create_goal("rt_goal", "Real-time goal", 0.9)
        
        metrics = self.tracker.get_real_time_metrics()
        
        assert 'current_performance' in metrics
        assert 'current_learning_rate' in metrics
        assert 'recent_avg_performance' in metrics
        assert 'performance_trend' in metrics
        assert 'active_goals_count' in metrics
        assert 'total_learning_events' in metrics
        assert 'last_update' in metrics
        
        assert metrics['current_performance'] == 1.0  # Last recorded performance
        assert metrics['active_goals_count'] == 1
        assert metrics['total_learning_events'] == 5
        
    def test_max_history_limit(self):
        """Test that learning events respect max_history limit."""
        tracker = LearningProgressTracker(max_history=5)
        
        # Add more events than the limit
        for i in range(10):
            tracker.record_learning_event(
                task_type="limit_test",
                performance_score=0.5,
                learning_rate=0.01,
                episode_count=i,
                convergence_metric=0.1
            )
            
        assert len(tracker.learning_events) == 5
        
    def test_multiple_task_types(self):
        """Test handling of multiple task types."""
        task_types = ["classification", "regression", "clustering", "reinforcement"]
        
        for task_type in task_types:
            for i in range(5):
                self.tracker.record_learning_event(
                    task_type=task_type,
                    performance_score=0.5 + i * 0.1,
                    learning_rate=0.01,
                    episode_count=i * 10,
                    convergence_metric=0.05
                )
                
        # Check that all task types are tracked
        for task_type in task_types:
            timestamps, scores = self.tracker.get_learning_curve(task_type)
            assert len(timestamps) == 5
            assert len(scores) == 5
            
        # Check task performance tracking
        assert len(self.tracker.task_performance) == 4
        for task_type in task_types:
            assert task_type in self.tracker.task_performance
            assert len(self.tracker.task_performance[task_type]) == 5
            
    def test_performance_trend_edge_cases(self):
        """Test performance trend analysis edge cases."""
        # Test with insufficient data
        self.tracker.record_learning_event(
            task_type="insufficient_data",
            performance_score=0.5,
            learning_rate=0.01,
            episode_count=1,
            convergence_metric=0.1
        )
        
        trend = self.tracker.get_performance_trend("insufficient_data")
        assert trend['trend'] == 0.0
        assert trend['confidence'] == 0.0
        
        # Test with non-existent task
        trend = self.tracker.get_performance_trend("non_existent_task")
        assert trend['trend'] == 0.0
        assert trend['confidence'] == 0.0
        
    def test_goal_update_edge_cases(self):
        """Test goal update edge cases."""
        # Update non-existent goal
        self.tracker.update_goal_progress("non_existent", 0.5)
        # Should not raise error, just ignore
        
        # Test goal over-completion
        self.tracker.create_goal("over_goal", "Over completion test", 1.0)
        self.tracker.update_goal_progress("over_goal", 1.5)  # 150% performance
        
        goal = self.tracker.goals["over_goal"]
        assert goal.current_performance == 1.5
        assert goal.completion_percentage == 100.0  # Capped at 100%
        assert goal.is_completed is True
        
    def test_thread_safety(self):
        """Test thread safety of learning tracker."""
        import threading
        
        def record_events():
            for i in range(50):
                self.tracker.record_learning_event(
                    task_type="thread_test",
                    performance_score=0.5 + i * 0.01,
                    learning_rate=0.01,
                    episode_count=i,
                    convergence_metric=0.05
                )
                
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=record_events)
            threads.append(thread)
            thread.start()
            
        # Wait for completion
        for thread in threads:
            thread.join()
            
        # Verify all events were recorded
        assert len(self.tracker.learning_events) == 150  # 3 threads * 50 events
        
    def test_convergence_detection(self):
        """Test convergence detection accuracy."""
        # Add events that don't converge
        for i in range(10):
            self.tracker.record_learning_event(
                task_type="no_convergence",
                performance_score=0.5,
                learning_rate=0.01,
                episode_count=i,
                convergence_metric=0.1  # Above convergence threshold
            )
            
        # Add events that converge
        for i in range(5):
            self.tracker.record_learning_event(
                task_type="convergence",
                performance_score=0.8,
                learning_rate=0.01,
                episode_count=i,
                convergence_metric=0.005  # Below convergence threshold
            )
            
        analysis = self.tracker.get_convergence_analysis()
        
        # Should only detect convergences from the second set
        assert analysis['total_convergences'] == 5
        assert 'convergence' in analysis['task_statistics']
        assert analysis['task_statistics']['convergence']['count'] == 5
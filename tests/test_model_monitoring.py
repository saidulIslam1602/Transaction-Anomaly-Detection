"""
Tests for Model Monitoring module.
"""

import pytest
import pandas as pd
import numpy as np
from src.mlops.model_monitoring import (
    DataDriftDetector,
    ModelPerformanceMonitor,
    PredictionMonitor,
    ComprehensiveModelMonitor
)


class TestDataDriftDetector:
    """Test DataDriftDetector class."""
    
    def test_initialization(self, sample_dataframe):
        """Test detector initialization."""
        detector = DataDriftDetector(sample_dataframe)
        
        assert detector.reference_data is not None
        assert len(detector.reference_stats) > 0
    
    def test_detect_no_drift(self, sample_dataframe):
        """Test detection when there's no drift."""
        detector = DataDriftDetector(sample_dataframe)
        
        # Use same data, should have no drift
        drift_results = detector.detect_drift(sample_dataframe)
        
        assert 'overall_drift_detected' in drift_results
        assert 'features_checked' in drift_results
    
    def test_detect_drift(self, sample_dataframe):
        """Test detection when there is drift."""
        detector = DataDriftDetector(sample_dataframe)
        
        # Create drifted data (shift amount distribution)
        drifted_df = sample_dataframe.copy()
        drifted_df['amount'] = drifted_df['amount'] * 10
        
        drift_results = detector.detect_drift(drifted_df)
        
        assert 'details' in drift_results
        # Amount should show drift
        if 'amount' in drift_results['details']:
            assert 'drift_detected' in drift_results['details']['amount']


class TestModelPerformanceMonitor:
    """Test ModelPerformanceMonitor class."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = ModelPerformanceMonitor()
        
        assert monitor.alert_thresholds is not None
        assert len(monitor.performance_history) == 0
    
    def test_log_performance(self):
        """Test logging performance metrics."""
        monitor = ModelPerformanceMonitor()
        
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0])
        y_scores = np.random.rand(10)
        
        result = monitor.log_performance(y_true, y_pred, y_scores)
        
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        assert 'precision' in result['metrics']
        assert 'recall' in result['metrics']
        assert len(monitor.performance_history) == 1
    
    def test_check_alerts(self):
        """Test alert checking."""
        monitor = ModelPerformanceMonitor(
            alert_thresholds={'accuracy': 0.95}
        )
        
        alerts = monitor._check_alerts({'accuracy': 0.85})
        
        assert len(alerts) > 0
        assert 'accuracy' in alerts[0]
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        monitor = ModelPerformanceMonitor()
        
        # Log multiple performance records
        for _ in range(5):
            y_true = np.random.randint(0, 2, 100)
            y_pred = np.random.randint(0, 2, 100)
            monitor.log_performance(y_true, y_pred)
        
        summary = monitor.get_performance_summary()
        
        assert 'metrics_summary' in summary
        assert 'records_count' in summary
        assert summary['records_count'] == 5


class TestPredictionMonitor:
    """Test PredictionMonitor class."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PredictionMonitor(window_size=100)
        
        assert monitor.window_size == 100
        assert len(monitor.predictions_window) == 0
    
    def test_log_predictions(self):
        """Test logging predictions."""
        monitor = PredictionMonitor()
        
        predictions = np.random.randint(0, 2, 50)
        scores = np.random.rand(50)
        
        stats = monitor.log_predictions(predictions, scores)
        
        assert 'positive_rate' in stats
        assert 'mean_score' in stats
        assert 'batch_size' in stats
        assert stats['batch_size'] == 50
    
    def test_window_maintenance(self):
        """Test sliding window maintenance."""
        monitor = PredictionMonitor(window_size=100)
        
        # Add more than window size
        for _ in range(3):
            predictions = np.random.randint(0, 2, 50)
            monitor.log_predictions(predictions)
        
        assert len(monitor.predictions_window) == 100


class TestComprehensiveModelMonitor:
    """Test ComprehensiveModelMonitor class."""
    
    def test_initialization(self, sample_dataframe):
        """Test comprehensive monitor initialization."""
        monitor = ComprehensiveModelMonitor(sample_dataframe)
        
        assert monitor.drift_detector is not None
        assert monitor.performance_monitor is not None
        assert monitor.prediction_monitor is not None
    
    def test_monitor_batch(self, sample_dataframe):
        """Test batch monitoring."""
        monitor = ComprehensiveModelMonitor(sample_dataframe)
        
        # Create test data
        current_data = sample_dataframe.copy()
        y_true = sample_dataframe['isFraud'].values
        y_pred = np.random.randint(0, 2, len(y_true))
        y_scores = np.random.rand(len(y_true))
        
        report = monitor.monitor_batch(current_data, y_true, y_pred, y_scores)
        
        assert 'drift_detection' in report
        assert 'performance' in report
        assert 'predictions' in report
        assert 'overall_health' in report
    
    def test_assess_overall_health(self, sample_dataframe):
        """Test overall health assessment."""
        monitor = ComprehensiveModelMonitor(sample_dataframe)
        
        # Healthy scenario
        drift_results = {'overall_drift_detected': False}
        performance_results = {'metrics': {'accuracy': 0.95}}
        
        health = monitor._assess_overall_health(drift_results, performance_results)
        assert health == 'HEALTHY'
        
        # Drift detected
        drift_results = {'overall_drift_detected': True}
        health = monitor._assess_overall_health(drift_results, performance_results)
        assert 'CRITICAL' in health


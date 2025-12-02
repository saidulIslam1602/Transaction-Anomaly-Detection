"""
Model monitoring and drift detection for production ML systems.

This module provides comprehensive monitoring of model performance, data drift,
and concept drift to ensure models maintain accuracy in production.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import json

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detect data drift in model inputs.
    
    Monitors feature distributions and statistical properties to detect
    when input data distribution changes significantly.
    """
    
    def __init__(self, reference_data: pd.DataFrame, significance_level: float = 0.05):
        """
        Initialize drift detector with reference data.
        
        Args:
            reference_data: Reference dataset for comparison
            significance_level: Statistical significance level for drift tests
        """
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.reference_stats = self._compute_statistics(reference_data)
        
        logger.info("Data drift detector initialized")
    
    def _compute_statistics(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Compute statistical summaries for reference data."""
        stats_dict = {}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            stats_dict[column] = {
                'mean': data[column].mean(),
                'std': data[column].std(),
                'min': data[column].min(),
                'max': data[column].max(),
                'q25': data[column].quantile(0.25),
                'q50': data[column].quantile(0.50),
                'q75': data[column].quantile(0.75)
            }
        
        return stats_dict
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current dataset to check for drift
            
        Returns:
            Dictionary with drift detection results
        """
        if not SCIPY_AVAILABLE:
            logger.warning("SciPy not available. Drift detection limited.")
            return {}
        
        drift_results = {}
        features_with_drift = []
        
        for column in self.reference_stats.keys():
            if column not in current_data.columns:
                continue
            
            # Kolmogorov-Smirnov test for distribution drift
            reference_values = self.reference_data[column].dropna()
            current_values = current_data[column].dropna()
            
            ks_statistic, p_value = stats.ks_2samp(reference_values, current_values)
            
            drift_detected = p_value < self.significance_level
            
            # Calculate PSI (Population Stability Index)
            psi = self._calculate_psi(reference_values, current_values)
            
            drift_results[column] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'drift_detected': drift_detected,
                'psi': float(psi),
                'current_mean': float(current_values.mean()),
                'reference_mean': float(self.reference_stats[column]['mean']),
                'mean_change_pct': float(
                    (current_values.mean() - self.reference_stats[column]['mean']) /
                    (self.reference_stats[column]['mean'] + 1e-10) * 100
                )
            }
            
            if drift_detected:
                features_with_drift.append(column)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'features_checked': len(drift_results),
            'features_with_drift': len(features_with_drift),
            'drift_features': features_with_drift,
            'overall_drift_detected': len(features_with_drift) > 0,
            'details': drift_results
        }
        
        if features_with_drift:
            logger.warning(f"Data drift detected in {len(features_with_drift)} features: {features_with_drift}")
        
        return summary
    
    def _calculate_psi(self,
                      reference: pd.Series,
                      current: pd.Series,
                      buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures the shift in distribution:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change
        """
        # Create bins based on reference data
        breakpoints = np.linspace(
            reference.min(),
            reference.max(),
            buckets + 1
        )
        
        # Calculate distributions
        reference_dist = np.histogram(reference, bins=breakpoints)[0] / len(reference)
        current_dist = np.histogram(current, bins=breakpoints)[0] / len(current)
        
        # Avoid division by zero
        reference_dist = np.where(reference_dist == 0, 0.0001, reference_dist)
        current_dist = np.where(current_dist == 0, 0.0001, current_dist)
        
        # Calculate PSI
        psi = np.sum((current_dist - reference_dist) * np.log(current_dist / reference_dist))
        
        return psi


class ModelPerformanceMonitor:
    """
    Monitor model performance metrics over time.
    
    Tracks accuracy, precision, recall, and other metrics to detect
    model degradation in production.
    """
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None, config: Optional[Dict] = None):
        """
        Initialize performance monitor.
        
        Args:
            alert_thresholds: Dictionary of metric thresholds for alerts
            config: Configuration dictionary with monitoring settings
        """
        # Use provided alert_thresholds, or load from config, or use defaults
        if alert_thresholds is not None:
            self.alert_thresholds = alert_thresholds
        elif config is not None:
            self.alert_thresholds = config.get('monitoring', {}).get('performance_monitoring', {}).get('alert_thresholds', {
                'accuracy': 0.90,
                'precision': 0.85,
                'recall': 0.80,
                'f1_score': 0.85,
                'auc': 0.90
            })
        else:
            self.alert_thresholds = {
                'accuracy': 0.90,
                'precision': 0.85,
                'recall': 0.80,
                'f1_score': 0.85,
                'auc': 0.90
            }
        
        self.performance_history = []
        
        # Load trend detection thresholds from config
        self.config = config or {}
        monitoring_config = self.config.get('model_monitoring', {})
        self.trend_improving = monitoring_config.get('performance_thresholds', {}).get('trend_improving', 0.01)
        self.trend_degrading = monitoring_config.get('performance_thresholds', {}).get('trend_degrading', -0.01)
        
        logger.info("Model performance monitor initialized")
    
    def log_performance(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       y_scores: Optional[np.ndarray] = None,
                       metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Log model performance for a batch of predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores (probabilities)
            metadata: Additional metadata (e.g., timestamp, model version)
            
        Returns:
            Dictionary with performance metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
        }
        
        if y_scores is not None and len(np.unique(y_true)) > 1:
            metrics['auc'] = float(roc_auc_score(y_true, y_scores))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            })
        
        # Add metadata
        performance_record = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {},
            'sample_size': len(y_true)
        }
        
        # Check for alerts
        alerts = self._check_alerts(metrics)
        if alerts:
            performance_record['alerts'] = alerts
            logger.warning(f"Performance alerts triggered: {alerts}")
        
        # Store in history
        self.performance_history.append(performance_record)
        
        return performance_record
    
    def _check_alerts(self, metrics: Dict[str, float]) -> List[str]:
        """Check if any metrics are below alert thresholds."""
        alerts = []
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics and metrics[metric] < threshold:
                alerts.append(f"{metric} ({metrics[metric]:.3f}) below threshold ({threshold})")
        
        return alerts
    
    def get_performance_summary(self,
                               time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get summary of model performance over a time window.
        
        Args:
            time_window: Time window for summary (None for all history)
            
        Returns:
            Dictionary with performance summary statistics
        """
        if not self.performance_history:
            return {'status': 'no_data'}
        
        # Filter by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            relevant_records = [
                r for r in self.performance_history
                if datetime.fromisoformat(r['timestamp']) >= cutoff_time
            ]
        else:
            relevant_records = self.performance_history
        
        if not relevant_records:
            return {'status': 'no_data_in_window'}
        
        # Calculate statistics for each metric
        metrics_over_time = defaultdict(list)
        
        for record in relevant_records:
            for metric, value in record['metrics'].items():
                metrics_over_time[metric].append(value)
        
        summary = {
            'time_window': str(time_window) if time_window else 'all',
            'records_count': len(relevant_records),
            'metrics_summary': {}
        }
        
        for metric, values in metrics_over_time.items():
            summary['metrics_summary'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'current': float(values[-1]),
                'trend': self._calculate_trend(values)
            }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a metric using config thresholds."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > self.trend_improving:
            return 'improving'
        elif slope < self.trend_degrading:
            return 'degrading'
        else:
            return 'stable'
    
    def export_metrics(self, filepath: str) -> None:
        """Export performance history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        logger.info(f"Performance metrics exported to {filepath}")


class PredictionMonitor:
    """
    Monitor individual predictions for anomalies and patterns.
    
    Tracks prediction distributions and identifies unusual prediction patterns.
    """
    
    def __init__(self, window_size: int = 1000, config: Optional[Dict] = None):
        """
        Initialize prediction monitor.
        
        Args:
            window_size: Size of sliding window for monitoring
            config: Configuration dictionary with monitoring settings
        """
        self.window_size = window_size
        self.predictions_window = []
        self.scores_window = []
        
        # Load config values
        self.config = config or {}
        pred_monitoring = self.config.get('model_monitoring', {}).get('prediction_monitoring', {})
        self.anomaly_threshold_std = pred_monitoring.get('anomaly_threshold_std', 2.0)
        self.min_samples = pred_monitoring.get('min_samples_for_detection', 100)
        
        logger.info("Prediction monitor initialized")
    
    def log_predictions(self,
                       predictions: np.ndarray,
                       scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Log batch of predictions and analyze patterns.
        
        Args:
            predictions: Array of predictions
            scores: Array of prediction scores
            
        Returns:
            Dictionary with monitoring statistics
        """
        # Add to windows
        self.predictions_window.extend(predictions.tolist())
        if scores is not None:
            self.scores_window.extend(scores.tolist())
        
        # Maintain window size
        if len(self.predictions_window) > self.window_size:
            self.predictions_window = self.predictions_window[-self.window_size:]
            self.scores_window = self.scores_window[-self.window_size:]
        
        # Calculate statistics
        stats = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': len(predictions),
            'window_size': len(self.predictions_window),
            'positive_rate': float(np.mean(self.predictions_window)),
            'positive_count': int(np.sum(self.predictions_window))
        }
        
        if self.scores_window:
            stats.update({
                'mean_score': float(np.mean(self.scores_window)),
                'std_score': float(np.std(self.scores_window)),
                'score_percentiles': {
                    'p25': float(np.percentile(self.scores_window, 25)),
                    'p50': float(np.percentile(self.scores_window, 50)),
                    'p75': float(np.percentile(self.scores_window, 75)),
                    'p95': float(np.percentile(self.scores_window, 95))
                }
            })
        
        return stats
    
    def detect_prediction_anomalies(self,
                                   current_positive_rate: float,
                                   threshold_std: Optional[float] = None) -> Dict[str, Any]:
        """
        Detect anomalies in prediction patterns.
        
        Args:
            current_positive_rate: Current positive prediction rate
            threshold_std: Standard deviation threshold for anomaly (uses config if not provided)
            
        Returns:
            Dictionary with anomaly detection results
        """
        if len(self.predictions_window) < self.min_samples:
            return {'status': 'insufficient_data'}
        
        # Use provided threshold or config value
        if threshold_std is None:
            threshold_std = self.anomaly_threshold_std
        
        historical_rate = np.mean(self.predictions_window)
        historical_std = np.std(self.predictions_window)
        
        z_score = (current_positive_rate - historical_rate) / (historical_std + 1e-10)
        
        anomaly_detected = abs(z_score) > threshold_std
        
        result = {
            'anomaly_detected': anomaly_detected,
            'current_rate': float(current_positive_rate),
            'historical_rate': float(historical_rate),
            'z_score': float(z_score),
            'threshold': threshold_std
        }
        
        if anomaly_detected:
            direction = 'higher' if z_score > 0 else 'lower'
            logger.warning(f"Prediction anomaly detected: {direction} than expected")
        
        return result


class ComprehensiveModelMonitor:
    """
    Comprehensive monitoring system combining drift detection and performance monitoring.
    
    Provides a unified interface for monitoring all aspects of model health.
    """
    
    def __init__(self,
                 reference_data: pd.DataFrame,
                 alert_thresholds: Optional[Dict] = None,
                 config: Optional[Dict] = None):
        """
        Initialize comprehensive monitor.
        
        Args:
            reference_data: Reference dataset for drift detection
            alert_thresholds: Performance alert thresholds
            config: Configuration dictionary with monitoring settings
        """
        # Use significance level from config if available
        significance_level = 0.05
        if config:
            significance_level = config.get('model_monitoring', {}).get('drift_detection', {}).get('significance_level', 0.05)
        
        self.drift_detector = DataDriftDetector(reference_data, significance_level=significance_level)
        self.performance_monitor = ModelPerformanceMonitor(alert_thresholds, config=config)
        
        # Use window size from config if available
        window_size = 1000
        if config:
            window_size = config.get('monitoring', {}).get('prediction_monitoring', {}).get('window_size', 1000)
        
        self.prediction_monitor = PredictionMonitor(window_size=window_size, config=config)
        
        logger.info("Comprehensive model monitor initialized")
    
    def monitor_batch(self,
                     current_data: pd.DataFrame,
                     y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive monitoring on a batch of data and predictions.
        
        Args:
            current_data: Current input data
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores
            
        Returns:
            Dictionary with comprehensive monitoring results
        """
        # Drift detection
        drift_results = self.drift_detector.detect_drift(current_data)
        
        # Performance monitoring
        performance_results = self.performance_monitor.log_performance(
            y_true, y_pred, y_scores
        )
        
        # Prediction monitoring
        prediction_results = self.prediction_monitor.log_predictions(y_pred, y_scores)
        
        # Combine results
        monitoring_report = {
            'timestamp': datetime.now().isoformat(),
            'drift_detection': drift_results,
            'performance': performance_results,
            'predictions': prediction_results,
            'overall_health': self._assess_overall_health(
                drift_results,
                performance_results
            )
        }
        
        return monitoring_report
    
    def _assess_overall_health(self,
                              drift_results: Dict,
                              performance_results: Dict) -> str:
        """Assess overall model health status."""
        if drift_results.get('overall_drift_detected'):
            return 'CRITICAL: Data drift detected'
        
        if 'alerts' in performance_results:
            return 'WARNING: Performance degradation'
        
        return 'HEALTHY'
    
    def generate_monitoring_report(self,
                                  time_window: Optional[timedelta] = None) -> str:
        """
        Generate human-readable monitoring report.
        
        Args:
            time_window: Time window for report
            
        Returns:
            Formatted monitoring report string
        """
        performance_summary = self.performance_monitor.get_performance_summary(time_window)
        
        report = []
        report.append("=" * 60)
        report.append("MODEL MONITORING REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Time Window: {time_window or 'All History'}")
        report.append("")
        
        if 'metrics_summary' in performance_summary:
            report.append("PERFORMANCE METRICS")
            report.append("-" * 60)
            
            for metric, stats in performance_summary['metrics_summary'].items():
                report.append(f"\n{metric.upper()}:")
                report.append(f"  Current: {stats['current']:.4f}")
                report.append(f"  Mean: {stats['mean']:.4f}")
                report.append(f"  Std: {stats['std']:.4f}")
                report.append(f"  Trend: {stats['trend']}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

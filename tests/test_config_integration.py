"""
Test configuration integration across all modules.

This test verifies that all components properly load and use configuration values
instead of hardcoded values.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from services.business_metrics import BusinessMetricsCalculator
from services.merchant_services import (
    MerchantRiskIntelligenceService,
    MerchantAlertPrioritization,
    MerchantOnboardingRiskAssessment
)
from services.feature_store import FeatureStore
from data.preprocessor import TransactionPreprocessor
from mlops.model_monitoring import (
    ModelPerformanceMonitor,
    PredictionMonitor,
    ComprehensiveModelMonitor
)
from models.ml_anomaly_detection import AnomalyDetector
from utils.helpers import load_config


def test_config_loads():
    """Test that configuration file loads successfully."""
    config = load_config()
    
    assert config is not None
    assert isinstance(config, dict)
    assert 'business_metrics' in config
    assert 'merchant_services' in config
    assert 'preprocessing' in config


def test_business_metrics_uses_config():
    """Test BusinessMetricsCalculator uses config values."""
    # Custom config with known values
    config = {
        'business_metrics': {
            'cost_per_alert_review': 25.0,
            'industry_benchmarks': {
                'avg_fraud_rate': 0.05
            }
        }
    }
    
    calc = BusinessMetricsCalculator(config=config)
    
    assert calc.cost_per_alert_review == 25.0
    assert calc.industry_benchmarks['avg_fraud_rate'] == 0.05


def test_merchant_services_uses_config():
    """Test MerchantRiskIntelligenceService uses config values."""
    config = {
        'merchant_services': {
            'risk_thresholds': {
                'high_risk': 9.0,
                'medium_risk': 5.0
            },
            'recommendations': {
                'fraud_rate_threshold': 0.10
            }
        },
        'business_metrics': {
            'industry_benchmarks': {}
        }
    }
    
    service = MerchantRiskIntelligenceService(config=config)
    
    assert service.high_risk_threshold == 9.0
    assert service.medium_risk_threshold == 5.0
    assert service.fraud_rate_threshold == 0.10


def test_alert_prioritization_uses_config():
    """Test MerchantAlertPrioritization uses config values."""
    config = {
        'merchant_services': {
            'alert_prioritization': {
                'amount_thresholds': {
                    'critical': 50000
                },
                'priority_levels': {
                    'critical': 90
                }
            }
        }
    }
    
    prioritizer = MerchantAlertPrioritization(config=config)
    
    assert prioritizer.amount_critical == 50000
    assert prioritizer.critical_threshold == 90


def test_onboarding_assessment_uses_config():
    """Test MerchantOnboardingRiskAssessment uses config values."""
    config = {
        'merchant_services': {
            'onboarding': {
                'risk_score_thresholds': {
                    'reject': 70
                },
                'business_age_thresholds': {
                    'new_business_years': 2
                }
            },
            'onboarding_assessment': {
                'high_risk_industries': [],
                'medium_risk_industries': [],
                'high_risk_countries': [],
                'enhanced_monitoring_countries': []
            }
        }
    }
    
    assessor = MerchantOnboardingRiskAssessment(config=config)
    
    assert assessor.reject_threshold == 70
    assert assessor.new_business_years == 2


def test_feature_store_uses_config():
    """Test FeatureStore uses config values."""
    config = {
        'feature_store': {
            'online_cache_size': 2000
        }
    }
    
    store = FeatureStore(config=config)
    
    assert store.cache_size_limit == 2000


def test_preprocessor_uses_config():
    """Test TransactionPreprocessor uses config values."""
    config = {
        'preprocessing': {
            'outlier_detection': {
                'iqr_multiplier': 5.0,
                'epsilon': 0.001
            }
        }
    }
    
    preprocessor = TransactionPreprocessor(config=config)
    
    assert preprocessor.iqr_multiplier == 5.0
    assert preprocessor.epsilon == 0.001


def test_model_monitoring_uses_config():
    """Test ModelPerformanceMonitor uses config values."""
    config = {
        'monitoring': {
            'performance_monitoring': {
                'alert_thresholds': {
                    'accuracy': 0.95
                }
            }
        },
        'model_monitoring': {
            'performance_thresholds': {
                'trend_improving': 0.02,
                'trend_degrading': -0.02
            }
        }
    }
    
    monitor = ModelPerformanceMonitor(config=config)
    
    assert monitor.alert_thresholds['accuracy'] == 0.95
    assert monitor.trend_improving == 0.02
    assert monitor.trend_degrading == -0.02


def test_prediction_monitor_uses_config():
    """Test PredictionMonitor uses config values."""
    config = {
        'model_monitoring': {
            'prediction_monitoring': {
                'anomaly_threshold_std': 3.0,
                'min_samples_for_detection': 200
            }
        }
    }
    
    monitor = PredictionMonitor(config=config)
    
    assert monitor.anomaly_threshold_std == 3.0
    assert monitor.min_samples == 200


def test_anomaly_detector_accepts_config():
    """Test AnomalyDetector accepts config parameter."""
    config = {
        'ml_models': {
            'xgboost': {
                'enabled': True,
                'n_estimators': 150
            }
        }
    }
    
    detector = AnomalyDetector(config=config)
    
    assert detector.config is not None
    assert detector.config['ml_models']['xgboost']['n_estimators'] == 150


def test_default_values_when_no_config():
    """Test that default values are used when config is not provided."""
    # Test without config
    calc = BusinessMetricsCalculator()
    assert calc.cost_per_alert_review == 10.0  # Default value
    
    service = MerchantRiskIntelligenceService()
    assert service.high_risk_threshold == 7.0  # Default value
    
    store = FeatureStore()
    assert store.cache_size_limit == 1000  # Default value


def test_partial_config():
    """Test that partial config works with defaults for missing values."""
    config = {
        'business_metrics': {
            'cost_per_alert_review': 15.0
            # Missing industry_benchmarks
        }
    }
    
    calc = BusinessMetricsCalculator(config=config)
    
    # Custom value used
    assert calc.cost_per_alert_review == 15.0
    
    # Default benchmarks used
    assert 'avg_fraud_rate' in calc.industry_benchmarks


def test_full_config_integration():
    """Test loading full config and initializing all components."""
    config = load_config()
    
    # Initialize all components with config
    calc = BusinessMetricsCalculator(config=config)
    service = MerchantRiskIntelligenceService(config=config)
    prioritizer = MerchantAlertPrioritization(config=config)
    assessor = MerchantOnboardingRiskAssessment(config=config)
    store = FeatureStore(config=config)
    preprocessor = TransactionPreprocessor(config=config)
    monitor = ModelPerformanceMonitor(config=config)
    detector = AnomalyDetector(config=config)
    
    # Verify all components initialized successfully
    assert calc.cost_per_alert_review > 0
    assert service.high_risk_threshold > 0
    assert prioritizer.amount_critical > 0
    assert assessor.reject_threshold > 0
    assert store.cache_size_limit > 0
    assert preprocessor.iqr_multiplier > 0
    assert len(monitor.alert_thresholds) > 0
    assert detector.config is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


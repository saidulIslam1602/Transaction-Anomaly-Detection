"""
Tests for business metrics calculator.
"""

import pytest
import pandas as pd
import numpy as np

from src.services.business_metrics import BusinessMetricsCalculator


@pytest.fixture
def sample_transactions():
    """Create sample transaction data with fraud labels."""
    np.random.seed(42)
    n = 1000
    
    df = pd.DataFrame({
        'step': np.random.randint(1, 744, n),
        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT'], n),
        'amount': np.random.lognormal(5, 2, n),
        'nameOrig': [f'C{i}' for i in range(n)],
        'nameDest': [f'M{i%50}' for i in range(n)],
        'isFraud': np.random.choice([0, 1], n, p=[0.95, 0.05]),
        'high_risk_flag': np.random.choice([True, False], n, p=[0.1, 0.9]),
        'final_risk_score': np.random.uniform(0, 10, n)
    })
    
    # Ensure some detected fraud
    df.loc[df['isFraud'] == 1, 'high_risk_flag'] = np.random.choice(
        [True, False], 
        size=df['isFraud'].sum(), 
        p=[0.8, 0.2]
    )
    
    return df


def test_business_metrics_calculator_init():
    """Test business metrics calculator initialization."""
    calculator = BusinessMetricsCalculator()
    assert calculator is not None


def test_calculate_fraud_detection_rate(sample_transactions):
    """Test fraud detection rate calculation."""
    calculator = BusinessMetricsCalculator()
    metrics = calculator.calculate_fraud_detection_rate(sample_transactions)
    
    assert 'total_transactions' in metrics
    assert 'total_fraud' in metrics
    assert 'detection_rate_pct' in metrics
    assert 'false_positive_rate_pct' in metrics
    assert metrics['total_transactions'] == len(sample_transactions)


def test_calculate_merchant_risk_distribution(sample_transactions):
    """Test merchant risk distribution calculation."""
    calculator = BusinessMetricsCalculator()
    merchant_metrics = calculator.calculate_merchant_risk_distribution(sample_transactions)
    
    assert not merchant_metrics.empty
    assert 'merchant_id' in merchant_metrics.columns
    assert 'avg_risk_score' in merchant_metrics.columns
    assert 'risk_category' in merchant_metrics.columns


def test_calculate_transaction_volume_trends(sample_transactions):
    """Test transaction volume trends calculation."""
    calculator = BusinessMetricsCalculator()
    trends = calculator.calculate_transaction_volume_trends(sample_transactions, frequency='D')
    
    assert not trends.empty
    assert 'period' in trends.columns
    assert 'transaction_count' in trends.columns
    assert 'total_volume' in trends.columns


def test_calculate_detection_efficiency_metrics(sample_transactions):
    """Test detection efficiency metrics calculation."""
    calculator = BusinessMetricsCalculator()
    metrics = calculator.calculate_detection_efficiency_metrics(sample_transactions)
    
    assert 'total_alerts' in metrics
    assert 'alert_rate_pct' in metrics
    assert 'estimated_review_cost' in metrics


def test_calculate_model_performance_metrics():
    """Test model performance metrics calculation."""
    calculator = BusinessMetricsCalculator()
    
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
    y_scores = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.1, 0.6, 0.9])
    
    metrics = calculator.calculate_model_performance_metrics(y_true, y_pred, y_scores)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'auc' in metrics
    assert metrics['accuracy'] >= 0 and metrics['accuracy'] <= 1


def test_generate_business_summary_report(sample_transactions, tmp_path):
    """Test business summary report generation."""
    calculator = BusinessMetricsCalculator()
    output_path = tmp_path / 'business_report.json'
    
    report = calculator.generate_business_summary_report(
        sample_transactions,
        output_path=str(output_path)
    )
    
    assert 'report_date' in report
    assert 'detection_metrics' in report
    assert 'efficiency_metrics' in report
    assert 'summary' in report
    assert output_path.exists()


def test_calculate_fraud_detection_rate_missing_columns():
    """Test fraud detection rate with missing columns."""
    calculator = BusinessMetricsCalculator()
    df = pd.DataFrame({'amount': [100, 200, 300]})
    
    metrics = calculator.calculate_fraud_detection_rate(df)
    assert metrics == {}  # Should return empty dict when columns missing


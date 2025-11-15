"""
Integration tests for the full pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from services.feature_store import FeatureStore
from mlops.model_monitoring import ComprehensiveModelMonitor
from services.bi_export import BIExportService
from services.business_metrics import BusinessMetricsCalculator
import tempfile
import shutil


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete pipeline."""
    
    def test_feature_extraction_pipeline(self, sample_dataframe):
        """Test feature extraction pipeline."""
        store = FeatureStore()
        
        # Compute features
        features_df = store.batch_compute_features(sample_dataframe.head(100))
        
        assert len(features_df) == 100
        assert 'amount' in features_df.columns
        assert features_df.notna().all().all()  # No NaN values
    
    def test_monitoring_pipeline(self, sample_dataframe):
        """Test monitoring pipeline."""
        # Split data
        train_df = sample_dataframe.iloc[:800]
        test_df = sample_dataframe.iloc[800:]
        
        # Initialize monitor
        monitor = ComprehensiveModelMonitor(train_df)
        
        # Generate predictions (mock)
        y_true = test_df['isFraud'].values
        y_pred = np.random.randint(0, 2, len(y_true))
        y_scores = np.random.rand(len(y_true))
        
        # Monitor batch
        report = monitor.monitor_batch(test_df, y_true, y_pred, y_scores)
        
        assert 'overall_health' in report
        assert report['timestamp'] is not None
    
    def test_end_to_end_latency(self, sample_transaction):
        """Test end-to-end latency for real-time prediction."""
        import time
        
        store = FeatureStore()
        
        start_time = time.time()
        
        # Extract features
        features = store.get_features(sample_transaction)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Should be under 100ms for real-time serving
        assert latency_ms < 100
        assert len(features) > 0
    
    def test_bi_export_integration(self, sample_dataframe):
        """Test BI export integration with full pipeline."""
        # Add required columns for BI export
        sample_dataframe['high_risk_flag'] = np.random.choice([True, False], len(sample_dataframe), p=[0.1, 0.9])
        sample_dataframe['final_risk_score'] = np.random.uniform(0, 10, len(sample_dataframe))
        
        # Create temporary directory for exports
        temp_dir = tempfile.mkdtemp()
        try:
            export_service = BIExportService(output_dir=temp_dir)
            
            # Export transactions
            transaction_file = export_service.export_transactions_for_bi(
                sample_dataframe,
                format='parquet'
            )
            assert Path(transaction_file).exists()
            
            # Export merchant metrics
            merchant_file = export_service.export_merchant_metrics(
                sample_dataframe,
                format='parquet'
            )
            assert Path(merchant_file).exists()
            
            # Export all views
            exports = export_service.export_all_views(sample_dataframe, formats=['parquet'])
            assert len(exports) >= 3  # transactions, merchants, trends, performance
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_business_metrics_integration(self, sample_dataframe):
        """Test business metrics integration with full pipeline."""
        # Add required columns
        sample_dataframe['high_risk_flag'] = np.random.choice([True, False], len(sample_dataframe), p=[0.1, 0.9])
        sample_dataframe['final_risk_score'] = np.random.uniform(0, 10, len(sample_dataframe))
        
        calculator = BusinessMetricsCalculator()
        
        # Calculate detection metrics
        detection_metrics = calculator.calculate_fraud_detection_rate(sample_dataframe)
        assert 'detection_rate_pct' in detection_metrics
        assert 'false_positive_rate_pct' in detection_metrics
        
        # Calculate merchant metrics
        merchant_metrics = calculator.calculate_merchant_risk_distribution(sample_dataframe)
        assert not merchant_metrics.empty
        assert 'merchant_id' in merchant_metrics.columns
        
        # Generate business summary
        summary = calculator.generate_business_summary_report(sample_dataframe)
        assert 'detection_metrics' in summary
        assert 'efficiency_metrics' in summary
        assert 'summary' in summary


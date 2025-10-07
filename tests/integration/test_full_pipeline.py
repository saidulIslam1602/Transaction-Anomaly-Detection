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


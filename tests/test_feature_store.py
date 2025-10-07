"""
Tests for Feature Store module.
"""

import pytest
import pandas as pd
import numpy as np
from src.services.feature_store import FeatureStore, Feature, RealTimeFeatureServer


class TestFeature:
    """Test Feature class."""
    
    def test_feature_creation(self):
        """Test creating a feature."""
        feature = Feature(
            name="amount",
            dtype="float",
            description="Transaction amount",
            tags=["numerical", "important"]
        )
        
        assert feature.name == "amount"
        assert feature.dtype == "float"
        assert len(feature.tags) == 2
        assert feature.created_at is not None


class TestFeatureStore:
    """Test FeatureStore class."""
    
    def test_initialization(self, tmp_path):
        """Test feature store initialization."""
        store = FeatureStore(storage_path=str(tmp_path))
        assert store.storage_path == str(tmp_path)
        assert len(store.features) == 0
    
    def test_register_feature(self, tmp_path):
        """Test registering a feature."""
        store = FeatureStore(storage_path=str(tmp_path))
        feature = Feature("amount", "float", "Transaction amount")
        
        store.register_feature(feature)
        assert "amount" in store.features
    
    def test_compute_transaction_features(self, sample_transaction):
        """Test computing basic transaction features."""
        store = FeatureStore()
        features = store.compute_transaction_features(sample_transaction)
        
        assert 'amount' in features
        assert 'origin_balance_change' in features
        assert 'dest_balance_change' in features
        assert 'amount_to_balance_ratio' in features
        assert features['amount'] == 1000.0
    
    def test_compute_aggregation_features(self, sample_transaction):
        """Test computing aggregation features."""
        store = FeatureStore()
        
        # Add some transactions to cache
        for i in range(5):
            store.compute_aggregation_features("C123456789", sample_transaction)
        
        features = store.compute_aggregation_features("C123456789", sample_transaction)
        
        assert 'txn_count_24h' in features
        assert 'total_amount_24h' in features
        assert features['txn_count_24h'] > 0
    
    def test_get_features(self, sample_transaction):
        """Test getting all features for a transaction."""
        store = FeatureStore()
        features = store.get_features(sample_transaction)
        
        assert isinstance(features, dict)
        assert 'amount' in features
        assert 'origin_balance_change' in features
    
    def test_batch_compute_features(self, sample_dataframe):
        """Test batch feature computation."""
        store = FeatureStore()
        feature_df = store.batch_compute_features(sample_dataframe.head(10))
        
        assert isinstance(feature_df, pd.DataFrame)
        assert len(feature_df) == 10
        assert 'amount' in feature_df.columns


class TestRealTimeFeatureServer:
    """Test RealTimeFeatureServer class."""
    
    def test_initialization(self):
        """Test feature server initialization."""
        store = FeatureStore()
        server = RealTimeFeatureServer(store)
        
        assert server.request_count == 0
        assert server.total_latency == 0.0
    
    def test_serve_features(self, sample_transaction):
        """Test serving features with latency tracking."""
        store = FeatureStore()
        server = RealTimeFeatureServer(store)
        
        features, latency = server.serve_features(sample_transaction)
        
        assert isinstance(features, dict)
        assert isinstance(latency, float)
        assert latency > 0
        assert server.request_count == 1
    
    def test_performance_metrics(self, sample_transaction):
        """Test performance metrics tracking."""
        store = FeatureStore()
        server = RealTimeFeatureServer(store)
        
        # Serve features multiple times
        for _ in range(10):
            server.serve_features(sample_transaction)
        
        metrics = server.get_performance_metrics()
        
        assert metrics['request_count'] == 10
        assert metrics['avg_latency_ms'] > 0
        assert metrics['total_latency_ms'] > 0


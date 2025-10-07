"""
Feature store for real-time and batch feature serving.

This module provides a centralized feature store for consistent feature
computation across training and serving, with support for online and offline features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import logging
import json
import pickle

logger = logging.getLogger(__name__)


class Feature:
    """
    Represents a feature with metadata.
    """
    
    def __init__(self,
                 name: str,
                 dtype: str,
                 description: str = "",
                 tags: Optional[List[str]] = None):
        """
        Initialize feature metadata.
        
        Args:
            name: Feature name
            dtype: Data type
            description: Feature description
            tags: Optional tags for categorization
        """
        self.name = name
        self.dtype = dtype
        self.description = description
        self.tags = tags or []
        self.created_at = datetime.now()


class FeatureStore:
    """
    Centralized feature store for consistent feature serving.
    
    Manages feature computation, storage, and retrieval for both
    training (batch) and inference (real-time) scenarios.
    """
    
    def __init__(self, storage_path: str = "./feature_store"):
        """
        Initialize feature store.
        
        Args:
            storage_path: Path to store feature data
        """
        self.storage_path = storage_path
        self.features = {}
        self.feature_groups = {}
        self.online_cache = {}
        
        logger.info(f"Feature store initialized at {storage_path}")
    
    def register_feature(self, feature: Feature) -> None:
        """
        Register a feature in the store.
        
        Args:
            feature: Feature object to register
        """
        self.features[feature.name] = feature
        logger.info(f"Feature registered: {feature.name}")
    
    def register_feature_group(self,
                              group_name: str,
                              feature_names: List[str],
                              description: str = "") -> None:
        """
        Register a group of related features.
        
        Args:
            group_name: Name of the feature group
            feature_names: List of feature names in the group
            description: Group description
        """
        self.feature_groups[group_name] = {
            'features': feature_names,
            'description': description,
            'created_at': datetime.now().isoformat()
        }
        logger.info(f"Feature group registered: {group_name}")
    
    def compute_transaction_features(self, transaction: Dict) -> Dict[str, Any]:
        """
        Compute real-time features for a transaction.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            Dictionary of computed features
        """
        features = {}
        
        # Basic features
        features['amount'] = float(transaction.get('amount', 0))
        features['type'] = str(transaction.get('type', 'UNKNOWN'))
        
        # Balance features
        old_bal_orig = float(transaction.get('oldbalanceOrg', 0))
        new_bal_orig = float(transaction.get('newbalanceOrig', 0))
        old_bal_dest = float(transaction.get('oldbalanceDest', 0))
        new_bal_dest = float(transaction.get('newbalanceDest', 0))
        
        features['origin_balance_before'] = old_bal_orig
        features['origin_balance_after'] = new_bal_orig
        features['dest_balance_before'] = old_bal_dest
        features['dest_balance_after'] = new_bal_dest
        
        # Derived features
        features['origin_balance_change'] = new_bal_orig - old_bal_orig
        features['dest_balance_change'] = new_bal_dest - old_bal_dest
        features['amount_to_balance_ratio'] = (
            features['amount'] / (old_bal_orig + 1e-6)
        )
        
        # Error flags
        features['error_balance_orig'] = int(
            new_bal_orig != old_bal_orig - features['amount']
        )
        features['error_balance_dest'] = int(
            new_bal_dest != old_bal_dest + features['amount']
        )
        
        # Zero balance flags
        features['is_zero_balance_orig'] = int(old_bal_orig == 0)
        features['is_zero_balance_dest'] = int(old_bal_dest == 0)
        
        return features
    
    def compute_aggregation_features(self,
                                    account_id: str,
                                    transaction: Dict,
                                    window_hours: int = 24) -> Dict[str, Any]:
        """
        Compute aggregation features over time window.
        
        Args:
            account_id: Account identifier
            transaction: Current transaction
            window_hours: Time window for aggregation
            
        Returns:
            Dictionary of aggregation features
        """
        features = {}
        
        # Get historical transactions for account (from cache)
        history = self.online_cache.get(account_id, [])
        
        # Filter by time window
        current_time = transaction.get('step', 0)
        window_transactions = [
            t for t in history
            if current_time - t.get('step', 0) <= window_hours
        ]
        
        if window_transactions:
            amounts = [t.get('amount', 0) for t in window_transactions]
            features[f'txn_count_{window_hours}h'] = len(window_transactions)
            features[f'total_amount_{window_hours}h'] = sum(amounts)
            features[f'avg_amount_{window_hours}h'] = np.mean(amounts)
            features[f'max_amount_{window_hours}h'] = max(amounts)
            features[f'min_amount_{window_hours}h'] = min(amounts)
            features[f'std_amount_{window_hours}h'] = np.std(amounts) if len(amounts) > 1 else 0
        else:
            features[f'txn_count_{window_hours}h'] = 0
            features[f'total_amount_{window_hours}h'] = 0
            features[f'avg_amount_{window_hours}h'] = 0
            features[f'max_amount_{window_hours}h'] = 0
            features[f'min_amount_{window_hours}h'] = 0
            features[f'std_amount_{window_hours}h'] = 0
        
        # Update cache with current transaction
        if account_id not in self.online_cache:
            self.online_cache[account_id] = []
        
        self.online_cache[account_id].append(transaction)
        
        # Keep cache size manageable
        if len(self.online_cache[account_id]) > 1000:
            self.online_cache[account_id] = self.online_cache[account_id][-1000:]
        
        return features
    
    def get_features(self,
                    transaction: Dict,
                    feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get features for a transaction.
        
        Args:
            transaction: Transaction dictionary
            feature_names: Optional list of specific features to retrieve
            
        Returns:
            Dictionary of feature values
        """
        # Compute all features
        basic_features = self.compute_transaction_features(transaction)
        
        account_id = transaction.get('nameOrig', 'unknown')
        agg_features = self.compute_aggregation_features(account_id, transaction)
        
        all_features = {**basic_features, **agg_features}
        
        # Filter to requested features if specified
        if feature_names:
            all_features = {k: v for k, v in all_features.items() if k in feature_names}
        
        return all_features
    
    def batch_compute_features(self,
                              df: pd.DataFrame,
                              feature_group: Optional[str] = None) -> pd.DataFrame:
        """
        Compute features for a batch of transactions.
        
        Args:
            df: DataFrame with transactions
            feature_group: Optional feature group to compute
            
        Returns:
            DataFrame with computed features
        """
        logger.info(f"Computing features for {len(df)} transactions")
        
        feature_dfs = []
        
        for _, row in df.iterrows():
            transaction = row.to_dict()
            features = self.get_features(transaction)
            feature_dfs.append(features)
        
        feature_df = pd.DataFrame(feature_dfs, index=df.index)
        
        logger.info(f"Computed {len(feature_df.columns)} features")
        
        return feature_df
    
    def save_features(self, features_df: pd.DataFrame, name: str) -> None:
        """
        Save computed features to disk.
        
        Args:
            features_df: DataFrame with features
            name: Name for the saved features
        """
        import os
        os.makedirs(self.storage_path, exist_ok=True)
        
        filepath = os.path.join(self.storage_path, f"{name}_features.parquet")
        features_df.to_parquet(filepath)
        
        logger.info(f"Features saved to {filepath}")
    
    def load_features(self, name: str) -> pd.DataFrame:
        """
        Load saved features from disk.
        
        Args:
            name: Name of the saved features
            
        Returns:
            DataFrame with features
        """
        import os
        filepath = os.path.join(self.storage_path, f"{name}_features.parquet")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Features not found: {filepath}")
        
        features_df = pd.read_parquet(filepath)
        logger.info(f"Features loaded from {filepath}")
        
        return features_df
    
    def export_feature_metadata(self) -> Dict[str, Any]:
        """
        Export feature metadata for documentation.
        
        Returns:
            Dictionary with feature metadata
        """
        metadata = {
            'features': {},
            'feature_groups': self.feature_groups,
            'exported_at': datetime.now().isoformat()
        }
        
        for name, feature in self.features.items():
            metadata['features'][name] = {
                'dtype': feature.dtype,
                'description': feature.description,
                'tags': feature.tags,
                'created_at': feature.created_at.isoformat()
            }
        
        return metadata


class RealTimeFeatureServer:
    """
    Real-time feature serving for low-latency inference.
    
    Optimized for sub-100ms feature computation and retrieval.
    """
    
    def __init__(self, feature_store: FeatureStore):
        """
        Initialize real-time feature server.
        
        Args:
            feature_store: FeatureStore instance
        """
        self.feature_store = feature_store
        self.request_count = 0
        self.total_latency = 0.0
        
        logger.info("Real-time feature server initialized")
    
    def serve_features(self,
                      transaction: Dict,
                      feature_names: Optional[List[str]] = None) -> Tuple[Dict[str, Any], float]:
        """
        Serve features with latency tracking.
        
        Args:
            transaction: Transaction dictionary
            feature_names: Optional list of features to serve
            
        Returns:
            Tuple of (features dictionary, latency in milliseconds)
        """
        start_time = datetime.now()
        
        features = self.feature_store.get_features(transaction, feature_names)
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Track metrics
        self.request_count += 1
        self.total_latency += latency_ms
        
        return features, latency_ms
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get feature serving performance metrics."""
        if self.request_count == 0:
            return {'avg_latency_ms': 0.0, 'request_count': 0}
        
        return {
            'avg_latency_ms': self.total_latency / self.request_count,
            'request_count': self.request_count,
            'total_latency_ms': self.total_latency
        }

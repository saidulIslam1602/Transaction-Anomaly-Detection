"""
Pytest configuration and fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_transaction():
    """Sample transaction data for testing."""
    return {
        'step': 1,
        'type': 'PAYMENT',
        'amount': 1000.0,
        'nameOrig': 'C123456789',
        'oldbalanceOrg': 5000.0,
        'newbalanceOrig': 4000.0,
        'nameDest': 'M987654321',
        'oldbalanceDest': 2000.0,
        'newbalanceDest': 3000.0,
        'isFraud': 0
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame with transactions for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'step': np.random.randint(1, 100, n_samples),
        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], n_samples),
        'amount': np.random.exponential(1000, n_samples),
        'nameOrig': [f'C{i:09d}' for i in range(n_samples)],
        'oldbalanceOrg': np.random.exponential(5000, n_samples),
        'newbalanceOrig': np.random.exponential(4000, n_samples),
        'nameDest': [f'M{i:09d}' for i in range(n_samples)],
        'oldbalanceDest': np.random.exponential(2000, n_samples),
        'newbalanceDest': np.random.exponential(3000, n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def fraud_transaction():
    """Sample fraudulent transaction."""
    return {
        'step': 1,
        'type': 'TRANSFER',
        'amount': 50000.0,
        'nameOrig': 'C999999999',
        'oldbalanceOrg': 50000.0,
        'newbalanceOrig': 0.0,
        'nameDest': 'C888888888',
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0,
        'isFraud': 1
    }


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        'ml_models': {
            'xgboost': {'enabled': True, 'max_depth': 6},
            'lightgbm': {'enabled': True, 'num_leaves': 31}
        },
        'rule_engine': {
            'large_transaction_threshold': 10000.0,
            'structuring_threshold': 9000.0
        },
        'llm': {'enabled': False},
        'rag': {'enabled': False}
    }


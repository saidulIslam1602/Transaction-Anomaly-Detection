"""
Tests for BI export service.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.services.bi_export import BIExportService


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        'step': np.random.randint(1, 744, n),
        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT'], n),
        'amount': np.random.lognormal(5, 2, n),
        'nameOrig': [f'C{i}' for i in range(n)],
        'oldbalanceOrg': np.random.uniform(0, 100000, n),
        'newbalanceOrig': np.random.uniform(0, 100000, n),
        'nameDest': [f'M{i%10}' for i in range(n)],
        'oldbalanceDest': np.random.uniform(0, 100000, n),
        'newbalanceDest': np.random.uniform(0, 100000, n),
        'isFraud': np.random.choice([0, 1], n, p=[0.9, 0.1]),
        'high_risk_flag': np.random.choice([True, False], n, p=[0.1, 0.9]),
        'final_risk_score': np.random.uniform(0, 10, n)
    })


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_bi_export_service_init(temp_output_dir):
    """Test BI export service initialization."""
    service = BIExportService(output_dir=temp_output_dir)
    assert service.output_dir == Path(temp_output_dir)
    assert service.output_dir.exists()


def test_export_transactions_for_bi_parquet(sample_transactions, temp_output_dir):
    """Test exporting transactions in Parquet format."""
    service = BIExportService(output_dir=temp_output_dir)
    filepath = service.export_transactions_for_bi(sample_transactions, format='parquet')
    
    assert Path(filepath).exists()
    assert filepath.endswith('.parquet')
    
    # Verify file can be read
    df = pd.read_parquet(filepath)
    assert len(df) > 0
    assert 'transaction_date' in df.columns


def test_export_transactions_for_bi_csv(sample_transactions, temp_output_dir):
    """Test exporting transactions in CSV format."""
    service = BIExportService(output_dir=temp_output_dir)
    filepath = service.export_transactions_for_bi(sample_transactions, format='csv')
    
    assert Path(filepath).exists()
    assert filepath.endswith('.csv')
    
    # Verify file can be read
    df = pd.read_csv(filepath)
    assert len(df) > 0


def test_export_merchant_metrics(sample_transactions, temp_output_dir):
    """Test exporting merchant metrics."""
    service = BIExportService(output_dir=temp_output_dir)
    filepath = service.export_merchant_metrics(sample_transactions, format='parquet')
    
    assert Path(filepath).exists()
    
    # Verify file can be read
    df = pd.read_parquet(filepath)
    assert len(df) > 0
    assert 'merchant_id' in df.columns


def test_export_volume_trends(sample_transactions, temp_output_dir):
    """Test exporting volume trends."""
    service = BIExportService(output_dir=temp_output_dir)
    filepath = service.export_volume_trends(sample_transactions, frequency='D', format='parquet')
    
    assert Path(filepath).exists()
    
    # Verify file can be read
    df = pd.read_parquet(filepath)
    assert len(df) > 0


def test_export_all_views(sample_transactions, temp_output_dir):
    """Test exporting all views."""
    service = BIExportService(output_dir=temp_output_dir)
    exports = service.export_all_views(sample_transactions, formats=['parquet'])
    
    assert len(exports) > 0
    for name, filepath in exports.items():
        assert Path(filepath).exists()


def test_export_invalid_format(sample_transactions, temp_output_dir):
    """Test error handling for invalid format."""
    service = BIExportService(output_dir=temp_output_dir)
    
    with pytest.raises(ValueError):
        service.export_transactions_for_bi(sample_transactions, format='invalid')


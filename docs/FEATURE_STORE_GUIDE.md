# Feature Store Guide

Guide to using the centralized feature store for consistent feature computation across training and serving.

## Overview

The Feature Store provides:
- **Consistent Features**: Same features in training and production
- **Real-time Serving**: Sub-100ms feature computation
- **Feature Versioning**: Track feature changes over time
- **Online/Offline Features**: Support for both real-time and batch serving

## Quick Start

### Python API

```python
from src.services.feature_store import FeatureStore

# Initialize
store = FeatureStore(storage_path="./feature_store")

# Get features for a transaction
transaction = {
    'step': 1,
    'type': 'PAYMENT',
    'amount': 1000.0,
    'nameOrig': 'C123',
    'oldbalanceOrg': 5000.0,
    'newbalanceOrig': 4000.0,
    'nameDest': 'M456',
    'oldbalanceDest': 2000.0,
    'newbalanceDest': 3000.0
}

features = store.compute_transaction_features(transaction)
print(features)
```

## Available Features

### Transaction-Level Features

- `is_large_transaction`: Flag for transactions > $10,000
- `is_cash_transaction`: Flag for CASH_IN/CASH_OUT transactions
- `orig_balance_ratio`: Amount relative to origin balance
- `dest_balance_ratio`: Amount relative to destination balance
- `balance_mismatch`: Flag for balance calculation errors

### Temporal Features

- `txn_count_1h`: Transaction count in last hour
- `total_amount_1h`: Total amount in last hour
- `txn_count_24h`: Transaction count in last 24 hours
- `total_amount_24h`: Total amount in last 24 hours
- `txn_count_7d`: Transaction count in last 7 days

### Account-Level Features

- `account_fraud_rate`: Historical fraud rate for account
- `account_fan_out`: Number of unique destinations

## Usage Patterns

### Real-Time Feature Serving

```python
from src.services.feature_store import FeatureStore

store = FeatureStore()

# For real-time predictions
transaction = get_transaction_from_api()
features = store.get_features(transaction)
risk_score = model.predict(features)
```

### Batch Feature Computation

```python
import pandas as pd
from src.services.feature_store import FeatureStore

store = FeatureStore()

# Load batch of transactions
df = pd.read_csv('transactions.csv')

# Compute features for all transactions
feature_list = []
for _, row in df.iterrows():
    features = store.compute_transaction_features(row.to_dict())
    feature_list.append(features)

features_df = pd.DataFrame(feature_list)
```

### Feature Aggregations

```python
# Get aggregated features for time windows
features = store.compute_transaction_features(
    transaction,
    aggregation_windows=[1, 24, 168]  # 1h, 24h, 1 week
)
```

## Feature Groups

Features are organized into groups:

- **transaction_features**: Transaction-level features
- **temporal_features**: Time-based aggregations
- **account_features**: Account-level metrics
- **network_features**: Graph-based features

## Feature Versioning

Features are versioned to track changes:

```python
# Register a new feature version
feature = Feature(
    name="transaction_velocity",
    dtype="float",
    description="Transaction velocity in last hour",
    tags=["temporal", "velocity"]
)
store.register_feature(feature)
```

## Integration with Models

### Training

```python
# Use feature store for consistent features
store = FeatureStore()
X_train = store.get_batch_features(train_transactions)
y_train = train_transactions['isFraud']

model.fit(X_train, y_train)
```

### Serving

```python
# Same feature computation in production
features = store.get_features(transaction)
prediction = model.predict(features)
```

## Performance

- **Latency**: < 100ms for single transaction
- **Throughput**: 10,000+ transactions/second
- **Caching**: Online features cached in Redis

## Best Practices

1. **Use Feature Store**: Always use feature store for consistency
2. **Version Features**: Track feature changes over time
3. **Document Features**: Add descriptions and tags
4. **Test Features**: Validate feature computation
5. **Monitor Performance**: Track feature serving latency

## Troubleshooting

### Issue: Features not matching training

**Solution**: Ensure you're using the same feature store version and configuration

### Issue: Slow feature computation

**Solution**: 
- Check if aggregations are cached
- Use batch computation for multiple transactions
- Optimize aggregation windows

### Issue: Missing features

**Solution**: 
- Check feature store configuration
- Verify required input columns are present
- Review feature registration

## Related Documentation

- [Self-Service Guide](SELF_SERVICE_GUIDE.md)
- [Query Examples](QUERY_EXAMPLES.md)
- [API Documentation](../src/services/feature_store.py)


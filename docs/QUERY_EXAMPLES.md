# SQL Query Examples

Common SQL queries for analyzing transaction data in Databricks or other SQL environments.

## Basic Queries

### Get Recent Transactions

```sql
SELECT *
FROM silver_transactions
WHERE ingestion_date >= CURRENT_DATE() - INTERVAL 7 DAYS
ORDER BY step DESC
LIMIT 100;
```

### Transaction Volume by Type

```sql
SELECT 
    type,
    COUNT(*) as transaction_count,
    SUM(amount) as total_volume,
    AVG(amount) as avg_amount,
    MAX(amount) as max_amount
FROM silver_transactions
GROUP BY type
ORDER BY total_volume DESC;
```

## Fraud Analysis

### Fraud Detection Rate

```sql
SELECT 
    COUNT(*) as total_transactions,
    SUM(isFraud) as total_fraud,
    SUM(CASE WHEN high_risk_flag = 1 THEN 1 ELSE 0 END) as detected_fraud,
    SUM(CASE WHEN high_risk_flag = 1 AND isFraud = 1 THEN 1 ELSE 0 END) as true_positives,
    SUM(CASE WHEN high_risk_flag = 1 AND isFraud = 0 THEN 1 ELSE 0 END) as false_positives,
    ROUND(
        SUM(CASE WHEN high_risk_flag = 1 AND isFraud = 1 THEN 1 ELSE 0 END) * 100.0 / 
        NULLIF(SUM(isFraud), 0), 
        2
    ) as detection_rate_pct
FROM combined_results;
```

### False Positive Rate by Transaction Type

```sql
SELECT 
    type,
    COUNT(*) as total_alerts,
    SUM(CASE WHEN isFraud = 0 THEN 1 ELSE 0 END) as false_positives,
    ROUND(
        SUM(CASE WHEN isFraud = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) as false_positive_rate_pct
FROM combined_results
WHERE high_risk_flag = 1
GROUP BY type
ORDER BY false_positive_rate_pct DESC;
```

## Merchant Analysis

### Top Risky Merchants

```sql
SELECT 
    nameDest as merchant_id,
    COUNT(*) as transaction_count,
    SUM(amount) as total_volume,
    AVG(final_risk_score) as avg_risk_score,
    MAX(final_risk_score) as max_risk_score,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as fraud_count,
    ROUND(
        SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) as fraud_rate_pct
FROM combined_results
WHERE nameDest IS NOT NULL
GROUP BY nameDest
HAVING transaction_count >= 10
ORDER BY avg_risk_score DESC
LIMIT 20;
```

### Merchant Transaction Patterns

```sql
SELECT 
    nameDest as merchant_id,
    type,
    COUNT(*) as transaction_count,
    AVG(amount) as avg_amount,
    SUM(amount) as total_amount
FROM silver_transactions
WHERE nameDest IS NOT NULL
GROUP BY nameDest, type
ORDER BY merchant_id, total_amount DESC;
```

## Time-Based Analysis

### Daily Transaction Volume

```sql
SELECT 
    DATE_ADD('2020-01-01', CAST(step / 24 AS INT)) as transaction_date,
    COUNT(*) as transaction_count,
    SUM(amount) as total_volume,
    AVG(amount) as avg_amount,
    SUM(CASE WHEN high_risk_flag = 1 THEN 1 ELSE 0 END) as high_risk_count
FROM combined_results
GROUP BY transaction_date
ORDER BY transaction_date;
```

### Hourly Transaction Patterns

```sql
SELECT 
    step % 24 as hour_of_day,
    COUNT(*) as transaction_count,
    AVG(amount) as avg_amount,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as fraud_count
FROM silver_transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;
```

## Feature Analysis

### Risk Score Distribution

```sql
SELECT 
    CASE 
        WHEN final_risk_score < 2 THEN 'LOW (0-2)'
        WHEN final_risk_score < 5 THEN 'MEDIUM (2-5)'
        WHEN final_risk_score < 8 THEN 'HIGH (5-8)'
        ELSE 'CRITICAL (8-10)'
    END as risk_category,
    COUNT(*) as transaction_count,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as fraud_count,
    ROUND(
        SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) as fraud_rate_pct
FROM combined_results
GROUP BY risk_category
ORDER BY 
    CASE risk_category
        WHEN 'LOW (0-2)' THEN 1
        WHEN 'MEDIUM (2-5)' THEN 2
        WHEN 'HIGH (5-8)' THEN 3
        WHEN 'CRITICAL (8-10)' THEN 4
    END;
```

### Large Transaction Analysis

```sql
SELECT 
    CASE 
        WHEN amount < 1000 THEN '< $1K'
        WHEN amount < 10000 THEN '$1K - $10K'
        WHEN amount < 50000 THEN '$10K - $50K'
        ELSE '> $50K'
    END as amount_range,
    COUNT(*) as transaction_count,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as fraud_count,
    ROUND(
        SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) as fraud_rate_pct
FROM silver_transactions
GROUP BY amount_range
ORDER BY 
    CASE amount_range
        WHEN '< $1K' THEN 1
        WHEN '$1K - $10K' THEN 2
        WHEN '$10K - $50K' THEN 3
        WHEN '> $50K' THEN 4
    END;
```

## Performance Metrics

### Model Performance Summary

```sql
SELECT 
    'Rule-Based' as detection_method,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN rule_based_flag = 1 AND isFraud = 1 THEN 1 ELSE 0 END) as true_positives,
    SUM(CASE WHEN rule_based_flag = 1 AND isFraud = 0 THEN 1 ELSE 0 END) as false_positives,
    SUM(CASE WHEN rule_based_flag = 0 AND isFraud = 1 THEN 1 ELSE 0 END) as false_negatives
FROM combined_results

UNION ALL

SELECT 
    'ML-Based' as detection_method,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN ml_flag = 1 AND isFraud = 1 THEN 1 ELSE 0 END) as true_positives,
    SUM(CASE WHEN ml_flag = 1 AND isFraud = 0 THEN 1 ELSE 0 END) as false_positives,
    SUM(CASE WHEN ml_flag = 0 AND isFraud = 1 THEN 1 ELSE 0 END) as false_negatives
FROM combined_results

UNION ALL

SELECT 
    'Combined' as detection_method,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN high_risk_flag = 1 AND isFraud = 1 THEN 1 ELSE 0 END) as true_positives,
    SUM(CASE WHEN high_risk_flag = 1 AND isFraud = 0 THEN 1 ELSE 0 END) as false_positives,
    SUM(CASE WHEN high_risk_flag = 0 AND isFraud = 1 THEN 1 ELSE 0 END) as false_negatives
FROM combined_results;
```

## Using dbt Models

If you're using dbt, you can query the marts models:

### Query Fact Table

```sql
SELECT *
FROM {{ ref('fct_transactions') }}
WHERE ingestion_date >= CURRENT_DATE() - INTERVAL 7 DAYS
LIMIT 100;
```

### Query Merchant Dimension

```sql
SELECT *
FROM {{ ref('dim_merchants') }}
WHERE risk_category IN ('HIGH', 'CRITICAL')
ORDER BY fraud_rate DESC;
```

## Tips

1. **Use Indexes**: For large tables, ensure proper indexing on frequently queried columns
2. **Partitioning**: Query partitioned tables by date for better performance
3. **Sampling**: For exploratory analysis, use `TABLESAMPLE` to work with smaller datasets
4. **Caching**: Cache frequently used aggregations in materialized views

## Related Documentation

- [Self-Service Guide](SELF_SERVICE_GUIDE.md)
- [Feature Store Guide](FEATURE_STORE_GUIDE.md)
- [dbt Models](../dbt/README.md)


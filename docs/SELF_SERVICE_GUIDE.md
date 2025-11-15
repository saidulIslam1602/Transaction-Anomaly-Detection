# Self-Service Analytics Guide

This guide helps non-technical users and product teams get started with analyzing transaction data using the Transaction Anomaly Detection system.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Using the Business Dashboard](#using-the-business-dashboard)
3. [Querying the Feature Store](#querying-the-feature-store)
4. [Exporting Data for BI Tools](#exporting-data-for-bi-tools)
5. [Common Analysis Patterns](#common-analysis-patterns)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

- Access to the transaction data (CSV files or database)
- Basic understanding of data analysis concepts
- (Optional) Access to Power BI, Looker, or similar BI tools

### Quick Start

1. **Use the Business Dashboard** (Easiest option)
   ```bash
   streamlit run dashboards/business_dashboard.py
   ```
   Upload your CSV file and explore the interactive dashboards.

2. **Export Data for BI Tools**
   ```bash
   python scripts/export_for_bi.py --input data/transactions.csv --output bi_exports/
   ```
   Import the exported files into your BI tool.

## Using the Business Dashboard

The Business Dashboard is the easiest way to explore transaction data without writing code.

### Features

- **Overview Tab**: System-wide metrics and trends
- **Fraud Detection Tab**: Performance metrics and risk analysis
- **Merchant Analytics Tab**: Merchant-level insights
- **Export Tab**: Download data for external analysis

### Step-by-Step

1. Open the dashboard:
   ```bash
   streamlit run dashboards/business_dashboard.py
   ```

2. Upload your data:
   - Click "Browse files" in the sidebar
   - Select your CSV file with transaction data
   - Wait for data to load

3. Explore the tabs:
   - Start with "Overview" to see high-level metrics
   - Check "Fraud Detection" for performance analysis
   - Review "Merchant Analytics" for merchant insights

4. Export data:
   - Go to "Export Data" tab
   - Select your preferred format (Parquet, CSV, or Excel)
   - Click export buttons to download files

## Querying the Feature Store

For more advanced analysis, you can query the feature store directly.

### Using Python

```python
from src.services.feature_store import FeatureStore

# Initialize feature store
store = FeatureStore()

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

features = store.get_features(transaction)
print(features)
```

### Using SQL (Databricks)

If you have access to Databricks, you can query the feature store tables:

```sql
-- Query transaction features
SELECT * FROM feature_store_transactions
WHERE ingestion_date = CURRENT_DATE()
LIMIT 100;

-- Get merchant-level aggregations
SELECT 
    nameDest as merchant_id,
    COUNT(*) as transaction_count,
    AVG(amount) as avg_amount,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as fraud_count
FROM feature_store_transactions
GROUP BY nameDest
ORDER BY fraud_count DESC;
```

## Exporting Data for BI Tools

### Using the CLI Script

```bash
# Export all views
python scripts/export_for_bi.py --input data/transactions.csv --output bi_exports/ --all

# Export specific format
python scripts/export_for_bi.py --input data/transactions.csv --format csv
```

### Available Exports

1. **Transaction Data**: Complete transaction records optimized for BI
2. **Merchant Metrics**: Pre-aggregated merchant-level metrics
3. **Volume Trends**: Time-series data for trend analysis
4. **Detection Performance**: Performance metrics and KPIs

### Importing into BI Tools

#### Power BI
1. Open Power BI Desktop
2. Get Data → From File → From Folder
3. Select the exported Parquet or CSV files
4. Transform and load data

#### Looker
1. Create a new connection to your data source
2. Upload the exported CSV files
3. Create explores based on the data

## Common Analysis Patterns

### Pattern 1: Fraud Detection Rate by Merchant

**Question**: Which merchants have the highest fraud rates?

**Solution**:
1. Use Business Dashboard → Merchant Analytics tab
2. Or export merchant metrics and analyze in BI tool
3. Sort by `fraud_rate` or `avg_risk_score`

### Pattern 2: Transaction Volume Trends

**Question**: How has transaction volume changed over time?

**Solution**:
1. Business Dashboard → Overview tab → Transaction Volume Trends
2. Or export volume trends and create time-series chart in BI tool

### Pattern 3: False Positive Analysis

**Question**: What percentage of alerts are false positives?

**Solution**:
1. Business Dashboard → Fraud Detection tab
2. Check "False Positive Rate" metric
3. Review confusion matrix

### Pattern 4: High-Risk Transaction Patterns

**Question**: What patterns characterize high-risk transactions?

**Solution**:
1. Export transaction data
2. Filter by `high_risk_flag = True`
3. Analyze distributions of:
   - Transaction types
   - Amount ranges
   - Time patterns
   - Merchant categories

## Troubleshooting

### Issue: Dashboard won't load

**Solution**:
- Check that Streamlit is installed: `pip install streamlit`
- Verify Python version (3.8+)
- Check file path and permissions

### Issue: Export fails

**Solution**:
- Ensure output directory exists and is writable
- Check that required columns are present in input data
- Verify file format (CSV expected)

### Issue: Missing columns in data

**Solution**:
- Check [Data Format](#data-format) section in dashboard README
- Some features are optional - dashboard will work with available columns
- Missing required columns will show warnings

### Issue: Slow performance

**Solution**:
- For large datasets (>1M rows), use Parquet format
- Consider sampling data for initial exploration
- Use BI tools for large-scale analysis

## Getting Help

- **Documentation**: See [README.md](../README.md) for technical details
- **Examples**: Check [examples/](../examples/) directory for code samples
- **Query Examples**: See [QUERY_EXAMPLES.md](QUERY_EXAMPLES.md)

## Next Steps

1. Explore the [Business Dashboard](../dashboards/README.md)
2. Review [Query Examples](QUERY_EXAMPLES.md)
3. Check [Feature Store Guide](FEATURE_STORE_GUIDE.md)
4. Try the [Product Team Examples](../examples/)


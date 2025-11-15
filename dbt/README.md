# dbt Models for Transaction Anomaly Detection

This directory contains dbt models for data transformation and modeling in the Transaction Anomaly Detection system.

## Overview

The dbt project implements a medallion architecture (Bronze → Silver → Gold) using dbt models:

- **Staging models** (`staging/`): Clean and standardize data from Bronze layer
- **Intermediate models** (`intermediate/`): Create reusable features and transformations
- **Marts models** (`marts/`): Business-ready fact and dimension tables for analytics

## Project Structure

```
dbt/
├── dbt_project.yml          # dbt project configuration
├── profiles.yml             # Connection profiles (Databricks)
├── models/
│   ├── sources.yml         # Source table definitions
│   ├── staging/
│   │   └── stg_transactions.sql
│   ├── intermediate/
│   │   └── int_transaction_features.sql
│   └── marts/
│       ├── fct_transactions.sql
│       └── dim_merchants.sql
└── README.md
```

## Setup

### Prerequisites

1. Install dbt:
```bash
pip install dbt-databricks
```

2. Configure Databricks connection:

   **Option A: Use environment variables (Recommended)**
   
   Set these environment variables:
   ```bash
   export DATABRICKS_HOST="your-workspace.cloud.databricks.com"
   export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/your-warehouse-id"
   export DATABRICKS_TOKEN="your-personal-access-token"
   ```
   
   Or use the setup script:
   ```bash
   ./scripts/setup_credentials.sh
   ```
   
   **Option B: Edit profiles.yml directly (Not recommended)**
   
   Edit `dbt/profiles.yml` and replace placeholder values:
   - `your-databricks-host.cloud.databricks.com` → Your actual host
   - `your-warehouse-id` → Your actual warehouse ID
   - `your-token` → Your actual token
   
   **⚠️ Warning**: If you edit profiles.yml directly, make sure it's NOT committed to version control!

### Running dbt Models

```bash
# Navigate to dbt directory
cd dbt

# Run all models
dbt run

# Run specific model
dbt run --select stg_transactions

# Run models with dependencies
dbt run --select fct_transactions+

# Test models
dbt test

# Generate documentation
dbt docs generate
dbt docs serve
```

## Models

### Staging Models

**stg_transactions**: Cleans and standardizes raw transaction data
- Removes invalid records
- Adds derived columns (balance changes, error flags)
- Implements Silver layer transformations

### Intermediate Models

**int_transaction_features**: Creates transaction-level features
- Transaction type flags
- Balance indicators
- Risk-related features

### Marts Models

**fct_transactions**: Fact table for transaction analytics
- Complete transaction details
- All feature flags and metrics
- Ready for BI tools and dashboards

**dim_merchants**: Dimension table for merchant analytics
- Merchant-level aggregations
- Risk categorization
- Volume metrics

## Integration with Databricks

These dbt models can be run in Databricks using dbt-databricks adapter. The models reference Delta Lake tables created by the Databricks notebooks in `databricks/notebooks/`.

## Usage in BI Tools

The marts models (`fct_transactions`, `dim_merchants`) are optimized for use in:
- Power BI
- Looker
- Tableau
- Other BI tools

These tables provide pre-aggregated, business-ready data that can be directly connected to BI tools.

## Best Practices

1. **Incremental Models**: For large datasets, consider using incremental materialization
2. **Testing**: Add dbt tests to validate data quality
3. **Documentation**: Keep model documentation up to date
4. **Version Control**: All models are version controlled in Git

## Related Documentation

- [dbt Documentation](https://docs.getdbt.com/)
- [dbt-databricks Adapter](https://github.com/databricks/dbt-databricks)
- [Databricks Notebooks](../databricks/notebooks/)


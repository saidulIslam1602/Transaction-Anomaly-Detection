# Product Team Collaboration Examples

This document provides examples of how this project enables product teams to work with data effectively.

## Overview

The Transaction Anomaly Detection system is designed to enable product teams through:
- Self-service analytics tools
- Pre-aggregated data views
- Interactive dashboards
- Clear documentation and examples
- API access for integration

## Collaboration Patterns

### Pattern 1: Self-Service Analytics

**Scenario**: Product team wants to analyze transaction trends

**Solution**: Business Dashboard

```python
# Product team uses the dashboard
streamlit run dashboards/business_dashboard.py

# They can:
# 1. Upload their data
# 2. Explore interactive visualizations
# 3. Export data for further analysis
# 4. Generate business reports
```

**Benefits**:
- No coding required
- Immediate insights
- Export capabilities for BI tools

### Pattern 2: BI Tool Integration

**Scenario**: Product team wants to create Power BI dashboards

**Solution**: BI Export Service

```python
from src.services.bi_export import BIExportService

# Export pre-aggregated views
export_service = BIExportService()
exports = export_service.export_all_views(df, formats=['parquet'])

# Product team imports into Power BI
# - transactions_bi_*.parquet → Transaction fact table
# - merchant_metrics_*.parquet → Merchant dimension
# - volume_trends_*.parquet → Time-series data
```

**Benefits**:
- Pre-aggregated data saves time
- Optimized formats for BI tools
- Consistent data across teams

### Pattern 3: API Integration

**Scenario**: Product team wants to integrate fraud detection into their application

**Solution**: REST API

```python
import requests

# Real-time fraud detection
response = requests.post(
    "https://aml-api-prod.azurewebsites.net/predict",
    json=transaction_data
)

result = response.json()
# Use result['is_fraud'], result['risk_score'], etc.
```

**Benefits**:
- Real-time predictions
- Easy integration
- No need to understand ML models

### Pattern 4: SQL Queries

**Scenario**: Product team wants to answer specific business questions

**Solution**: Query Examples and Documentation

```sql
-- Example: Which merchants have highest fraud rates?
SELECT 
    nameDest as merchant_id,
    COUNT(*) as transaction_count,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) as fraud_count,
    ROUND(
        SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) as fraud_rate_pct
FROM combined_results
WHERE nameDest IS NOT NULL
GROUP BY nameDest
HAVING transaction_count >= 10
ORDER BY fraud_rate_pct DESC;
```

**Benefits**:
- Direct data access
- Flexible analysis
- Reusable query patterns

## Real-World Use Cases

### Use Case 1: Merchant Onboarding Decision

**Question**: Should we onboard this merchant?

**Process**:
1. Product team queries merchant risk metrics
2. Reviews fraud rate and transaction patterns
3. Uses dashboard to visualize risk profile
4. Makes data-driven onboarding decision

**Tools Used**:
- Business Dashboard → Merchant Analytics tab
- SQL queries for detailed analysis
- BI exports for stakeholder presentations

### Use Case 2: Transaction Limit Setting

**Question**: What transaction limits should we set?

**Process**:
1. Product team analyzes transaction amount distributions
2. Reviews fraud rates by amount ranges
3. Uses business metrics to understand risk patterns
4. Sets limits based on data insights

**Tools Used**:
- Business Dashboard → Fraud Detection tab
- SQL queries for amount analysis
- Business metrics calculator

### Use Case 3: System Performance Monitoring

**Question**: How is our fraud detection system performing?

**Process**:
1. Product team checks detection metrics
2. Reviews false positive rates
3. Monitors trends over time
4. Identifies areas for improvement

**Tools Used**:
- Business Dashboard → Overview tab
- API metrics endpoint
- Business summary reports

### Use Case 4: User Behavior Analysis

**Question**: What are the transaction patterns of high-risk users?

**Process**:
1. Product team queries user transaction patterns
2. Analyzes risk score distributions
3. Identifies behavioral patterns
4. Develops targeted interventions

**Tools Used**:
- SQL queries for user analysis
- Product metrics calculator
- Business Dashboard visualizations

## Documentation for Product Teams

### Self-Service Guide

**Location**: `docs/SELF_SERVICE_GUIDE.md`

**Contents**:
- Getting started instructions
- Dashboard usage guide
- Query examples
- Export procedures
- Troubleshooting

### Query Examples

**Location**: `docs/QUERY_EXAMPLES.md`, `examples/product_team_queries.sql`

**Contents**:
- Common SQL queries
- Business question patterns
- Use case examples

### API Documentation

**Location**: `examples/API_USAGE_EXAMPLES.md`

**Contents**:
- API endpoint documentation
- Integration examples
- Error handling
- Best practices

## Best Practices for Collaboration

### 1. Make Data Accessible

- Provide multiple access methods (dashboard, API, SQL)
- Pre-aggregate common views
- Document everything clearly

### 2. Enable Self-Service

- Create interactive tools
- Provide examples and templates
- Support multiple skill levels

### 3. Focus on Business Value

- Calculate business metrics, not just technical metrics
- Create visualizations that tell stories
- Provide actionable insights

### 4. Iterate Based on Feedback

- Gather feedback from product teams
- Improve tools based on usage
- Add features as needed

## Success Metrics

- **Adoption**: Number of product team members using the tools
- **Self-Service**: Percentage of questions answered without analyst help
- **Time to Insight**: Time from question to answer
- **Satisfaction**: Product team feedback and satisfaction scores

## Related Documentation

- [Self-Service Guide](SELF_SERVICE_GUIDE.md)
- [Business Dashboard Guide](DASHBOARD_GUIDE.md)
- [Query Examples](QUERY_EXAMPLES.md)
- [API Usage Examples](../examples/API_USAGE_EXAMPLES.md)


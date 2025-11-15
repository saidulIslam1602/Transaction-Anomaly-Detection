# Business Dashboard

Interactive Streamlit dashboard for product teams and business stakeholders to explore transaction data, fraud patterns, and system performance.

## Features

- **Overview Dashboard**: System-wide metrics and transaction trends
- **Fraud Detection Analytics**: Performance metrics, confusion matrix, risk score distributions
- **Merchant Analytics**: Merchant risk profiling and transaction patterns
- **Data Export**: Export pre-aggregated views for Power BI, Looker, and other BI tools

## Setup

### Install Dependencies

```bash
pip install -r dashboards/requirements.txt
```

### Run Dashboard

```bash
streamlit run dashboards/business_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Usage

1. **Upload Data**: Use the sidebar to upload a CSV file with transaction data
2. **Explore Tabs**: Navigate through different analytics views
3. **Export Data**: Use the Export tab to download data in various formats for BI tools

## Data Format

The dashboard expects a CSV file with the following columns:
- `step`: Time step (hour)
- `type`: Transaction type
- `amount`: Transaction amount
- `nameOrig`: Origin account
- `nameDest`: Destination account (merchant)
- `oldbalanceOrg`, `newbalanceOrig`: Origin balance
- `oldbalanceDest`, `newbalanceDest`: Destination balance
- `isFraud`: Fraud label (0 or 1)
- `high_risk_flag`: High risk flag (boolean)
- `final_risk_score`: Risk score (0-10)

## Export Formats

- **Parquet**: Optimized for large datasets and Power BI
- **CSV**: Universal format for all BI tools
- **Excel**: Easy to use in Excel and Google Sheets

## Integration with BI Tools

Exported data can be directly imported into:
- Power BI
- Looker
- Tableau
- Google Data Studio
- Other BI tools

## Related Documentation

- [Self-Service Guide](../docs/SELF_SERVICE_GUIDE.md)
- [BI Export Service](../src/services/bi_export.py)
- [Business Metrics](../src/services/business_metrics.py)


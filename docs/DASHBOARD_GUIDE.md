# Business Dashboard Guide

Step-by-step guide to using the interactive Business Dashboard for transaction analysis.

## Overview

The Business Dashboard is a Streamlit application that provides:
- Interactive visualizations
- Real-time data exploration
- Export capabilities for BI tools
- No coding required

## Getting Started

### Launch Dashboard

```bash
# Install dependencies
pip install -r dashboards/requirements.txt

# Run dashboard
streamlit run dashboards/business_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Dashboard Tabs

### 1. Overview Tab

**Purpose**: High-level system metrics and trends

**Features**:
- Total transactions count
- Total transaction volume
- Fraud cases detected
- High-risk alerts
- Transaction volume trends (time-series)
- Transaction type distribution (pie chart)

**Use Cases**:
- Daily/weekly system health checks
- Understanding transaction patterns
- Identifying volume anomalies

### 2. Fraud Detection Tab

**Purpose**: Analyze fraud detection performance

**Features**:
- Detection rate metrics
- False positive rate
- Precision metrics
- Confusion matrix visualization
- Risk score distribution

**Use Cases**:
- Evaluating model performance
- Understanding false positive patterns
- Optimizing detection thresholds

### 3. Merchant Analytics Tab

**Purpose**: Merchant-level insights and risk profiling

**Features**:
- Top risky merchants table
- Risk category distribution
- Volume vs Risk scatter plot
- Merchant transaction patterns

**Use Cases**:
- Identifying high-risk merchants
- Merchant onboarding decisions
- Risk-based merchant management

### 4. Export Data Tab

**Purpose**: Export data for external analysis

**Features**:
- Export transaction data
- Export merchant metrics
- Export volume trends
- Export all views at once
- Multiple formats (Parquet, CSV, Excel)
- Business summary report generation

**Use Cases**:
- Preparing data for Power BI
- Creating Looker dashboards
- Sharing data with stakeholders
- Generating reports

## Step-by-Step Workflows

### Workflow 1: Daily Health Check

1. Upload today's transaction data
2. Go to Overview tab
3. Check key metrics:
   - Total transactions (compare to baseline)
   - Fraud cases (check for anomalies)
   - High-risk alerts (monitor volume)
4. Review transaction volume trends
5. Export summary report

### Workflow 2: Fraud Detection Analysis

1. Upload transaction data with fraud labels
2. Go to Fraud Detection tab
3. Review detection metrics:
   - Detection rate (should be > 90%)
   - False positive rate (should be < 5%)
   - Precision (should be > 85%)
4. Analyze confusion matrix
5. Review risk score distribution
6. Export performance metrics

### Workflow 3: Merchant Risk Assessment

1. Upload transaction data
2. Go to Merchant Analytics tab
3. Review top risky merchants
4. Analyze risk category distribution
5. Review volume vs risk scatter plot
6. Export merchant metrics for further analysis

### Workflow 4: BI Tool Integration

1. Upload transaction data
2. Go to Export Data tab
3. Select export format (Parquet recommended for Power BI)
4. Click "Export All Views"
5. Download exported files
6. Import into BI tool:
   - Power BI: Get Data → From File → From Folder
   - Looker: Upload CSV files
   - Tableau: Connect to CSV/Parquet files

## Tips and Tricks

### Performance

- **Large Datasets**: Use Parquet format for better performance
- **Sampling**: For initial exploration, sample your data
- **Caching**: Dashboard caches calculations for faster reloads

### Data Quality

- **Required Columns**: Ensure all required columns are present
- **Data Types**: Verify numeric columns are numeric
- **Missing Values**: Dashboard handles missing values gracefully

### Visualizations

- **Interactive**: Click and drag to zoom in charts
- **Hover**: Hover over data points for details
- **Export**: Right-click charts to save images

## Troubleshooting

### Dashboard won't start

**Solution**:
```bash
# Check Streamlit installation
pip install streamlit

# Verify Python version (3.8+)
python --version

# Check for port conflicts
streamlit run dashboards/business_dashboard.py --server.port 8502
```

### Data not loading

**Solution**:
- Check CSV file format
- Verify column names match expected format
- Check file encoding (UTF-8 recommended)
- Review error messages in sidebar

### Charts not displaying

**Solution**:
- Check browser console for errors
- Try different browser
- Clear browser cache
- Verify Plotly is installed

### Export fails

**Solution**:
- Check output directory permissions
- Verify disk space
- Check file format compatibility
- Review error messages

## Keyboard Shortcuts

- `R`: Rerun dashboard
- `C`: Clear cache
- `?`: Show keyboard shortcuts
- `Esc`: Close dialogs

## Related Documentation

- [Self-Service Guide](SELF_SERVICE_GUIDE.md)
- [Dashboard README](../dashboards/README.md)
- [BI Export Service](../src/services/bi_export.py)


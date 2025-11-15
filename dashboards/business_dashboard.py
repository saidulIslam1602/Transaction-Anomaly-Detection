"""
Business Dashboard for Transaction Anomaly Detection.

Interactive Streamlit dashboard for product teams and business stakeholders
to explore transaction data, fraud patterns, and system performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.services.business_metrics import BusinessMetricsCalculator
from src.services.product_metrics import ProductMetricsCalculator
from src.services.bi_export import BIExportService
from src.services.automated_reporting import AutomatedReportingService

# Page configuration
st.set_page_config(
    page_title="Transaction Anomaly Detection - Business Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize calculators
@st.cache_resource
def get_calculators():
    return BusinessMetricsCalculator(), ProductMetricsCalculator(), BIExportService(), AutomatedReportingService()

business_calc, product_calc, bi_export, reporting = get_calculators()

# Title
st.title("📊 Transaction Anomaly Detection - Business Dashboard")
st.markdown("**Interactive analytics for product teams and business stakeholders**")

# Sidebar
st.sidebar.header("Dashboard Controls")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload transaction data (CSV)",
    type=['csv'],
    help="Upload a CSV file with transaction data to analyze"
)

# Load sample data if no file uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {len(df)} transactions")
else:
    st.sidebar.info("💡 Upload a CSV file to get started, or use sample data below")
    # Create sample data for demonstration
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'step': np.random.randint(1, 744, n),
        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'], n),
        'amount': np.random.lognormal(5, 2, n),
        'nameOrig': [f'C{i}' for i in range(n)],
        'oldbalanceOrg': np.random.uniform(0, 100000, n),
        'newbalanceOrig': np.random.uniform(0, 100000, n),
        'nameDest': [f'M{i%50}' for i in range(n)],
        'oldbalanceDest': np.random.uniform(0, 100000, n),
        'newbalanceDest': np.random.uniform(0, 100000, n),
        'isFraud': np.random.choice([0, 1], n, p=[0.95, 0.05]),
        'high_risk_flag': np.random.choice([True, False], n, p=[0.1, 0.9]),
        'final_risk_score': np.random.uniform(0, 10, n)
    })
    st.sidebar.warning("Using sample data for demonstration")

# Main dashboard
if df is not None and len(df) > 0:
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "🔍 Fraud Detection",
        "🏪 Merchant Analytics",
        "📤 Export Data",
        "📊 Automated Reports"
    ])
    
    with tab1:
        st.header("System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Transactions",
                f"{len(df):,}",
                delta=None
            )
        
        with col2:
            total_volume = df['amount'].sum() if 'amount' in df.columns else 0
            st.metric(
                "Total Volume",
                f"${total_volume:,.0f}",
                delta=None
            )
        
        with col3:
            fraud_count = df['isFraud'].sum() if 'isFraud' in df.columns else 0
            st.metric(
                "Fraud Cases",
                f"{fraud_count:,}",
                delta=f"{(fraud_count/len(df)*100):.2f}%"
            )
        
        with col4:
            high_risk = df['high_risk_flag'].sum() if 'high_risk_flag' in df.columns else 0
            st.metric(
                "High Risk Alerts",
                f"{high_risk:,}",
                delta=f"{(high_risk/len(df)*100):.2f}%"
            )
        
        # Transaction volume over time
        st.subheader("Transaction Volume Trends")
        
        if 'step' in df.columns:
            trends = business_calc.calculate_transaction_volume_trends(df, frequency='D')
            if not trends.empty:
                fig = px.line(
                    trends,
                    x='period',
                    y='transaction_count',
                    title='Daily Transaction Volume',
                    labels={'transaction_count': 'Number of Transactions', 'period': 'Date'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Transaction type distribution
        st.subheader("Transaction Type Distribution")
        
        if 'type' in df.columns:
            type_dist = df['type'].value_counts()
            fig = px.pie(
                values=type_dist.values,
                names=type_dist.index,
                title="Transactions by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Fraud Detection Performance")
        
        if 'isFraud' in df.columns and 'high_risk_flag' in df.columns:
            # Detection metrics
            detection_metrics = business_calc.calculate_fraud_detection_rate(df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Detection Rate",
                    f"{detection_metrics.get('detection_rate_pct', 0):.2f}%"
                )
            
            with col2:
                st.metric(
                    "False Positive Rate",
                    f"{detection_metrics.get('false_positive_rate_pct', 0):.2f}%"
                )
            
            with col3:
                st.metric(
                    "Precision",
                    f"{detection_metrics.get('precision', 0):.2f}%"
                )
            
            # Confusion matrix visualization
            st.subheader("Detection Performance Matrix")
            
            tp = len(df[(df['high_risk_flag'] == True) & (df['isFraud'] == 1)])
            fp = len(df[(df['high_risk_flag'] == True) & (df['isFraud'] == 0)])
            tn = len(df[(df['high_risk_flag'] == False) & (df['isFraud'] == 0)])
            fn = len(df[(df['high_risk_flag'] == False) & (df['isFraud'] == 1)])
            
            confusion_matrix = pd.DataFrame({
                'Predicted: Low Risk': [tn, fn],
                'Predicted: High Risk': [fp, tp]
            }, index=['Actual: Legitimate', 'Actual: Fraud'])
            
            fig = px.imshow(
                confusion_matrix.values,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=confusion_matrix.columns,
                y=confusion_matrix.index,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk score distribution
            if 'final_risk_score' in df.columns:
                st.subheader("Risk Score Distribution")
                fig = px.histogram(
                    df,
                    x='final_risk_score',
                    color='isFraud' if 'isFraud' in df.columns else None,
                    nbins=50,
                    title="Distribution of Risk Scores"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Merchant Analytics")
        
        if 'nameDest' in df.columns:
            merchant_metrics = business_calc.calculate_merchant_risk_distribution(df)
            
            if not merchant_metrics.empty:
                st.subheader("Top Risky Merchants")
                
                # Display top merchants
                top_merchants = merchant_metrics.head(20)
                
                # Select available columns
                available_cols = []
                for col in ['merchant_id', 'avg_risk_score', 'transaction_count', 
                           'total_volume', 'fraud_rate', 'risk_category']:
                    if col in top_merchants.columns:
                        available_cols.append(col)
                
                if available_cols:
                    st.dataframe(
                        top_merchants[available_cols],
                        use_container_width=True
                    )
                else:
                    st.dataframe(top_merchants, use_container_width=True)
                
                # Risk category distribution
                if 'risk_category' in merchant_metrics.columns:
                    st.subheader("Merchant Risk Distribution")
                    risk_dist = merchant_metrics['risk_category'].value_counts()
                    fig = px.bar(
                        x=risk_dist.index,
                        y=risk_dist.values,
                        title="Merchants by Risk Category",
                        labels={'x': 'Risk Category', 'y': 'Number of Merchants'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Volume vs Risk scatter
                if 'total_volume' in merchant_metrics.columns and 'avg_risk_score' in merchant_metrics.columns:
                    st.subheader("Transaction Volume vs Risk Score")
                    scatter_kwargs = {
                        'x': 'total_volume',
                        'y': 'avg_risk_score',
                        'title': "Merchant Risk vs Volume",
                        'labels': {'total_volume': 'Total Volume', 'avg_risk_score': 'Average Risk Score'}
                    }
                    if 'transaction_count' in merchant_metrics.columns:
                        scatter_kwargs['size'] = 'transaction_count'
                    if 'risk_category' in merchant_metrics.columns:
                        scatter_kwargs['color'] = 'risk_category'
                    if 'merchant_id' in merchant_metrics.columns:
                        scatter_kwargs['hover_data'] = ['merchant_id']
                    
                    fig = px.scatter(merchant_metrics, **scatter_kwargs)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Export Data for BI Tools")
        
        st.markdown("""
        Export pre-aggregated views optimized for Power BI, Looker, and other BI tools.
        """)
        
        export_format = st.selectbox(
            "Select export format",
            ['parquet', 'csv', 'excel']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Transaction Data"):
                try:
                    filepath = bi_export.export_transactions_for_bi(df, format=export_format)
                    st.success(f"✅ Exported to: {filepath}")
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            "Download File",
                            f.read(),
                            filepath.split('/')[-1],
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            if st.button("Export Merchant Metrics"):
                try:
                    filepath = bi_export.export_merchant_metrics(df, format=export_format)
                    st.success(f"✅ Exported to: {filepath}")
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            "Download File",
                            f.read(),
                            filepath.split('/')[-1],
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            if st.button("Export Volume Trends"):
                try:
                    filepath = bi_export.export_volume_trends(df, format=export_format)
                    st.success(f"✅ Exported to: {filepath}")
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            "Download File",
                            f.read(),
                            filepath.split('/')[-1],
                            mime="application/octet-stream"
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            if st.button("Export All Views"):
                try:
                    exports = bi_export.export_all_views(df, formats=[export_format])
                    st.success(f"✅ Exported {len(exports)} views")
                    for name, filepath in exports.items():
                        st.text(f"  - {name}: {filepath}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Business summary report
        st.subheader("Generate Business Summary Report")
        if st.button("Generate Report"):
            try:
                report = business_calc.generate_business_summary_report(df)
                st.json(report)
                
                # Download report
                import json
                report_json = json.dumps(report, indent=2, default=str)
                st.download_button(
                    "Download Report (JSON)",
                    report_json,
                    f"business_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab5:
        st.header("Automated Reporting")
        st.markdown("Generate scheduled business reports for stakeholders")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generate Report")
            report_type = st.selectbox(
                "Report Type",
                ['daily', 'weekly', 'monthly'],
                help="Select the type of report to generate"
            )
            
            if st.button("Generate Report", type="primary"):
                try:
                    with st.spinner(f"Generating {report_type} report..."):
                        if report_type == 'daily':
                            result = reporting.generate_daily_report(df)
                        elif report_type == 'weekly':
                            result = reporting.generate_weekly_report(df)
                        else:
                            result = reporting.generate_monthly_report(df)
                        
                        st.success(f"✅ {report_type.title()} report generated!")
                        
                        # Display key metrics
                        metrics = result['metrics']
                        st.metric("Total Transactions", f"{metrics.get('total_transactions', 0):,}")
                        st.metric("Total Volume", f"${metrics.get('total_volume', 0):,.2f}")
                        st.metric("Fraud Cases", metrics.get('fraud_count', 0))
                        st.metric("Fraud Rate", f"{metrics.get('fraud_rate', 0):.2f}%")
                        
                        # Download links
                        st.subheader("Download Report")
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            with open(result['html_path'], 'rb') as f:
                                st.download_button(
                                    "📄 HTML Report",
                                    f.read(),
                                    result['html_path'].split('/')[-1],
                                    mime="text/html"
                                )
                        
                        with col_b:
                            with open(result['json_path'], 'rb') as f:
                                st.download_button(
                                    "📊 JSON Data",
                                    f.read(),
                                    result['json_path'].split('/')[-1],
                                    mime="application/json"
                                )
                        
                        with col_c:
                            with open(result['csv_path'], 'rb') as f:
                                st.download_button(
                                    "📋 CSV Summary",
                                    f.read(),
                                    result['csv_path'].split('/')[-1],
                                    mime="text/csv"
                                )
                        
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        with col2:
            st.subheader("Schedule Reports")
            st.markdown("""
            **Schedule automated reports using cron:**
            
            ```bash
            # Daily report at 9 AM
            0 9 * * * python scripts/schedule_reports.py --type daily
            
            # Weekly report every Monday
            0 9 * * 1 python scripts/schedule_reports.py --type weekly
            
            # Monthly report on 1st of month
            0 9 1 * * python scripts/schedule_reports.py --type monthly
            ```
            """)
            
            st.info("💡 Reports are saved to the `reports/` directory")

else:
    st.info("👆 Please upload a CSV file in the sidebar to get started")

# Footer
st.markdown("---")
st.markdown("**Transaction Anomaly Detection System** | Built for product teams and business stakeholders")


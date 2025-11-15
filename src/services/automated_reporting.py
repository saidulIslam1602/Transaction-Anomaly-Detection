"""
Automated reporting service for scheduled business reports.

This module provides functionality to generate and schedule automated reports
for product teams and business stakeholders.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from jinja2 import Template
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.services.business_metrics import BusinessMetricsCalculator
from src.services.product_metrics import ProductMetricsCalculator
from src.services.bi_export import BIExportService

logger = logging.getLogger(__name__)


class AutomatedReportingService:
    """
    Service for generating and scheduling automated reports.
    
    Supports daily, weekly, and monthly reports with customizable
    metrics and visualizations for different stakeholders.
    """
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize automated reporting service.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.business_calc = BusinessMetricsCalculator()
        self.product_calc = ProductMetricsCalculator()
        self.bi_export = BIExportService()
        
        logger.info(f"Automated reporting service initialized at {output_dir}")
    
    def generate_daily_report(self,
                             df: pd.DataFrame,
                             report_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate daily business report.
        
        Args:
            df: Transaction data for the day
            report_date: Date for the report (defaults to today)
            
        Returns:
            Dictionary with report data and file paths
        """
        if report_date is None:
            report_date = datetime.now()
        
        logger.info(f"Generating daily report for {report_date.date()}")
        
        # Calculate key metrics
        metrics = {
            'report_date': report_date.strftime('%Y-%m-%d'),
            'report_type': 'daily',
            'total_transactions': len(df),
            'total_volume': float(df['amount'].sum()) if 'amount' in df.columns else 0,
            'avg_transaction_amount': float(df['amount'].mean()) if 'amount' in df.columns else 0,
            'fraud_count': int(df['isFraud'].sum()) if 'isFraud' in df.columns else 0,
            'fraud_rate': float(df['isFraud'].mean() * 100) if 'isFraud' in df.columns else 0,
        }
        
        # Transaction type breakdown
        if 'type' in df.columns:
            type_breakdown = df['type'].value_counts().to_dict()
            metrics['transaction_types'] = type_breakdown
        
        # High-risk transactions
        if 'high_risk_flag' in df.columns:
            metrics['high_risk_count'] = int(df['high_risk_flag'].sum())
            metrics['high_risk_rate'] = float(df['high_risk_flag'].mean() * 100)
        
        # Merchant metrics
        if 'nameDest' in df.columns:
            merchant_metrics = self.business_calc.calculate_merchant_risk_distribution(df)
            if not merchant_metrics.empty:
                metrics['top_risky_merchants'] = merchant_metrics.head(10).to_dict('records')
                metrics['total_merchants'] = len(merchant_metrics)
        
        # Volume trends
        if 'step' in df.columns:
            try:
                trends = self.business_calc.calculate_transaction_volume_trends(df, frequency='H')
                if not trends.empty:
                    metrics['peak_hour'] = trends.loc[trends['transaction_count'].idxmax(), 'period'] if len(trends) > 0 else None
                    metrics['peak_volume'] = float(trends['total_volume'].max()) if len(trends) > 0 else 0
            except Exception as e:
                logger.warning(f"Could not calculate volume trends: {e}")
                # Fallback: simple aggregation by step
                if 'amount' in df.columns:
                    hourly = df.groupby('step').agg({'amount': ['count', 'sum']}).reset_index()
                    hourly.columns = ['step', 'transaction_count', 'total_volume']
                    if len(hourly) > 0:
                        peak_idx = hourly['transaction_count'].idxmax()
                        metrics['peak_hour'] = int(hourly.loc[peak_idx, 'step'])
                        metrics['peak_volume'] = float(hourly.loc[peak_idx, 'total_volume'])
        
        # Generate report file
        report_data = self._create_report_file(metrics, 'daily', report_date)
        
        return report_data
    
    def generate_weekly_report(self,
                              df: pd.DataFrame,
                              week_start: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Generate weekly business report.
        
        Args:
            df: Transaction data for the week
            week_start: Start date of the week (defaults to Monday of current week)
            
        Returns:
            Dictionary with report data and file paths
        """
        if week_start is None:
            today = datetime.now()
            week_start = today - timedelta(days=today.weekday())
        
        week_end = week_start + timedelta(days=6)
        
        logger.info(f"Generating weekly report for {week_start.date()} to {week_end.date()}")
        
        # Calculate key metrics
        metrics = {
            'report_period': f"{week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}",
            'report_type': 'weekly',
            'total_transactions': len(df),
            'total_volume': float(df['amount'].sum()) if 'amount' in df.columns else 0,
            'avg_daily_transactions': len(df) / 7,
            'avg_daily_volume': float(df['amount'].sum() / 7) if 'amount' in df.columns else 0,
            'fraud_count': int(df['isFraud'].sum()) if 'isFraud' in df.columns else 0,
            'fraud_rate': float(df['isFraud'].mean() * 100) if 'isFraud' in df.columns else 0,
        }
        
        # Daily breakdown
        if 'step' in df.columns:
            try:
                daily_trends = self.business_calc.calculate_transaction_volume_trends(df, frequency='D')
                if not daily_trends.empty:
                    metrics['daily_breakdown'] = daily_trends.to_dict('records')
                    metrics['busiest_day'] = daily_trends.loc[daily_trends['transaction_count'].idxmax(), 'period'] if len(daily_trends) > 0 else None
            except Exception as e:
                logger.warning(f"Could not calculate daily trends: {e}")
                # Fallback: simple aggregation
                if 'amount' in df.columns:
                    daily = df.groupby(df['step'] // 24).agg({'amount': ['count', 'sum']}).reset_index()
                    daily.columns = ['day', 'transaction_count', 'total_volume']
                    metrics['daily_breakdown'] = daily.to_dict('records')
        
        # Transaction type analysis
        if 'type' in df.columns:
            type_df = df.groupby('type').agg({
                'amount': ['count', 'sum', 'mean'],
                'isFraud': 'sum' if 'isFraud' in df.columns else 'count'
            })
            # Convert to serializable format
            type_analysis = {}
            for col in type_df.columns:
                col_name = f"{col[0]}_{col[1]}" if isinstance(col, tuple) else str(col)
                type_analysis[col_name] = type_df[col].to_dict()
            metrics['transaction_type_analysis'] = type_analysis
        
        # Top merchants
        if 'nameDest' in df.columns:
            merchant_metrics = self.business_calc.calculate_merchant_risk_distribution(df)
            if not merchant_metrics.empty:
                metrics['top_merchants'] = merchant_metrics.head(20).to_dict('records')
        
        # Week-over-week comparison (if historical data available)
        metrics['comparison_note'] = "Week-over-week comparison requires historical data"
        
        # Generate report file
        report_data = self._create_report_file(metrics, 'weekly', week_start)
        
        return report_data
    
    def generate_monthly_report(self,
                               df: pd.DataFrame,
                               month: Optional[int] = None,
                               year: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate monthly business report.
        
        Args:
            df: Transaction data for the month
            month: Month number (1-12, defaults to current month)
            year: Year (defaults to current year)
            
        Returns:
            Dictionary with report data and file paths
        """
        if month is None:
            month = datetime.now().month
        if year is None:
            year = datetime.now().year
        
        month_start = datetime(year, month, 1)
        if month == 12:
            month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = datetime(year, month + 1, 1) - timedelta(days=1)
        
        logger.info(f"Generating monthly report for {month_start.strftime('%B %Y')}")
        
        # Calculate comprehensive metrics
        metrics = {
            'report_period': month_start.strftime('%B %Y'),
            'report_type': 'monthly',
            'total_transactions': len(df),
            'total_volume': float(df['amount'].sum()) if 'amount' in df.columns else 0,
            'avg_daily_transactions': len(df) / month_end.day,
            'avg_daily_volume': float(df['amount'].sum() / month_end.day) if 'amount' in df.columns else 0,
            'fraud_count': int(df['isFraud'].sum()) if 'isFraud' in df.columns else 0,
            'fraud_rate': float(df['isFraud'].mean() * 100) if 'isFraud' in df.columns else 0,
        }
        
        # Weekly breakdown
        if 'step' in df.columns:
            try:
                weekly_trends = self.business_calc.calculate_transaction_volume_trends(df, frequency='W')
                if not weekly_trends.empty:
                    metrics['weekly_breakdown'] = weekly_trends.to_dict('records')
            except Exception as e:
                logger.warning(f"Could not calculate weekly trends: {e}")
                # Fallback: simple aggregation
                if 'amount' in df.columns:
                    weekly = df.groupby(df['step'] // (24 * 7)).agg({'amount': ['count', 'sum']}).reset_index()
                    weekly.columns = ['week', 'transaction_count', 'total_volume']
                    metrics['weekly_breakdown'] = weekly.to_dict('records')
        
        # Merchant analysis
        if 'nameDest' in df.columns:
            merchant_metrics = self.business_calc.calculate_merchant_risk_distribution(df)
            if not merchant_metrics.empty:
                metrics['merchant_summary'] = {
                    'total_merchants': len(merchant_metrics),
                    'high_risk_merchants': int((merchant_metrics.get('risk_category', pd.Series()) == 'HIGH').sum()) if 'risk_category' in merchant_metrics.columns else 0,
                    'top_merchants': merchant_metrics.head(30).to_dict('records')
                }
        
        # Product metrics
        if 'nameOrig' in df.columns:
            try:
                user_patterns = self.product_calc.calculate_user_transaction_patterns(df)
                if not user_patterns.empty:
                    metrics['user_insights'] = {
                        'total_users': len(user_patterns),
                        'avg_transactions_per_user': float(user_patterns['transaction_count'].mean()),
                        'top_users': user_patterns.head(10).to_dict('records')
                    }
            except Exception as e:
                logger.warning(f"Could not calculate user patterns: {e}")
                # Fallback: simple user aggregation
                if 'nameOrig' in df.columns and 'amount' in df.columns:
                    user_simple = df.groupby('nameOrig').agg({
                        'amount': ['count', 'sum']
                    }).reset_index()
                    user_simple.columns = ['user_id', 'transaction_count', 'total_volume']
                    metrics['user_insights'] = {
                        'total_users': len(user_simple),
                        'avg_transactions_per_user': float(user_simple['transaction_count'].mean()),
                        'top_users': user_simple.nlargest(10, 'transaction_count').to_dict('records')
                    }
        
        # Generate report file
        report_data = self._create_report_file(metrics, 'monthly', month_start)
        
        return report_data
    
    def _create_report_file(self,
                           metrics: Dict[str, Any],
                           report_type: str,
                           report_date: datetime) -> Dict[str, Any]:
        """
        Create report files (JSON, HTML, CSV summary).
        
        Args:
            metrics: Report metrics dictionary
            report_type: Type of report (daily/weekly/monthly)
            report_date: Date for the report
            
        Returns:
            Dictionary with file paths and report data
        """
        timestamp = report_date.strftime('%Y%m%d')
        filename_base = f"{report_type}_report_{timestamp}"
        
        # Save JSON report (clean metrics for JSON serialization)
        json_metrics = self._clean_for_json(metrics)
        json_path = self.output_dir / f"{filename_base}.json"
        with open(json_path, 'w') as f:
            json.dump(json_metrics, f, indent=2, default=str)
        
        # Generate HTML report
        html_path = self.output_dir / f"{filename_base}.html"
        html_content = self._generate_html_report(metrics, report_type, report_date)
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        # Generate CSV summary
        csv_path = self.output_dir / f"{filename_base}_summary.csv"
        summary_df = pd.DataFrame([{
            'metric': k,
            'value': str(v) if not isinstance(v, (dict, list)) else json.dumps(v)
        } for k, v in metrics.items() if not isinstance(v, (dict, list)) or k in ['total_transactions', 'total_volume', 'fraud_count', 'fraud_rate']])
        summary_df.to_csv(csv_path, index=False)
        
        logger.info(f"Report files created: {json_path}, {html_path}, {csv_path}")
        
        return {
            'json_path': str(json_path),
            'html_path': str(html_path),
            'csv_path': str(csv_path),
            'metrics': metrics,
            'report_type': report_type,
            'report_date': report_date.isoformat()
        }
    
    def _generate_html_report(self,
                             metrics: Dict[str, Any],
                             report_type: str,
                             report_date: datetime) -> str:
        """Generate HTML report with styling."""
        
        # Format numbers
        def fmt_num(v):
            if isinstance(v, (int, float)):
                return f"{v:,.0f}"
            return str(v)
        
        def fmt_curr(v):
            if isinstance(v, (int, float)):
                return f"${v:,.2f}"
            return str(v)
        
        # Build HTML content
        total_txns = fmt_num(metrics.get('total_transactions', 0))
        total_vol = fmt_curr(metrics.get('total_volume', 0))
        fraud_cnt = fmt_num(metrics.get('fraud_count', 0))
        fraud_rate = f"{metrics.get('fraud_rate', 0):.2f}%"
        
        # Transaction types table
        type_table = ""
        if 'transaction_types' in metrics:
            type_table = "<h2>Transaction Types</h2><table><tr><th>Type</th><th>Count</th></tr>"
            for ttype, count in metrics['transaction_types'].items():
                type_table += f"<tr><td>{ttype}</td><td>{fmt_num(count)}</td></tr>"
            type_table += "</table>"
        
        # Top merchants table
        merchants_table = ""
        if 'top_risky_merchants' in metrics and metrics['top_risky_merchants']:
            merchants_table = "<h2>Top Risky Merchants</h2><table><tr><th>Merchant ID</th><th>Risk Score</th><th>Transactions</th><th>Volume</th></tr>"
            for m in metrics['top_risky_merchants'][:10]:
                mid = m.get('merchant_id', 'N/A')
                risk = f"{m.get('avg_risk_score', 0):.2f}"
                txn_cnt = fmt_num(m.get('transaction_count', 0))
                vol = fmt_curr(m.get('total_volume', 0))
                merchants_table += f"<tr><td>{mid}</td><td>{risk}</td><td>{txn_cnt}</td><td>{vol}</td></tr>"
            merchants_table += "</table>"
        
        period_info = ""
        if 'report_period' in metrics:
            period_info = f"<p><strong>Period:</strong> {metrics['report_period']}</p>"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_type.title()} Report - {report_date.strftime('%Y-%m-%d')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report_type.title()} Business Report</h1>
        <p><strong>Report Date:</strong> {report_date.strftime('%Y-%m-%d')}</p>
        {period_info}
        
        <h2>Key Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Transactions</div>
                <div class="metric-value">{total_txns}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Volume</div>
                <div class="metric-value">{total_vol}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Fraud Cases</div>
                <div class="metric-value">{fraud_cnt}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Fraud Rate</div>
                <div class="metric-value">{fraud_rate}</div>
            </div>
        </div>
        
        {type_table}
        {merchants_table}
        
        <div class="footer">
            <p>Generated by Transaction Anomaly Detection System</p>
            <p>Report Type: {report_type.title()} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def _clean_for_json(self, obj: Any) -> Any:
        """Recursively clean object for JSON serialization."""
        if isinstance(obj, dict):
            return {str(k): self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return obj
    
    def schedule_report(self,
                       report_type: str,
                       schedule: str,
                       data_source: str) -> Dict[str, Any]:
        """
        Schedule automated reports (creates configuration file).
        
        Args:
            report_type: Type of report (daily/weekly/monthly)
            schedule: Schedule string (e.g., 'daily', 'weekly', 'monthly')
            data_source: Path to data source or database connection
            
        Returns:
            Dictionary with schedule configuration
        """
        schedule_config = {
            'report_type': report_type,
            'schedule': schedule,
            'data_source': data_source,
            'created_at': datetime.now().isoformat(),
            'enabled': True
        }
        
        config_path = self.output_dir / 'scheduled_reports.json'
        
        # Load existing schedules
        if config_path.exists():
            with open(config_path, 'r') as f:
                schedules = json.load(f)
        else:
            schedules = []
        
        schedules.append(schedule_config)
        
        # Save schedules
        with open(config_path, 'w') as f:
            json.dump(schedules, f, indent=2)
        
        logger.info(f"Scheduled {report_type} report: {schedule}")
        
        return schedule_config


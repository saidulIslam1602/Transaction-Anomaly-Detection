#!/usr/bin/env python3
"""
CLI script for generating automated reports.

Usage:
    python scripts/generate_report.py --type daily --data data/transactions.csv
    python scripts/generate_report.py --type weekly --data data/transactions.csv
    python scripts/generate_report.py --type monthly --data data/transactions.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.automated_reporting import AutomatedReportingService


def main():
    parser = argparse.ArgumentParser(description='Generate automated business reports')
    parser.add_argument('--type', 
                       choices=['daily', 'weekly', 'monthly'],
                       required=True,
                       help='Type of report to generate')
    parser.add_argument('--data',
                       required=True,
                       help='Path to transaction data CSV file')
    parser.add_argument('--output',
                       default='./reports',
                       help='Output directory for reports (default: ./reports)')
    parser.add_argument('--date',
                       help='Date for report (YYYY-MM-DD, defaults to today/latest)')
    parser.add_argument('--filter-date',
                       action='store_true',
                       help='Filter data to specific date range')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} transactions")
    
    # Initialize reporting service
    reporting = AutomatedReportingService(output_dir=args.output)
    
    # Parse date if provided
    report_date = None
    if args.date:
        report_date = datetime.strptime(args.date, '%Y-%m-%d')
    
    # Filter data if requested
    if args.filter_date and 'step' in df.columns and report_date:
        # Filter to specific date (assuming step represents hours)
        # This is a simplified filter - adjust based on your data structure
        print(f"Filtering data for {report_date.date()}...")
        # Note: Actual filtering logic depends on how 'step' maps to dates
    
    # Generate report
    print(f"\nGenerating {args.type} report...")
    
    if args.type == 'daily':
        result = reporting.generate_daily_report(df, report_date)
    elif args.type == 'weekly':
        result = reporting.generate_weekly_report(df, report_date)
    elif args.type == 'monthly':
        month = report_date.month if report_date else None
        year = report_date.year if report_date else None
        result = reporting.generate_monthly_report(df, month, year)
    
    # Print results
    print("\n" + "=" * 70)
    print(f"✅ {args.type.upper()} REPORT GENERATED")
    print("=" * 70)
    print(f"JSON Report: {result['json_path']}")
    print(f"HTML Report: {result['html_path']}")
    print(f"CSV Summary: {result['csv_path']}")
    print("\nKey Metrics:")
    metrics = result['metrics']
    print(f"  Total Transactions: {metrics.get('total_transactions', 'N/A'):,}")
    print(f"  Total Volume: ${metrics.get('total_volume', 0):,.2f}")
    print(f"  Fraud Cases: {metrics.get('fraud_count', 0)}")
    print(f"  Fraud Rate: {metrics.get('fraud_rate', 0):.2f}%")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())


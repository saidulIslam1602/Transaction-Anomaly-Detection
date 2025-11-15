#!/usr/bin/env python3
"""
Schedule automated reports using cron or system scheduler.

This script can be run via cron to generate scheduled reports.
Example cron entries:
    # Daily report at 9 AM
    0 9 * * * /usr/bin/python3 /path/to/scripts/schedule_reports.py --type daily
    
    # Weekly report every Monday at 9 AM
    0 9 * * 1 /usr/bin/python3 /path/to/scripts/schedule_reports.py --type weekly
    
    # Monthly report on 1st of month at 9 AM
    0 9 1 * * /usr/bin/python3 /path/to/scripts/schedule_reports.py --type monthly
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.automated_reporting import AutomatedReportingService


def main():
    parser = argparse.ArgumentParser(description='Generate scheduled automated reports')
    parser.add_argument('--type',
                       choices=['daily', 'weekly', 'monthly'],
                       required=True,
                       help='Type of report to generate')
    parser.add_argument('--data',
                       default='data/transactions.csv',
                       help='Path to transaction data CSV file')
    parser.add_argument('--output',
                       default='./reports',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    try:
        # Load data
        df = pd.read_csv(args.data)
        
        # Initialize reporting service
        reporting = AutomatedReportingService(output_dir=args.output)
        
        # Generate report
        if args.type == 'daily':
            result = reporting.generate_daily_report(df)
        elif args.type == 'weekly':
            result = reporting.generate_weekly_report(df)
        elif args.type == 'monthly':
            result = reporting.generate_monthly_report(df)
        
        print(f"✅ {args.type.upper()} report generated: {result['html_path']}")
        return 0
        
    except Exception as e:
        print(f"❌ Error generating report: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())


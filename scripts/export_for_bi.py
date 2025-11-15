#!/usr/bin/env python3
"""
CLI script for exporting data to BI tools.

Usage:
    python scripts/export_for_bi.py --input data/transactions.csv --output bi_exports/
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.services.bi_export import BIExportService
from src.services.business_metrics import BusinessMetricsCalculator


def main():
    parser = argparse.ArgumentParser(
        description='Export transaction data for BI tools (Power BI, Looker, etc.)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file with transaction data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./bi_exports',
        help='Output directory for exported files'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        default='parquet',
        choices=['parquet', 'csv', 'excel'],
        help='Export format (default: parquet)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Export all views (transactions, merchants, trends, performance)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} transactions")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Initialize export service
    export_service = BIExportService(output_dir=args.output)
    
    # Export based on options
    if args.all:
        print(f"Exporting all views in {args.format} format...")
        exports = export_service.export_all_views(df, formats=[args.format])
        print(f"\n✅ Exported {len(exports)} views:")
        for name, filepath in exports.items():
            print(f"  - {name}: {filepath}")
    else:
        print(f"Exporting transaction data in {args.format} format...")
        filepath = export_service.export_transactions_for_bi(df, format=args.format)
        print(f"✅ Exported to: {filepath}")
    
    # Generate business summary
    print("\nGenerating business summary report...")
    calculator = BusinessMetricsCalculator()
    report = calculator.generate_business_summary_report(
        df,
        output_path=str(Path(args.output) / 'business_summary.json')
    )
    print(f"✅ Business summary saved to {args.output}/business_summary.json")
    
    print("\n✅ Export complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())


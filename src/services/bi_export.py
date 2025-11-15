"""
Business Intelligence export service.

This module provides functionality to export data in formats suitable for
Power BI, Looker, and other BI tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BIExportService:
    """
    Service for exporting data to BI tools like Power BI and Looker.
    
    Provides pre-aggregated views and exports in various formats
    (CSV, Parquet, Excel) optimized for business intelligence tools.
    """
    
    def __init__(self, output_dir: str = "./bi_exports"):
        """
        Initialize BI export service.
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"BI export service initialized with output directory: {output_dir}")
    
    def export_transactions_for_bi(self,
                                  df: pd.DataFrame,
                                  format: str = 'parquet',
                                  filename: Optional[str] = None) -> str:
        """
        Export transaction data optimized for BI tools.
        
        Args:
            df: DataFrame with transaction data
            format: Export format ('parquet', 'csv', 'excel')
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        # Prepare data for BI tools
        bi_df = df.copy()
        
        # Ensure date columns are properly formatted
        if 'step' in bi_df.columns and bi_df['step'].dtype in ['int64', 'int32']:
            base_date = pd.Timestamp('2020-01-01')
            bi_df['transaction_date'] = base_date + pd.to_timedelta(bi_df['step'], unit='h')
            bi_df['transaction_hour'] = bi_df['step'] % 24
            bi_df['transaction_day_of_week'] = (bi_df['step'] // 24) % 7
        else:
            bi_df['transaction_date'] = pd.to_datetime(bi_df.get('step', datetime.now()))
        
        # Select relevant columns for BI
        bi_columns = [
            'transaction_date',
            'type',
            'amount',
            'nameOrig',
            'nameDest',
            'oldbalanceOrg',
            'newbalanceOrig',
            'oldbalanceDest',
            'newbalanceDest'
        ]
        
        # Add risk columns if available
        if 'final_risk_score' in bi_df.columns:
            bi_columns.append('final_risk_score')
        if 'high_risk_flag' in bi_df.columns:
            bi_columns.append('high_risk_flag')
        if 'risk_level' in bi_df.columns:
            bi_columns.append('risk_level')
        
        bi_df = bi_df[[col for col in bi_columns if col in bi_df.columns]]
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transactions_bi_{timestamp}.{format}"
        
        filepath = self.output_dir / filename
        
        # Export based on format
        if format == 'parquet':
            bi_df.to_parquet(filepath, index=False, engine='pyarrow')
        elif format == 'csv':
            bi_df.to_csv(filepath, index=False)
        elif format == 'excel':
            bi_df.to_excel(filepath, index=False, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(bi_df)} transactions to {filepath}")
        return str(filepath)
    
    def export_merchant_metrics(self,
                               df: pd.DataFrame,
                               format: str = 'parquet',
                               filename: Optional[str] = None) -> str:
        """
        Export merchant-level aggregated metrics for BI tools.
        
        Args:
            df: DataFrame with transaction data
            format: Export format ('parquet', 'csv', 'excel')
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        from src.services.business_metrics import BusinessMetricsCalculator
        
        calculator = BusinessMetricsCalculator()
        try:
            merchant_metrics = calculator.calculate_merchant_risk_distribution(df)
        except Exception as e:
            logger.warning(f"Error calculating merchant metrics: {e}")
            # Fallback: create basic merchant aggregation
            if 'nameDest' in df.columns and 'amount' in df.columns:
                merchant_metrics = df.groupby('nameDest').agg({
                    'amount': ['count', 'sum', 'mean'],
                    'nameOrig': 'nunique'
                }).reset_index()
                merchant_metrics.columns = ['merchant_id', 'transaction_count', 'total_volume', 'avg_amount', 'unique_customers']
            else:
                return ""
        
        if merchant_metrics.empty:
            logger.warning("No merchant metrics to export")
            return ""
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"merchant_metrics_{timestamp}.{format}"
        
        filepath = self.output_dir / filename
        
        # Export based on format
        if format == 'parquet':
            merchant_metrics.to_parquet(filepath, index=False, engine='pyarrow')
        elif format == 'csv':
            merchant_metrics.to_csv(filepath, index=False)
        elif format == 'excel':
            merchant_metrics.to_excel(filepath, index=False, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported merchant metrics to {filepath}")
        return str(filepath)
    
    def export_volume_trends(self,
                           df: pd.DataFrame,
                           frequency: str = 'D',
                           format: str = 'parquet',
                           filename: Optional[str] = None) -> str:
        """
        Export transaction volume trends for time-series analysis in BI tools.
        
        Args:
            df: DataFrame with transaction data
            frequency: Time frequency ('D', 'W', 'M')
            format: Export format ('parquet', 'csv', 'excel')
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        try:
            from src.services.business_metrics import BusinessMetricsCalculator
            
            calculator = BusinessMetricsCalculator()
            trends = calculator.calculate_transaction_volume_trends(df, frequency=frequency)
            
            if trends.empty:
                logger.warning("No trend data to export")
                # Create minimal trend data from available columns
                if 'step' in df.columns:
                    trends = df.groupby('step').agg({
                        'amount': ['count', 'sum', 'mean']
                    }).reset_index()
                    trends.columns = ['period', 'transaction_count', 'total_volume', 'avg_amount']
                else:
                    return ""
        except Exception as e:
            logger.warning(f"Error calculating trends, creating basic aggregation: {e}")
            # Fallback: create basic aggregation
            if 'step' in df.columns and 'amount' in df.columns:
                trends = df.groupby('step').agg({
                    'amount': ['count', 'sum', 'mean']
                }).reset_index()
                trends.columns = ['period', 'transaction_count', 'total_volume', 'avg_amount']
            else:
                return ""
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"volume_trends_{frequency}_{timestamp}.{format}"
        
        filepath = self.output_dir / filename
        
        # Export based on format
        if format == 'parquet':
            trends.to_parquet(filepath, index=False, engine='pyarrow')
        elif format == 'csv':
            trends.to_csv(filepath, index=False)
        elif format == 'excel':
            trends.to_excel(filepath, index=False, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported volume trends to {filepath}")
        return str(filepath)
    
    def export_detection_performance(self,
                                   df: pd.DataFrame,
                                   format: str = 'parquet',
                                   filename: Optional[str] = None) -> str:
        """
        Export detection performance metrics for BI dashboards.
        
        Args:
            df: DataFrame with detection results
            format: Export format ('parquet', 'csv', 'excel')
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        # Create performance summary
        if 'high_risk_flag' not in df.columns or 'isFraud' not in df.columns:
            logger.warning("Missing required columns for performance export")
            return ""
        
        performance_df = pd.DataFrame({
            'metric': [
                'total_transactions',
                'total_fraud',
                'detected_fraud',
                'false_positives',
                'true_positives',
                'false_negatives'
            ],
            'value': [
                len(df),
                df['isFraud'].sum(),
                df[df['high_risk_flag'] & df['isFraud']].shape[0],
                df[df['high_risk_flag'] & ~df['isFraud']].shape[0],
                df[df['high_risk_flag'] & df['isFraud']].shape[0],
                df[~df['high_risk_flag'] & df['isFraud']].shape[0]
            ]
        })
        
        # Add calculated rates
        performance_df = pd.concat([
            performance_df,
            pd.DataFrame({
                'metric': [
                    'detection_rate_pct',
                    'false_positive_rate_pct',
                    'precision_pct',
                    'recall_pct'
                ],
                'value': [
                    (performance_df[performance_df['metric'] == 'detected_fraud']['value'].values[0] /
                     performance_df[performance_df['metric'] == 'total_fraud']['value'].values[0] * 100)
                    if performance_df[performance_df['metric'] == 'total_fraud']['value'].values[0] > 0 else 0,
                    (performance_df[performance_df['metric'] == 'false_positives']['value'].values[0] /
                     len(df) * 100),
                    (performance_df[performance_df['metric'] == 'true_positives']['value'].values[0] /
                     (performance_df[performance_df['metric'] == 'true_positives']['value'].values[0] +
                      performance_df[performance_df['metric'] == 'false_positives']['value'].values[0]) * 100)
                    if (performance_df[performance_df['metric'] == 'true_positives']['value'].values[0] +
                        performance_df[performance_df['metric'] == 'false_positives']['value'].values[0]) > 0 else 0,
                    (performance_df[performance_df['metric'] == 'true_positives']['value'].values[0] /
                     performance_df[performance_df['metric'] == 'total_fraud']['value'].values[0] * 100)
                    if performance_df[performance_df['metric'] == 'total_fraud']['value'].values[0] > 0 else 0
                ]
            })
        ], ignore_index=True)
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_performance_{timestamp}.{format}"
        
        filepath = self.output_dir / filename
        
        # Export based on format
        if format == 'parquet':
            performance_df.to_parquet(filepath, index=False, engine='pyarrow')
        elif format == 'csv':
            performance_df.to_csv(filepath, index=False)
        elif format == 'excel':
            performance_df.to_excel(filepath, index=False, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported detection performance to {filepath}")
        return str(filepath)
    
    def export_all_views(self,
                        df: pd.DataFrame,
                        formats: List[str] = ['parquet', 'csv']) -> Dict[str, List[str]]:
        """
        Export all pre-aggregated views for BI tools.
        
        Args:
            df: DataFrame with transaction data
            formats: List of formats to export ('parquet', 'csv', 'excel')
            
        Returns:
            Dictionary mapping view names to exported file paths
        """
        exports = {}
        
        for format in formats:
            try:
                exports[f'transactions_{format}'] = self.export_transactions_for_bi(df, format=format)
                exports[f'merchant_metrics_{format}'] = self.export_merchant_metrics(df, format=format)
                exports[f'volume_trends_daily_{format}'] = self.export_volume_trends(df, frequency='D', format=format)
                exports[f'volume_trends_weekly_{format}'] = self.export_volume_trends(df, frequency='W', format=format)
                exports[f'detection_performance_{format}'] = self.export_detection_performance(df, format=format)
            except Exception as e:
                logger.error(f"Error exporting {format}: {str(e)}")
        
        logger.info(f"Exported {len(exports)} views in {len(formats)} formats")
        return exports


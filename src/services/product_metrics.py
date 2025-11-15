"""
Product team metrics and insights.

This module provides product-focused metrics and insights that help product teams
understand user behavior, transaction patterns, and system impact.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ProductMetricsCalculator:
    """
    Calculate product-focused metrics for product teams.
    
    These metrics help product teams understand:
    - User transaction patterns
    - Feature adoption and usage
    - Product impact metrics
    - User experience indicators
    """
    
    def __init__(self):
        """Initialize the product metrics calculator."""
        logger.info("Product metrics calculator initialized")
    
    def calculate_user_transaction_patterns(self,
                                           df: pd.DataFrame,
                                           user_column: str = 'nameOrig') -> pd.DataFrame:
        """
        Analyze transaction patterns per user.
        
        Args:
            df: DataFrame with transaction data
            user_column: Column name for user identifier
            
        Returns:
            DataFrame with user transaction patterns
        """
        if user_column not in df.columns:
            logger.warning(f"User column '{user_column}' not found")
            return pd.DataFrame()
        
        user_patterns = df.groupby(user_column).agg({
            'amount': ['count', 'sum', 'mean', 'std'],
            'type': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
            'high_risk_flag': 'sum' if 'high_risk_flag' in df.columns else 'count'
        }).reset_index()
        
        user_patterns.columns = [
            'user_id',
            'transaction_count',
            'total_volume',
            'avg_transaction',
            'transaction_std',
            'most_common_type',
            'high_risk_count'
        ]
        
        user_patterns['high_risk_rate'] = (
            user_patterns['high_risk_count'] / user_patterns['transaction_count'] * 100
        )
        
        return user_patterns.sort_values('transaction_count', ascending=False)
    
    def calculate_transaction_type_distribution(self,
                                              df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate distribution of transaction types.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Dictionary with transaction type metrics
        """
        if 'type' not in df.columns:
            return {}
        
        type_dist = df.groupby('type').agg({
            'amount': ['count', 'sum', 'mean'],
            'high_risk_flag': 'sum' if 'high_risk_flag' in df.columns else 'count'
        })
        
        type_dist.columns = ['count', 'total_volume', 'avg_amount', 'high_risk_count']
        type_dist['percentage'] = (type_dist['count'] / len(df) * 100).round(2)
        type_dist['high_risk_rate'] = (
            type_dist['high_risk_count'] / type_dist['count'] * 100
        ).round(2)
        
        return type_dist.to_dict('index')
    
    def calculate_time_based_insights(self,
                                     df: pd.DataFrame,
                                     time_column: str = 'step') -> Dict[str, Any]:
        """
        Calculate time-based transaction insights.
        
        Args:
            df: DataFrame with transaction data
            time_column: Column name for time dimension
            
        Returns:
            Dictionary with time-based insights
        """
        if time_column not in df.columns:
            return {}
        
        # Convert step to hour of day (assuming step represents hours)
        if df[time_column].dtype in ['int64', 'int32']:
            df_copy = df.copy()
            df_copy['hour_of_day'] = df_copy[time_column] % 24
        else:
            df_copy = df.copy()
            df_copy['hour_of_day'] = pd.to_datetime(df_copy[time_column]).dt.hour
        
        hourly_patterns = df_copy.groupby('hour_of_day').agg({
            'amount': ['count', 'sum'],
            'high_risk_flag': 'sum' if 'high_risk_flag' in df.columns else 'count'
        })
        
        hourly_patterns.columns = ['transaction_count', 'total_volume', 'high_risk_count']
        
        peak_hour = hourly_patterns['transaction_count'].idxmax()
        quiet_hour = hourly_patterns['transaction_count'].idxmin()
        
        return {
            'peak_hour': int(peak_hour),
            'quiet_hour': int(quiet_hour),
            'hourly_patterns': hourly_patterns.to_dict('index'),
            'peak_hour_volume': float(hourly_patterns.loc[peak_hour, 'transaction_count'])
        }
    
    def calculate_feature_adoption_metrics(self,
                                         df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate metrics related to feature adoption and usage.
        
        Args:
            df: DataFrame with transaction and feature data
            
        Returns:
            Dictionary with feature adoption metrics
        """
        metrics = {}
        
        # Check for various features
        if 'final_risk_score' in df.columns:
            metrics['risk_scoring_adoption'] = {
                'users_with_risk_scores': int(df['nameOrig'].nunique() if 'nameOrig' in df.columns else 0),
                'avg_risk_score': float(df['final_risk_score'].mean()),
                'high_risk_users_pct': float(
                    (df[df['final_risk_score'] >= 7.0]['nameOrig'].nunique() / 
                     df['nameOrig'].nunique() * 100) if 'nameOrig' in df.columns else 0
                )
            }
        
        if 'ml_score' in df.columns:
            metrics['ml_detection_usage'] = {
                'transactions_with_ml_scores': int(df['ml_score'].notna().sum()),
                'ml_detection_rate': float(df['ml_score'].notna().sum() / len(df) * 100)
            }
        
        return metrics
    
    def generate_product_insights_report(self,
                                       df: pd.DataFrame,
                                       output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive product insights report.
        
        Args:
            df: DataFrame with transaction data
            output_path: Optional path to save report
            
        Returns:
            Dictionary with product insights
        """
        report = {
            'report_date': datetime.now().isoformat(),
            'transaction_type_distribution': self.calculate_transaction_type_distribution(df),
            'time_based_insights': self.calculate_time_based_insights(df),
            'feature_adoption': self.calculate_feature_adoption_metrics(df)
        }
        
        # Add user patterns summary
        user_patterns = self.calculate_user_transaction_patterns(df)
        if not user_patterns.empty:
            report['user_patterns'] = {
                'total_users': len(user_patterns),
                'active_users': int((user_patterns['transaction_count'] > 0).sum()),
                'avg_transactions_per_user': float(user_patterns['transaction_count'].mean()),
                'top_10_users': user_patterns.head(10).to_dict('records')
            }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Product insights report saved to {output_path}")
        
        return report


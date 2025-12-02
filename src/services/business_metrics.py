"""
Business metrics and KPI calculations for transaction anomaly detection.

This module provides business-focused metrics that product teams and stakeholders
can use to understand system performance and make data-driven decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BusinessMetricsCalculator:
    """
    Calculate business-focused metrics and KPIs for fraud detection system.
    
    These metrics help product teams understand:
    - System performance from a business perspective
    - Cost and efficiency metrics
    - Merchant and transaction patterns
    - Detection quality and impact
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the business metrics calculator.
        
        Args:
            config: Configuration dictionary with business metrics settings
        """
        self.config = config or {}
        self.cost_per_alert_review = self.config.get('business_metrics', {}).get('cost_per_alert_review', 10.0)
        self.industry_benchmarks = self.config.get('business_metrics', {}).get('industry_benchmarks', {
            'avg_fraud_rate': 0.02,
            'avg_risk_score': 3.5,
            'avg_transaction_amount': 5000.0
        })
        logger.info("Business metrics calculator initialized")
    
    def calculate_fraud_detection_rate(self,
                                      df: pd.DataFrame,
                                      time_window: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate fraud detection rate and related metrics.
        
        Args:
            df: DataFrame with transactions and fraud flags
            time_window: Optional time window ('daily', 'weekly', 'monthly')
            
        Returns:
            Dictionary with detection rate metrics
        """
        if 'isFraud' not in df.columns or 'high_risk_flag' not in df.columns:
            logger.warning("Missing required columns for fraud detection rate calculation")
            return {}
        
        total_transactions = len(df)
        total_fraud = df['isFraud'].sum() if 'isFraud' in df.columns else 0
        detected_fraud = df[df['high_risk_flag'] == True]['isFraud'].sum() if 'isFraud' in df.columns else 0
        
        detection_rate = (detected_fraud / total_fraud * 100) if total_fraud > 0 else 0
        false_positive_rate = ((df['high_risk_flag'].sum() - detected_fraud) / total_transactions * 100) if total_transactions > 0 else 0
        
        metrics = {
            'total_transactions': int(total_transactions),
            'total_fraud': int(total_fraud),
            'detected_fraud': int(detected_fraud),
            'detection_rate_pct': round(detection_rate, 2),
            'false_positive_rate_pct': round(false_positive_rate, 2),
            'true_positive_rate_pct': round(detection_rate, 2),
            'precision': round(detected_fraud / df['high_risk_flag'].sum() * 100, 2) if df['high_risk_flag'].sum() > 0 else 0
        }
        
        if time_window:
            metrics['time_window'] = time_window
        
        logger.info(f"Fraud detection rate: {detection_rate:.2f}%")
        return metrics
    
    def calculate_merchant_risk_distribution(self,
                                            df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate risk distribution across merchants.
        
        Args:
            df: DataFrame with merchant and risk information
            
        Returns:
            DataFrame with merchant risk metrics
        """
        if 'nameDest' not in df.columns or 'final_risk_score' not in df.columns:
            logger.warning("Missing required columns for merchant risk distribution")
            return pd.DataFrame()
        
        merchant_metrics = df.groupby('nameDest').agg({
            'final_risk_score': ['mean', 'max', 'count'],
            'high_risk_flag': 'sum',
            'amount': ['sum', 'mean', 'count']
        }).reset_index()
        
        merchant_metrics.columns = [
            'merchant_id',
            'avg_risk_score',
            'max_risk_score',
            'transaction_count',
            'high_risk_count',
            'total_amount',
            'avg_amount',
            'unique_transactions'
        ]
        
        merchant_metrics['high_risk_rate'] = (
            merchant_metrics['high_risk_count'] / merchant_metrics['transaction_count'] * 100
        )
        merchant_metrics['risk_category'] = pd.cut(
            merchant_metrics['avg_risk_score'],
            bins=[0, 2, 5, 8, 10],
            labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        )
        
        return merchant_metrics.sort_values('avg_risk_score', ascending=False)
    
    def calculate_transaction_volume_trends(self,
                                          df: pd.DataFrame,
                                          time_column: str = 'step',
                                          frequency: str = 'D') -> pd.DataFrame:
        """
        Calculate transaction volume trends over time.
        
        Args:
            df: DataFrame with transactions
            time_column: Column name for time dimension
            frequency: Time frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
            
        Returns:
            DataFrame with volume trends
        """
        if time_column not in df.columns:
            logger.warning(f"Time column '{time_column}' not found")
            return pd.DataFrame()
        
        # Convert step to datetime if needed (assuming step represents hours)
        if df[time_column].dtype in ['int64', 'int32']:
            # Create a base date and add hours
            base_date = pd.Timestamp('2020-01-01')
            df_with_date = df.copy()
            df_with_date['date'] = base_date + pd.to_timedelta(df[time_column], unit='h')
        else:
            df_with_date = df.copy()
            df_with_date['date'] = pd.to_datetime(df[time_column])
        
        # Group by time period
        trends = df_with_date.groupby(pd.Grouper(key='date', freq=frequency)).agg({
            'amount': ['count', 'sum', 'mean'],
            'high_risk_flag': 'sum' if 'high_risk_flag' in df.columns else 'count'
        }).reset_index()
        
        trends.columns = [
            'period',
            'transaction_count',
            'total_volume',
            'avg_transaction_amount',
            'high_risk_count'
        ]
        
        trends['high_risk_rate'] = (
            trends['high_risk_count'] / trends['transaction_count'] * 100
        )
        
        return trends.fillna(0)
    
    def calculate_detection_efficiency_metrics(self,
                                             df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate efficiency metrics for fraud detection.
        
        Args:
            df: DataFrame with transaction and detection data
            
        Returns:
            Dictionary with efficiency metrics
        """
        if 'high_risk_flag' not in df.columns:
            return {}
        
        total_alerts = df['high_risk_flag'].sum()
        total_transactions = len(df)
        
        # Calculate cost metrics (configurable cost per alert review)
        total_review_cost = total_alerts * self.cost_per_alert_review
        
        # Calculate average detection time (if available)
        if 'detection_time_ms' in df.columns:
            avg_detection_time = df[df['high_risk_flag']]['detection_time_ms'].mean()
        else:
            avg_detection_time = None
        
        metrics = {
            'total_alerts': int(total_alerts),
            'alert_rate_pct': round(total_alerts / total_transactions * 100, 2),
            'estimated_review_cost': round(total_review_cost, 2),
            'cost_per_transaction': round(total_review_cost / total_transactions, 4),
            'avg_detection_time_ms': round(avg_detection_time, 2) if avg_detection_time else None
        }
        
        return metrics
    
    def calculate_model_performance_metrics(self,
                                           y_true: np.ndarray,
                                           y_pred: np.ndarray,
                                           y_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate business-relevant model performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores (probabilities)
            
        Returns:
            Dictionary with performance metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix
        )
        
        metrics = {
            'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
            'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            'recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            'f1_score': round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        }
        
        if y_scores is not None:
            try:
                metrics['auc'] = round(float(roc_auc_score(y_true, y_scores)), 4)
            except ValueError:
                metrics['auc'] = None
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['false_positive_rate'] = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0
        
        return metrics
    
    def generate_business_summary_report(self,
                                       df: pd.DataFrame,
                                       output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive business summary report.
        
        Args:
            df: DataFrame with transaction and detection data
            output_path: Optional path to save report as JSON
            
        Returns:
            Dictionary with complete business summary
        """
        report = {
            'report_date': datetime.now().isoformat(),
            'detection_metrics': self.calculate_fraud_detection_rate(df),
            'efficiency_metrics': self.calculate_detection_efficiency_metrics(df),
            'summary': {}
        }
        
        # Add summary statistics
        if 'amount' in df.columns:
            report['summary']['total_transaction_volume'] = float(df['amount'].sum())
            report['summary']['avg_transaction_amount'] = float(df['amount'].mean())
            report['summary']['median_transaction_amount'] = float(df['amount'].median())
        
        if 'high_risk_flag' in df.columns:
            report['summary']['total_high_risk_transactions'] = int(df['high_risk_flag'].sum())
            report['summary']['high_risk_rate_pct'] = round(
                df['high_risk_flag'].sum() / len(df) * 100, 2
            )
        
        # Add merchant metrics
        merchant_risk = self.calculate_merchant_risk_distribution(df)
        if not merchant_risk.empty:
            report['merchant_metrics'] = {
                'total_merchants': len(merchant_risk),
                'high_risk_merchants': int((merchant_risk['risk_category'] == 'HIGH').sum() + 
                                          (merchant_risk['risk_category'] == 'CRITICAL').sum()),
                'top_10_risky_merchants': merchant_risk.head(10).to_dict('records')
            }
        
        # Add volume trends
        trends = self.calculate_transaction_volume_trends(df)
        if not trends.empty:
            report['volume_trends'] = {
                'total_periods': len(trends),
                'avg_daily_volume': float(trends['transaction_count'].mean()) if len(trends) > 0 else 0,
                'trend_data': trends.tail(30).to_dict('records')  # Last 30 periods
            }
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Business summary report saved to {output_path}")
        
        return report


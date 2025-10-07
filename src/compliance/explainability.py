"""
Explainable AI framework for regulatory compliance.

This module provides tools for model explainability, interpretability,
and audit trails to meet regulatory requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import json

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Provides explanations for model predictions.
    
    Generates interpretable explanations using SHAP values and
    feature importance analysis.
    """
    
    def __init__(self, model, model_type: str = 'tree'):
        """
        Initialize model explainer.
        
        Args:
            model: Trained model to explain
            model_type: Type of model ('tree', 'linear', 'kernel')
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
        
        if SHAP_AVAILABLE:
            self._initialize_explainer()
        else:
            logger.warning("SHAP not available. Install with: pip install shap")
        
        logger.info(f"Model explainer initialized for {model_type} model")
    
    def _initialize_explainer(self) -> None:
        """Initialize appropriate SHAP explainer based on model type."""
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model)
        else:
            # Use KernelExplainer as fallback
            logger.info("Using KernelExplainer (may be slow)")
    
    def explain_prediction(self,
                          features: pd.DataFrame,
                          feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            features: Feature values for the prediction
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary with explanation details
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return self._fallback_explanation(features, feature_names)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(features)
        
        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        if len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
            shap_values = shap_values[0]
        
        # Get feature names
        if feature_names is None:
            feature_names = features.columns.tolist() if hasattr(features, 'columns') else [
                f"feature_{i}" for i in range(len(shap_values))
            ]
        
        # Create explanation
        feature_contributions = []
        for fname, shap_val in zip(feature_names, shap_values):
            feature_contributions.append({
                'feature': fname,
                'shap_value': float(shap_val),
                'importance': float(abs(shap_val))
            })
        
        # Sort by importance
        feature_contributions.sort(key=lambda x: x['importance'], reverse=True)
        
        explanation = {
            'top_features': feature_contributions[:10],
            'all_features': feature_contributions,
            'base_value': float(self.explainer.expected_value) if hasattr(
                self.explainer, 'expected_value'
            ) else None,
            'prediction_explanation': self._generate_text_explanation(
                feature_contributions[:5]
            )
        }
        
        return explanation
    
    def _generate_text_explanation(self, top_features: List[Dict]) -> str:
        """Generate human-readable explanation."""
        if not top_features:
            return "No significant features identified."
        
        parts = ["This prediction was primarily influenced by:"]
        
        for feat in top_features:
            direction = "increased" if feat['shap_value'] > 0 else "decreased"
            parts.append(
                f"- {feat['feature']} {direction} the risk score (contribution: {abs(feat['shap_value']):.3f})"
            )
        
        return "\n".join(parts)
    
    def _fallback_explanation(self,
                             features: pd.DataFrame,
                             feature_names: Optional[List[str]]) -> Dict[str, Any]:
        """Provide fallback explanation when SHAP is not available."""
        if feature_names is None:
            feature_names = features.columns.tolist() if hasattr(features, 'columns') else []
        
        return {
            'message': 'SHAP not available. Using fallback explanation.',
            'features_used': feature_names,
            'recommendation': 'Install SHAP for detailed explanations'
        }


class DecisionAuditLog:
    """
    Audit logging for model decisions.
    
    Maintains detailed logs of all model decisions for regulatory
    compliance and investigation purposes.
    """
    
    def __init__(self, log_file: str = "decision_audit.jsonl"):
        """
        Initialize audit log.
        
        Args:
            log_file: Path to audit log file
        """
        self.log_file = log_file
        self.in_memory_log = []
        
        logger.info(f"Decision audit log initialized: {log_file}")
    
    def log_decision(self,
                    transaction_id: str,
                    input_data: Dict,
                    prediction: Any,
                    confidence: float,
                    explanation: Dict,
                    model_version: str = "1.0",
                    user_action: Optional[str] = None) -> None:
        """
        Log a model decision.
        
        Args:
            transaction_id: Unique transaction identifier
            input_data: Input features used for prediction
            prediction: Model prediction
            confidence: Prediction confidence score
            explanation: Explanation of the decision
            model_version: Version of the model used
            user_action: Optional action taken by user/system
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'transaction_id': transaction_id,
            'prediction': str(prediction),
            'confidence': float(confidence),
            'model_version': model_version,
            'input_features': self._sanitize_input(input_data),
            'explanation_summary': self._summarize_explanation(explanation),
            'user_action': user_action
        }
        
        # Add to in-memory log
        self.in_memory_log.append(log_entry)
        
        # Write to file
        self._write_to_file(log_entry)
    
    def _sanitize_input(self, input_data: Dict) -> Dict:
        """Remove sensitive information from input data."""
        # Create copy
        sanitized = input_data.copy()
        
        # Remove or mask sensitive fields
        sensitive_fields = ['nameOrig', 'nameDest', 'account_id', 'customer_id']
        for field in sensitive_fields:
            if field in sanitized:
                # Hash instead of removing for audit trail
                sanitized[field] = f"MASKED_{hash(str(sanitized[field])) % 10000:04d}"
        
        return sanitized
    
    def _summarize_explanation(self, explanation: Dict) -> Dict:
        """Create summary of explanation for logging."""
        if 'top_features' in explanation:
            return {
                'top_features': [
                    f"{f['feature']}: {f['shap_value']:.3f}"
                    for f in explanation['top_features'][:3]
                ]
            }
        return {'summary': 'No explanation available'}
    
    def _write_to_file(self, log_entry: Dict) -> None:
        """Append log entry to file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {str(e)}")
    
    def query_logs(self,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  transaction_id: Optional[str] = None) -> List[Dict]:
        """
        Query audit logs with filters.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            transaction_id: Specific transaction ID
            
        Returns:
            List of matching log entries
        """
        results = []
        
        for entry in self.in_memory_log:
            # Filter by time
            entry_time = datetime.fromisoformat(entry['timestamp'])
            if start_time and entry_time < start_time:
                continue
            if end_time and entry_time > end_time:
                continue
            
            # Filter by transaction ID
            if transaction_id and entry['transaction_id'] != transaction_id:
                continue
            
            results.append(entry)
        
        return results


class ComplianceReporter:
    """
    Automated compliance report generation.
    
    Generates reports for regulatory authorities including AML reports
    and suspicious activity summaries.
    """
    
    def __init__(self):
        """Initialize compliance reporter."""
        logger.info("Compliance reporter initialized")
    
    def generate_aml_report(self,
                          transactions: pd.DataFrame,
                          time_period: str,
                          threshold_amount: float = 10000) -> Dict[str, Any]:
        """
        Generate AML compliance report.
        
        Args:
            transactions: DataFrame with transactions
            time_period: Time period for the report
            threshold_amount: Reporting threshold
            
        Returns:
            Dictionary with AML report
        """
        # Filter flagged transactions
        flagged = transactions[transactions.get('high_risk_flag', False)]
        large_transactions = transactions[transactions['amount'] >= threshold_amount]
        
        report = {
            'report_type': 'AML_COMPLIANCE',
            'period': time_period,
            'generated_at': datetime.now().isoformat(),
            
            'summary': {
                'total_transactions': len(transactions),
                'flagged_transactions': len(flagged),
                'flagged_percentage': (len(flagged) / len(transactions) * 100) if len(transactions) > 0 else 0,
                'large_transactions': len(large_transactions),
                'total_flagged_amount': float(flagged['amount'].sum()) if len(flagged) > 0 else 0
            },
            
            'risk_distribution': self._calculate_risk_distribution(transactions),
            
            'top_risk_accounts': self._identify_top_risk_accounts(transactions),
            
            'suspicious_patterns': self._identify_suspicious_patterns(flagged),
            
            'regulatory_thresholds': {
                'threshold_amount': threshold_amount,
                'transactions_above_threshold': len(large_transactions),
                'accounts_above_threshold': large_transactions['nameOrig'].nunique() if len(large_transactions) > 0 else 0
            }
        }
        
        return report
    
    def _calculate_risk_distribution(self, transactions: pd.DataFrame) -> Dict:
        """Calculate distribution of risk scores."""
        if 'final_risk_score' not in transactions.columns:
            return {}
        
        risk_scores = transactions['final_risk_score']
        
        return {
            'mean': float(risk_scores.mean()),
            'median': float(risk_scores.median()),
            'std': float(risk_scores.std()),
            'percentiles': {
                'p25': float(risk_scores.quantile(0.25)),
                'p50': float(risk_scores.quantile(0.50)),
                'p75': float(risk_scores.quantile(0.75)),
                'p95': float(risk_scores.quantile(0.95))
            }
        }
    
    def _identify_top_risk_accounts(self,
                                   transactions: pd.DataFrame,
                                   top_n: int = 10) -> List[Dict]:
        """Identify accounts with highest risk."""
        if 'final_risk_score' not in transactions.columns:
            return []
        
        account_risk = transactions.groupby('nameOrig').agg({
            'final_risk_score': 'mean',
            'amount': ['sum', 'count']
        }).reset_index()
        
        account_risk.columns = ['account', 'avg_risk', 'total_amount', 'txn_count']
        account_risk = account_risk.nlargest(top_n, 'avg_risk')
        
        return account_risk.to_dict('records')
    
    def _identify_suspicious_patterns(self, flagged: pd.DataFrame) -> List[str]:
        """Identify common patterns in flagged transactions."""
        patterns = []
        
        if len(flagged) == 0:
            return patterns
        
        # Pattern 1: Frequent small transactions
        small_frequent = flagged[flagged['amount'] < 1000]
        if len(small_frequent) / len(flagged) > 0.3:
            patterns.append("High proportion of small transactions flagged")
        
        # Pattern 2: Specific transaction types
        if 'type' in flagged.columns:
            dominant_type = flagged['type'].mode()[0] if len(flagged) > 0 else None
            if dominant_type:
                type_percentage = (flagged['type'] == dominant_type).mean() * 100
                if type_percentage > 50:
                    patterns.append(f"Over 50% of flagged transactions are {dominant_type}")
        
        # Pattern 3: Time clustering
        if 'step' in flagged.columns and len(flagged) > 10:
            time_std = flagged['step'].std()
            if time_std < 100:
                patterns.append("Flagged transactions clustered in time")
        
        return patterns
    
    def export_report(self, report: Dict, filepath: str, format: str = 'json') -> None:
        """
        Export compliance report to file.
        
        Args:
            report: Report dictionary
            filepath: Output file path
            format: Output format ('json' or 'pdf')
        """
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            logger.warning(f"Format {format} not yet implemented")
        
        logger.info(f"Compliance report exported to {filepath}")

import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support, roc_auc_score,
    average_precision_score, confusion_matrix
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_model(model: Any, 
              model_name: str, 
              output_dir: str = 'models') -> str:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        model_name: Name to save the model as
        output_dir: Directory to save the model in
        
    Returns:
        Path to the saved model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.joblib"
    filepath = os.path.join(output_dir, filename)
    
    # Save the model
    joblib.dump(model, filepath)
    logger.info(f"Model saved to {filepath}")
    
    return filepath


def load_model(filepath: str) -> Any:
    """
    Load a saved model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object
    """
    model = joblib.load(filepath)
    logger.info(f"Model loaded from {filepath}")
    return model


def save_results(results: Dict, 
               output_file: str) -> None:
    """
    Save analysis results to disk.
    
    Args:
        results: Dictionary of results to save
        output_file: File to save results to (JSON format)
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {output_file}")


def evaluate_binary_classifier(y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             y_scores: Optional[np.ndarray] = None) -> Dict:
    """
    Evaluate a binary classifier and return various metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_scores: Predicted probability scores (for AUC, etc.)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate basic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )
    
    # Calculate classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate AUC if scores are provided
    auc = None
    avg_precision = None
    if y_scores is not None:
        auc = roc_auc_score(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
    
    # Combine all metrics
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'auc': auc,
        'average_precision': avg_precision,
        'classification_report': report
    }
    
    return metrics


def calculate_risk_score(flags: Dict[str, pd.Series], 
                       weights: Dict[str, float]) -> pd.Series:
    """
    Calculate a weighted risk score from multiple flag series.
    
    Args:
        flags: Dictionary of flag names and corresponding boolean Series
        weights: Dictionary of flag names and corresponding weights
        
    Returns:
        Series with weighted risk scores
    """
    # Validate inputs
    if not set(flags.keys()) == set(weights.keys()):
        raise ValueError("Flag keys and weight keys must match")
    
    # Get the index from the first flag
    index = next(iter(flags.values())).index
    
    # Calculate weighted sum
    risk_score = pd.Series(0.0, index=index)
    
    for flag_name, flag_series in flags.items():
        weight = weights[flag_name]
        risk_score += flag_series.astype(int) * weight
    
    return risk_score


def generate_alert_report(transaction_data: pd.DataFrame,
                        suspicious_flags: pd.Series,
                        risk_scores: pd.Series,
                        output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Generate a report of suspicious transactions for further investigation.
    
    Args:
        transaction_data: Original transaction data
        suspicious_flags: Boolean Series indicating suspicious transactions
        risk_scores: Series with risk scores for transactions
        output_file: Optional CSV file to save the report to
        
    Returns:
        DataFrame with alert report
    """
    # Filter suspicious transactions
    suspicious_txns = transaction_data[suspicious_flags].copy()
    
    # Add risk score
    suspicious_txns['risk_score'] = risk_scores[suspicious_flags]
    
    # Sort by risk score (descending)
    suspicious_txns = suspicious_txns.sort_values('risk_score', ascending=False)
    
    # Add alert ID
    suspicious_txns['alert_id'] = [f"ALERT-{i:06d}" for i in range(1, len(suspicious_txns) + 1)]
    
    # Add timestamp
    suspicious_txns['alert_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Reorder columns
    columns = ['alert_id', 'alert_timestamp', 'risk_score'] + [
        col for col in suspicious_txns.columns 
        if col not in ['alert_id', 'alert_timestamp', 'risk_score']
    ]
    suspicious_txns = suspicious_txns[columns]
    
    # Save report if output_file is provided
    if output_file:
        suspicious_txns.to_csv(output_file, index=False)
        logger.info(f"Alert report saved to {output_file}")
    
    logger.info(f"Generated alert report with {len(suspicious_txns)} suspicious transactions")
    return suspicious_txns


def sample_transactions(df: pd.DataFrame, 
                      n_normal: int = 1000, 
                      n_fraud: int = 1000,
                      random_state: int = 42) -> pd.DataFrame:
    """
    Sample a balanced dataset from the full dataset.
    
    Args:
        df: DataFrame with transaction data
        n_normal: Number of normal transactions to sample
        n_fraud: Number of fraudulent transactions to sample
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with sampled transactions
    """
    if 'isFraud' not in df.columns:
        raise ValueError("DataFrame must contain 'isFraud' column")
    
    # Split into normal and fraud
    normal = df[df['isFraud'] == 0]
    fraud = df[df['isFraud'] == 1]
    
    # Sample from each
    if len(normal) > n_normal:
        normal_sample = normal.sample(n=n_normal, random_state=random_state)
    else:
        normal_sample = normal
    
    if len(fraud) > n_fraud:
        fraud_sample = fraud.sample(n=n_fraud, random_state=random_state)
    else:
        fraud_sample = fraud
    
    # Combine samples
    sample = pd.concat([normal_sample, fraud_sample], ignore_index=True)
    
    # Shuffle the combined sample
    sample = sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    logger.info(f"Sampled {len(sample)} transactions ({len(fraud_sample)} fraud, {len(normal_sample)} normal)")
    return sample


def get_high_risk_accounts(df: pd.DataFrame, 
                         n_accounts: int = 100) -> List[str]:
    """
    Identify high-risk accounts based on fraud frequency.
    
    Args:
        df: DataFrame with transaction data
        n_accounts: Number of high-risk accounts to return
        
    Returns:
        List of high-risk account IDs
    """
    if 'isFraud' not in df.columns:
        raise ValueError("DataFrame must contain 'isFraud' column")
    
    # Get accounts involved in fraud
    fraud_df = df[df['isFraud'] == 1]
    
    # Count fraud transactions per account
    orig_counts = fraud_df['nameOrig'].value_counts()
    dest_counts = fraud_df['nameDest'].value_counts()
    
    # Combine counts
    all_counts = pd.concat([orig_counts, dest_counts])
    combined_counts = all_counts.groupby(all_counts.index).sum()
    
    # Get top accounts
    high_risk_accounts = combined_counts.sort_values(ascending=False).head(n_accounts).index.tolist()
    
    logger.info(f"Identified {len(high_risk_accounts)} high-risk accounts")
    return high_risk_accounts


def create_time_windows(df: pd.DataFrame, 
                       window_size: int = 10) -> pd.DataFrame:
    """
    Add a time window column to the DataFrame.
    
    Args:
        df: DataFrame with transaction data
        window_size: Size of time windows in steps
        
    Returns:
        DataFrame with added time_window column
    """
    if 'step' not in df.columns:
        raise ValueError("DataFrame must contain 'step' column")
    
    # Create a copy
    result_df = df.copy()
    
    # Add time window column
    result_df['time_window'] = result_df['step'] // window_size
    
    logger.info(f"Created time windows of size {window_size} steps")
    return result_df


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary with configuration settings
    """
    try:
        import yaml
        
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return {} 
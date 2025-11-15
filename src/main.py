import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Optional MLflow import (graceful degradation)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except (ImportError, Exception) as e:
    MLFLOW_AVAILABLE = False
    # Don't log here as logger might not be set up yet

# Import project modules
from data.preprocessor import TransactionPreprocessor
from models.rule_based_scenarios import AMLRuleEngine, AdaptiveThresholdCalculator
from models.ml_anomaly_detection import AnomalyDetector
from models.network_analysis import TransactionNetworkAnalyzer
from visualization.visualizer import AMLVisualizer
from utils.helpers import (
    save_model, save_results, evaluate_binary_classifier,
    sample_transactions, get_high_risk_accounts, generate_alert_report,
    calculate_risk_score, load_config
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aml_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check for advanced dependencies
ADVANCED_MODELS_AVAILABLE = True
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    ADVANCED_MODELS_AVAILABLE = False
    logger.warning("TensorFlow not available. Deep learning models will be disabled.")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    
    try:
        import torch_geometric
        TORCH_GEOMETRIC_AVAILABLE = True
    except ImportError:
        TORCH_GEOMETRIC_AVAILABLE = False
        ADVANCED_MODELS_AVAILABLE = False
        logger.warning("PyTorch Geometric not available. GNN-based analysis will be disabled.")
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_GEOMETRIC_AVAILABLE = False
    ADVANCED_MODELS_AVAILABLE = False
    logger.warning("PyTorch not available. GNN-based analysis will be disabled.")

logger.info(f"Advanced models available: {ADVANCED_MODELS_AVAILABLE}")
logger.info(f"TensorFlow available: {TF_AVAILABLE}")
logger.info(f"PyTorch available: {TORCH_AVAILABLE}")
logger.info(f"PyTorch Geometric available: {TORCH_GEOMETRIC_AVAILABLE}")


class TransactionAnomalyDetectionSystem:
    """
    Main class for the Transaction Anomaly Detection System.
    
    This class orchestrates the entire system, combining rule-based scenarios,
    machine learning models, and network analysis to detect suspicious activities.
    """
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = 'output',
                 mlflow_tracking_uri: Optional[str] = None,
                 sample_size: Optional[int] = None):
        """
        Initialize the system.
        
        Args:
            data_path: Path to transaction data CSV file
            output_dir: Directory to save outputs
            mlflow_tracking_uri: URI for MLflow tracking server
            sample_size: Number of transactions to sample (for testing)
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.sample_size = sample_size
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = TransactionPreprocessor()
        self.rule_engine = AMLRuleEngine()
        self.anomaly_detector = AnomalyDetector(
            contamination=0.01,
            tracking_uri=mlflow_tracking_uri
        )
        self.network_analyzer = TransactionNetworkAnalyzer()
        self.visualizer = AMLVisualizer()
        
        # Set up MLflow tracking
        if mlflow_tracking_uri and MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        elif mlflow_tracking_uri and not MLFLOW_AVAILABLE:
            logger.warning("MLflow tracking URI provided but MLflow is not available. Tracking disabled.")
        
        logger.info(f"Transaction Anomaly Detection System initialized with data from {data_path}")
    
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess the transaction data.
        
        Returns:
            Tuple of (full dataset, sampled dataset)
        """
        logger.info("Loading and preprocessing data")
        
        # Load data
        df = self.preprocessor.load_data(self.data_path, sample=self.sample_size)
        
        # Preprocess data
        df = self.preprocessor.preprocess(df)
        
        # Create a smaller balanced sample for model training
        if 'isFraud' in df.columns:
            sample_df = sample_transactions(df, n_normal=10000, n_fraud=10000)
        else:
            sample_df = df.sample(min(20000, len(df))) if len(df) > 20000 else df
        
        logger.info(f"Data preprocessing complete. Full dataset: {len(df)} rows, Sample: {len(sample_df)} rows")
        
        return df, sample_df
    
    def run_rule_based_detection(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run rule-based scenarios for AML detection.
        
        Args:
            df: Preprocessed transaction data
            
        Returns:
            Tuple of (results DataFrame, summary DataFrame)
        """
        logger.info("Running rule-based detection scenarios")
        
        # Get high-risk accounts if fraud labels are available
        high_risk_accounts = None
        if 'isFraud' in df.columns:
            high_risk_accounts = get_high_risk_accounts(df, n_accounts=100)
        
        # Run all scenarios
        results_df, summary_df = self.rule_engine.run_all_scenarios(df, high_risk_accounts)
        
        # Save results
        results_file = os.path.join(self.output_dir, 'rule_based_results.csv')
        summary_file = os.path.join(self.output_dir, 'rule_based_summary.csv')
        
        results_df.to_csv(results_file, index=False)
        summary_df.to_csv(summary_file, index=False)
        
        # Visualize results
        plot_file = os.path.join(self.output_dir, 'rule_based_summary.png')
        self.visualizer.plot_scenario_results(summary_df, save_path=plot_file)
        
        logger.info(f"Rule-based detection complete. Flagged {results_df['is_suspicious'].sum()} suspicious transactions")
        
        return results_df, summary_df
    
    def run_ml_detection(self, 
                       full_df: pd.DataFrame, 
                       sample_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run machine learning-based anomaly detection.
        
        Args:
            full_df: Full preprocessed transaction data
            sample_df: Sampled data for training
            
        Returns:
            Dictionary with ML detection results
        """
        logger.info("Running machine learning-based anomaly detection")
        
        # Prepare data for ML
        encoded_sample = self.preprocessor.encode_and_scale(sample_df)
        encoded_full = self.preprocessor.encode_and_scale(full_df, train=False)
        
        # Split into features and target (if available)
        has_labels = 'isFraud' in encoded_sample.columns
        
        if has_labels:
            logger.info("Using supervised anomaly detection (fraud labels available)")
            
            # Prepare train/test data
            X_train, X_test, y_train, y_test = self.preprocessor.prepare_train_test_data(encoded_sample)
            
            # Train supervised models
            supervised_models = self.anomaly_detector.train_supervised_models(X_train, y_train, X_test, y_test)
            
            # Get feature importances
            feature_imp = self.anomaly_detector.feature_importances.get('xgboost')
            if feature_imp is not None:
                # Convert dict to DataFrame if needed
                if isinstance(feature_imp, dict):
                    feature_imp = pd.DataFrame(list(feature_imp.items()), columns=['feature', 'importance'])
                imp_file = os.path.join(self.output_dir, 'feature_importance.png')
                self.visualizer.plot_feature_importance(feature_imp, save_path=imp_file)
            
            # Generate predictions on full dataset
            X_full = encoded_full.drop(['isFraud', 'nameOrig', 'nameDest', 'step'], errors='ignore', axis=1)
            y_scores = {}
            y_preds = {}
            
            for model_name, model_info in supervised_models.items():
                model = model_info['model']
                threshold = model_info['best_threshold']
                
                # Get predictions
                y_scores[model_name] = model.predict_proba(X_full)[:, 1]
                y_preds[model_name] = (y_scores[model_name] > threshold).astype(int)
                
                # Save model
                save_model(model, model_name, output_dir=os.path.join(self.output_dir, 'models'))
            
            # Save ensemble prediction
            ensemble_scores = sum(y_scores.values()) / len(y_scores)
            ensemble_preds = (ensemble_scores > 0.5).astype(int)
            
            # Explain predictions with SHAP for a sample
            sample_to_explain = X_test.iloc[:100]
            shap_results = self.anomaly_detector.explain_predictions('xgboost', sample_to_explain)
            
            # Plot SHAP values
            shap_file = os.path.join(self.output_dir, 'shap_summary.png')
            self.visualizer.plot_shap_summary(
                shap_results['shap_values'], 
                sample_to_explain,
                save_path=shap_file
            )
            
            # Save results
            detection_results = {
                'supervised_models': supervised_models,
                'y_scores': y_scores,
                'y_preds': y_preds,
                'ensemble_scores': ensemble_scores,
                'ensemble_preds': ensemble_preds
            }
            
        else:
            logger.info("Using unsupervised anomaly detection (no fraud labels)")
            
            # Drop non-feature columns
            X_sample = encoded_sample.drop(['nameOrig', 'nameDest', 'step'], errors='ignore', axis=1)
            X_full = encoded_full.drop(['nameOrig', 'nameDest', 'step'], errors='ignore', axis=1)
            
            # Train isolation forest
            iso_forest = self.anomaly_detector.train_isolation_forest(X_sample)
            
            # Generate predictions on full dataset
            anomaly_scores = -iso_forest['model'].score_samples(X_full)
            anomaly_threshold = iso_forest['threshold']
            anomaly_flags = anomaly_scores > anomaly_threshold
            
            # Visualize anomaly scores
            scores_file = os.path.join(self.output_dir, 'anomaly_scores.png')
            self.visualizer.plot_anomaly_scores(
                anomaly_scores, 
                anomaly_threshold,
                save_path=scores_file
            )
            
            # Save model
            save_model(iso_forest['model'], 'isolation_forest', output_dir=os.path.join(self.output_dir, 'models'))
            
            # Save results
            detection_results = {
                'unsupervised_model': iso_forest,
                'anomaly_scores': anomaly_scores,
                'anomaly_threshold': anomaly_threshold,
                'anomaly_flags': anomaly_flags
            }
        
        logger.info(f"Machine learning detection complete")
        return detection_results
    
    def run_network_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run network analysis for complex pattern detection.
        
        Args:
            df: Preprocessed transaction data
            
        Returns:
            Dictionary with network analysis results
        """
        logger.info("Running network analysis")
        
        # Build transaction network
        G = self.network_analyzer.build_transaction_network(df)
        
        # Detect cycles (potential money laundering)
        cycles = self.network_analyzer.detect_cycles()
        
        # Detect fan patterns
        fan_out, fan_in = self.network_analyzer.detect_fan_patterns()
        
        # Calculate centrality metrics
        centrality = self.network_analyzer.calculate_centrality()
        
        # Detect communities
        communities = self.network_analyzer.detect_communities()
        
        # Identify suspicious accounts
        suspicious_accounts = self.network_analyzer.identify_suspicious_accounts()
        
        # Flag suspicious transactions
        network_flags = self.network_analyzer.flag_suspicious_transactions(df, suspicious_accounts)
        
        # Visualize the network
        self.network_analyzer.visualize_network(
            highlight_nodes=list(suspicious_accounts)[:100],
            highlight_communities=True,
            filename=os.path.join(self.output_dir, 'transaction_network.png')
        )
        
        # Save results
        network_results = {
            'suspicious_accounts': list(suspicious_accounts),
            'cycles_count': len(cycles),
            'fan_out_count': len(fan_out),
            'fan_in_count': len(fan_in),
            'communities_count': len(set(communities.values())),
            'network_flags': network_flags
        }
        
        logger.info(f"Network analysis complete. Identified {len(suspicious_accounts)} suspicious accounts")
        return network_results
    
    def combine_all_results(self,
                          df: pd.DataFrame,
                          rule_results: pd.DataFrame,
                          ml_results: Dict[str, Any],
                          network_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Combine results from all detection methods.
        
        Args:
            df: Original transaction data
            rule_results: Results from rule-based detection
            ml_results: Results from ML-based detection
            network_results: Results from network analysis
            
        Returns:
            DataFrame with combined results and final risk scores
        """
        logger.info("Combining results from all detection methods")
        
        # Create a copy of the original data
        combined_df = df.copy()
        
        # Add rule-based flags
        combined_df['rule_based_flag'] = rule_results['is_suspicious']
        combined_df['rule_based_score'] = rule_results['aml_risk_score']
        
        # Add ML flags
        if 'ensemble_preds' in ml_results:
            combined_df['ml_flag'] = ml_results['ensemble_preds']
            combined_df['ml_score'] = ml_results['ensemble_scores']
        else:
            combined_df['ml_flag'] = ml_results['anomaly_flags']
            combined_df['ml_score'] = ml_results['anomaly_scores'] / ml_results['anomaly_threshold']
        
        # Add network flags
        combined_df['network_flag'] = network_results['network_flags']
        
        # Calculate final risk score
        flags = {
            'rule_based': combined_df['rule_based_flag'],
            'ml': combined_df['ml_flag'],
            'network': combined_df['network_flag']
        }
        
        weights = {
            'rule_based': 3.0,  # Higher weight for rule-based (regulatory focus)
            'ml': 2.0,          # Medium weight for ML predictions
            'network': 2.0       # Medium weight for network analysis
        }
        
        combined_df['final_risk_score'] = calculate_risk_score(flags, weights)
        
        # Flag high-risk transactions
        combined_df['high_risk_flag'] = combined_df['final_risk_score'] >= 2.0
        
        # Generate alert report
        alert_report = generate_alert_report(
            combined_df,
            combined_df['high_risk_flag'],
            combined_df['final_risk_score'],
            output_file=os.path.join(self.output_dir, 'alert_report.csv')
        )
        
        # Save combined results
        combined_df.to_csv(os.path.join(self.output_dir, 'combined_results.csv'), index=False)
        
        # Calculate statistics
        stats = {
            'total_transactions': len(combined_df),
            'rule_based_flags': combined_df['rule_based_flag'].sum(),
            'ml_flags': combined_df['ml_flag'].sum(),
            'network_flags': combined_df['network_flag'].sum(),
            'high_risk_flags': combined_df['high_risk_flag'].sum(),
            'high_risk_percentage': combined_df['high_risk_flag'].mean() * 100
        }
        
        # Save statistics
        save_results(stats, os.path.join(self.output_dir, 'detection_statistics.json'))
        
        logger.info(f"Results combined. Flagged {stats['high_risk_flags']} high-risk transactions ({stats['high_risk_percentage']:.2f}%)")
        
        return combined_df
    
    def evaluate_system(self, 
                      combined_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate system performance against known fraud labels.
        
        Args:
            combined_df: Combined results DataFrame
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Only evaluate if fraud labels are available
        if 'isFraud' not in combined_df.columns:
            logger.info("No fraud labels available. Skipping evaluation.")
            return {}
        
        logger.info("Evaluating system performance")
        
        # Evaluate rule-based detection
        rule_metrics = evaluate_binary_classifier(
            combined_df['isFraud'].values,
            combined_df['rule_based_flag'].values,
            combined_df['rule_based_score'].values
        )
        
        # Evaluate ML detection
        ml_metrics = evaluate_binary_classifier(
            combined_df['isFraud'].values,
            combined_df['ml_flag'].values,
            combined_df['ml_score'].values
        )
        
        # Evaluate network analysis
        network_metrics = evaluate_binary_classifier(
            combined_df['isFraud'].values,
            combined_df['network_flag'].values
        )
        
        # Evaluate combined system
        combined_metrics = evaluate_binary_classifier(
            combined_df['isFraud'].values,
            combined_df['high_risk_flag'].values,
            combined_df['final_risk_score'].values
        )
        
        # Combine all metrics
        all_metrics = {
            'rule_based': rule_metrics,
            'ml_based': ml_metrics,
            'network_based': network_metrics,
            'combined_system': combined_metrics
        }
        
        # Save evaluation results
        save_results(all_metrics, os.path.join(self.output_dir, 'evaluation_metrics.json'))
        
        # Plot ROC curves
        if 'auc' in rule_metrics and rule_metrics['auc'] is not None:
            scores = {
                'Rule-based': combined_df['rule_based_score'].values,
                'ML-based': combined_df['ml_score'].values,
                'Combined': combined_df['final_risk_score'].values
            }
            
            self.visualizer.plot_roc_curves(
                combined_df['isFraud'].values,
                scores,
                save_path=os.path.join(self.output_dir, 'roc_curves.png')
            )
            
            self.visualizer.plot_precision_recall_curves(
                combined_df['isFraud'].values,
                scores,
                save_path=os.path.join(self.output_dir, 'pr_curves.png')
            )
        
        # Plot confusion matrix
        self.visualizer.plot_confusion_matrix(
            combined_df['isFraud'].values,
            combined_df['high_risk_flag'].values,
            normalize=True,
            save_path=os.path.join(self.output_dir, 'confusion_matrix.png')
        )
        
        logger.info(f"System evaluation complete. Combined AUC: {combined_metrics.get('auc', 'N/A')}")
        
        return all_metrics
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete AML detection pipeline.
        
        Returns:
            Dictionary with all results
        """
        start_time = datetime.now()
        logger.info(f"Starting full AML detection pipeline at {start_time}")
        
        try:
            # Step 1: Load and preprocess data
            full_df, sample_df = self.load_and_preprocess_data()
            
            # Step 2: Run rule-based detection
            rule_results, rule_summary = self.run_rule_based_detection(full_df)
            
            # Step 3: Run ML-based detection
            ml_results = self.run_ml_detection(full_df, sample_df)
            
            # Step 4: Run network analysis
            network_results = self.run_network_analysis(full_df)
            
            # Step 5: Combine all results
            combined_df = self.combine_all_results(
                full_df, rule_results, ml_results, network_results
            )
            
            # Step 6: Evaluate system (if labels available)
            evaluation_metrics = self.evaluate_system(combined_df)
            
            # Return all results
            final_results = {
                'rule_summary': rule_summary.to_dict(),
                'ml_results': {k: v for k, v in ml_results.items() if not isinstance(v, (pd.DataFrame, np.ndarray))},
                'network_results': network_results,
                'evaluation_metrics': evaluation_metrics,
                'output_dir': self.output_dir
            }
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            logger.info(f"Full pipeline completed in {duration:.2f} minutes")
            
            return final_results
        
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Transaction Anomaly Detection System')
    
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to transaction data CSV file')
    
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Directory to save outputs')
    
    parser.add_argument('--mlflow-uri', type=str, default=None,
                        help='MLflow tracking URI')
    
    parser.add_argument('--sample', '-s', type=int, default=None,
                        help='Number of transactions to sample (for testing)')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Initialize the system
    system = TransactionAnomalyDetectionSystem(
        data_path=args.data,
        output_dir=args.output,
        mlflow_tracking_uri=args.mlflow_uri,
        sample_size=args.sample
    )
    
    # Run the full pipeline
    results = system.run_full_pipeline()
    
    logger.info(f"Results saved to {args.output}")
    print(f"Results saved to {args.output}") 
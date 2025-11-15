import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AMLVisualizer:
    """
    Visualization tools for AML and fraud detection analysis.
    
    This class provides methods to create various visualizations for
    transaction data analysis and model evaluation.
    """
    
    def __init__(self, 
                 theme: str = 'darkgrid', 
                 figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            theme: Seaborn theme for matplotlib visualizations
            figsize: Default figure size for matplotlib plots
        """
        self.theme = theme
        self.figsize = figsize
        
        # Set up visual style
        sns.set_theme(style=theme)
        
    def plot_transaction_distributions(self, 
                                     df: pd.DataFrame, 
                                     by_fraud: bool = True,
                                     save_path: Optional[str] = None) -> None:
        """
        Plot distributions of transaction amounts and other numerical features.
        
        Args:
            df: DataFrame with transaction data
            by_fraud: Whether to split distributions by fraud label
            save_path: Path to save the figure
        """
        plt.figure(figsize=self.figsize)
        
        # Plot transaction amount distribution
        if by_fraud and 'isFraud' in df.columns:
            plt.subplot(2, 2, 1)
            sns.histplot(
                data=df,
                x='amount',
                hue='isFraud',
                log_scale=True,
                bins=50,
                alpha=0.7
            )
            plt.title('Transaction Amount Distribution by Fraud Label')
            plt.xlabel('Amount (log scale)')
            
            # Plot transaction amount by type
            plt.subplot(2, 2, 2)
            sns.boxplot(
                data=df,
                x='type',
                y='amount',
                hue='isFraud'
            )
            plt.title('Transaction Amount by Type and Fraud Label')
            plt.yscale('log')
            plt.xlabel('Transaction Type')
            plt.ylabel('Amount (log scale)')
            
            # Plot balance changes
            plt.subplot(2, 2, 3)
            balance_change = df['oldbalanceOrg'] - df['newbalanceOrig']
            sns.histplot(
                x=balance_change,
                hue=df['isFraud'],
                log_scale=True,
                bins=50,
                alpha=0.7
            )
            plt.title('Balance Change Distribution by Fraud Label')
            plt.xlabel('Balance Change (log scale)')
            
            # Plot step (time) distribution
            plt.subplot(2, 2, 4)
            sns.histplot(
                data=df,
                x='step',
                hue='isFraud',
                bins=30,
                alpha=0.7
            )
            plt.title('Transaction Time Distribution by Fraud Label')
            plt.xlabel('Time Step')
            
        else:
            plt.subplot(2, 2, 1)
            sns.histplot(
                data=df,
                x='amount',
                log_scale=True,
                bins=50,
                alpha=0.7
            )
            plt.title('Transaction Amount Distribution')
            plt.xlabel('Amount (log scale)')
            
            # Plot transaction amount by type
            plt.subplot(2, 2, 2)
            sns.boxplot(
                data=df,
                x='type',
                y='amount'
            )
            plt.title('Transaction Amount by Type')
            plt.yscale('log')
            plt.xlabel('Transaction Type')
            plt.ylabel('Amount (log scale)')
            
            # Plot balance changes
            plt.subplot(2, 2, 3)
            balance_change = df['oldbalanceOrg'] - df['newbalanceOrig']
            sns.histplot(
                x=balance_change,
                log_scale=True,
                bins=50,
                alpha=0.7
            )
            plt.title('Balance Change Distribution')
            plt.xlabel('Balance Change (log scale)')
            
            # Plot step (time) distribution
            plt.subplot(2, 2, 4)
            sns.histplot(
                data=df,
                x='step',
                bins=30,
                alpha=0.7
            )
            plt.title('Transaction Time Distribution')
            plt.xlabel('Time Step')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Transaction distributions saved to {save_path}")
        
        plt.close()  # Close figure to free memory
    
    def plot_correlation_matrix(self, 
                              df: pd.DataFrame,
                              save_path: Optional[str] = None) -> None:
        """
        Plot correlation matrix of numerical features.
        
        Args:
            df: DataFrame with transaction data
            save_path: Path to save the figure
        """
        # Select only numerical columns
        num_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr = num_df.corr()
        
        # Plot
        plt.figure(figsize=self.figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        heatmap = sns.heatmap(
            corr,
            mask=mask,
            cmap='coolwarm',
            vmin=-1, 
            vmax=1,
            annot=True,
            fmt='.2f',
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={'shrink': .5}
        )
        plt.title('Feature Correlation Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        plt.close()
    
    def plot_transaction_flows(self,
                             df: pd.DataFrame,
                             n_accounts: int = 10,
                             save_path: Optional[str] = None) -> None:
        """
        Plot Sankey diagram of transaction flows between accounts.
        
        Args:
            df: DataFrame with transaction data
            n_accounts: Number of top accounts to include
            save_path: Path to save the figure
        """
        # Get top accounts by transaction count
        top_orig = df['nameOrig'].value_counts().head(n_accounts).index.tolist()
        top_dest = df['nameDest'].value_counts().head(n_accounts).index.tolist()
        
        # Filter transactions involving top accounts
        df_top = df[(df['nameOrig'].isin(top_orig)) & (df['nameDest'].isin(top_dest))]
        
        # Aggregate transaction amounts
        flows = df_top.groupby(['nameOrig', 'nameDest']).agg({
            'amount': 'sum',
            'isFraud': 'sum'
        }).reset_index()
        
        # Create node labels and indices
        all_accounts = list(set(flows['nameOrig'].tolist() + flows['nameDest'].tolist()))
        node_indices = {account: i for i, account in enumerate(all_accounts)}
        
        # Create Sankey diagram data
        source = [node_indices[account] for account in flows['nameOrig']]
        target = [node_indices[account] for account in flows['nameDest']]
        value = flows['amount'].tolist()
        
        # Color links based on fraud presence
        color = ['rgba(255,0,0,0.7)' if fraud > 0 else 'rgba(0,0,255,0.3)' 
                for fraud in flows['isFraud']]
        
        # Create figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=all_accounts
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color=color
            )
        )])
        
        fig.update_layout(
            title_text="Transaction Flows Between Top Accounts",
            font_size=10,
            height=600
        )
        
        if save_path:
            try:
                # Try to save as PNG (requires kaleido)
                fig.write_image(save_path, width=1200, height=800)
                logger.info(f"Transaction flow diagram saved to {save_path}")
            except Exception as e:
                # Fallback to HTML if kaleido not available
                logger.warning(f"Could not save Plotly image as PNG: {e}. Saving as HTML instead.")
                html_path = save_path.replace('.png', '.html')
                fig.write_html(html_path)
                logger.info(f"Transaction flow diagram saved to {html_path}")
    
    def plot_model_performance(self,
                              models_metrics: Dict[str, Dict[str, float]],
                              metric: str = 'auc',
                              save_path: Optional[str] = None) -> None:
        """
        Plot performance comparison of different models.
        
        Args:
            models_metrics: Dictionary of model metrics
            metric: Metric to plot
            save_path: Path to save the figure
        """
        model_names = list(models_metrics.keys())
        metric_values = [metrics.get(metric, 0) for _, metrics in models_metrics.items()]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, metric_values, color='skyblue')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom'
            )
        
        plt.title(f'Model Performance Comparison ({metric.upper()})')
        plt.ylabel(metric.upper())
        plt.ylim(0, 1.1 * max(metric_values))
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance comparison saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curves(self,
                       y_true: np.ndarray,
                       y_scores: Dict[str, np.ndarray],
                       save_path: Optional[str] = None) -> None:
        """
        Plot ROC curves for multiple models.
        
        Args:
            y_true: True binary labels
            y_scores: Dictionary of model prediction scores
            save_path: Path to save the figure
        """
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=self.figsize)
        
        for model_name, scores in y_scores.items():
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, 
                tpr, 
                lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.3f})'
            )
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curves(self,
                                   y_true: np.ndarray,
                                   y_scores: Dict[str, np.ndarray],
                                   save_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curves for multiple models.
        
        Args:
            y_true: True binary labels
            y_scores: Dictionary of model prediction scores
            save_path: Path to save the figure
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        plt.figure(figsize=self.figsize)
        
        for model_name, scores in y_scores.items():
            precision, recall, _ = precision_recall_curve(y_true, scores)
            avg_precision = average_precision_score(y_true, scores)
            
            plt.plot(
                recall, 
                precision, 
                lw=2, 
                label=f'{model_name} (AP = {avg_precision:.3f})'
            )
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curves saved to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self,
                              feature_importance: pd.DataFrame,
                              title: str = "Feature Importance",
                              top_n: int = 20,
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance from a model.
        
        Args:
            feature_importance: DataFrame with feature names and importance scores
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save the figure
        """
        # Sort and get top features
        sorted_features = feature_importance.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=self.figsize)
        sns.barplot(
            x='importance',
            y='feature',
            data=sorted_features,
            palette='viridis'
        )
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def plot_scenario_results(self,
                            summary_df: pd.DataFrame,
                            save_path: Optional[str] = None) -> None:
        """
        Plot summary of AML scenario results.
        
        Args:
            summary_df: DataFrame with scenario results
            save_path: Path to save the figure
        """
        plt.figure(figsize=self.figsize)
        
        plt.subplot(1, 2, 1)
        sns.barplot(
            y='scenario',
            x='flagged_count',
            data=summary_df,
            palette='coolwarm'
        )
        plt.title('Number of Flagged Transactions by Scenario')
        plt.xlabel('Flagged Count')
        plt.ylabel('')
        
        plt.subplot(1, 2, 2)
        sns.barplot(
            y='scenario',
            x='percentage',
            data=summary_df,
            palette='coolwarm'
        )
        plt.title('Percentage of Flagged Transactions by Scenario')
        plt.xlabel('Percentage (%)')
        plt.ylabel('')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Scenario results plot saved to {save_path}")
        
        plt.close()
    
    def plot_anomaly_scores(self,
                          scores: np.ndarray,
                          threshold: float,
                          y_true: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Plot distribution of anomaly scores with decision threshold.
        
        Args:
            scores: Array of anomaly scores
            threshold: Threshold for anomaly detection
            y_true: True binary labels (if available)
            save_path: Path to save the figure
        """
        plt.figure(figsize=self.figsize)
        
        if y_true is not None:
            plt.hist(
                [scores[y_true == 0], scores[y_true == 1]],
                bins=50,
                alpha=0.7,
                label=['Normal', 'Fraud/Anomaly'],
                color=['blue', 'red']
            )
            plt.legend()
        else:
            plt.hist(scores, bins=50, alpha=0.7, color='blue')
        
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
        plt.legend()
        plt.title('Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Anomaly scores plot saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            normalize: bool = False,
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix for binary classification.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            normalize: Whether to normalize by row
            title: Plot title
            save_path: Path to save the figure
        """
        from sklearn.metrics import confusion_matrix
        
        # Compute confusion matrix
        # sklearn format: [[TN, FP], [FN, TP]]
        # where rows = true labels (0=Normal, 1=Fraud), cols = predicted labels
        cm = confusion_matrix(y_true, y_pred)
        
        # Debug: log the confusion matrix
        logger.debug(f"Confusion matrix: {cm}")
        logger.debug(f"y_true: {np.unique(y_true, return_counts=True)}")
        logger.debug(f"y_pred: {np.unique(y_pred, return_counts=True)}")
        
        if normalize:
            # Normalize by row (each row sums to 1.0)
            row_sums = cm.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1
            cm = cm.astype('float') / row_sums
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        # Plot confusion matrix
        # Rows (y-axis) = True labels: [Normal, Fraud/Suspicious]
        # Columns (x-axis) = Predicted labels: [Normal, Fraud/Suspicious]
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt,
            cmap="Blues",
            cbar=False,
            square=True,
            xticklabels=['Normal', 'Fraud/Suspicious'],
            yticklabels=['Normal', 'Fraud/Suspicious'],
            vmin=0.0,
            vmax=1.0 if normalize else None
        )
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title)
        
        # Ensure tight layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
        
    def plot_shap_summary(self,
                        shap_values: np.ndarray,
                        features: pd.DataFrame,
                        max_display: int = 20,
                        save_path: Optional[str] = None) -> None:
        """
        Plot SHAP summary plot for model explanations.
        
        Args:
            shap_values: SHAP values from explainer
            features: Feature data
            max_display: Maximum number of features to display
            save_path: Path to save the figure
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not available. Skipping SHAP plot.")
            return
        
        try:
            plt.figure(figsize=self.figsize)
            shap.summary_plot(
                shap_values, 
                features,
                max_display=max_display,
                show=False
            )
            
            plt.title("SHAP Feature Importance")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
        except Exception as e:
            logger.warning(f"Error creating SHAP plot: {e}")
        finally:
            plt.close() 
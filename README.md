# Transaction Anomaly Detection System

## Project Overview

This project implements an advanced transaction anomaly detection system for Anti-Money Laundering (AML) compliance. It combines traditional rule-based monitoring with cutting-edge machine learning techniques and network analysis to identify suspicious transactions and patterns.

## Key Features

### 1. Traditional Rule-Based Scenarios
- **Large Transaction Detection**: Identify transactions exceeding regulatory thresholds
- **Structuring Detection**: Detect attempts to avoid reporting thresholds by breaking transactions
- **Rapid Movement**: Flag funds moving quickly through multiple accounts (layering)  
- **Unusual Activity**: Identify transactions deviating from historical patterns
- **Smurfing Detection**: Find multiple small transactions to same destination
- **High-Risk Entity Monitoring**: Flag transactions involving high-risk countries/entities

### 2. Advanced Adaptive Thresholding
- **Customer Segmentation**: Automatically groups customers based on transaction behaviors
- **Segment-Specific Thresholds**: Calculates different thresholds for different customer segments
- **Statistical Outlier Detection**: Employs Z-score, IQR, and MAD-based methods for anomaly detection
- **Dynamic Adjustment**: Thresholds adapt based on historical patterns and transaction context

### 3. Machine Learning Models
- **Isolation Forest**: Unsupervised detection of outlier transactions
- **XGBoost & LightGBM**: Supervised learning for fraud classification
- **Random Forest**: Feature importance analysis and classification
- **Deep Learning Autoencoders**: Neural network-based anomaly detection
- **LSTM-Autoencoder**: Temporal anomaly detection for transaction sequences

### 4. Graph-Based Network Analysis
- **Graph Neural Networks (GNN)**: Advanced pattern recognition in transaction networks
- **Cycle Detection**: Identify circular money movement patterns
- **Fan-in/Fan-out Analysis**: Detect money mule and collection account patterns
- **Community Detection**: Find groups of related accounts
- **Centrality Analysis**: Identify key nodes in money laundering networks

### 5. Visualization and Reporting
- **Transaction Network Visualization**: Interactive graph-based visualizations
- **Risk Scoring**: Multi-factor risk assessment for transactions and accounts
- **Alert Generation**: Customizable alert system for suspicious activities
- **Model Evaluation**: Performance metrics and visualization for models

## Project Structure

```
Transaction Anomaly Detection System/
├── data/                           # Data storage
│   └── PS_20174392719_1491204439457_log.csv  # Transaction dataset
├── notebooks/                      # Jupyter notebooks
│   ├── 1_exploratory_data_analysis.ipynb     # Data exploration
│   ├── 2_aml_scenarios_and_risk_scoring.ipynb # Rule-based analysis
│   ├── 3_ml_anomaly_detection.ipynb          # ML-based analysis
│   └── 4_network_analysis.ipynb              # Graph-based analysis
├── src/                            # Source code
│   ├── data/                       # Data processing modules
│   │   └── preprocessor.py         # Data preprocessing
│   ├── models/                     # Model implementations
│   │   ├── ml_anomaly_detection.py # ML-based models
│   │   ├── network_analysis.py     # Network/graph-based models
│   │   └── rule_based_scenarios.py # Rule-based detection
│   ├── utils/                      # Utility functions
│   │   └── helpers.py              # Helper functions
│   ├── visualization/              # Visualization tools
│   │   └── visualizer.py           # Visualization functions
│   └── main.py                     # Main execution script
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## Technical Highlights

- **MLflow Integration**: Model versioning, parameter tracking, and experiment management
- **Graph Neural Networks**: Advanced graph embedding techniques for network analysis
- **LSTM-based Sequence Analysis**: Temporal pattern detection in transaction sequences
- **Adaptive Thresholding**: Dynamically adjusting detection rules based on customer segments
- **Ensemble Methods**: Combining multiple detection techniques for higher accuracy
- **SHAP Values**: Explainable AI for understanding model decisions
- **Deep Learning Models**: Neural network-based pattern recognition

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Transaction-Anomaly-Detection.git
cd Transaction-Anomaly-Detection
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
The dataset used is the "IEEE-CIS Fraud Detection" dataset from Kaggle or similar financial transaction datasets.

## Usage

### Running the Main Analysis:
```bash
python src/main.py
```

### Exploring the Notebooks:
Run Jupyter Notebook or Jupyter Lab to explore the analysis notebooks:
```bash
jupyter notebook
# or
jupyter lab
```

## Model Performance

The system combines multiple detection methods for optimal performance:

- **Rule-based detection**: High precision for known patterns
- **Machine learning models**: ~95% AUC for supervised models
- **Unsupervised anomaly detection**: Effective at detecting novel patterns
- **Network analysis**: Identifies complex relationships invisible to other methods
- **Deep learning models**: Capture complex non-linear patterns in transaction data

## Future Enhancements

- Real-time stream processing integration
- Entity embedding for customer risk profiling
- Advanced alert management system
- Additional visualization dashboards
- API for integration with other systems 
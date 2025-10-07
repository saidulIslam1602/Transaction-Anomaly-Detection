# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training on Databricks
# MAGIC 
# MAGIC Train anomaly detection models using Databricks ML Runtime and MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import shap

# Set MLflow experiment
mlflow.set_experiment("/Shared/transaction-anomaly-detection")

# Load features
feature_store_path = "/mnt/delta/feature_store/transaction_features"
df = spark.read.format("delta").load(feature_store_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare Data

# COMMAND ----------

# Convert to Pandas for scikit-learn
pdf = df.toPandas()

# Select features
feature_cols = [col for col in pdf.columns if col not in 
                ['nameOrig', 'nameDest', 'isFraud', 'ingestion_time', 'ingestion_date', 'step']]

X = pdf[feature_cols]
y = pdf['isFraud']

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Fraud rate in training: {y_train.mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train XGBoost Model

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_model") as run:
    # Define model
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_prob)
    
    # Log parameters
    mlflow.log_params(model.get_params())
    
    # Log metrics
    mlflow.log_metrics({
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "xgboost_model")
    
    # Log feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    mlflow.log_dict(importance_df.to_dict(), "feature_importance.json")
    
    print(f"XGBoost Model - AUC: {auc:.4f}, F1: {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train LightGBM Model

# COMMAND ----------

with mlflow.start_run(run_name="lightgbm_model") as run:
    # Define model
    model = LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.05,
        random_state=42,
        is_unbalance=True
    )
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_prob)
    
    # Log everything
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    })
    mlflow.sklearn.log_model(model, "lightgbm_model")
    
    print(f"LightGBM Model - AUC: {auc:.4f}, F1: {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Explainability with SHAP

# COMMAND ----------

# Load best model (XGBoost)
best_model = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/xgboost_model")

# Calculate SHAP values
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled[:100])

# Save SHAP summary plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test[:100], feature_names=feature_cols, show=False)
plt.savefig("/dbfs/tmp/shap_summary.png", bbox_inches='tight', dpi=150)
plt.close()

with mlflow.start_run(run_id=run.info.run_id):
    mlflow.log_artifact("/dbfs/tmp/shap_summary.png")

print("SHAP analysis complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Best Model

# COMMAND ----------

# Register the best model to Model Registry
model_name = "transaction-anomaly-detection"

model_uri = f"runs:/{run.info.run_id}/xgboost_model"
model_version = mlflow.register_model(model_uri, model_name)

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)

print(f"Model registered: {model_name} v{model_version.version}")


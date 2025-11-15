# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering for Anomaly Detection
# MAGIC 
# MAGIC This notebook creates features for the anomaly detection models.
# MAGIC 
# MAGIC **Features Created:**
# MAGIC 1. Transaction-level features
# MAGIC 2. Temporal aggregations (rolling windows)
# MAGIC 3. Network features (graph-based)
# MAGIC 4. Behavioral patterns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import DeltaTable
import mlflow

# Load data from silver layer
silver_path = "/mnt/delta/silver/transactions"
feature_store_path = "/mnt/delta/feature_store/transaction_features"

df = spark.read.format("delta").load(silver_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transaction-Level Features

# COMMAND ----------

df_features = (df
    # Basic features
    .withColumn("is_large_transaction", (col("amount") > 10000).cast("int"))
    .withColumn("is_cash_transaction", 
                (col("type").isin(["CASH_OUT", "CASH_IN"])).cast("int"))
    
    # Balance features
    .withColumn("orig_balance_ratio", 
                col("amount") / (col("oldbalanceOrg") + lit(1.0)))
    .withColumn("dest_balance_ratio",
                col("amount") / (col("oldbalanceDest") + lit(1.0)))
    .withColumn("balance_mismatch", 
                (col("error_balance_orig") | col("error_balance_dest")).cast("int"))
    
    # Zero balance indicators
    .withColumn("is_zero_orig_before", (col("oldbalanceOrg") == 0).cast("int"))
    .withColumn("is_zero_orig_after", (col("newbalanceOrig") == 0).cast("int"))
    .withColumn("is_zero_dest_before", (col("oldbalanceDest") == 0).cast("int"))
    .withColumn("is_zero_dest_after", (col("newbalanceDest") == 0).cast("int"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Temporal Features (Rolling Windows)

# COMMAND ----------

# Define windows for origin accounts
window_1h = Window.partitionBy("nameOrig").orderBy("step").rangeBetween(-1, 0)
window_24h = Window.partitionBy("nameOrig").orderBy("step").rangeBetween(-24, 0)
window_7d = Window.partitionBy("nameOrig").orderBy("step").rangeBetween(-168, 0)

df_features = (df_features
    # 1 hour window
    .withColumn("txn_count_1h", count("*").over(window_1h))
    .withColumn("total_amount_1h", sum("amount").over(window_1h))
    .withColumn("avg_amount_1h", avg("amount").over(window_1h))
    .withColumn("max_amount_1h", max("amount").over(window_1h))
    
    # 24 hour window
    .withColumn("txn_count_24h", count("*").over(window_24h))
    .withColumn("total_amount_24h", sum("amount").over(window_24h))
    .withColumn("avg_amount_24h", avg("amount").over(window_24h))
    .withColumn("std_amount_24h", stddev("amount").over(window_24h))
    
    # 7 day window
    .withColumn("txn_count_7d", count("*").over(window_7d))
    .withColumn("total_amount_7d", sum("amount").over(window_7d))
    .withColumn("unique_dest_7d", approx_count_distinct("nameDest").over(window_7d))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Network Features

# COMMAND ----------

# Account transaction patterns
account_stats = (df
    .groupBy("nameOrig")
    .agg(
        count("*").alias("total_transactions"),
        countDistinct("nameDest").alias("unique_destinations"),
        sum("amount").alias("total_sent"),
        avg("amount").alias("avg_sent"),
        sum("isFraud").alias("fraud_transactions")
    )
    .withColumn("fraud_rate", col("fraud_transactions") / col("total_transactions"))
    .withColumn("fan_out_score", col("unique_destinations") / col("total_transactions"))
)

# Join back to main dataframe
df_features = df_features.join(
    account_stats.select(
        col("nameOrig"),
        col("fraud_rate").alias("account_fraud_rate"),
        col("fan_out_score").alias("account_fan_out")
    ),
    on="nameOrig",
    how="left"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Behavioral Features

# COMMAND ----------

# Transaction type distribution per account
type_distribution = (df
    .groupBy("nameOrig", "type")
    .agg(count("*").alias("type_count"))
    .groupBy("nameOrig")
    .pivot("type")
    .agg(first("type_count"))
    .fillna(0)
)

# Calculate type diversity (entropy)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

df_features = df_features.join(
    type_distribution.select("nameOrig", *[col(c).alias(f"type_count_{c}") for c in type_distribution.columns if c != "nameOrig"]),
    on="nameOrig",
    how="left"
).fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Feature Store

# COMMAND ----------

# Select final features
feature_columns = [
    "step", "type", "amount", "nameOrig", "nameDest",
    "is_large_transaction", "is_cash_transaction",
    "orig_balance_ratio", "dest_balance_ratio", "balance_mismatch",
    "is_zero_orig_before", "is_zero_orig_after",
    "is_zero_dest_before", "is_zero_dest_after",
    "txn_count_1h", "total_amount_1h", "avg_amount_1h", "max_amount_1h",
    "txn_count_24h", "total_amount_24h", "avg_amount_24h", "std_amount_24h",
    "txn_count_7d", "total_amount_7d", "unique_dest_7d",
    "account_fraud_rate", "account_fan_out",
    "isFraud", "ingestion_time", "ingestion_date"
]

df_final = df_features.select(feature_columns)

# Write to feature store
(df_final.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("ingestion_date")
    .save(feature_store_path)
)

# Create table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS feature_store_transactions
    USING DELTA
    LOCATION '{feature_store_path}'
""")

print(f"Features saved to: {feature_store_path}")
print(f"Total features: {len(feature_columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Statistics

# COMMAND ----------

# Calculate feature statistics
feature_stats = df_final.describe()
display(feature_stats)

# Check feature distributions
numeric_features = [f.name for f in df_final.schema.fields if isinstance(f.dataType, (DoubleType, IntegerType))]

for feature in numeric_features[:5]:  # Show first 5 features
    display(df_final.select(feature).summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alternative: Using dbt for Data Modeling
# MAGIC 
# MAGIC For more structured data modeling, you can also use dbt models:
# MAGIC 
# MAGIC ```bash
# MAGIC # Run dbt models (if dbt is configured)
# MAGIC dbt run --select int_transaction_features
# MAGIC ```
# MAGIC 
# MAGIC See `dbt/README.md` for dbt setup and usage.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log to MLflow

# COMMAND ----------

# Log feature metadata
import mlflow

with mlflow.start_run(run_name="feature_engineering"):
    mlflow.log_param("feature_count", len(feature_columns))
    mlflow.log_param("record_count", df_final.count())
    mlflow.log_param("feature_store_path", feature_store_path)
    
    # Log feature statistics
    stats_dict = {}
    for row in feature_stats.collect():
        metric_name = row['summary']
        for col_name in feature_stats.columns[1:]:
            if metric_name in ['count', 'mean', 'stddev']:
                try:
                    stats_dict[f"{col_name}_{metric_name}"] = float(row[col_name])
                except:
                    pass
    
    mlflow.log_metrics(stats_dict)

print("Feature engineering complete!")


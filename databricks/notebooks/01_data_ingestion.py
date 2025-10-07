# Databricks notebook source
# MAGIC %md
# MAGIC # Transaction Data Ingestion Pipeline
# MAGIC 
# MAGIC This notebook handles the ingestion of transaction data from Azure Blob Storage into Delta Lake.
# MAGIC 
# MAGIC **Steps:**
# MAGIC 1. Read raw transaction data from Azure Blob Storage
# MAGIC 2. Validate and clean the data
# MAGIC 3. Write to Delta Lake with partitioning
# MAGIC 4. Create optimized tables for analytics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import DeltaTable
import datetime

# Configure Azure Blob Storage access
storage_account_name = dbutils.secrets.get(scope="aml-secrets", key="storage-account-name")
storage_account_key = dbutils.secrets.get(scope="aml-secrets", key="storage-account-key")

spark.conf.set(
    f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
    storage_account_key
)

# Define paths
bronze_path = f"wasbs://ml-data@{storage_account_name}.blob.core.windows.net/bronze/transactions"
silver_path = "/mnt/delta/silver/transactions"
gold_path = "/mnt/delta/gold/transactions"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Schema

# COMMAND ----------

transaction_schema = StructType([
    StructField("step", IntegerType(), True),
    StructField("type", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("nameOrig", StringType(), True),
    StructField("oldbalanceOrg", DoubleType(), True),
    StructField("newbalanceOrig", DoubleType(), True),
    StructField("nameDest", StringType(), True),
    StructField("oldbalanceDest", DoubleType(), True),
    StructField("newbalanceDest", DoubleType(), True),
    StructField("isFraud", IntegerType(), True),
    StructField("isFlaggedFraud", IntegerType(), True)
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Raw Data (Bronze Layer)

# COMMAND ----------

df_raw = (spark.read
    .format("csv")
    .option("header", "true")
    .schema(transaction_schema)
    .load(bronze_path)
)

print(f"Loaded {df_raw.count()} transactions from bronze layer")
display(df_raw.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

# Check for nulls
null_counts = df_raw.select([count(when(col(c).isNull(), c)).alias(c) for c in df_raw.columns])
display(null_counts)

# Check for duplicates
duplicate_count = df_raw.count() - df_raw.dropDuplicates().count()
print(f"Found {duplicate_count} duplicate records")

# Value ranges
df_raw.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transform and Clean Data (Silver Layer)

# COMMAND ----------

df_silver = (df_raw
    # Add ingestion timestamp
    .withColumn("ingestion_time", current_timestamp())
    .withColumn("ingestion_date", current_date())
    
    # Add derived features
    .withColumn("balance_change_orig", col("newbalanceOrig") - col("oldbalanceOrg"))
    .withColumn("balance_change_dest", col("newbalanceDest") - col("oldbalanceDest"))
    .withColumn("amount_to_balance_ratio", 
                col("amount") / (col("oldbalanceOrg") + lit(1.0)))
    
    # Add error flags
    .withColumn("error_balance_orig", 
                when(col("newbalanceOrig") != col("oldbalanceOrg") - col("amount"), 1)
                .otherwise(0))
    .withColumn("error_balance_dest",
                when(col("newbalanceDest") != col("oldbalanceDest") + col("amount"), 1)
                .otherwise(0))
    
    # Remove duplicates
    .dropDuplicates()
    
    # Filter invalid records
    .filter(col("amount") >= 0)
    .filter(col("oldbalanceOrg") >= 0)
)

print(f"Cleaned data: {df_silver.count()} transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Delta Lake (Silver Layer)

# COMMAND ----------

# Write to Delta Lake with partitioning
(df_silver.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("ingestion_date", "type")
    .option("overwriteSchema", "true")
    .save(silver_path)
)

print(f"Written to silver layer: {silver_path}")

# Create table
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS silver_transactions
    USING DELTA
    LOCATION '{silver_path}'
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Optimize Delta Table

# COMMAND ----------

# Optimize and Z-order
spark.sql(f"""
    OPTIMIZE silver_transactions
    ZORDER BY (nameOrig, nameDest, amount)
""")

# Vacuum old files (keep 7 days)
spark.sql("""
    VACUUM silver_transactions RETAIN 168 HOURS
""")

print("Delta table optimized")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Gold Layer Aggregations

# COMMAND ----------

# Account-level aggregations
df_gold_accounts = (spark.read
    .format("delta")
    .load(silver_path)
    .groupBy("nameOrig", "type")
    .agg(
        count("*").alias("transaction_count"),
        sum("amount").alias("total_amount"),
        avg("amount").alias("avg_amount"),
        max("amount").alias("max_amount"),
        sum("isFraud").alias("fraud_count"),
        (sum("isFraud") / count("*")).alias("fraud_rate")
    )
)

# Write gold layer
gold_accounts_path = f"{gold_path}/accounts"
(df_gold_accounts.write
    .format("delta")
    .mode("overwrite")
    .save(gold_accounts_path)
)

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS gold_account_metrics
    USING DELTA
    LOCATION '{gold_accounts_path}'
""")

print("Gold layer created successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Metrics

# COMMAND ----------

# Calculate and display metrics
metrics = {
    "total_transactions": df_silver.count(),
    "unique_accounts": df_silver.select("nameOrig").distinct().count(),
    "fraud_rate": df_silver.filter(col("isFraud") == 1).count() / df_silver.count(),
    "avg_transaction_amount": df_silver.select(avg("amount")).collect()[0][0],
    "data_quality_score": 1.0 - (null_counts.first()[0] / df_silver.count())
}

print("Data Ingestion Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value}")

# Log metrics to MLflow
import mlflow
mlflow.log_metrics(metrics)


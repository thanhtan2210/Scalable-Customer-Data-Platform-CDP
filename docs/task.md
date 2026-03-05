# Scalable Customer Data Platform (CDP)
Objective: Build a scalable customer data processing platform (target: 50M users) to serve Data Engineering (DE), Data Science (DS), and MLOps requirements.

## Overview
- **Goal:** Ingest and build a clean feature table for downstream modeling and analytics.
- **Key Constraints:** Use S3 for raw and feature storage (Parquet), Spark for heavy ETL tasks, implement a robust partitioning strategy, and evaluate Iceberg/Delta Lake for ACID compliance and schema evolution.
- **Resume Highlight:** "Data Warehousing (BigQuery/Simulating Iceberg architecture) — built S3 Parquet pipeline with Spark for data cleaning & deduplication."

## Phase 1 — Data Engineering (DE)
- **Purpose:** Standardize raw data, remove noise and duplicates, and create an efficiently partitioned feature table for downstream jobs.
- **Key Steps:**
  1. **Ingestion:** Stream logs from S3, connect to Postgres for master/customer data (CDC), and write to the landing zone (S3 raw/landing) in Parquet format.
  2. **Cleaning (Spark):**
     - Schema enforcement (explicit schemas and types).
     - Null handling and type conversions (e.g., converting 'Total Charges' to numeric).
     - Normalize categorical values using canonical mapping.
     - Outlier detection and handling for numeric features (capping/winsorization).
  3. **Duplicate Handling:**
     - Deduplicate based on stable keys (CustomerID + event_timestamp / last_updated).
     - Use window functions to select the latest record:
       - Partition by `CustomerID`, order by `last_updated` descending, and filter where `row_number() == 1`.
  4. **Partitioning & Storage:**
     - Write Parquet to S3 with a partition layout such as: `s3://bucket/cdp/customer_features/year=YYYY/month=MM/day=DD/`
     - Partition keys: Date (`event_date`) and potentially low-cardinality keys like region or product line.
     - **Note:** Avoid partitioning by high-cardinality keys like `CustomerID`.
  5. **Metadata & Format:**
     - Start with Parquet; evaluate Delta Lake or Apache Iceberg for ACID transactions, merge capabilities, and schema evolution.
     - For local testing, simulate Iceberg/Delta using Docker + Spark with a configured catalog.
  6. **Orchestration:**
     - Use Apache Airflow to schedule jobs (daily incrementals or hourly micro-batches).
     - **Workflow:** Ingest -> Validate (Great Expectations) -> Transform (Spark) -> Write Feature Table -> Register Metadata (Glue/Metastore or Iceberg catalog).
- **Practical Checks:**
  - Validate row counts and null rates; detect sudden data drift.
  - Compact small Parquet files after multiple incremental writes.

### Spark Cleaning & Deduplication Example (PySpark)
Example PySpark job: Read, clean, deduplicate, and write Parquet partitioned by date.

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("cdp-clean-dedup") \
    .getOrCreate()

# Read raw data from S3
raw_df = spark.read.parquet("s3a://my-bucket/landing/telco_logs/")

# Enforce schema conversions & cleaning
df = raw_df \
    .withColumn("TotalCharges", F.col("TotalCharges").cast("double")) \
    .withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double")) \
    .withColumn("TenureMonths", F.col("TenureMonths").cast("int")) \
    .withColumn("event_date", F.to_date(F.col("event_timestamp")))

# Simple normalizations
df = df.replace({'Yes': 'Yes', 'No': 'No'}, subset=['Churn Label'])

# Deduplicate by CustomerID keeping the latest record
win = Window.partitionBy("CustomerID").orderBy(F.col("last_updated").desc())
df_dedup = df.withColumn("rn", F.row_number().over(win)).filter(F.col("rn") == 1).drop("rn")

# Write to S3 as partitioned Parquet
df_dedup \
    .repartition(200) \
    .write \
    .mode("overwrite") \
    .partitionBy("event_date") \
    .parquet("s3a://my-bucket/cdp/feature_tables/customer_features/")
```

### Best Practices for Partitioning & File Sizing:
- Aim for partition files between 100–500 MB for efficient reads.
- Use date partitions (year/month/day) for time-based filtering.
- Aggregate small files using a nightly coalesce or compaction job.
- If using Iceberg/Delta, utilize table-level optimization commands (Compaction/Optimize).

### Delta Lake / Apache Iceberg Notes (Local with Docker):
- **Why:** ACID compliance, time travel, schema evolution, and MERGE support.
- **Local Testing:** Run Spark with `delta-core` or `iceberg-spark-runtime` jars.
  - **Delta Lake:** Add `delta-core` jar to Spark session and write using `format("delta")`.
  - **Iceberg:** Configure Spark session with `spark.sql.catalog.local = 'org.apache.iceberg.spark.SparkCatalog'` and `catalog-impl=hadoop`.
- **Infrastructure Limitation:** Keep Parquet on S3 and simulate Iceberg via partitioned layouts and metadata.

### Airflow DAG (Skeleton)
Use `SparkSubmitOperator` to execute Spark jobs on a daily schedule.

```python
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

default_args = {
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG('cdp_daily',
         default_args=default_args,
         schedule_interval='@daily',
         start_date=days_ago(1)) as dag:

    spark_clean = SparkSubmitOperator(
        application='/opt/spark/jobs/cdp_clean_dedup.py',
        task_id='spark_clean_dedup',
        conn_id='spark_default',
        application_args=['--env', 'prod']
    )

    spark_clean
```

### Validation & QA:
- Run Great Expectations (GE) after ETL to validate null rates and schema integrity.
- Store validation metrics in a monitoring table and set up alerts for anomalies.

## Phase 2 — Data Science (DS)
- Utilize the feature table from the DE phase.
- **Standard Pipeline:** Train/test split, baseline model (Logistic Regression), advanced models (XGBoost/LightGBM), hyperparameter tuning, probability calibration, and SHAP explanations.
- **Persistence:** Save model and preprocessing pipelines using `joblib` or ONNX.

## Phase 3 — MLOps / Serving & BI
- **Export:** Model and feature pipeline deployment.
- **Serving:** FastAPI or specialized model servers (SageMaker / TorchServe).
- **Online Inference:** Retrieve latest features via a Feature Store or by filtering S3 Parquet files.
- **Batch/Streaming Scores:** Schedule daily batch jobs to score all customers and write predictions back to the CDP (S3/Warehouse/DB).

## Tech Stack Recommendations
- **Ingestion & Orchestration:** Apache Airflow, Kafka (optional).
- **ETL:** PySpark on EMR / Dataproc / Kubernetes.
- **Storage:** S3 with Parquet (evolving to Iceberg/Delta Lake for ACID).
- **Warehouse (Analytics):** BigQuery (or simulate with Iceberg + Presto/Trino).
- **Feature Store/Serving:** Feature tables in S3/Iceberg + online layer (Redis / Feast).
- **Modeling:** scikit-learn / XGBoost / LightGBM, joblib.
- **Serving:** FastAPI + Docker + Kubernetes.
- **Validation:** Great Expectations, Unit Tests for ETL.
- **Resume Bullet:** "Built a Scalable Customer Data Platform (CDP) using PySpark & Airflow; developed robust cleaning and deduplication pipelines and stored partitioned Parquet on S3 (Delta/Iceberg simulation), enabling downstream model training for churn predictions."

## Deliverables
- Spark ETL jobs (Clean / Dedup / Feature Computation).
- Airflow DAGs.
- Partitioned Parquet on S3 or Iceberg tables.
- DE Data Quality checks (Great Expectations).
- Feature table schema and documentation.

## Notes & Fallbacks
- If Iceberg/Delta is not feasible: Use Parquet on S3 with strong partitioning and metadata management in AWS Glue/Metastore.
- Record steps for Resume: Highlight S3 Parquet with simulated Iceberg architecture if using BigQuery as a warehouse.

# Scalable Customer Data Platform (CDP)
Mục tiêu: xây dựng nền tảng xử lý dữ liệu khách hàng có khả năng mở rộng (50M users), phục vụ DE, DS, MLOps.

## Overview / Tổng quan
- Goal: Ingest & build a clean feature table for downstream modeling and analytics.
- Key constraints: S3 as raw + feature storage (Parquet), Spark for heavy ETL, partitioning strategy, eventual Iceberg/Delta Lake for ACID & schema-evolution.
- Trong CV: "Data Warehousing (BigQuery/Simulating Iceberg architecture) — built S3 Parquet pipeline with Spark for data cleaning & dedup".

## Giai đoạn 1 — Data Engineering (DE)
- Purpose / Tác dụng: Chuẩn hóa dữ liệu thô, loại bỏ nhiễu và trùng lặp, tạo feature table phân vùng hiệu quả cho downstream jobs.
- Key steps:
  1. Ingestion: stream logs from S3, connect to Postgres for master/customer data (CDC), write landing zone (S3 raw/landing) as Parquet.
  2. Cleaning (Spark):
     - Schema enforcement (explicit schema, types).
     - Null handling and conversions (e.g., Total Charges numeric).
     - Normalize categorical values with canonical mapping.
     - Outlier detection/handling for numeric features (capping / winsorize).
  3. Duplicate handling:
     - Dedup based on stable keys (CustomerID + event_timestamp / last_updated).
     - Use windowing to pick latest record:
       - Partition by CustomerID, order by last_updated desc, row_number() == 1.
  4. Partitioning & Storage:
     - Write Parquet to S3 with partition layout like: s3://bucket/cdp/customer_features/year=YYYY/month=MM/day=DD/
     - Partition keys: date (event_date) & maybe region or product line (low-cardinality).
     - Avoid partition by high-cardinality keys (CustomerID).
  5. Metadata & Format:
     - Start on Parquet; evaluate Delta Lake or Apache Iceberg for ACID/merge / schema evolution.
     - For local testing, simulate Iceberg/Delta with Docker + Spark (set up catalog).
  6. Orchestration:
     - Use Airflow to schedule jobs (daily incremental, hourly micro-batches).
     - Jobs: ingest -> validate (Great Expectations) -> transform (Spark) -> write feature table -> register metadata (Glue/Metastore or Iceberg catalog).
- Practical checks:
  - Validate row counts, null rates, detect sudden changes (data drift).
  - Compact small Parquet files after many small writes.

### Spark Cleaning & Dedup Example (PySpark)
- Example PySpark job: read, clean, deduplicate, write Parquet partitioned by date.
```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("cdp-clean-dedup") \
    .getOrCreate()

# read raw data (s3 path)
raw_df = spark.read.parquet("s3a://my-bucket/landing/telco_logs/")

# enforce schema conversions & cleaning
df = raw_df \
    .withColumn("TotalCharges", F.col("TotalCharges").cast("double")) \
    .withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double")) \
    .withColumn("TenureMonths", F.col("TenureMonths").cast("int")) \
    .withColumn("event_date", F.to_date(F.col("event_timestamp")))

# simple normalizations
df = df.replace({'Yes': 'Yes', 'No': 'No'}, subset=['Churn Label'])

# deduplicate by CustomerID keeping latest
win = Window.partitionBy("CustomerID").orderBy(F.col("last_updated").desc())
df_dedup = df.withColumn("rn", F.row_number().over(win)).filter(F.col("rn") == 1).drop("rn")

# write to S3 as partitioned Parquet (partition by date)
df_dedup \
    .repartition(200) \
    .write \
    .mode("overwrite") \
    .partitionBy("event_date") \
    .parquet("s3a://my-bucket/cdp/feature_tables/customer_features/")
```

### Best practices for partitioning + file sizing:
- Aim for partition files ~100–500 MB for efficient reads.
- Use date partitions (year/month/day) for time-based filtering.
- Aggregate small files using coalesce/compaction job nightly.
- If using Iceberg/Delta, use table-level optimization (compaction/optimize commands).

### Delta Lake / Apache Iceberg notes (local with Docker):
- Why: ACID, time travel, schema evolution, MERGE.
- Local testing: run Spark with delta-core or iceberg-spark runtime jar:
  - Example with Delta Lake (Spark + delta-core):
    - Add delta-core jar to Spark session and write as `format("delta")`.
  - Example with Iceberg, configure Spark session:
    - `spark.sql.catalog.local = 'org.apache.iceberg.spark.SparkCatalog'` with `catalog-impl` set to `hadoop`.
- If infra limited: keep Parquet on S3 and simulate Iceberg via partitioned layout + metadata.

### Airflow DAG (skeleton)
- Use SparkSubmitOperator to execute Spark job; schedule daily.
```python
# filepath: d:\Bon Bon\SourceCode\AI-project\Customer_Churn_Prediction\airflow_dags\cdp_dag.py
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
- Use Great Expectations (GE) runs after ETL to validate null rates & schema.
- Store validation metrics into a monitoring table; alert on anomalies.

## Giai đoạn 2 — Data Science (DS)
- Use the feature table from DE; standard pipeline: train/test split, model baseline (Logistic), model improvements (XGBoost/LightGBM), hyperparameter tune, calibration, SHAP explanations.
- Persist model + preprocessing (joblib / ONNX).

## Giai đoạn 3 — MLOps / Serving & BI
- Export model & feature pipeline; serve via FastAPI or model server (SageMaker / TorchServe).
- Online inference: retrieve latest features (via feature store or read Parquet & filter).
- Batch/Streaming scores: schedule daily batch job to score all customers and write predictions to CDP (S3/warehouse/DB).

## Tech Stack recommendation
- Ingestion & Orchestration: Airflow, Kafka (optional)
- ETL: PySpark on EMR / Dataproc / Kubernetes spark
- Storage: S3 with Parquet (-> Iceberg/Delta Lake for ACID)
- Warehouse (analytics): BigQuery (or simulate with Iceberg + Presto/Trino)
- Feature Store/Serving: Feature table in S3/Iceberg + online layer (Redis / Feast)
- Model: scikit-learn / XGBoost / LightGBM, joblib
- Serving: FastAPI + container (Docker) + Kubernetes
- Validation: Great Expectations, unit tests for ETL
- CV line for JD: "Built Scalable Customer Data Platform (CDP) using PySpark & Airflow; wrote robust cleaning & dedup pipelines and stored partitioned Parquet on S3 (Delta/Iceberg simulation), enabling downstream model training for churn predictions."

## Deliverables
- Spark jobs for ETL (clean/dedup/feature compute)
- Airflow DAGs
- Partitioned Parquet on S3 or Iceberg table
- DE data quality checks (Great Expectations)
- Feature table schema & documentation

## Notes / Fallback
- If Iceberg/Delta not feasible in current infra: use Parquet on S3 + strong partitioning + metadata in Glue/Metastore.
- Record steps in CV: highlight S3 Parquet with simulated Iceberg architecture if used BigQuery as warehouse.

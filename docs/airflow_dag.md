# Airflow DAG

DAG: dags/telco_pipeline.py

Tasks:
1. csv_to_parquet: Convert Excel to Parquet locally
2. ingest_to_minio: Upload raw parquet to MinIO (s3://datalake/raw/...)
3. spark_transform: Run launcher.py to execute the Spark job
4. train_mlflow: Train a model and log to MLflow registry

Notes:
- Install Airflow separately using constraints (see docs/airflow_install.md) to avoid dependency conflicts.
- Ensure the project path is importable by the Airflow environment.
- Configure MinIO and MLflow via environment variables or .env.
- For production, consider separate environments per task (e.g., Spark on cluster).

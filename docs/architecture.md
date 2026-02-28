# Project Architecture

Storage & Tracking: MinIO (S3-compatible) + MLflow (artifacts in MinIO)
Processing: Pandas utilities (src/etl) + Spark batch job (spark_jobs)
Serving: FastAPI loading model from MLflow model registry
Orchestration: Airflow DAG (dags/telco_pipeline.py)
Testing & CI: pytest + ruff + black + mypy in GitHub Actions

## Data Flow
1. Convert Excel to Parquet locally (scripts/csv_to_parquet.py)
2. Upload raw Parquet to MinIO (scripts/ingest_to_minio.py)
3. Spark reads raw (s3a://) and writes processed features (s3a://)
4. Training reads processed features and logs model to MLflow
5. API loads model from MLflow for predictions

## Code Layout
src/etl: modular, testable transforms
src/models: training scripts (local + MLflow)
src/api: FastAPI app + MLflow loading
dags: Airflow end-to-end pipeline
deploy/api: Dockerfile and minimal requirements for API container
docs: guides for setup, DAG, deployment, config, and data quality
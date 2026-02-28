"""Airflow DAG: Telco end-to-end pipeline with SLA monitoring

Tasks (daily):
- csv_to_parquet: Convert XLSX to parquet (local)
- ingest_to_minio: Upload raw parquet to MinIO
- data_quality: Run Great Expectations checkpoint for validation
- spark_transform: Run Spark job to produce processed features
- train_mlflow: Train model and log to MLflow

Requires Airflow with project dependencies available.
Install: pip install apache-airflow==2.7.3
Or use constraints: https://airflow.apache.org/docs/apache-airflow/stable/installation/installing-from-pypi.html
"""
from src.main import run_pipeline
from src.models.train_mlflow import train as train_mlflow_model
from scripts.ingest_to_minio import upload_to_minio as ingest_to_minio_upload
from scripts.csv_to_parquet import convert as csv_to_parquet_convert
import sys
from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.exceptions import AirflowException

from pathlib import Path

# Resolve project root (assumes DAG file lives under repo/dags)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure Python can import project modules when running from Airflow context
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import callable tasks


def run_data_quality_check(**context):
    """Run Great Expectations checkpoint for raw data validation."""
    import subprocess

    cmd = (
        f"cd {PROJECT_ROOT.as_posix()} && "
        "great_expectations checkpoint run raw_telco_checkpoint"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        raise AirflowException(
            f"Data quality check failed:\n{result.stderr}"
        )

    return result.stdout


def run_etl_with_validation(**context):
    """Run ETL pipeline with schema validation and SLA checks."""
    try:
        df = run_pipeline(
            input_csv=str(PROJECT_ROOT / "data" / "raw" /
                          "Telco_customer_churn.xlsx"),
            out_dir=str(PROJECT_ROOT / "data" / "parquet" / "processed"),
            partition_col="load_date",
            validate=True,  # Enable schema validation
            track_lineage=True,  # Track data lineage
            track_metrics=True,  # Collect metrics and validate SLAs
        )

        context["task_instance"].xcom_push(
            key="processed_rows", value=len(df)
        )
        return f"ETL completed: {len(df)} rows processed"
    except Exception as e:
        raise AirflowException(f"ETL pipeline failed: {e}") from e


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
    "start_date": datetime(2025, 1, 1),
}


with DAG(
    dag_id="telco_end_to_end_pipeline",
    default_args=default_args,
    description="Daily telco CDP pipeline: ingest -> quality -> spark -> train",
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["telco", "cdp", "mlflow", "great-expectations"],
) as dag:

    csv_to_parquet = PythonOperator(
        task_id="csv_to_parquet",
        python_callable=csv_to_parquet_convert,
        provide_context=True,
    )

    ingest_to_minio = PythonOperator(
        task_id="ingest_to_minio",
        python_callable=ingest_to_minio_upload,
        provide_context=True,
    )

    data_quality = PythonOperator(
        task_id="data_quality_checkpoint",
        python_callable=run_data_quality_check,
        provide_context=True,
    )

    spark_transform = BashOperator(
        task_id="spark_transform",
        bash_command=f"cd {PROJECT_ROOT.as_posix()} && python launcher.py",
    )

    etl_with_validation = PythonOperator(
        task_id="etl_with_validation",
        python_callable=run_etl_with_validation,
        provide_context=True,
    )

    train_mlflow = PythonOperator(
        task_id="train_mlflow",
        python_callable=train_mlflow_model,
        provide_context=True,
    )

    # Define task dependencies
    csv_to_parquet >> ingest_to_minio >> data_quality >> spark_transform >> etl_with_validation >> train_mlflow

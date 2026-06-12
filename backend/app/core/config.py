"""Centralized configuration for the CDP project.

Reads from environment variables with safe defaults for local dev.
Attempts to load a .env file if python-dotenv is available.
"""

from __future__ import annotations

import os


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        # dotenv is optional; ignore if not installed
        pass


_load_dotenv_if_available()

# MinIO / S3 / Cloudflare R2 Storage Configuration
STORAGE_MODE: str = os.getenv("STORAGE_MODE", "local")  # 'local' or 's3'
S3_ENDPOINT_URL: str = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY_ID: str = os.getenv("S3_ACCESS_KEY_ID", "admin")
S3_SECRET_ACCESS_KEY: str = os.getenv("S3_SECRET_ACCESS_KEY", "password")
S3_REGION: str = os.getenv("S3_REGION", "auto")
S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "cdp-datalake-assets")

# Legacy S3 Config mapping (for MLflow/Spark fallback)
MLFLOW_S3_ENDPOINT_URL: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", S3_ENDPOINT_URL)
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", S3_ACCESS_KEY_ID)
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", S3_SECRET_ACCESS_KEY)

# MLflow Tracking Configuration
MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_TRACKING_USERNAME: str | None = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD: str | None = os.getenv("MLFLOW_TRACKING_PASSWORD") # DagsHub Token
MODEL_NAME: str = os.getenv("MODEL_NAME", "universal_churn_model")
MODEL_STAGE: str = os.getenv("MODEL_STAGE", "None")
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1")

# Spark paths (example)
RAW_PARQUET_PATH: str = os.getenv(
    "RAW_PARQUET_PATH", "s3://datalake/raw/telco_churn.parquet"
)
PROCESSED_FEATURES_PATH: str = os.getenv(
    "PROCESSED_FEATURES_PATH", "s3://datalake/processed/features"
)

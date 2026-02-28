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

# MinIO / S3
MLFLOW_S3_ENDPOINT_URL: str = os.getenv(
    "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "admin")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "password")

# MLflow
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME: str = os.getenv("MODEL_NAME", "TelcoChurnModel")
MODEL_STAGE: str = os.getenv("MODEL_STAGE", "None")
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1")

# Spark paths (example)
RAW_PARQUET_PATH: str = os.getenv(
    "RAW_PARQUET_PATH", "s3://datalake/raw/telco_churn.parquet")
PROCESSED_FEATURES_PATH: str = os.getenv(
    "PROCESSED_FEATURES_PATH", "s3://datalake/processed/features")

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
MLFLOW_TRACKING_PASSWORD: str | None = os.getenv(
    "MLFLOW_TRACKING_PASSWORD"
)  # DagsHub Token

# Database Configuration (Supabase / PostgreSQL)
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")

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

# Target Detection Entropy Scoring Configuration (Phase 1 Continuous Scoring)
ENTROPY_LIMIT_LOW: float = float(os.getenv("ENTROPY_LIMIT_LOW", "0.5"))
ENTROPY_LIMIT_MED: float = float(os.getenv("ENTROPY_LIMIT_MED", "0.75"))
ENTROPY_LIMIT_HIGH: float = float(os.getenv("ENTROPY_LIMIT_HIGH", "0.95"))

ENTROPY_SCORE_LOW: float = float(os.getenv("ENTROPY_SCORE_LOW", "0.2"))
ENTROPY_SCORE_MED: float = float(os.getenv("ENTROPY_SCORE_MED", "0.6"))
ENTROPY_SCORE_HIGH: float = float(os.getenv("ENTROPY_SCORE_HIGH", "0.8"))

# Composite Target Synthesis Configuration
CPI_MIN_COLUMNS: int = int(os.getenv("CPI_MIN_COLUMNS", "2"))
CPI_VARIANCE_THRESHOLD: float = float(os.getenv("CPI_VARIANCE_THRESHOLD", "0.8"))
CPI_AUTO_THRESHOLD: int = int(os.getenv("CPI_AUTO_THRESHOLD", "2"))
COMPOSITE_SYNTHESIS_ENABLED: bool = (
    os.getenv("COMPOSITE_SYNTHESIS_ENABLED", "True").lower() == "true"
)

# App & Environment settings
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "local")
DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
API_KEY: str = os.getenv("API_KEY", "test-api-key")
IS_PRODUCTION: bool = os.getenv("ENV", "development") == "production"
ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")

# JWT Authentication settings
SECRET_KEY: str = os.getenv(
    "SECRET_KEY",
    "dev-secret-key-minimum-32-characters!!"
)
JWT_ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)
REFRESH_TOKEN_EXPIRE_DAYS: int = int(
    os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7")
)

# Drift & Auto-Retraining settings
DRIFT_AUTO_RETRAIN: bool = os.getenv("DRIFT_AUTO_RETRAIN", "False").lower() == "true"
DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", "0.2"))
DRIFT_CHECK_INTERVAL_SEC: int = int(os.getenv("DRIFT_CHECK_INTERVAL_SEC", "3600"))

# Database Pool Settings
DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "20"))
DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))
DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "15"))

# Training Configuration
MAX_TRAINING_MINUTES: int = int(os.getenv("MAX_TRAINING_MINUTES", "30"))

ENABLE_DRIFT_SCHEDULER: bool = (
    os.getenv("ENABLE_DRIFT_SCHEDULER", "true").lower() == "true"
)

import sys

settings = sys.modules[__name__]

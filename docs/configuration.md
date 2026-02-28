# Configuration

Centralize environment settings in src/config.py, which reads from environment variables and optionally loads a .env if python-dotenv is installed.

## Keys
- MLFLOW_S3_ENDPOINT_URL
- AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
- MLFLOW_TRACKING_URI
- MODEL_NAME / MODEL_VERSION / MODEL_STAGE
- RAW_PARQUET_PATH / PROCESSED_FEATURES_PATH

## .env
Use .env.example as a template and create .env in project root.

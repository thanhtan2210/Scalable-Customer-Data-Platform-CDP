# Getting Started

## Prerequisites
- Python 3.10
- Java (for Spark)
- Docker (for MinIO + MLflow) optional

## Setup
1. Create and activate a virtual environment (Windows PowerShell):
   - python -m venv .venv
   - & .\.venv\Scripts\Activate.ps1
2. Install dependencies: pip install -r requirements.txt
3. Copy .env.example to .env and adjust values.
4. Start infra (optional): docker compose up -d

## Run Pipeline (Local)
- python scripts/csv_to_parquet.py
- python scripts/ingest_to_minio.py
- python launcher.py  (runs Spark transform)
- python -m src.models.train_mlflow  (or import and call train())
- uvicorn src.api.main:app --reload

## Tests
- & .\.venv\Scripts\Activate.ps1
- python -m pytest -q
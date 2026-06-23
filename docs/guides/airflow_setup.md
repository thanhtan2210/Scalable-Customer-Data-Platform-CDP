# Apache Airflow Setup & DAGs

## 1. Installing Apache Airflow
Airflow has a large dependency graph. To avoid resolver conflicts, install it separately using the official constraints file, not from the project's main requirements.txt.

### Python 3.10 example
Replace `<VERSION>` with the Airflow version you want (e.g., `2.10.3`).

```powershell
# Inside your venv
pip install "apache-airflow==<VERSION>" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-<VERSION>/constraints-3.10.txt"
```

This pins all providers and transitive dependencies to versions verified by the Airflow project (e.g., packaging, colorlog, provider bundles).

### Notes
- If you already installed Airflow without constraints, uninstall first:
  ```powershell
  pip uninstall -y apache-airflow apache-airflow-core apache-airflow-providers-* colorlog packaging
  ```
- Then re-install using the constraint command above.
- For other Python versions, change the constraints file suffix accordingly (e.g., `constraints-3.11.txt`).
- To run the DAG added in this repo, Airflow only needs to have access to the project folder on its PYTHONPATH and the project dependencies.

---

## 2. Airflow DAG Configuration

**DAG File**: `dags/telco_pipeline.py`

### DAG Tasks Flow:
1. `csv_to_parquet`: Convert Excel to Parquet locally
2. `ingest_to_minio`: Upload raw parquet to MinIO (`s3://datalake/raw/...`)
3. `spark_transform`: Run `launcher.py` to execute the Spark job
4. `train_mlflow`: Train a model and log to MLflow registry

### Operating Notes:
- Ensure the project path is importable by the Airflow environment.
- Configure MinIO and MLflow via environment variables or `.env`.
- For production, consider separate environments per task (e.g., Spark on cluster).

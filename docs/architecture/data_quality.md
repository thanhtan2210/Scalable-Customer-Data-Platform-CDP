# Data Quality & Lineage

## Data Quality
- Great Expectations project scaffolded under `great_expectations/`.
- Suite: `raw_telco_suite` checks basic row count threshold on raw data.
- Checkpoint: `raw_telco_checkpoint` reads `s3://datalake/raw/telco_churn.parquet` via pandas with S3 credentials.
- Keep lightweight Pandera checks for fast unit tests.

## Lineage (Proposal)
- Use OpenLineage with Airflow to emit lineage for each task (ingest, transform, train).

## How to Run Manually
```powershell
& ".\.venv\Scripts\Activate.ps1"
Push-Location "."
great_expectations checkpoint run raw_telco_checkpoint
```

## Next Steps
- Add richer expectations (schema, non-null, ranges) to `great_expectations/expectations/raw_telco_suite.json`.
- Add a processed-features suite and checkpoint once Spark output is stable.
- Wire OpenLineage in Airflow via configuration and emit runs to Marquez.

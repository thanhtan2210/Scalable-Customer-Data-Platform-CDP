# Dashboard (Optional)

Add a simple Streamlit app to visualize churn KPIs and filter cohorts:
- Read from data/processed/ or directly from MinIO via s3fs
- Show churn rate over tenure_bin, monthly_bin
- Provide slicers for partner/dependents/senior citizen

## Streamlit App
- File: Sales_Dashboard/streamlit_app.py
- Run using the project's virtual environment:
```powershell
& ".\.venv\Scripts\Activate.ps1"
Push-Location "."
streamlit run Sales_Dashboard/streamlit_app.py
```

## Notes
- The app reads from `PROCESSED_FEATURES_PATH` (default `s3://datalake/processed/features`).
- Configure MinIO via environment variables (see docs/configuration.md) or `.env`.
- If reading from MinIO fails, the app falls back to local data `data/parquet/raw/telco_churn.parquet` if present.

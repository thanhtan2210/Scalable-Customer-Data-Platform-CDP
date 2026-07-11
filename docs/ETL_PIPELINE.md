# Comprehensive ETL Data Pipeline Documentation

This document describes the design, implementation, and features of the production-ready Pandas-based ETL data pipeline.

## ⚙️ Core Modules

The ETL pipeline consists of several key modules situated under `backend/app/core/etl/`:

1. **Cleaning Engine (`cleaning.py`)**: Responsible for data cleaning steps:
   - нормализация/resolve column aliases (e.g., matching alternate formats like `Total Charges` and `TotalCharges`).
   - Type conversions: coercing string columns to float/numeric and handling invalid values.
   - Dropping invalid rows (e.g., removing records with null values in core features).
   - Boolean mapping: standardizing categorical answers (Yes/No) to binary representation (1/0).
   - Feature creation (e.g., binning tenure and monthly charges).
   - Writing outputs as compressed Parquet files with optional date partitioning.

2. **Schema Enforcement (`schema.py`)**: Restricts incoming and processed formats using **Pandera DataFrameSchema**:
   - `RawTelcoSchema`: strict type check and check restrictions (e.g., verifying enums on Gender and Partner).
   - `ProcessedFeatureSchema`: post-transformation check to guarantee compliance with model specifications.

3. **Data Lineage Tracking (`lineage.py`)**: Persists transformation metadata across execution stages into `data/lineage.jsonl`:
   - Records timestamp, source name, target name, rows before/after, row loss percentage, and feature metadata.

4. **Observability & SLA Validation (`observability.py`)**:
   - Collects quality metrics (null rates, duplicates, row/column counts, latency).
   - Validates metrics against custom SLAs (e.g., minimum expected rows, maximum allowed null rates, duplicates).

5. **Incremental Processing (`incremental.py`)**:
   - Implements high-watermark dates based on last successful executions.
   - Restricts data reading using datetime thresholds to support efficient delta loads.

---

## 📈 Quality SLAs

The default validation rules are defined as follows:

| Stage | Metric Constraint | Limit |
|---|---|---|
| **Raw Telco Data** | Minimum Row Count | `≥ 6000` |
| | Max Duplicate Count | `≤ 10` |
| | Max Null Percentage | `TotalCharges ≤ 10%` |
| **Processed Features** | Minimum Row Count | `≥ 5500` |
| | Max Duplicate Count | `0` (Strictly unique) |
| | Max Null Percentage | `Churn Reason ≤ 80%` (Expected) <br> `Other columns ≤ 5%` |

---

## 💾 Lineage Logging Format (Example)

Lineage entries are appended to `data/lineage.jsonl` as structured JSON strings:

```json
{
  "timestamp": "2026-02-28T18:16:19.855340",
  "source": "raw_telco",
  "target": "converted",
  "transformation": "convert_types",
  "schema_before": {"CustomerID": "object", "MonthlyCharges": "float64", "TotalCharges": "float64"},
  "schema_after": {"CustomerID": "object", "MonthlyCharges": "float64", "TotalCharges": "float64"},
  "rows_before": 7043,
  "rows_after": 7043,
  "row_loss_pct": 0.0,
  "metadata": {}
}
```

---

## 🚀 Execution CLI

Trigger the ETL pipeline via the unified project launcher:

```powershell
python launcher.py pipeline
```

Expected output:
```text
⚙️ Running Entire Data Pipeline (Comprehensive Pandas ETL)...
📥 Loading input raw data: data/raw/cleaned_telco.csv
💾 Saving processed parquet to: data/parquet/processed
✅ Pipeline executed successfully. Processed 7043 rows.
```

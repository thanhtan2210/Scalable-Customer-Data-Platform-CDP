# Data Engineering Gap Analysis & Recommendations

**Date:** January 28, 2026  
**Focus:** What's missing for a production-grade Data Engineering platform

---

## Current DE State

### ✅ What's Implemented
| Component                  | Status | Details                                                         |
| -------------------------- | ------ | --------------------------------------------------------------- |
| **CSV → Parquet Ingest**   | ✅      | `scripts/csv_to_parquet.py` reads Excel, converts types         |
| **Basic ETL Cleaning**     | ✅      | `src/etl/cleaning.py` with testable functions                   |
| **Spark Job**              | ✅      | `spark_jobs/clean_data_spark.py` normalizes schema, casts types |
| **Pandas Pipeline**        | ✅      | `src/main.py` orchestrator chains functions                     |
| **Data Quality Framework** | ✅      | Great Expectations scaffolded with sample checkpoint            |
| **Parquet Storage**        | ✅      | Local + MinIO (S3-compatible) support                           |
| **Partition Strategy**     | ✅      | Year/month/day partition layout designed                        |
| **Airflow DAG**            | ✅      | `dags/telco_pipeline.py` with 5-task pipeline                   |
| **Testing**                | ✅      | `tests/test_cleaning.py` validates ETL functions                |
| **CI/CD**                  | ✅      | GitHub Actions runs tests + Spark job on every push             |

---

## 🔴 Critical Gaps (Must-Have for Production)

### 1. **Data Lineage & Metadata Tracking**
**Why:** Without tracking where data came from, transformations applied, and schema evolution, debugging data issues becomes nearly impossible in production.

**Current State:** None

**What's Missing:**
- No data lineage tool (OpenMetadata, Collibra, Apache Atlas)
- No metadata store (catalog of datasets, schemas, ownership)
- No column-level lineage (which column came from which transformation)
- No versioning of transformations or SQL logic

**Recommendation:**
```python
# Add a simple metadata registry to track data transformations
# File: src/etl/lineage.py

import json
from datetime import datetime
from pathlib import Path

class DataLineageRegistry:
    """Simple in-memory lineage tracker for local pipelines."""
    
    def __init__(self, registry_path: str = "data/lineage.json"):
        self.registry_path = Path(registry_path)
        self.lineage = self._load_registry()
    
    def log_transformation(self, 
        source_table: str,
        target_table: str, 
        transformation: str,
        schema_before: dict,
        schema_after: dict,
        row_count_before: int,
        row_count_after: int
    ):
        """Log a transformation step."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": source_table,
            "target": target_table,
            "transformation": transformation,
            "schema_before": schema_before,
            "schema_after": schema_after,
            "rows_before": row_count_before,
            "rows_after": row_count_after,
        }
        self.lineage.append(entry)
        self._save_registry()
        return entry
    
    def get_lineage_for_column(self, table: str, column: str):
        """Trace a column back to its source."""
        chain = []
        for entry in reversed(self.lineage):
            if entry["target"] == table and column in entry["schema_after"]:
                chain.append(entry)
                table = entry["source"]
        return list(reversed(chain))
```

**Next Step:** Integrate into `src/main.py` orchestrator to log each transformation.

---

### 2. **Incremental Data Processing (Delta/Merge Logic)**
**Why:** Daily production pipelines can't afford to reprocess all 7K+ customers every day. Incremental processing saves compute and storage.

**Current State:** All-or-nothing batch (reprocess entire dataset)

**What's Missing:**
- No change data capture (CDC) to detect new/modified records
- No incremental merge logic (UPDATE when customer already exists)
- No partition pruning (only read today's data, not all history)
- No watermarking (track progress of incremental runs)

**Recommendation:**
```python
# File: src/etl/incremental.py

import pandas as pd
from datetime import datetime

def load_incremental(
    raw_path: str,
    processed_path: str,
    partition_col: str = "load_date",
    lookback_days: int = 7
) -> tuple[pd.DataFrame, datetime]:
    """Load only new/changed data since last run.
    
    Strategy:
    1. Find max load_date in processed table (high watermark)
    2. Read raw data only from past 7 days
    3. Merge with processed table, updating existing customer records
    """
    # Get watermark (last successful load)
    try:
        existing = pd.read_parquet(processed_path)
        max_date = pd.to_datetime(existing[partition_col]).max()
    except:
        max_date = pd.Timestamp("2000-01-01")
        existing = None
    
    # Load only recent data
    raw = pd.read_parquet(raw_path)
    cutoff = max_date - pd.Timedelta(days=lookback_days)
    raw_new = raw[pd.to_datetime(raw[partition_col]) >= cutoff].copy()
    
    if existing is not None:
        # Merge: existing + new (dedup by CustomerID, keep latest)
        merged = pd.concat([existing, raw_new], ignore_index=True)
        merged = merged.sort_values(partition_col).drop_duplicates(
            subset=["CustomerID"], keep="last"
        )
        return merged, max_date
    else:
        return raw_new, datetime.utcnow()

def get_changed_records(
    before: pd.DataFrame,
    after: pd.DataFrame,
    key: str = "CustomerID"
) -> pd.DataFrame:
    """Identify which records changed between two versions."""
    before_set = set(before[key])
    after_set = set(after[key])
    
    new_keys = after_set - before_set
    changed = after[after[key].isin(new_keys | before_set)].copy()
    changed["_change_type"] = changed[key].apply(
        lambda x: "NEW" if x not in before_set else "UPDATED"
    )
    return changed
```

**Next Step:** Update `src/main.py` to support incremental flag: `run_pipeline(..., mode="incremental")`

---

### 3. **Data Validation & Schema Enforcement**
**Why:** Without strict validation, bad data propagates downstream (garbage in = garbage out).

**Current State:** Basic type conversion, no schema enforcement

**What's Missing:**
- No Pydantic/Pandera models defining required fields
- No constraint validation (e.g., "tenure must be ≥ 0")
- No enum enforcement (e.g., "Contract must be in [Month-to-month, One year, Two year]")
- No custom validators (e.g., "if Senior Citizen = Yes, age must be ≥ 65")

**Recommendation:**
```python
# File: src/etl/schema.py

from pandera import Column, DataFrameSchema, Check, Index
import pandera as pa

# Define the canonical schema for raw data
RawTelcoSchema = DataFrameSchema({
    "CustomerID": Column(str, unique=True, nullable=False),
    "Gender": Column(str, isin=["Male", "Female"], nullable=False),
    "SeniorCitizen": Column(int, isin=[0, 1], nullable=False),
    "Partner": Column(str, isin=["Yes", "No"], nullable=False),
    "Dependents": Column(str, isin=["Yes", "No"], nullable=False),
    "Tenure Months": Column(int, Check.greater_than_or_equal_to(0), nullable=False),
    "PhoneService": Column(str, isin=["Yes", "No"], nullable=False),
    "MonthlyCharges": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
    "TotalCharges": Column(float, Check.greater_than_or_equal_to(0), nullable=True),
    "Churn": Column(int, isin=[0, 1], nullable=False),
}, index=Index(str, name="CustomerID"))

# Define the schema for processed features
ProcessedFeatureSchema = DataFrameSchema({
    "CustomerID": Column(str, unique=True, nullable=False),
    "tenure_bin": Column(str, isin=["0-6", "7-12", "13-24", "25-48", "49+"], nullable=True),
    "monthly_bin": Column(pa.Category, nullable=True),
    "Churn": Column(int, isin=[0, 1], nullable=False),
    # ... other processed columns
})

def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw data against schema."""
    try:
        return RawTelcoSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Schema validation failed: {e}")

def validate_processed(df: pd.DataFrame) -> pd.DataFrame:
    """Validate processed data against schema."""
    try:
        return ProcessedFeatureSchema.validate(df)
    except pa.errors.SchemaError as e:
        raise ValueError(f"Processed schema validation failed: {e}")
```

**Next Step:** Call `validate_raw()` in `csv_to_parquet.py`, `validate_processed()` in Spark job.

---

### 4. **Data Observability & Monitoring (SLAs)**
**Why:** You can't fix what you don't measure. Need alerts when data quality degrades or pipelines fail.

**Current State:** Basic logging, no metrics or SLAs

**What's Missing:**
- No data quality metrics (row counts, null %, duplicates %)
- No data drift detection (unexpected value distributions)
- No pipeline SLAs (must complete by X time, must have Y rows)
- No alerting (Slack, email when violations occur)

**Recommendation:**
```python
# File: src/etl/observability.py

import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DataQualityMetrics:
    """Track metrics for each pipeline run."""
    timestamp: datetime
    table_name: str
    row_count: int
    null_counts: dict  # column -> null count
    duplicate_count: int
    schema_columns: list
    processing_time_sec: float
    status: str  # "success" or "failed"
    

class MetricsCollector:
    def __init__(self, output_file: str = "data/metrics.jsonl"):
        self.output_file = output_file
        self.logger = logging.getLogger(__name__)
    
    def collect(self, df: pd.DataFrame, table_name: str, elapsed_time: float) -> DataQualityMetrics:
        """Collect quality metrics from a DataFrame."""
        metrics = DataQualityMetrics(
            timestamp=datetime.utcnow(),
            table_name=table_name,
            row_count=len(df),
            null_counts={col: df[col].isnull().sum() for col in df.columns},
            duplicate_count=len(df[df.duplicated(subset=["CustomerID"])]),
            schema_columns=list(df.columns),
            processing_time_sec=elapsed_time,
            status="success"
        )
        self._log_metrics(metrics)
        return metrics
    
    def _log_metrics(self, metrics: DataQualityMetrics):
        """Write metrics to JSONL (queryable format)."""
        import json
        with open(self.output_file, "a") as f:
            f.write(json.dumps({
                "timestamp": metrics.timestamp.isoformat(),
                "table": metrics.table_name,
                "row_count": metrics.row_count,
                "null_pct": {k: v/metrics.row_count for k, v in metrics.null_counts.items()},
                "duplicates": metrics.duplicate_count,
                "elapsed_sec": metrics.processing_time_sec,
            }) + "\n")
        
        # Log summaries
        self.logger.info(
            f"{metrics.table_name}: {metrics.row_count} rows, "
            f"nulls: {max(metrics.null_counts.values()) if metrics.null_counts else 0}, "
            f"time: {metrics.processing_time_sec:.2f}s"
        )

class SLAValidator:
    """Check if pipeline meets SLAs."""
    
    def __init__(self, slas: dict):
        # slas = {
        #     "raw_telco": {"min_rows": 7000, "max_nulls_pct": 0.05},
        #     "processed_features": {"min_rows": 6500, "max_duplicates": 0}
        # }
        self.slas = slas
    
    def validate(self, metrics: DataQualityMetrics) -> list[str]:
        """Return list of SLA violations."""
        violations = []
        sla = self.slas.get(metrics.table_name, {})
        
        if metrics.row_count < sla.get("min_rows", 0):
            violations.append(
                f"Row count {metrics.row_count} below minimum {sla['min_rows']}"
            )
        
        max_nulls_pct = sla.get("max_nulls_pct", 1.0)
        for col, null_count in metrics.null_counts.items():
            null_pct = null_count / metrics.row_count
            if null_pct > max_nulls_pct:
                violations.append(
                    f"Column {col} null pct {null_pct:.1%} exceeds {max_nulls_pct:.1%}"
                )
        
        return violations
```

**Next Step:** Wire into Airflow DAG tasks to fail pipeline if SLAs violated.

---

### 5. **Partitioning Strategy Validation**
**Why:** Poor partitioning causes slow queries. Need to verify partition scheme is being applied correctly.

**Current State:** Partition layout defined but not validated

**What's Missing:**
- No validation that files are written with partition columns
- No monitoring of partition skew (some partitions have 1000x more data)
- No pruning verification (queries actually skip partitions)

**Recommendation:**
```python
# File: src/etl/partitioning.py

import pandas as pd
import os
from pathlib import Path

def validate_partition_layout(
    parquet_dir: str,
    expected_partitions: list[str] = ["year", "month", "day"]
) -> dict:
    """Verify partition directory structure is correct.
    
    Expected structure:
    parquet_dir/
      year=2025/
        month=01/
          day=28/
            data.parquet
    """
    base = Path(parquet_dir)
    partition_structure = {}
    
    for partition in expected_partitions:
        paths = list(base.glob(f"{partition}=*"))
        partition_structure[partition] = {
            "count": len(paths),
            "values": sorted([p.name.split("=")[1] for p in paths])
        }
    
    return partition_structure

def detect_partition_skew(parquet_dir: str) -> dict:
    """Find partitions with significantly more/fewer rows."""
    from pathlib import Path
    import pandas as pd
    
    partitions = {}
    
    # Iterate through all partition dirs
    for year_dir in Path(parquet_dir).glob("year=*"):
        for month_dir in year_dir.glob("month=*"):
            for day_dir in month_dir.glob("day=*"):
                # Count rows in this partition
                parquet_files = list(day_dir.glob("*.parquet"))
                if not parquet_files:
                    continue
                
                try:
                    df = pd.read_parquet(day_dir)
                    partition_key = f"{year_dir.name}/{month_dir.name}/{day_dir.name}"
                    partitions[partition_key] = len(df)
                except:
                    pass
    
    # Detect skew (e.g., 1000x difference)
    if not partitions:
        return {}
    
    min_rows = min(partitions.values())
    max_rows = max(partitions.values())
    skew_ratio = max_rows / (min_rows + 1)
    
    skewed = {
        k: v for k, v in partitions.items()
        if v > max_rows * 0.8 or v < min_rows * 1.2
    }
    
    return {
        "skew_ratio": skew_ratio,
        "min_partition": min_rows,
        "max_partition": max_rows,
        "problematic_partitions": skewed
    }
```

**Next Step:** Add to Great Expectations checkpoint to warn on partition skew.

---

### 6. **Airflow Production Deployment**
**Why:** DAG is scaffolded but Airflow isn't installable with current constraints.

**Current State:** Airflow DAG exists but dependencies conflict

**What's Missing:**
- Airflow not in `requirements.txt` (requires separate `constraints.txt`)
- No Airflow Helm chart or Docker image for orchestration
- No Airflow worker pool configuration
- No SLA/alerting integration
- No retry logic for transient failures

**Recommendation:**
```bash
# Create separate Airflow requirements
# File: airflow-requirements.txt

# Pin versions to avoid conflicts with main requirements
apache-airflow==2.7.3
apache-airflow-providers-apache-spark==4.1.2
apache-airflow-providers-amazon==8.0.1
apache-airflow-providers-http==4.4.3
# Follow Airflow constraints
```

**Next Step:** Add Docker Compose service for Airflow (Webserver + Scheduler + Executor).

---

### 7. **Feature Store or Columnar Format Upgrade**
**Why:** Parquet works but lacks ACID, schema evolution, and time-travel that production systems need.

**Current State:** Parquet on MinIO (great baseline, but limited)

**What's Missing:**
- No ACID transactions (data can be partially written if crash occurs)
- No schema evolution without rewriting all files
- No time-travel (can't query historical versions)
- No efficient schema updates (DELETEs require full rewrite)

**Recommendation (Low Priority):**
```python
# Migrate from Parquet to Apache Iceberg
# Benefits: ACID, time-travel, schema evolution, rollback

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TelcoIceberg") \
    .config("spark.jars.packages", "org.apache.iceberg:iceberg-spark-runtime-3.4_2.12:1.4.0") \
    .config("spark.sql.catalog.iceberg", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.iceberg.warehouse", "s3a://datalake/iceberg") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .getOrCreate()

# Write as Iceberg
df.write \
    .mode("overwrite") \
    .format("iceberg") \
    .partitionedBy("year", "month", "day") \
    .option("write-format", "parquet") \
    .save("s3a://datalake/processed/features_iceberg")

# Time-travel query (go back to specific commit)
spark.read \
    .format("iceberg") \
    .option("as-of-timestamp", "2025-01-01T00:00:00Z") \
    .load("s3a://datalake/processed/features_iceberg")
```

---

## 🟡 Medium Priority Gaps

### 8. **Data Discovery & Cataloging**
**Missing:** No central place to browse available datasets, schemas, lineage
**Recommendation:** Integrate OpenMetadata or use Glue Data Catalog

### 9. **Cost Optimization**
**Missing:** No monitoring of storage usage, compute cost per job
**Recommendation:** Add S3 metrics, Spark job profiling

### 10. **Testing for ETL**
**Current:** Basic unit tests for cleaning functions  
**Missing:** Integration tests (test with real MinIO), parameterized tests for edge cases
**Recommendation:** Add pytest fixtures for MinIO, test with various schema variations

---

## 🟢 Quick Wins (< 1 day each)

| Gap                   | Impact | Effort | Recommendation                                         |
| --------------------- | ------ | ------ | ------------------------------------------------------ |
| Data lineage tracking | Medium | Low    | Add `src/etl/lineage.py` + log in `main.py`            |
| Metrics collection    | Medium | Low    | Add `src/etl/observability.py` + wire to pipeline      |
| Schema validation     | High   | Low    | Add Pandera models in `src/etl/schema.py`              |
| Partition validation  | Low    | Low    | Add `src/etl/partitioning.py` validation               |
| Integration tests     | Medium | Medium | Add `tests/test_etl_integration.py` with MinIO fixture |

---

## 🔴 Strategic Priorities (What to do first)

### **Phase 1 (Week 1):** Foundation
1. ✅ Add Pandera schema validation (`src/etl/schema.py`)
2. ✅ Add metrics collection (`src/etl/observability.py`)
3. ✅ Add data lineage tracking (`src/etl/lineage.py`)
4. ✅ Update `src/main.py` to use all three
5. ✅ Add integration tests for MinIO

### **Phase 2 (Week 2-3):** Incremental Processing
1. Implement incremental processing (`src/etl/incremental.py`)
2. Update `dags/telco_pipeline.py` to use incremental mode
3. Add SLA monitoring to Airflow

### **Phase 3 (Month 2):** Advanced Features
1. Migrate from Parquet to Apache Iceberg (optional but powerful)
2. Deploy Airflow to production (Kubernetes or Managed Service)
3. Integrate data cataloging (OpenMetadata or Glue)

---

## Summary

Your CDP has a **solid foundation for DE**. To reach **enterprise-grade**, prioritize:

1. **Schema Validation** (Pandera) — catch data issues early ✅ HIGH IMPACT, LOW EFFORT
2. **Metrics & Observability** — know what's broken ✅ HIGH IMPACT, LOW EFFORT  
3. **Data Lineage** — understand data flow ✅ MEDIUM IMPACT, LOW EFFORT
4. **Incremental Processing** — scale to 50M efficiently ✅ HIGH IMPACT, MEDIUM EFFORT
5. **Airflow Production** — schedule + orchestrate reliably ⚠️ CRITICAL FOR PROD

Focus on **Phase 1 first** — can be done in 2-3 days and dramatically improves robustness.


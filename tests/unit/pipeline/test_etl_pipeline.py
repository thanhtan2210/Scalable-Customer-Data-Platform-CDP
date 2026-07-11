import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import pytest
import pandera as pa

from backend.app.core.etl.schema import validate_raw, validate_processed
from backend.app.core.etl.lineage import DataLineageRegistry
from backend.app.core.etl.observability import MetricsCollector, SLAValidator, DataQualityMetrics
from backend.app.core.etl.incremental import IncrementalProcessor

def test_validate_raw_success():
    df = pd.DataFrame({
        "CustomerID": ["C1", "C2"],
        "Gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [12, 24],
        "PhoneService": ["Yes", "No"],
        "InternetService": ["DSL", "Fiber Optic"],
        "MonthlyCharges": [50.5, 75.0],
        "TotalCharges": [606.0, 1800.0],
        "Churn": [0, 1]
    })
    validated = validate_raw(df)
    assert not validated.empty

def test_validate_raw_missing_column_raises_error():
    df = pd.DataFrame({
        "CustomerID": ["C1"],
        "Gender": ["Male"]
    })
    with pytest.raises(ValueError, match="Raw schema validation failed"):
        validate_raw(df)

def test_validate_processed_success():
    df = pd.DataFrame({
        "CustomerID": ["C1", "C2"],
        "Gender": [1, 0],
        "SeniorCitizen": [0, 1],
        "Partner": [1, 0],
        "Dependents": [0, 1],
        "tenure": [12, 24],
        "MonthlyCharges": [50.5, 75.0],
        "TotalCharges": [606.0, 1800.0],
        "Churn": [0, 1]
    })
    validated = validate_processed(df)
    assert not validated.empty

def test_lineage_registry(tmp_path):
    registry_file = tmp_path / "lineage.jsonl"
    registry = DataLineageRegistry(registry_path=str(registry_file))
    
    registry.log_transformation(
        source_table="raw_telco",
        target_table="converted",
        transformation="convert_types",
        schema_before={"col1": "object"},
        schema_after={"col1": "float64"},
        row_count_before=100,
        row_count_after=98,
        metadata={"info": "test"}
    )
    
    assert registry_file.exists()
    
    # Reload registry
    new_registry = DataLineageRegistry(registry_path=str(registry_file))
    assert len(new_registry.lineage) == 1
    assert new_registry.lineage[0]["transformation"] == "convert_types"
    assert new_registry.lineage[0]["row_loss_pct"] == 2.0

def test_metrics_collector_and_sla_validator(tmp_path):
    metrics_file = tmp_path / "metrics.jsonl"
    collector = MetricsCollector(metrics_file=str(metrics_file))
    
    df = pd.DataFrame({
        "CustomerID": ["C1", "C2", "C1"], # 1 duplicate CustomerID
        "MonthlyCharges": [50.0, 75.0, 50.0],
        "TotalCharges": [100.0, None, 100.0] # 1 null
    })
    
    metrics = collector.collect(df, table_name="test_table", elapsed_time=1.5)
    assert metrics.row_count == 3
    assert metrics.duplicate_count == 1
    assert metrics.null_counts["TotalCharges"] == 1
    
    # SLA Validator
    slas = {
        "test_table": {
            "min_rows": 5,
            "max_nulls_pct": {"TotalCharges": 0.10},
            "max_duplicates": 0
        }
    }
    validator = SLAValidator(slas=slas)
    violations = validator.validate(metrics)
    
    assert len(violations) == 3
    assert any("Row count" in v for v in violations)
    assert any("TotalCharges null pct" in v for v in violations)
    assert any("Duplicate count" in v for v in violations)

def test_incremental_processor(tmp_path):
    watermark_dir = tmp_path / "watermarks"
    processor = IncrementalProcessor(watermark_dir=str(watermark_dir))
    
    # No watermark initially
    assert processor.get_watermark("table1") is None
    
    # Set watermark
    now = datetime(2026, 7, 10, 12, 0, 0)
    processor.set_watermark("table1", now)
    assert processor.get_watermark("table1") == now

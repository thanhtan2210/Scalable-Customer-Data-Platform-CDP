"""Tests for critical gap implementations."""

import json
import pandas as pd
import pytest

from backend.app.core.etl.schema import (
    validate_raw,
    validate_processed,
    get_schema_report,
    RawTelcoSchema,
)
from backend.app.core.etl.lineage import DataLineageRegistry
from backend.app.core.etl.incremental import IncrementalProcessor
from backend.app.core.etl.observability import MetricsCollector, SLAValidator
from backend.app.core.etl.partitioning import (
    validate_partition_layout,
    detect_partition_skew,
    validate_partition_completeness,
    get_partition_statistics,
)


class TestSchemaValidation:
    """Test Pandera schema validation."""

    def test_validate_raw_success(self, tmp_df):
        """Valid raw data should pass validation."""
        # tmp_df is a valid Telco DataFrame
        result = validate_raw(tmp_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_validate_raw_missing_column(self, tmp_df):
        """Missing required column should raise error."""
        df = tmp_df.drop(columns=["CustomerID"])
        with pytest.raises(ValueError, match="schema validation failed"):
            validate_raw(df)

    def test_validate_processed_success(self, tmp_df):
        """Valid processed data should pass validation."""
        # Simulate processed data
        df = tmp_df.copy()
        df["tenure"] = df.get("Tenure Months", df.get("tenure", 0))
        # Convert categorical raw values to processed numeric flags
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
        df["Partner"] = df["Partner"].map({"Yes": 1, "No": 0})
        df["Dependents"] = df["Dependents"].map({"Yes": 1, "No": 0})
        result = validate_processed(df)
        assert isinstance(result, pd.DataFrame)

    def test_get_schema_report(self, tmp_df):
        """Schema report should identify valid and invalid columns."""
        report = get_schema_report(tmp_df, RawTelcoSchema)
        assert "valid" in report
        assert "errors" in report
        assert "column_stats" in report


class TestDataLineage:
    """Test data lineage tracking."""

    def test_lineage_registry_init(self, tmp_path):
        """Registry should initialize and create file."""
        registry_path = tmp_path / "lineage.jsonl"
        registry = DataLineageRegistry(str(registry_path))
        assert isinstance(registry, DataLineageRegistry)

    def test_log_transformation(self, tmp_path):
        """Should log transformations to file."""
        registry = DataLineageRegistry(str(tmp_path / "lineage.jsonl"))

        entry = registry.log_transformation(
            source_table="raw",
            target_table="processed",
            transformation="filter_nulls",
            schema_before={"id": "int64", "value": "float64"},
            schema_after={"id": "int64", "value": "float64"},
            row_count_before=100,
            row_count_after=95,
        )

        assert entry["source"] == "raw"
        assert entry["target"] == "processed"
        assert entry["rows_before"] == 100
        assert entry["rows_after"] == 95

    def test_lineage_persistence(self, tmp_path):
        """Lineage should persist to disk."""
        registry_path = tmp_path / "lineage.jsonl"
        registry = DataLineageRegistry(str(registry_path))

        registry.log_transformation(
            source_table="raw",
            target_table="processed",
            transformation="test",
            schema_before={},
            schema_after={},
            row_count_before=10,
            row_count_after=10,
        )

        # Read file and verify
        lines = registry_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["transformation"] == "test"

    def test_get_lineage_chain(self, tmp_path):
        """Should trace table lineage back to source."""
        registry = DataLineageRegistry(str(tmp_path / "lineage.jsonl"))

        registry.log_transformation("raw", "step1", "tx1", {}, {}, 100, 95)
        registry.log_transformation("step1", "step2", "tx2", {}, {}, 95, 90)
        registry.log_transformation("step2", "final", "tx3", {}, {}, 90, 90)

        chain = registry.get_lineage_chain("final")
        assert len(chain) == 3
        assert chain[0]["source"] == "raw"
        assert chain[-1]["target"] == "final"


class TestIncrementalProcessing:
    """Test incremental processing."""

    def test_incremental_init(self, tmp_path):
        """Processor should initialize."""
        processor = IncrementalProcessor(str(tmp_path / "watermarks"))
        assert isinstance(processor, IncrementalProcessor)

    def test_watermark_get_set(self, tmp_path):
        """Should persist and retrieve watermarks."""
        from datetime import datetime

        processor = IncrementalProcessor(str(tmp_path / "watermarks"))

        ts = datetime(2025, 1, 28, 12, 0, 0)
        processor.set_watermark("test_table", ts)

        retrieved = processor.get_watermark("test_table")
        assert retrieved is not None
        assert retrieved.year == 2025

    def test_get_changed_records(self, tmp_path):
        """Should identify new/updated records."""
        processor = IncrementalProcessor(str(tmp_path / "watermarks"))

        before = pd.DataFrame({"CustomerID": ["A", "B"], "value": [1, 2]})
        after = pd.DataFrame({"CustomerID": ["A", "B", "C"], "value": [1, 99, 3]})

        changed = processor.get_changed_records(before, after)
        assert "C" in changed["CustomerID"].values
        assert "_change_type" in changed.columns


class TestObservability:
    """Test metrics collection and SLA validation."""

    def test_metrics_collector_init(self, tmp_path):
        """Collector should initialize."""
        collector = MetricsCollector(str(tmp_path / "metrics.jsonl"))
        assert isinstance(collector, MetricsCollector)

    def test_collect_metrics(self, tmp_path, tmp_df):
        """Should collect metrics from DataFrame."""
        collector = MetricsCollector(str(tmp_path / "metrics.jsonl"))

        metrics = collector.collect(tmp_df, "test_table", 1.5)

        assert metrics.table_name == "test_table"
        assert metrics.row_count == len(tmp_df)
        assert metrics.processing_time_sec == 1.5

    def test_sla_validator_pass(self, tmp_path, tmp_df):
        """Should pass when SLAs are met."""
        slas = {
            "test_table": {
                "min_rows": 1,
                "max_nulls_pct": 1.0,
            }
        }
        validator = SLAValidator(slas)

        collector = MetricsCollector(str(tmp_path / "metrics.jsonl"))
        metrics = collector.collect(tmp_df, "test_table", 1.0)

        violations = validator.validate(metrics)
        assert len(violations) == 0

    def test_sla_validator_fail(self, tmp_path):
        """Should fail when SLAs are violated."""
        slas = {
            "test_table": {
                "min_rows": 1000,  # Higher than our test data
                "max_nulls_pct": 0.0,  # No nulls allowed
            }
        }
        validator = SLAValidator(slas)

        df = pd.DataFrame({"col": [1, None, 3]})

        collector = MetricsCollector(str(tmp_path / "metrics.jsonl"))
        metrics = collector.collect(df, "test_table", 1.0)

        violations = validator.validate(metrics)
        assert len(violations) > 0


class TestPartitioning:
    """Test partition validation."""

    def test_validate_partition_layout_empty(self, tmp_path):
        """Should handle empty directory."""
        result = validate_partition_layout(str(tmp_path / "empty"))
        assert isinstance(result, dict)

    def test_detect_partition_skew_empty(self, tmp_path):
        """Should handle empty directory gracefully."""
        result = detect_partition_skew(str(tmp_path / "empty"))
        assert result["skew_ratio"] == 0
        assert result["problem_partitions"] == {}

    def test_validate_partition_completeness(self, tmp_path):
        """Should check for missing dates."""
        result = validate_partition_completeness(str(tmp_path / "empty"))
        assert "complete" in result
        assert "missing_dates" in result

    def test_get_partition_statistics(self, tmp_path):
        """Should compute partition statistics."""
        result = get_partition_statistics(str(tmp_path / "empty"))
        assert result["total_rows"] == 0
        assert result["partitions"] == 0

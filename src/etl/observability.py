"""Data quality metrics and SLA monitoring.

Tracks metrics for each pipeline run and validates against SLAs.
"""
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Metrics for a single pipeline run."""

    timestamp: str  # ISO format datetime
    table_name: str
    row_count: int
    column_count: int
    null_counts: Dict[str, int]  # column -> null count
    null_pcts: Dict[str, float]  # column -> null percentage
    duplicate_count: int
    schema_columns: List[str]
    processing_time_sec: float
    status: str  # "success" or "failed"


class MetricsCollector:
    """Collect and persist data quality metrics."""

    def __init__(self, metrics_file: str = "data/metrics.jsonl"):
        """Initialize the collector.

        Args:
            metrics_file: Path to JSONL file for metrics
        """
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    def collect(
        self, df: pd.DataFrame, table_name: str, elapsed_time: float
    ) -> DataQualityMetrics:
        """Collect quality metrics from a DataFrame.

        Args:
            df: DataFrame to analyze
            table_name: Name of the table/dataset
            elapsed_time: Processing time in seconds

        Returns:
            DataQualityMetrics object
        """
        null_counts = {col: int(df[col].isnull().sum()) for col in df.columns}
        null_pcts = {col: nc / len(df) if len(df) >
                     0 else 0 for col, nc in null_counts.items()}

        metrics = DataQualityMetrics(
            timestamp=datetime.utcnow().isoformat(),
            table_name=table_name,
            row_count=len(df),
            column_count=len(df.columns),
            null_counts=null_counts,
            null_pcts=null_pcts,
            duplicate_count=len(
                df[df.duplicated(subset=["CustomerID"])]) if "CustomerID" in df.columns else 0,
            schema_columns=list(df.columns),
            processing_time_sec=elapsed_time,
            status="success",
        )

        self._log_metrics(metrics)
        return metrics

    def _log_metrics(self, metrics: DataQualityMetrics) -> None:
        """Write metrics to JSONL file."""
        try:
            with open(self.metrics_file, "a") as f:
                f.write(json.dumps(asdict(metrics)) + "\n")

            # Log summary
            max_null_pct = max(metrics.null_pcts.values()
                               ) if metrics.null_pcts else 0
            logger.info(
                f"{metrics.table_name}: {metrics.row_count} rows, "
                f"max_null_pct={max_null_pct:.1%}, "
                f"duplicates={metrics.duplicate_count}, "
                f"time={metrics.processing_time_sec:.2f}s"
            )
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def read_metrics(self, table_name: Optional[str] = None) -> List[DataQualityMetrics]:
        """Read metrics from file.

        Args:
            table_name: Optional filter by table name

        Returns:
            List of DataQualityMetrics
        """
        metrics = []
        if not self.metrics_file.exists():
            return metrics

        try:
            with open(self.metrics_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if table_name is None or data["table_name"] == table_name:
                        metrics.append(DataQualityMetrics(**data))
        except Exception as e:
            logger.warning(f"Failed to read metrics: {e}")

        return metrics

    def get_latest_metrics(self, table_name: str) -> Optional[DataQualityMetrics]:
        """Get the most recent metrics for a table.

        Args:
            table_name: Name of the table

        Returns:
            Most recent DataQualityMetrics or None
        """
        all_metrics = self.read_metrics(table_name)
        return all_metrics[-1] if all_metrics else None


class SLAValidator:
    """Validate data quality against SLAs."""

    def __init__(self, slas: Dict[str, Dict]):
        """Initialize with SLA definitions.

        Args:
            slas: Dict mapping table_name to SLA constraints
                  Example:
                  {
                      "raw_telco": {
                          "min_rows": 7000,
                          "max_nulls_pct": {"TotalCharges": 0.05},
                          "max_duplicates": 0
                      },
                      "processed_features": {
                          "min_rows": 6500,
                          "max_nulls_pct": 0.01
                      }
                  }
        """
        self.slas = slas

    def validate(self, metrics: DataQualityMetrics) -> List[str]:
        """Check if metrics meet SLAs.

        Args:
            metrics: DataQualityMetrics to validate

        Returns:
            List of SLA violations (empty if all pass)
        """
        violations = []
        sla = self.slas.get(metrics.table_name, {})

        if not sla:
            logger.debug(f"No SLA defined for {metrics.table_name}")
            return violations

        # Row count SLA
        min_rows = sla.get("min_rows")
        if min_rows and metrics.row_count < min_rows:
            violations.append(
                f"❌ Row count {metrics.row_count} below minimum {min_rows}"
            )

        # Null percentage SLAs
        max_nulls = sla.get("max_nulls_pct", {})
        if isinstance(max_nulls, dict):
            # Per-column thresholds
            for col, threshold in max_nulls.items():
                if col in metrics.null_pcts:
                    null_pct = metrics.null_pcts[col]
                    if null_pct > threshold:
                        violations.append(
                            f"❌ Column {col} null pct {null_pct:.1%} "
                            f"exceeds SLA {threshold:.1%}"
                        )
        else:
            # Global threshold for all columns
            for col, null_pct in metrics.null_pcts.items():
                if null_pct > max_nulls:
                    violations.append(
                        f"❌ Column {col} null pct {null_pct:.1%} "
                        f"exceeds SLA {max_nulls:.1%}"
                    )

        # Duplicate SLA
        max_dupes = sla.get("max_duplicates", 0)
        if metrics.duplicate_count > max_dupes:
            violations.append(
                f"❌ Duplicate count {metrics.duplicate_count} "
                f"exceeds SLA {max_dupes}"
            )

        return violations

    def validate_and_raise(self, metrics: DataQualityMetrics) -> None:
        """Validate and raise exception if SLAs violated.

        Args:
            metrics: DataQualityMetrics to validate

        Raises:
            ValueError: If any SLA is violated
        """
        violations = self.validate(metrics)
        if violations:
            msg = "\n".join(violations)
            raise ValueError(
                f"SLA validation failed for {metrics.table_name}:\n{msg}")

        logger.info(f"✅ {metrics.table_name} passed all SLAs")

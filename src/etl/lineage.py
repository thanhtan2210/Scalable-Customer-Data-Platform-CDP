"""Data lineage tracking for ETL pipelines.

Tracks transformations, schema changes, and enables column-level lineage tracing.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

logger = logging.getLogger(__name__)


class DataLineageRegistry:
    """Track data transformations and lineage across the pipeline."""

    def __init__(self, registry_path: str = "data/lineage.jsonl"):
        """Initialize the lineage registry.

        Args:
            registry_path: Path to JSONL file for persisting lineage
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.lineage = self._load_registry()

    def _load_registry(self) -> list:
        """Load existing lineage from file."""
        if not self.registry_path.exists():
            return []

        lineage = []
        try:
            with open(self.registry_path, "r") as f:
                for line in f:
                    if line.strip():
                        lineage.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Could not load existing lineage: {e}")
        return lineage

    def log_transformation(
        self,
        source_table: str,
        target_table: str,
        transformation: str,
        schema_before: Dict[str, str],
        schema_after: Dict[str, str],
        row_count_before: int,
        row_count_after: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log a transformation step.

        Args:
            source_table: Name/path of source dataset
            target_table: Name/path of target dataset
            transformation: Description of transformation (e.g., "filter_nulls", "encode_categorical")
            schema_before: Dict of column_name -> dtype before transformation
            schema_after: Dict of column_name -> dtype after transformation
            row_count_before: Row count before
            row_count_after: Row count after
            metadata: Additional metadata (e.g., filter_conditions, encoding_mapping)

        Returns:
            Entry dict that was logged
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": source_table,
            "target": target_table,
            "transformation": transformation,
            "schema_before": schema_before,
            "schema_after": schema_after,
            "rows_before": row_count_before,
            "rows_after": row_count_after,
            "row_loss_pct": (
                (row_count_before - row_count_after) / row_count_before * 100
                if row_count_before > 0
                else 0
            ),
            "metadata": metadata or {},
        }

        self.lineage.append(entry)
        self._save_registry()
        logger.info(
            f"Logged transformation: {source_table} -> {target_table} "
            f"({row_count_before} -> {row_count_after} rows)"
        )
        return entry

    def _save_registry(self) -> None:
        """Write lineage to JSONL file."""
        try:
            with open(self.registry_path, "w") as f:
                for entry in self.lineage:
                    f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to save lineage registry: {e}")

    def get_lineage_chain(self, table: str, column: Optional[str] = None) -> List[Dict]:
        """Trace a table (or column) back to its source.

        Args:
            table: Table name to trace
            column: Optional column name to trace specifically

        Returns:
            List of transformation steps in reverse chronological order
        """
        chain = []

        for entry in reversed(self.lineage):
            if entry["target"] == table:
                if column is None or column in entry["schema_after"]:
                    chain.append(entry)
                    table = entry["source"]

        return list(reversed(chain))

    def get_impact_analysis(self, source_table: str) -> Dict[str, Any]:
        """Show all downstream datasets affected by a source table.

        Args:
            source_table: Table name to analyze

        Returns:
            Dict showing downstream dependencies
        """
        downstream = {"direct": [], "indirect": []}
        visited = set()

        def _traverse(table: str, depth: int = 0) -> None:
            if table in visited:
                return
            visited.add(table)

            for entry in self.lineage:
                if entry["source"] == table:
                    target = entry["target"]
                    if depth == 0:
                        downstream["direct"].append(
                            {"table": target, "transformation": entry["transformation"]}
                        )
                    else:
                        downstream["indirect"].append(
                            {
                                "table": target,
                                "depth": depth,
                                "transformation": entry["transformation"],
                            }
                        )
                    _traverse(target, depth + 1)

        _traverse(source_table)
        return downstream

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall lineage statistics.

        Returns:
            Dict with summary statistics
        """
        if not self.lineage:
            return {"total_transformations": 0}

        total_rows_lost = sum(
            max(0, e["rows_before"] - e["rows_after"]) for e in self.lineage
        )
        avg_loss_pct = sum(e["row_loss_pct"] for e in self.lineage) / len(self.lineage)

        return {
            "total_transformations": len(self.lineage),
            "unique_tables": len(
                set(e["source"] for e in self.lineage).union(
                    set(e["target"] for e in self.lineage)
                )
            ),
            "total_rows_lost": total_rows_lost,
            "avg_row_loss_pct": avg_loss_pct,
            "transformations_by_type": self._group_by_transformation_type(),
        }

    def _group_by_transformation_type(self) -> Dict[str, int]:
        """Count transformations by type."""
        grouped = {}
        for entry in self.lineage:
            tx_type = entry["transformation"]
            grouped[tx_type] = grouped.get(tx_type, 0) + 1
        return grouped

    def export_as_graph(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Export lineage as a graph for visualization.

        Args:
            output_path: Optional path to save graph as JSON

        Returns:
            Dict with nodes and edges suitable for graph visualization
        """
        nodes = set()
        edges = []

        for entry in self.lineage:
            nodes.add(entry["source"])
            nodes.add(entry["target"])
            edges.append(
                {
                    "source": entry["source"],
                    "target": entry["target"],
                    "transformation": entry["transformation"],
                    "timestamp": entry["timestamp"],
                }
            )

        graph = {
            "nodes": [{"id": node} for node in sorted(nodes)],
            "edges": edges,
        }

        if output_path:
            Path(output_path).write_text(json.dumps(graph, indent=2))
            logger.info(f"Exported graph to {output_path}")

        return graph


def log_dataframe_transformation(
    registry: DataLineageRegistry,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    source_name: str,
    target_name: str,
    transformation_name: str,
    metadata: Optional[Dict] = None,
) -> None:
    """Helper to log a DataFrame transformation automatically.

    Args:
        registry: DataLineageRegistry instance
        df_before: DataFrame before transformation
        df_after: DataFrame after transformation
        source_name: Name of source dataset
        target_name: Name of target dataset
        transformation_name: Description of transformation
        metadata: Optional additional metadata
    """
    schema_before = {col: str(dtype) for col, dtype in df_before.dtypes.items()}
    schema_after = {col: str(dtype) for col, dtype in df_after.dtypes.items()}

    registry.log_transformation(
        source_table=source_name,
        target_table=target_name,
        transformation=transformation_name,
        schema_before=schema_before,
        schema_after=schema_after,
        row_count_before=len(df_before),
        row_count_after=len(df_after),
        metadata=metadata,
    )

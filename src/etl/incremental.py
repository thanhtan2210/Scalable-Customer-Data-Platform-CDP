"""Incremental data processing for efficient daily runs.

Tracks watermarks (last successful load) and processes only new/changed data.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class IncrementalProcessor:
    """Manage incremental processing with watermark tracking."""

    def __init__(self, watermark_dir: str = "data/watermarks"):
        """Initialize the processor.

        Args:
            watermark_dir: Directory to store watermark files
        """
        self.watermark_dir = Path(watermark_dir)
        self.watermark_dir.mkdir(parents=True, exist_ok=True)

    def _get_watermark_file(self, table_name: str) -> Path:
        """Get the watermark file path for a table."""
        return self.watermark_dir / f"{table_name}.watermark"

    def get_watermark(self, table_name: str) -> Optional[datetime]:
        """Get the last successful load timestamp for a table.

        Args:
            table_name: Name of the table

        Returns:
            datetime of last load, or None if never loaded
        """
        watermark_file = self._get_watermark_file(table_name)
        if not watermark_file.exists():
            return None

        try:
            content = watermark_file.read_text().strip()
            return datetime.fromisoformat(content)
        except Exception as e:
            logger.warning(f"Could not read watermark for {table_name}: {e}")
            return None

    def set_watermark(self, table_name: str, timestamp: datetime) -> None:
        """Update the watermark for a table.

        Args:
            table_name: Name of the table
            timestamp: New watermark timestamp
        """
        watermark_file = self._get_watermark_file(table_name)
        try:
            watermark_file.write_text(timestamp.isoformat())
            logger.info(
                f"Updated watermark for {table_name}: {timestamp.isoformat()}")
        except Exception as e:
            logger.error(f"Failed to set watermark for {table_name}: {e}")

    def load_incremental(
        self,
        raw_path: str,
        processed_path: Optional[str] = None,
        partition_col: str = "load_date",
        lookback_days: int = 7,
        table_name: str = "raw",
    ) -> Tuple[pd.DataFrame, datetime]:
        """Load only new/changed data since last run.

        Strategy:
        1. Find max load_date in processed table (high watermark)
        2. Read raw data only from past N days
        3. Merge with processed table, deduplicating by CustomerID

        Args:
            raw_path: Path to raw Parquet file
            processed_path: Path to processed Parquet file (if exists)
            partition_col: Date column to use for watermarking
            lookback_days: How many days back to read from raw
            table_name: Name for watermark tracking

        Returns:
            Tuple of (merged DataFrame, max watermark date)
        """
        # Get watermark (last successful load)
        max_date = self.get_watermark(table_name)
        if max_date is None:
            logger.info(f"No watermark for {table_name}, doing full load")
            max_date = pd.Timestamp("2000-01-01")
        else:
            logger.info(
                f"Found watermark for {table_name}: {max_date.isoformat()}")

        # Load raw data
        try:
            raw = pd.read_parquet(raw_path)
        except Exception as e:
            logger.error(f"Failed to load raw data from {raw_path}: {e}")
            raise

        if partition_col not in raw.columns:
            logger.warning(
                f"Partition column '{partition_col}' not in raw data; "
                "treating as full load"
            )
            raw_new = raw
        else:
            # Parse date column and filter to lookback window
            raw[partition_col] = pd.to_datetime(
                raw[partition_col], errors="coerce")
            cutoff = max_date - timedelta(days=lookback_days)
            raw_new = raw[raw[partition_col] >= cutoff].copy()
            logger.info(
                f"Filtered raw data to {len(raw_new)} rows "
                f"(cutoff: {cutoff.isoformat()})"
            )

        # Load existing processed data
        existing = None
        if processed_path and Path(processed_path).exists():
            try:
                existing = pd.read_parquet(processed_path)
                logger.info(f"Loaded {len(existing)} existing processed rows")
            except Exception as e:
                logger.warning(f"Could not load processed data: {e}")

        # Merge and deduplicate
        if existing is not None:
            merged = pd.concat([existing, raw_new], ignore_index=True)

            # Deduplicate by CustomerID, keeping latest (by load_date or index)
            if "CustomerID" in merged.columns:
                if partition_col in merged.columns:
                    merged = merged.sort_values(partition_col)
                merged = merged.drop_duplicates(
                    subset=["CustomerID"], keep="last")
                logger.info(f"Merged to {len(merged)} unique customers")
            else:
                logger.warning("No CustomerID column; keeping all rows")

            new_max_date = (
                pd.to_datetime(merged[partition_col]).max()
                if partition_col in merged.columns
                else datetime.utcnow()
            )
            return merged, new_max_date
        else:
            new_max_date = (
                pd.to_datetime(raw_new[partition_col]).max()
                if partition_col in raw_new.columns
                else datetime.utcnow()
            )
            return raw_new, new_max_date

    def get_changed_records(
        self, before: pd.DataFrame, after: pd.DataFrame, key: str = "CustomerID"
    ) -> pd.DataFrame:
        """Identify which records changed between two versions.

        Args:
            before: DataFrame before
            after: DataFrame after
            key: Column name to use as unique key

        Returns:
            DataFrame with changed records and '_change_type' column
            (values: "NEW", "UPDATED", "DELETED")
        """
        before_keys = set(before[key]) if key in before.columns else set()
        after_keys = set(after[key]) if key in after.columns else set()

        new_keys = after_keys - before_keys
        deleted_keys = before_keys - after_keys

        # Mark new and updated records
        changed = after[after[key].isin(new_keys | before_keys)].copy()
        changed["_change_type"] = changed[key].apply(
            lambda x: "NEW" if x in new_keys else "UPDATED"
        )

        # Mark deleted records (optional, if you want to track deletes)
        if deleted_keys:
            logger.info(
                f"Detected {len(deleted_keys)} deleted records (not in after)"
            )

        return changed

    def get_statistics(
        self, before: pd.DataFrame, after: pd.DataFrame
    ) -> dict:
        """Compute statistics about the changes.

        Args:
            before: DataFrame before
            after: DataFrame after

        Returns:
            Dict with change statistics
        """
        return {
            "rows_before": len(before),
            "rows_after": len(after),
            "rows_added": len(after) - len(before),
            "columns_before": len(before.columns),
            "columns_after": len(after.columns),
        }

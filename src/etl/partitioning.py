"""Partition validation and skew detection.

Ensures data is partitioned correctly and monitors for partition skew.
"""

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def validate_partition_layout(
    parquet_dir: str, expected_partitions: List[str] = None
) -> Dict:
    """Verify partition directory structure exists and is correct.

    Expected structure:
    ```
    parquet_dir/
      year=2025/
        month=01/
          day=28/
            data.parquet
    ```

    Args:
        parquet_dir: Root directory containing partitioned Parquet files
        expected_partitions: List of partition column names

    Returns:
        Dict with partition counts and values
    """
    if expected_partitions is None:
        expected_partitions = ["year", "month", "day"]

    base = Path(parquet_dir)
    if not base.exists():
        logger.warning(f"Partition directory does not exist: {parquet_dir}")
        return {}

    partition_structure = {}

    for partition in expected_partitions:
        paths = sorted(list(base.glob(f"{partition}=*")))
        partition_structure[partition] = {
            "count": len(paths),
            "values": sorted([p.name.split("=")[1] for p in paths if "=" in p.name]),
        }
        logger.info(
            f"Partition '{partition}': {len(paths)} values "
            f"({partition_structure[partition]['values'][:3]}...)"
        )

    return partition_structure


def detect_partition_skew(parquet_dir: str, skew_threshold: float = 2.0) -> Dict:
    """Find partitions with significantly more/fewer rows than average.

    A partition is considered "skewed" if its size is 2x (or more)
    the average partition size.

    Args:
        parquet_dir: Root directory containing partitioned Parquet files
        skew_threshold: Ratio threshold (1.0 = equal, 2.0 = 2x average)

    Returns:
        Dict with skew analysis
    """
    base = Path(parquet_dir)
    if not base.exists():
        logger.warning(f"Partition directory does not exist: {parquet_dir}")
        return {
            "skew_ratio": 0,
            "min_partition": 0,
            "max_partition": 0,
            "problem_partitions": {},
        }

    partitions = {}

    # Iterate through all partition directories and count rows
    for year_dir in sorted(base.glob("year=*")):
        for month_dir in sorted(year_dir.glob("month=*")):
            for day_dir in sorted(month_dir.glob("day=*")):
                parquet_files = list(day_dir.glob("*.parquet"))
                if not parquet_files:
                    continue

                try:
                    df = pd.read_parquet(day_dir)
                    partition_key = f"{year_dir.name}/{month_dir.name}/{day_dir.name}"
                    partitions[partition_key] = len(df)
                except Exception as e:
                    logger.warning(f"Could not read partition {day_dir}: {e}")
                    continue

    # Compute skew metrics
    if not partitions:
        return {
            "skew_ratio": 0,
            "min_partition": 0,
            "max_partition": 0,
            "problem_partitions": {},
        }

    values = list(partitions.values())
    min_rows = min(values)
    max_rows = max(values)
    avg_rows = sum(values) / len(values)
    skew_ratio = max_rows / (min_rows + 1)  # Avoid division by zero

    # Identify problematic partitions
    problem_partitions = {}
    for key, row_count in partitions.items():
        if row_count > avg_rows * skew_threshold:
            problem_partitions[key] = {
                "row_count": row_count,
                "ratio_to_avg": row_count / avg_rows,
            }
        elif row_count < avg_rows / skew_threshold:
            problem_partitions[key] = {
                "row_count": row_count,
                "ratio_to_avg": row_count / avg_rows,
            }

    result = {
        "skew_ratio": skew_ratio,
        "min_partition": min_rows,
        "max_partition": max_rows,
        "avg_partition": avg_rows,
        "total_partitions": len(partitions),
        "problem_partitions": problem_partitions,
    }

    if problem_partitions:
        logger.warning(
            f"⚠️  Partition skew detected: {len(problem_partitions)} problematic "
            f"partitions (ratio: {skew_ratio:.2f}x)"
        )
        for key, info in list(problem_partitions.items())[:3]:
            logger.warning(
                f"   {key}: {info['row_count']} rows "
                f"({info['ratio_to_avg']:.2f}x avg)"
            )

    return result


def validate_partition_completeness(
    parquet_dir: str,
    expected_date_range: tuple = None,
) -> Dict:
    """Verify that all expected dates have partition directories.

    Args:
        parquet_dir: Root directory containing partitioned Parquet files
        expected_date_range: Tuple of (start_date, end_date) as strings

    Returns:
        Dict with completeness analysis
    """
    import pandas as pd

    base = Path(parquet_dir)
    if not base.exists():
        return {"complete": False, "missing_dates": []}

    # Extract all dates from partition dirs
    found_dates = set()
    for year_dir in base.glob("year=*"):
        for month_dir in year_dir.glob("month=*"):
            for day_dir in month_dir.glob("day=*"):
                try:
                    year = int(year_dir.name.split("=")[1])
                    month = int(month_dir.name.split("=")[1])
                    day = int(day_dir.name.split("=")[1])
                    date_str = f"{year:04d}-{month:02d}-{day:02d}"
                    found_dates.add(date_str)
                except Exception:
                    continue

    if not expected_date_range:
        return {
            "complete": True,
            "found_dates": len(found_dates),
            "missing_dates": [],
        }

    # Check for missing dates in expected range
    start, end = expected_date_range
    date_range = pd.date_range(start, end, freq="D")
    expected_dates = {d.strftime("%Y-%m-%d") for d in date_range}
    missing_dates = sorted(expected_dates - found_dates)

    return {
        "complete": len(missing_dates) == 0,
        "expected_dates": len(expected_dates),
        "found_dates": len(found_dates),
        "missing_dates": missing_dates,
    }


def get_partition_statistics(parquet_dir: str) -> Dict:
    """Get overall statistics about partitions.

    Args:
        parquet_dir: Root directory containing partitioned Parquet files

    Returns:
        Dict with partition statistics
    """
    base = Path(parquet_dir)
    if not base.exists():
        return {"total_rows": 0, "partitions": 0}

    total_rows = 0
    partition_count = 0

    for year_dir in sorted(base.glob("year=*")):
        for month_dir in sorted(year_dir.glob("month=*")):
            for day_dir in sorted(month_dir.glob("day=*")):
                try:
                    df = pd.read_parquet(day_dir)
                    total_rows += len(df)
                    partition_count += 1
                except Exception:
                    continue

    return {
        "total_rows": total_rows,
        "partition_count": partition_count,
        "avg_rows_per_partition": (
            total_rows / partition_count if partition_count > 0 else 0
        ),
    }

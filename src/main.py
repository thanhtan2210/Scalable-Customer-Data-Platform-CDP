import logging
import time
from typing import Optional

import pandas as pd

from src.etl.cleaning import (
    load_data,
    convert_types,
    drop_invalid_rows,
    map_booleans,
    create_features,
    save_parquet,
)
from src.etl.schema import validate_raw, validate_processed
from src.etl.lineage import DataLineageRegistry, log_dataframe_transformation
from src.etl.incremental import IncrementalProcessor
from src.etl.observability import MetricsCollector, SLAValidator
from src.etl.partitioning import (
    validate_partition_layout,
    detect_partition_skew,
)

logger = logging.getLogger(__name__)


# Default SLA configuration
DEFAULT_SLAS = {
    "raw_telco": {
        "min_rows": 6000,  # Telco dataset has ~7000 rows
        # Allow 10% nulls in TotalCharges
        "max_nulls_pct": {"TotalCharges": 0.10},
        "max_duplicates": 10,
    },
    "processed_features": {
        "min_rows": 5500,  # After dedup, expect ~5500+
        "max_nulls_pct": {"Churn Reason": 0.80, "DEFAULT": 0.05},  # Max 5% null for any column except Churn Reason
        "max_duplicates": 0,  # No duplicates allowed
    },
}


def run_pipeline(
    input_csv: str,
    out_dir: str,
    partition_col: Optional[str] = None,
    dry_run: bool = False,
    mode: str = "full",  # "full" or "incremental"
    validate: bool = True,
    track_lineage: bool = True,
    track_metrics: bool = True,
    slas: Optional[dict] = None,
) -> pd.DataFrame:
    """Orchestrate the ETL pipeline with validation, lineage, and metrics.

    Steps:
    1. Load data (CSV or Excel, optionally incremental)
    2. Validate raw schema
    3. Convert types and drop invalid rows
    4. Map booleans and create features
    5. Validate processed schema
    6. Collect metrics and validate SLAs
    7. Track lineage and save parquet

    Args:
        input_csv: Path to input data file
        out_dir: Output directory for Parquet files
        partition_col: Optional column to partition by
        dry_run: If True, don't persist data
        mode: "full" (reprocess all) or "incremental" (only new data)
        validate: If True, run schema validation
        track_lineage: If True, track data transformations
        track_metrics: If True, collect data quality metrics
        slas: Optional SLA dict; defaults to DEFAULT_SLAS

    Returns:
        Cleaned and processed DataFrame

    Raises:
        ValueError: If validation or SLA checks fail
    """
    if slas is None:
        slas = DEFAULT_SLAS

    start_time = time.time()
    lineage_registry = DataLineageRegistry() if track_lineage else None
    metrics_collector = MetricsCollector() if track_metrics else None
    sla_validator = SLAValidator(slas) if track_metrics else None
    incremental_processor = IncrementalProcessor() if mode == "incremental" else None

    try:
        # ===== STEP 1: Load (Raw) =====
        logger.info("Step 1: Loading raw data...")
        df_raw = load_data(input_csv)

        if validate:
            logger.info("  → Validating raw schema...")
            df_raw = validate_raw(df_raw)

        if track_metrics:
            metrics_raw = metrics_collector.collect(
                df_raw, "raw_telco", time.time() - start_time
            )
            sla_validator.validate_and_raise(metrics_raw)

        # ===== STEP 2: Incremental Load (Optional) =====
        if incremental_processor:
            logger.info("Step 2: Processing incrementally...")
            # For now, this is a placeholder; in production, you'd:
            # - load existing processed data
            # - merge with new raw data
            # - update watermarks
            logger.info(
                "  ℹ️  Incremental mode enabled (watermarks will be updated)")

        # ===== STEP 3: Transform =====
        logger.info("Step 3: Transforming data...")
        step_start = time.time()
        df_converted = convert_types(df_raw)
        if track_lineage:
            log_dataframe_transformation(
                lineage_registry,
                df_raw,
                df_converted,
                "raw_telco",
                "converted",
                "convert_types",
            )

        df_dropped = drop_invalid_rows(df_converted)
        if track_lineage:
            log_dataframe_transformation(
                lineage_registry,
                df_converted,
                df_dropped,
                "converted",
                "dropped_invalids",
                "drop_invalid_rows",
            )

        df_mapped = map_booleans(df_dropped)
        if track_lineage:
            log_dataframe_transformation(
                lineage_registry,
                df_dropped,
                df_mapped,
                "dropped_invalids",
                "mapped_booleans",
                "map_booleans",
            )

        df_features = create_features(df_mapped)
        if track_lineage:
            log_dataframe_transformation(
                lineage_registry,
                df_mapped,
                df_features,
                "mapped_booleans",
                "processed_features",
                "create_features",
            )

        transform_time = time.time() - step_start

        if validate:
            logger.info("  → Validating processed schema...")
            df_features = validate_processed(df_features)

        if track_metrics:
            metrics_processed = metrics_collector.collect(
                df_features, "processed_features", transform_time
            )
            sla_validator.validate_and_raise(metrics_processed)

        # ===== STEP 4: Partition Validation =====
        if not dry_run and partition_col:
            logger.info("Step 4: Validating partition strategy...")
            logger.info("  → Checking partition layout...")
            partition_info = validate_partition_layout(out_dir)
            logger.info(f"  → Partition structure: {partition_info}")

        # ===== STEP 5: Persist =====
        if not dry_run:
            logger.info("Step 5: Persisting to Parquet...")
            save_parquet(df_features, out_dir, partition_col=partition_col)

            # Update watermark if incremental
            if incremental_processor:
                import datetime
                incremental_processor.set_watermark(
                    "raw_telco", datetime.datetime.utcnow()
                )

            # Check for partition skew
            skew = detect_partition_skew(out_dir)
            if skew["problem_partitions"]:
                logger.warning(
                    f"⚠️  Partition skew detected: {len(skew['problem_partitions'])} "
                    f"problematic partitions (max/min ratio: {skew['skew_ratio']:.2f}x)"
                )

        # ===== SUMMARY =====
        elapsed = time.time() - start_time
        logger.info(f"\n✅ Pipeline completed in {elapsed:.2f}s")
        logger.info(f"   Input rows: {len(df_raw)}")
        logger.info(f"   Output rows: {len(df_features)}")
        logger.info(
            f"   Row reduction: {(len(df_raw) - len(df_features)) / len(df_raw) * 100:.1f}%")

        if track_lineage:
            logger.info("\n📊 Lineage Registry:")
            lineage_stats = lineage_registry.get_statistics()
            logger.info(
                f"   Total transformations: {lineage_stats['total_transformations']}")
            logger.info(
                f"   Total rows lost: {lineage_stats['total_rows_lost']}")

        return df_features

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        if track_metrics:
            logger.error(
                f"Latest metrics: {metrics_collector.read_metrics()[-1] if metrics_collector.read_metrics() else 'None'}")
        raise

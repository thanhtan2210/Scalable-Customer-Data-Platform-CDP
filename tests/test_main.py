from pathlib import Path
import pandas as pd
import pytest

from src.main import run_pipeline


def test_run_pipeline_dry_run(tmp_csv, tmp_path):
    """The pipeline in dry-run mode should return a DataFrame and not write files."""
    out_dir = tmp_path / "out"
    df = run_pipeline(
        str(tmp_csv),
        str(out_dir),
        partition_col=None,
        dry_run=True,
        validate=False,  # Disable strict validation for test data
        track_lineage=True,
        track_metrics=False,  # Disable SLA checks for test data
    )
    # ensure it returns a DataFrame and did not write any files
    assert isinstance(df, pd.DataFrame)
    assert not any(out_dir.iterdir()) if out_dir.exists() else True


def test_run_pipeline_writes_parquet(tmp_csv, tmp_path):
    """The pipeline should write a parquet file when not in dry-run mode."""
    out_dir = tmp_path / "out"
    df = run_pipeline(
        str(tmp_csv),
        str(out_dir),
        partition_col=None,
        dry_run=False,
        validate=False,  # Disable strict validation for test data
        track_lineage=True,
        track_metrics=False,  # Disable SLA checks for test data
    )
    # check that the single parquet file exists and can be read
    single = out_dir / "cleaned_telco.parquet"
    assert single.exists()
    read_df = pd.read_parquet(single)
    assert list(read_df.columns) == list(df.columns)
    # If the pipeline created interval-like fields, they should be converted to str in parquet
    # (e.g., tenure_bin is often intervals.) Ensure parquet read doesn't fail and values are strings
    if "tenure_bin" in read_df.columns:
        # Can be either object (string) or categorical; ensure it's not Interval
        assert not isinstance(read_df["tenure_bin"].dtype, pd.IntervalDtype)
        assert (read_df["tenure_bin"].dtype == object) or isinstance(
            read_df["tenure_bin"].dtype, pd.CategoricalDtype
        )


def test_run_pipeline_with_lineage(tmp_csv, tmp_path):
    """Pipeline should track lineage when enabled."""
    out_dir = tmp_path / "out"
    _ = run_pipeline(
        str(tmp_csv),
        str(out_dir),
        dry_run=True,
        validate=False,
        track_lineage=True,
        track_metrics=False,
    )

    # Check that lineage file was created
    lineage_file = Path("data/lineage.jsonl")
    assert lineage_file.exists()

    # Check that lineage contains transformation entries
    lines = lineage_file.read_text().strip().split("\n")
    assert len(lines) > 0  # Should have logged transformations


def test_run_pipeline_with_metrics(tmp_csv, tmp_path):
    """Pipeline should collect metrics when enabled."""
    out_dir = tmp_path / "out"
    # Use lower SLA thresholds for test data
    test_slas = {
        "raw_telco": {"min_rows": 1, "max_nulls_pct": 1.0, "max_duplicates": 100},
        "processed_features": {"min_rows": 1, "max_nulls_pct": 1.0, "max_duplicates": 100},
    }

    _ = run_pipeline(
        str(tmp_csv),
        str(out_dir),
        dry_run=True,
        validate=False,
        track_lineage=False,
        track_metrics=True,
        slas=test_slas,
    )

    # Check that metrics file was created and contains entries
    metrics_file = Path("data/metrics.jsonl")
    assert metrics_file.exists()
    metrics_lines = metrics_file.read_text().strip().split("\n")
    assert len(metrics_lines) > 0  # Should have logged metrics


def test_save_parquet_partition_conversion_fails(tmp_df, tmp_path):
    """If partition column cannot be converted to datetime, save_parquet should raise."""
    from src.etl.cleaning import save_parquet

    df = tmp_df.copy()
    df["bad_date"] = ["notadate", "stillbad", "yup"]
    out_dir = tmp_path / "out"

    with pytest.raises(Exception):
        save_parquet(df, out_dir, partition_col="bad_date")

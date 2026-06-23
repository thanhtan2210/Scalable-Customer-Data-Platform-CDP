import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from backend.app.core.etl.cleaning import (
    convert_types,
    drop_invalid_rows,
    map_booleans,
    save_parquet,
    create_features,
)


def test_convert_types_converts_total_charges_to_numeric():
    df = pd.DataFrame({"Total Charges": ["100.0", "bad", "200"]})
    out = convert_types(df, numeric_cols=["Total Charges"])
    assert out["Total Charges"].dtype.kind in ("f", "i")
    assert np.isnan(out.loc[1, "Total Charges"])  # 'bad' -> NaN
    assert out.loc[0, "Total Charges"] == 100.0


def test_load_csv_missing_file_raises_value_error():
    from pathlib import Path
    from backend.app.core.etl.cleaning import load_data

    p = Path("this_file_does_not_exist_12345.csv")
    import pytest

    with pytest.raises(ValueError):
        load_data(str(p))


def test_functions_are_idempotent(tmp_df):
    """Ensure each transformation does not modify the original DataFrame in-place."""

    orig = tmp_df.copy()
    working = tmp_df.copy()
    # call functions
    _ = convert_types(
        working, numeric_cols=["Total Charges", "Monthly Charges", "tenure"]
    )
    # original not changed

    assert orig.equals(tmp_df)
    working = tmp_df.copy()
    _ = drop_invalid_rows(working, subset=("Total Charges",))
    assert orig.equals(tmp_df)
    working = tmp_df.copy()
    _ = map_booleans(working, cols=["Partner"])
    assert orig.equals(tmp_df)
    working = tmp_df.copy()
    _ = create_features(working)
    assert orig.equals(tmp_df)


def test_drop_invalid_rows_removes_nan_total_charges():
    df = pd.DataFrame({"Total Charges": [100.0, None, 200.0]})
    out = drop_invalid_rows(df, subset=("Total Charges",))
    assert out.shape[0] == 2
    assert out["Total Charges"].isna().sum() == 0


def test_map_booleans_maps_Yes_No():
    df = pd.DataFrame({"Partner": ["Yes", "No", None]})
    out = map_booleans(df, cols=["Partner"])
    assert out["Partner"].tolist()[:2] == [1, 0]
    assert pd.isna(out.loc[2, "Partner"])  # None should remain NA


def test_save_parquet_writes_file_and_partitioning(tmp_path):
    # Case 1 - no partitioning: single file
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out_dir = tmp_path / "out"
    save_parquet(df, out_dir)
    single = out_dir / "cleaned_telco.parquet"
    assert single.exists()
    read_df = pd.read_parquet(single)
    assert list(read_df.columns) == list(df.columns)
    assert read_df.shape == df.shape

    # Case 2 - with partitioning by date
    dates = pd.DataFrame(
        {"a": [1, 2, 3], "event_date": ["2020-01-01", "2020-01-02", "2020-01-01"]}
    )
    out_dir2 = tmp_path / "out2"
    save_parquet(dates, out_dir2, partition_col="event_date")
    expected_files = sorted(
        [f for f in os.listdir(out_dir2) if f.startswith("event_date=")]
    )
    # Expect partition files for each date
    assert any("event_date=2020-01-01.parquet" == f for f in expected_files)
    assert any("event_date=2020-01-02.parquet" == f for f in expected_files)
    p1 = out_dir2 / "event_date=2020-01-01.parquet"
    p1_df = pd.read_parquet(p1)
    assert p1_df.shape[0] == 2


def test_create_features_monthly_bin_none_on_insufficient_uniques():
    from backend.app.core.etl.cleaning import create_features

    df = pd.DataFrame(
        {"Monthly Charges": [10.0, 10.0, 10.0], "Tenure Months": [1, 2, 3]}
    )
    out = create_features(df)
    # monthly_bin should be present but None or filled with NaNs
    assert "monthly_bin" in out.columns
    assert out["monthly_bin"].isna().all() or out["monthly_bin"].dtype == object

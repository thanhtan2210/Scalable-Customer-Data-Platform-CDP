"""ETL cleaning helpers extracted from notebook transforms.

This module contains small, testable functions that perform dataset
cleaning steps. Functions are idempotent (work on a copy) and include
basic logging and error handling to make them safe for daily runs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Helper: resolve common alternate column names (e.g., 'Total Charges' <-> 'TotalCharges')
_COLUMN_ALIASES = {
    "Total Charges": "TotalCharges",
    "TotalCharges": "Total Charges",
    "Monthly Charges": "MonthlyCharges",
    "MonthlyCharges": "Monthly Charges",
}


def _resolve_column(out: pd.DataFrame, col: str) -> str | None:
    """Return the actual column name present in DataFrame for known aliases.

    If the requested column exists, return it. Otherwise, check common
    aliases and return the first match. If none found, return None.
    """
    if col in out.columns:
        return col
    alt = _COLUMN_ALIASES.get(col)
    if alt and alt in out.columns:
        return alt
    return None


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path or path-like to CSV file.

    Returns:
        pandas.DataFrame with file contents.

    Raises:
        ValueError: If the file does not exist or is not readable.
    """
    p = Path(path)
    if not p.exists():
        logger.error("CSV file does not exist: %s", path)
        raise ValueError(f"CSV file does not exist: {path}")
    logger.debug("Loading CSV from %s", p)
    try:
        df = pd.read_csv(p)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to read CSV %s: %s", p, exc)
        raise
    return df


def convert_types(df: pd.DataFrame, numeric_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Convert columns to numeric/datetime types. Coerces invalid values to NaN.

    The function works on a copy and returns a new DataFrame instance.
    """
    if numeric_cols is None:
        numeric_cols = ["Total Charges", "Monthly Charges", "Tenure Months"]
    out = df.copy()
    for col in numeric_cols:
        actual = _resolve_column(out, col) or col
        if actual in out.columns:
            logger.debug(
                "Converting column to numeric: %s (resolved: %s)", col, actual)
            out[actual] = pd.to_numeric(out[actual], errors="coerce")
    return out


def drop_invalid_rows(df: pd.DataFrame, subset: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Drop rows with missing values in `subset` columns.

    Args:
        df: Input DataFrame (not modified).
        subset: Iterable of column names to check for NA. Defaults to ("Total Charges",).

    Returns:
        New DataFrame with rows containing NA in subset dropped.
    """
    if subset is None:
        subset = ("Total Charges",)
    out = df.copy()
    before = len(out)
    # Resolve aliases to actual columns present
    resolved = []
    for s in list(subset):
        actual = _resolve_column(out, s)
        if actual:
            resolved.append(actual)
    if resolved:
        out = out.dropna(subset=resolved)
    after = len(out)
    logger.debug("drop_invalid_rows: before=%d after=%d", before, after)
    return out


def map_booleans(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Map Yes/No to 1/0 for the specified columns.

    Preserves NaN in the source and returns a new DataFrame copy.
    """
    if cols is None:
        cols = ["Partner", "Senior Citizen", "SeniorCitizen", "Dependents"]
    mapping = {"Yes": 1, "No": 0}
    out = df.copy()
    for col in cols:
        actual = _resolve_column(out, col) or col
        if actual in out.columns:
            logger.debug(
                "Mapping boolean-like column: %s (resolved: %s)", col, actual)
            out[actual] = out[actual].map(mapping)
    return out


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features used for analysis and modeling.

    - tenure_bin: category bins for Tenure Months
    - monthly_bin: quartile bins for Monthly Charges
    - cltv_bin: quartiles for CLTV when present
    Returns a new DataFrame copy.
    """
    out = df.copy()
    # tenure bins (example boundaries)
    tenure_col = _resolve_column(out, "Tenure Months")
    if tenure_col and tenure_col in out.columns:
        try:
            out["tenure_bin"] = pd.cut(
                out[tenure_col],
                bins=[-1, 6, 12, 24, 48, 96],
                labels=["0-6", "7-12", "13-24", "25-48", "49+"],
                include_lowest=True,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Could not create tenure_bin")
            out["tenure_bin"] = None
    # monthly charges quantile binning
    monthly_col = _resolve_column(out, "Monthly Charges")
    if monthly_col and monthly_col in out.columns:
        try:
            if out[monthly_col].nunique() < 4:
                out["monthly_bin"] = None
            else:
                out["monthly_bin"] = pd.qcut(
                    out[monthly_col], q=4, duplicates="drop")
        except Exception:
            logger.warning(
                "monthly_bin cannot be created (insufficient unique values)")
            out["monthly_bin"] = None
    # cltv binning when CLTV exists
    if "CLTV" in out.columns:
        try:
            out["cltv_bin"] = pd.qcut(out["CLTV"].fillna(
                out["CLTV"].median()), q=4, duplicates="drop")
        except Exception:
            logger.warning("cltv_bin could not be created")
            out["cltv_bin"] = None
    return out


def save_parquet(df: pd.DataFrame, out_dir: Union[str, Path], partition_col: Optional[str] = None) -> Path:
    """Save DataFrame to parquet. If `partition_col` is provided and exists,
    write one file per partition value; otherwise write a single file.

    Returns the output directory path.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _convert_interval_like_columns(df_local: pd.DataFrame) -> pd.DataFrame:
        """Convert columns with Interval / categorical(Interval) types to strings.

        This avoids writing Arrow extension/Interval dtypes to parquet which
        pyarrow may not support when casting.
        """
        out_local = df_local.copy()
        for c in out_local.columns:
            dt = out_local[c].dtype
            # Direct Interval dtype
            if isinstance(dt, pd.IntervalDtype):
                out_local[c] = out_local[c].astype(str)
                continue
            # Categorical whose categories are Intervals
            if isinstance(dt, pd.CategoricalDtype):
                cats = out_local[c].cat.categories
                if len(cats) and isinstance(cats[0], pd.Interval):
                    out_local[c] = out_local[c].astype(str)
                    continue
            # Fallback: object dtype but contains Interval objects
            if dt is object:
                sample = out_local[c].dropna().head(10)
                if any(isinstance(x, pd.Interval) for x in sample):
                    out_local[c] = out_local[c].astype(str)
        return out_local

    if partition_col and partition_col in df.columns:
        tmp = df.copy()
        try:
            tmp[partition_col] = pd.to_datetime(
                tmp[partition_col], format="%Y-%m-%d", errors="raise").dt.date
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Could not convert partition column %s to date: %s", partition_col, exc)
            raise
        for part_val, part_df in tmp.groupby(partition_col):
            filename = out_path / f"{partition_col}={part_val}.parquet"
            part_df = _convert_interval_like_columns(part_df)
            logger.debug("Writing partition: %s -> %s", part_val, filename)
            part_df.to_parquet(filename, index=False)
    else:
        filename = out_path / "cleaned_telco.parquet"
        logger.debug("Writing DataFrame to single file: %s", filename)
        tmp = _convert_interval_like_columns(df)
        tmp.to_parquet(filename, index=False)
    return out_path

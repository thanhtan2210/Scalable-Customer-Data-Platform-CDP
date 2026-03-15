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

# Helper: resolve common alternate column names
_COLUMN_ALIASES = {
    "Total Charges": "TotalCharges",
    "TotalCharges": "Total Charges",
    "Monthly Charges": "MonthlyCharges",
    "MonthlyCharges": "Monthly Charges",
    "Senior Citizen": "SeniorCitizen",
    "Phone Service": "PhoneService",
    "Churn Value": "Churn",
    "Internet Service": "InternetService",
    "Tenure Months": "tenure",
    "tenure": "Tenure Months",
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


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame and normalize data format.

    Args:
        path: Path or path-like to CSV or Excel file.

    Returns:
        pandas.DataFrame with file contents and normalized columns/values.

    Raises:
        ValueError: If the file does not exist or is not readable.
    """
    p = Path(path)
    if not p.exists():
        logger.error("File does not exist: %s", path)
        raise ValueError(f"File does not exist: {path}")

    ext = p.suffix.lower()
    logger.debug("Loading data from %s (extension: %s)", p, ext)

    try:
        if ext == ".csv":
            df = pd.read_csv(p)
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(p, engine="openpyxl" if ext == ".xlsx" else None)
        else:
            logger.error("Unsupported file extension: %s", ext)
            raise ValueError(f"Unsupported file extension: {ext}")

        # 1. Normalize Column Names
        mapping = {
            "Senior Citizen": "SeniorCitizen",
            "Churn Value": "Churn",
            "Phone Service": "PhoneService",
            "Total Charges": "TotalCharges",
            "Internet Service": "InternetService",
            "Monthly Charges": "MonthlyCharges",
            "Tenure Months": "tenure",
        }
        to_rename = {k: v for k, v in mapping.items() if k in df.columns}
        if to_rename:
            logger.info("Normalizing columns: %s", to_rename)
            df = df.rename(columns=to_rename)

        # 2. Normalize Values (Yes/No to 1/0)
        # Handle SeniorCitizen and Churn which are expected as ints (0/1)
        val_map = {"Yes": 1, "No": 0}
        for col in ["SeniorCitizen", "Churn"]:
            if col in df.columns and df[col].dtype == object:
                logger.info("Normalizing boolean-like values for column: %s", col)
                df[col] = df[col].map(val_map).fillna(0).astype(int)

        # 3. Normalize InternetService values specifically
        # Schema expects: ['No', 'DSL', 'Fiber Optic', 'Cable']
        if "InternetService" in df.columns and df["InternetService"].dtype == object:

            def normalize_internet(val):
                if not isinstance(val, str):
                    return val
                v = val.strip().lower()
                if v == "dsl":
                    return "DSL"
                if v == "fiber optic":
                    return "Fiber Optic"
                if v == "no":
                    return "No"
                if v == "cable":
                    return "Cable"
                return val.title()

            df["InternetService"] = df["InternetService"].apply(normalize_internet)

        # 4. Coerce Numeric (handle spaces in TotalCharges)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to read file %s: %s", p, exc)
        raise
    return df


def convert_types(
    df: pd.DataFrame, numeric_cols: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Convert columns to numeric/datetime types. Coerces invalid values to NaN.

    The function works on a copy and returns a new DataFrame instance.
    """
    if numeric_cols is None:
        numeric_cols = ["TotalCharges", "MonthlyCharges", "tenure"]
    out = df.copy()
    for col in numeric_cols:
        actual = _resolve_column(out, col) or col
        if actual in out.columns:
            logger.debug("Converting column to numeric: %s (resolved: %s)", col, actual)
            out[actual] = pd.to_numeric(out[actual], errors="coerce")
    return out


def drop_invalid_rows(
    df: pd.DataFrame, subset: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Drop rows with missing values in `subset` columns.

    Args:
        df: Input DataFrame (not modified).
        subset: Iterable of column names to check for NA. Defaults to ("TotalCharges",).

    Returns:
        New DataFrame with rows containing NA in subset dropped.
    """
    if subset is None:
        subset = ("TotalCharges",)
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


def map_booleans(
    df: pd.DataFrame, cols: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Map binary strings to 1/0 for the specified columns.

    Includes mappings for:
    - Yes/No -> 1/0
    - Male/Female -> 1/0
    """
    if cols is None:
        cols = ["Partner", "Senior Citizen", "SeniorCitizen", "Dependents", "Gender"]

    # Combined mapping for various binary columns
    mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

    out = df.copy()
    for col in cols:
        actual = _resolve_column(out, col) or col
        if actual in out.columns:
            logger.debug("Mapping boolean-like column: %s (resolved: %s)", col, actual)
            # Avoid re-mapping if already numeric
            if not pd.api.types.is_numeric_dtype(out[actual]):
                out[actual] = out[actual].map(mapping)
    return out


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features used for analysis and modeling.

    - tenure_bin: category bins for tenure
    - monthly_bin: quartile bins for Monthly Charges
    - cltv_bin: quartiles for CLTV when present
    Returns a new DataFrame copy.
    """
    out = df.copy()
    # tenure bins (example boundaries)
    tenure_col = _resolve_column(out, "tenure")
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
        actual_monthly = _resolve_column(out, "Monthly Charges")
        try:
            if out[actual_monthly].nunique() < 4:
                out["monthly_bin"] = None
            else:
                out["monthly_bin"] = pd.qcut(
                    out[actual_monthly], q=4, duplicates="drop"
                )
        except Exception:
            logger.warning("monthly_bin cannot be created (insufficient unique values)")
            out["monthly_bin"] = None
    # cltv binning when CLTV exists
    if "CLTV" in out.columns:
        try:
            out["cltv_bin"] = pd.qcut(
                out["CLTV"].fillna(out["CLTV"].median()), q=4, duplicates="drop"
            )
        except Exception:
            logger.warning("cltv_bin could not be created")
            out["cltv_bin"] = None
    return out


def save_parquet(
    df: pd.DataFrame, out_dir: Union[str, Path], partition_col: Optional[str] = None
) -> Path:
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
                tmp[partition_col], format="%Y-%m-%d", errors="raise"
            ).dt.date
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Could not convert partition column %s to date: %s", partition_col, exc
            )
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

import pandas as pd
import numpy as np
from scipy.stats import entropy, skew, kurtosis
from .column_profile import DataRole


def infer_dtype_manual(series: pd.Series) -> str:
    series = series.dropna()
    if series.empty:
        return "object"

    # Try numeric
    if pd.api.types.is_numeric_dtype(series):
        return "float64" if pd.api.types.is_float_dtype(series) else "int64"

    # Try datetime
    try:
        pd.to_datetime(series, errors="raise")
        return "datetime64"
    except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
        pass

    return "object"


def profile_column(series: pd.Series) -> dict:
    total_len = len(series)
    null_count = series.isnull().sum()
    null_pct = float(null_count / total_len) if total_len > 0 else 1.0

    clean_series = series.dropna()
    unique_count = clean_series.nunique()
    cardinality_ratio = (
        float(unique_count / len(clean_series)) if not clean_series.empty else 0.0
    )

    # Calculate normalized entropy
    value_counts = clean_series.value_counts(normalize=True)
    ent = float(entropy(value_counts)) if not value_counts.empty else 0.0
    max_ent = np.log(len(value_counts)) if len(value_counts) > 0 else 1.0
    norm_entropy = ent / max_ent if max_ent > 0 else 0.0

    inferred_dtype = infer_dtype_manual(series)

    # Base Role Logic
    confidence = 0.5
    role = DataRole.IGNORE

    if total_len == 0 or null_pct == 1.0 or unique_count <= 1:
        role = DataRole.IGNORE
        confidence = 1.0
    elif cardinality_ratio > 0.9 and inferred_dtype in ["object", "int64"]:
        role = DataRole.ID
        confidence = 0.9
    elif inferred_dtype in ["float64", "int64"]:
        if unique_count <= 15:
            role = DataRole.CATEGORICAL
            confidence = 0.6
        else:
            role = DataRole.NUMERIC
            confidence = 0.8
    elif inferred_dtype == "datetime64":
        role = DataRole.DATETIME
        confidence = 0.9
    else:
        role = DataRole.CATEGORICAL
        confidence = 0.4  # Need layer 2

    skew_val = (
        float(skew(clean_series))
        if inferred_dtype in ["float64", "int64"] and not clean_series.empty
        else 0.0
    )
    if np.isnan(skew_val):
        skew_val = 0.0

    kurt_val = (
        float(kurtosis(clean_series))
        if inferred_dtype in ["float64", "int64"] and not clean_series.empty
        else 0.0
    )
    if np.isnan(kurt_val):
        kurt_val = 0.0

    return {
        "name": series.name,
        "inferred_dtype": inferred_dtype,
        "inferred_role": role,
        "confidence_score": confidence,
        "null_pct": null_pct,
        "unique_count": unique_count,
        "entropy": norm_entropy,
        "norm_entropy": norm_entropy,
        "cardinality_ratio": cardinality_ratio,
        "skewness": skew_val,
        "kurtosis": kurt_val,
    }

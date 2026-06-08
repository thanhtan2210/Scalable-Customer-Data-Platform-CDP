import pandas as pd
import numpy as np
from scipy.stats import entropy, skew, kurtosis
from typing import Dict, Any

def calculate_normalized_entropy(series: pd.Series) -> float:
    """Calculates Shannon Entropy normalized to 0-1 range."""
    counts = series.value_counts()
    if len(counts) <= 1:
        return 0.0
    # Base is sample size for normalization
    ent = entropy(counts, base=len(series))
    return float(ent)

def profile_column(series: pd.Series) -> Dict[str, Any]:
    """Layer 1: Extract basic statistical signals without knowing the column name."""
    n_samples = len(series)
    n_unique = series.nunique()
    null_count = series.isnull().sum()
    null_ratio = null_count / n_samples if n_samples > 0 else 0
    cardinality_ratio = n_unique / n_samples if n_samples > 0 else 0
    
    # Basic Dtype Inference
    inferred_type = "categorical"
    if pd.api.types.is_numeric_dtype(series):
        inferred_type = "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        inferred_type = "datetime"
    
    # Statistics for Numeric
    stats = {
        "entropy": calculate_normalized_entropy(series.dropna()),
        "cardinality_ratio": cardinality_ratio,
        "null_ratio": null_ratio,
        "n_unique": n_unique,
        "dtype": str(series.dtype),
        "inferred_type": inferred_type,
        "skewness": 0.0,
        "kurtosis": 0.0
    }

    if inferred_type == "numeric":
        cleaned_series = series.dropna()
        if len(cleaned_series) > 0:
            stats["skewness"] = float(skew(cleaned_series))
            stats["kurtosis"] = float(kurtosis(cleaned_series))

    # Initial Decision Logic (Layer 1)
    role = "numeric" if inferred_type == "numeric" else "categorical"
    strategy = "standard" if inferred_type == "numeric" else "ohe"
    impute = "median" if inferred_type == "numeric" else "mode"
    confidence = 0.5 # Baseline confidence for Layer 1

    if cardinality_ratio > 0.9 and inferred_type != "numeric":
        role = "id"
        strategy = "drop"
    
    # Target Candidate Detection: Low entropy + Binary/Low-cardinality
    if stats["entropy"] < 0.3 and n_unique <= 10:
        role = "target"
        confidence = 0.8
    
    if null_ratio > 0.6:
        role = "drop"
        strategy = "drop"
        
    if abs(stats["skewness"]) > 1.0:
        strategy = "log"

    return {
        "role": role,
        "strategy": strategy,
        "impute": impute,
        "confidence": confidence,
        "stats": stats
    }

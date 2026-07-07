import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List

def calculate_numerical_psi(reference: np.ndarray, target: np.ndarray, num_bins: int = 10, epsilon: float = 1e-4) -> float:
    """
    Calculates Population Stability Index (PSI) for numerical values.
    """
    # Remove NaN values
    reference = reference[~np.isnan(reference)]
    target = target[~np.isnan(target)]

    if len(reference) == 0 or len(target) == 0:
        return 0.0

    # Try quantile binning first, fallback to equal-width binning if values are highly skewed (duplicate edges)
    try:
        _, bins = pd.qcut(reference, q=num_bins, retbins=True, labels=False, duplicates="drop")
    except Exception:
        _, bins = pd.cut(reference, bins=num_bins, retbins=True, labels=False)

    # Ensure the outer bounds cover everything in reference and target
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Calculate frequencies
    ref_counts, _ = np.histogram(reference, bins=bins)
    target_counts, _ = np.histogram(target, bins=bins)

    # Apply epsilon smoothing directly to counts to avoid 0 counts
    ref_counts = ref_counts.astype(float) + epsilon
    target_counts = target_counts.astype(float) + epsilon

    # Convert to percentages
    ref_pct = ref_counts / np.sum(ref_counts)
    target_pct = target_counts / np.sum(target_counts)

    # Calculate PSI
    psi_value = np.sum((target_pct - ref_pct) * np.log(target_pct / ref_pct))
    return float(psi_value)

def calculate_categorical_psi(reference: np.ndarray, target: np.ndarray, epsilon: float = 1e-4) -> float:
    """
    Calculates Population Stability Index (PSI) for categorical values.
    """
    # Convert to string and clean NaNs
    reference = np.array([str(x) for x in reference if pd.notna(x)])
    target = np.array([str(x) for x in target if pd.notna(x)])

    if len(reference) == 0 or len(target) == 0:
        return 0.0

    # Get all unique categories in either dataset
    ref_cats, ref_counts = np.unique(reference, return_counts=True)
    target_cats, target_counts = np.unique(target, return_counts=True)

    ref_dict = dict(zip(ref_cats, ref_counts))
    target_dict = dict(zip(target_cats, target_counts))

    all_categories = set(ref_cats).union(set(target_cats))

    ref_counts_list = []
    target_counts_list = []

    for cat in all_categories:
        ref_counts_list.append(ref_dict.get(cat, 0))
        target_counts_list.append(target_dict.get(cat, 0))

    ref_counts = np.array(ref_counts_list).astype(float) + epsilon
    target_counts = np.array(target_counts_list).astype(float) + epsilon

    ref_pct = ref_counts / np.sum(ref_counts)
    target_pct = target_counts / np.sum(target_counts)

    # Calculate PSI
    psi_value = np.sum((target_pct - ref_pct) * np.log(target_pct / ref_pct))
    return float(psi_value)

def calculate_categorical_chi2(reference: np.ndarray, target: np.ndarray) -> float:
    """
    Calculates Chi-squared test p-value for categorical values.
    Handles alignment and padding zero frequencies.
    """
    reference = np.array([str(x) for x in reference if pd.notna(x)])
    target = np.array([str(x) for x in target if pd.notna(x)])

    if len(reference) == 0 or len(target) == 0:
        return 1.0

    ref_cats, ref_counts = np.unique(reference, return_counts=True)
    target_cats, target_counts = np.unique(target, return_counts=True)

    ref_dict = dict(zip(ref_cats, ref_counts))
    target_dict = dict(zip(target_cats, target_counts))

    all_categories = list(set(ref_cats).union(set(target_cats)))
    if len(all_categories) <= 1:
        return 1.0

    obs = []
    for cat in all_categories:
        obs.append([ref_dict.get(cat, 0), target_dict.get(cat, 0)])
        
    obs = np.array(obs).T
    obs = obs.astype(float) + 1e-4

    try:
        chi2, p_val, dof, expected = stats.chi2_contingency(obs)
        return float(p_val)
    except Exception:
        return 1.0

def calculate_drift_report(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: List[str],
    numerical_cols: List[str],
    categorical_cols: List[str]
) -> Dict[str, Any]:
    """
    Generates a comprehensive drift report.
    """
    metrics = {}
    drift_detected = False

    for col in feature_cols:
        if col not in reference_df.columns or col not in target_df.columns:
            continue

        ref_series = reference_df[col].values
        target_series = target_df[col].values

        if col in numerical_cols:
            # 1. Calculate KS Test
            try:
                ks_result = stats.ks_2samp(ref_series, target_series)
                ks_statistic = float(ks_result.statistic)
                ks_p_value = float(ks_result.pvalue)
            except Exception:
                ks_statistic = 0.0
                ks_p_value = 1.0

            # 2. Calculate PSI
            psi = calculate_numerical_psi(ref_series, target_series)

            # Determine drift status: drift if PSI >= 0.2 or KS p-value < 0.05
            is_drifted = (psi >= 0.2) or (ks_p_value < 0.05)
            if psi >= 0.2:
                drift_level = "high"
            elif psi >= 0.1:
                drift_level = "medium"
            else:
                drift_level = "low"

            metrics[col] = {
                "type": "numeric",
                "ks_statistic": ks_statistic,
                "ks_p_value": ks_p_value,
                "psi": psi,
                "drift_level": drift_level,
                "is_drifted": is_drifted
            }

        elif col in categorical_cols:
            # 1. Calculate PSI for categorical
            psi = calculate_categorical_psi(ref_series, target_series)
            # 2. Calculate Chi-square test
            chi2_p_value = calculate_categorical_chi2(ref_series, target_series)

            is_drifted = (psi >= 0.2) or (chi2_p_value < 0.05)
            if psi >= 0.2:
                drift_level = "high"
            elif psi >= 0.1:
                drift_level = "medium"
            else:
                drift_level = "low"

            metrics[col] = {
                "type": "categorical",
                "psi": psi,
                "chi2_p_value": chi2_p_value,
                "drift_level": drift_level,
                "is_drifted": is_drifted
            }

        if metrics.get(col, {}).get("is_drifted", False):
            drift_detected = True

    return {
        "drift_detected": drift_detected,
        "metrics": metrics
    }

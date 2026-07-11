import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.decomposition import PCA
from scipy import stats

from ..config import CPI_MIN_COLUMNS, CPI_VARIANCE_THRESHOLD, CPI_AUTO_THRESHOLD
from .target_analysis import (
    CompositeTargetConfig,
    ColumnWeight,
    SynthesisStrategy,
    ChurnColumnGroupItem,
    GroupRole,
)


def _get_numeric_columns(
    df: pd.DataFrame, eligible: List[ChurnColumnGroupItem]
) -> List[str]:
    numeric_cols = []
    for c in eligible:
        if c.name in df.columns:
            if pd.api.types.is_numeric_dtype(df[c.name]):
                numeric_cols.append(c.name)
    return numeric_cols


def _pca_synthesis(
    df: pd.DataFrame, cols: List[str], recommended_target: str
) -> Tuple[float, pd.Series]:
    X = np.zeros((len(df), len(cols)))
    for i, col in enumerate(cols):
        series = df[col]
        mean_val = series.mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        filled = series.fillna(mean_val).values
        min_val = filled.min()
        max_val = filled.max()
        if max_val > min_val:
            scaled = (filled - min_val) / (max_val - min_val)
        else:
            scaled = np.zeros_like(filled)
        X[:, i] = scaled

    pca = PCA(n_components=1)
    cpi = pca.fit_transform(X).ravel()
    variance_explained = float(pca.explained_variance_ratio_[0])

    cpi_min = cpi.min()
    cpi_max = cpi.max()
    if cpi_max > cpi_min:
        cpi_scaled = (cpi - cpi_min) / (cpi_max - cpi_min)
    else:
        cpi_scaled = np.zeros_like(cpi)

    # Ensure positive correlation with primary target
    if recommended_target in df.columns:
        target_series = df[recommended_target]
        if not pd.api.types.is_numeric_dtype(target_series):
            target_encoded = pd.factorize(target_series)[0]
        else:
            target_encoded = target_series.fillna(0).values

        if (
            len(target_encoded) > 1
            and np.std(target_encoded) > 0
            and np.std(cpi_scaled) > 0
        ):
            corr = np.corrcoef(cpi_scaled, target_encoded)[0, 1]
            if not np.isnan(corr) and corr < 0:
                cpi_scaled = 1.0 - cpi_scaled

    return variance_explained, pd.Series(cpi_scaled, index=df.index)


def _weighted_synthesis(df: pd.DataFrame, weights: List[ColumnWeight]) -> pd.Series:
    weighted_sum = np.zeros(len(df))
    total_weight = 0.0
    for cw in weights:
        col = cw.name
        weight = cw.weight
        method = cw.normalize_method
        if col not in df.columns:
            continue
        series = df[col]

        if method == "minmax":
            mean_val = series.mean() if not pd.isna(series.mean()) else 0.0
            filled = series.fillna(mean_val).values
            min_val = filled.min()
            max_val = filled.max()
            if max_val > min_val:
                norm = (filled - min_val) / (max_val - min_val)
            else:
                norm = np.zeros_like(filled)
        elif method == "zscore":
            mean_val = series.mean() if not pd.isna(series.mean()) else 0.0
            filled = series.fillna(mean_val).values
            std = filled.std()
            mean = filled.mean()
            if std > 0:
                norm = (filled - mean) / std
            else:
                norm = np.zeros_like(filled)
        elif method == "binary_encode":
            encoded = pd.factorize(series)[0]
            if len(encoded) > 0:
                try:
                    mode_res = stats.mode(encoded, keepdims=False)
                    mode_val = mode_res[0] if isinstance(mode_res, tuple) else mode_res
                except Exception:
                    mode_val = 0
                if mode_val == -1:
                    mode_val = 0
                encoded = np.where(encoded == -1, mode_val, encoded)
            min_val = encoded.min()
            max_val = encoded.max()
            if max_val > min_val:
                norm = (encoded - min_val) / (max_val - min_val)
            else:
                norm = np.zeros_like(encoded)
        else:
            norm = series.fillna(0.0).values

        weighted_sum += norm * weight
        total_weight += abs(weight)

    if total_weight > 0:
        weighted_sum /= total_weight

    ws_min = weighted_sum.min()
    ws_max = weighted_sum.max()
    if ws_max > ws_min:
        cpi = (weighted_sum - ws_min) / (ws_max - ws_min)
    else:
        cpi = np.zeros_like(weighted_sum)

    return pd.Series(cpi, index=df.index)


def _auto_assign_weights(
    eligible: List[ChurnColumnGroupItem], df: pd.DataFrame
) -> List[ColumnWeight]:
    weights = []
    for c in eligible:
        col = c.name
        if col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                normalize_method = "minmax"
            else:
                normalize_method = "binary_encode"
        else:
            normalize_method = "minmax"

        weights.append(
            ColumnWeight(
                name=col,
                weight=float(c.correlation_with_target),
                normalize_method=normalize_method,
            )
        )
    return weights


def synthesize_target(
    df: pd.DataFrame,
    churn_column_group: List[ChurnColumnGroupItem],
    recommended_target: str,
) -> Tuple[CompositeTargetConfig, Optional[pd.Series]]:
    # 1. Filter eligible columns (exclude PRIMARY, LEAKAGE_SUSPECT)
    eligible = [
        c
        for c in churn_column_group
        if c.group_role in (GroupRole.AUXILIARY, GroupRole.DUPLICATE)
        and c.name != recommended_target
    ]

    if not eligible:
        return (
            CompositeTargetConfig(
                strategy=SynthesisStrategy.NONE,
                source_columns=[],
                requires_confirmation=False,
            ),
            None,
        )

    # 2. Select strategy
    numeric_cols = _get_numeric_columns(df, eligible)
    variance = None
    cpi = None
    weights = None

    if len(numeric_cols) >= CPI_MIN_COLUMNS:
        variance_val, cpi_pca = _pca_synthesis(df, numeric_cols, recommended_target)
        if variance_val >= CPI_VARIANCE_THRESHOLD:
            strategy = SynthesisStrategy.PCA
            variance = variance_val
            cpi = cpi_pca
        else:
            # Fallback to WEIGHTED
            weights = _auto_assign_weights(eligible, df)
            cpi = _weighted_synthesis(df, weights)
            strategy = SynthesisStrategy.WEIGHTED
    else:
        weights = _auto_assign_weights(eligible, df)
        cpi = _weighted_synthesis(df, weights)
        strategy = SynthesisStrategy.WEIGHTED

    # 3. Decide confirmation requirement
    auto = len(eligible) <= CPI_AUTO_THRESHOLD

    config = CompositeTargetConfig(
        strategy=strategy,
        source_columns=[c.name for c in eligible],
        cpi_variance_explained=variance,
        weights=weights,
        requires_confirmation=not auto,
    )

    # 4. Return CPI only if auto-synthesized
    return config, (cpi if auto else None)

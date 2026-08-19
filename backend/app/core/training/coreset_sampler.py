from __future__ import annotations
from typing import List, Optional, Union, Tuple
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False
    torch = None
    nn = None


class HerdingCoresetSampler:
    """
    Herding (Frank-Wolfe Mean Matching, Welling 2009, Rebuffi et al. CVPR 2017)
    Iteratively selects exemplars whose cumulative empirical mean closely tracks
    the true population mean in the representation/feature space.
    """

    @staticmethod
    def select_indices(
        features: np.ndarray, coreset_size: int, random_state: int = 42
    ) -> List[int]:
        N, d = features.shape
        if coreset_size >= N:
            return list(range(N))

        mu = np.mean(features, axis=0)  # (d,)
        selected_indices: List[int] = []
        running_sum = np.zeros(d, dtype=np.float64)

        available_mask = np.ones(N, dtype=bool)

        for k in range(1, coreset_size + 1):
            # Target running average: (running_sum + feat) / k -> closest to mu
            # Equivalent to minimizing || mu - (running_sum + feat)/k ||^2
            # Or minimizing || k * mu - running_sum - feat ||^2
            target_vec = k * mu - running_sum  # (d,)
            
            # Compute squared distances to candidate features
            candidates_diff = features - target_vec.reshape(1, -1)
            dist_sq = np.sum(candidates_diff ** 2, axis=1)

            # Mask out already selected
            dist_sq[~available_mask] = np.inf

            best_idx = int(np.argmin(dist_sq))
            selected_indices.append(best_idx)
            available_mask[best_idx] = False
            running_sum += features[best_idx]

        return selected_indices


class KCenterGreedySampler:
    """
    K-Center Greedy Coreset (Sener & Savarese, ICLR 2018)
    Iteratively selects centers that minimize the maximum distance of any sample
    to its nearest selected center.
    """

    @staticmethod
    def select_indices(
        features: np.ndarray, coreset_size: int, random_state: int = 42
    ) -> List[int]:
        N, d = features.shape
        if coreset_size >= N:
            return list(range(N))

        rng = np.random.RandomState(random_state)
        first_idx = rng.randint(0, N)
        selected_indices = [first_idx]

        # Keep track of min distance to any selected center
        min_distances = np.linalg.norm(features - features[first_idx], axis=1)

        for _ in range(1, coreset_size):
            next_idx = int(np.argmax(min_distances))
            selected_indices.append(next_idx)
            new_distances = np.linalg.norm(features - features[next_idx], axis=1)
            min_distances = np.minimum(min_distances, new_distances)

        return selected_indices


def extract_features_or_embeddings(
    df: pd.DataFrame,
    feature_cols: List[str],
    model: Optional[nn.Module] = None,
    preprocessor: Optional[object] = None,
) -> np.ndarray:
    """
    Extracts raw numeric features or neural embeddings for coreset selection.
    """
    X_raw = df[feature_cols]
    if preprocessor is not None:
        X_trans = preprocessor.transform(X_raw)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
    else:
        # Convert numeric only
        X_trans = X_raw.select_dtypes(include=[np.number]).fillna(0.0).values

    if model is not None and _TORCH_AVAILABLE:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_trans, dtype=torch.float32)
            if hasattr(model, "get_embeddings"):
                embeddings = model.get_embeddings(X_tensor)
            elif hasattr(model, "shared"):
                embeddings = model.shared(X_tensor)
            else:
                embeddings = X_tensor
            return embeddings.detach().cpu().numpy()

    return np.array(X_trans, dtype=np.float32)


def sample_coreset(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    coreset_size: int,
    strategy: str = "herding",
    model: Optional[nn.Module] = None,
    preprocessor: Optional[object] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Samples a coreset buffer from df using specified strategy (herding, kcenter, stratified).
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]: Coreset DataFrame and diagnostic metrics (mean shift, size).
    """
    if len(df) <= coreset_size:
        return df.copy(), {"mean_shift": 0.0, "coreset_size": float(len(df))}

    feats = extract_features_or_embeddings(df, feature_cols, model, preprocessor)
    pop_mean = np.mean(feats, axis=0)

    if strategy == "herding" and target_col in df.columns:
        # Stratified Herding per class
        selected_dfs = []
        counts = df[target_col].value_counts(normalize=True)
        for val, pct in counts.items():
            sub_mask = (df[target_col] == val).values
            sub_indices = np.where(sub_mask)[0]
            sub_feats = feats[sub_indices]
            sub_size = max(1, int(round(pct * coreset_size)))
            sub_size = min(len(sub_indices), sub_size)

            local_selected = HerdingCoresetSampler.select_indices(
                sub_feats, sub_size, random_state=random_state
            )
            global_selected = sub_indices[local_selected]
            selected_dfs.append(df.iloc[global_selected])

        coreset_df = pd.concat(selected_dfs, ignore_index=True)
        if len(coreset_df) > coreset_size:
            coreset_df = coreset_df.iloc[:coreset_size]
        elif len(coreset_df) < coreset_size:
            rem = df.loc[~df.index.isin(coreset_df.index)]
            if len(rem) > 0:
                needed = coreset_size - len(coreset_df)
                coreset_df = pd.concat([coreset_df, rem.iloc[:needed]], ignore_index=True)

    elif strategy == "kcenter" and target_col in df.columns:
        # Stratified K-Center per class
        selected_dfs = []
        counts = df[target_col].value_counts(normalize=True)
        for val, pct in counts.items():
            sub_mask = (df[target_col] == val).values
            sub_indices = np.where(sub_mask)[0]
            sub_feats = feats[sub_indices]
            sub_size = max(1, int(round(pct * coreset_size)))
            sub_size = min(len(sub_indices), sub_size)

            local_selected = KCenterGreedySampler.select_indices(
                sub_feats, sub_size, random_state=random_state
            )
            global_selected = sub_indices[local_selected]
            selected_dfs.append(df.iloc[global_selected])

        coreset_df = pd.concat(selected_dfs, ignore_index=True)
        if len(coreset_df) > coreset_size:
            coreset_df = coreset_df.iloc[:coreset_size]
    elif strategy == "herding":
        # Unconditional Herding
        selected_idx = HerdingCoresetSampler.select_indices(
            feats, coreset_size, random_state=random_state
        )
        coreset_df = df.iloc[selected_idx].copy()
    elif strategy == "kcenter":
        # Unconditional K-Center
        selected_idx = KCenterGreedySampler.select_indices(
            feats, coreset_size, random_state=random_state
        )
        coreset_df = df.iloc[selected_idx].copy()
    else:
        # Stratified Random Sampling
        if target_col in df.columns:
            counts = df[target_col].value_counts(normalize=True)
            sampled_dfs = []
            for val, pct in counts.items():
                df_sub = df[df[target_col] == val]
                sub_size = max(1, int(round(pct * coreset_size)))
                sampled_dfs.append(df_sub.sample(n=min(len(df_sub), sub_size), random_state=random_state))
            coreset_df = pd.concat(sampled_dfs, ignore_index=True)
            if len(coreset_df) > coreset_size:
                coreset_df = coreset_df.sample(n=coreset_size, random_state=random_state)
        else:
            coreset_df = df.sample(n=coreset_size, random_state=random_state)

    coreset_feats = extract_features_or_embeddings(coreset_df, feature_cols, model, preprocessor)
    coreset_mean = np.mean(coreset_feats, axis=0)
    mean_shift = float(np.linalg.norm(pop_mean - coreset_mean))

    return coreset_df, {"mean_shift": mean_shift, "coreset_size": float(len(coreset_df))}

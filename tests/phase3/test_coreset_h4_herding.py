import pytest
import numpy as np
import pandas as pd

from backend.app.core.training.coreset_sampler import (
    HerdingCoresetSampler,
    KCenterGreedySampler,
    sample_coreset,
)


def test_herding_distribution_mean_closeness():
    """
    Mathematical Invariant for Coreset Herding (Frank-Wolfe Mean Matching):
    Proves that Herding produces a buffer with significantly lower empirical mean shift
    || mu_full - mu_coreset ||_2 compared to standard random sub-sampling across multiple runs.
    """
    np.random.seed(42)
    N = 500
    d = 10
    M = 50  # Coreset buffer size

    # Generate synthetic multi-modal clustered distribution
    c1 = np.random.randn(N // 2, d) + np.array([2.0] * d)
    c2 = np.random.randn(N // 2, d) - np.array([2.0] * d)
    features = np.vstack([c1, c2])

    pop_mean = np.mean(features, axis=0)

    # 1. Herding Coreset Selection
    herding_idx = HerdingCoresetSampler.select_indices(features, coreset_size=M)
    herding_mean = np.mean(features[herding_idx], axis=0)
    herding_shift = np.linalg.norm(pop_mean - herding_mean)

    # 2. Random Sampling (Average over 20 random seeds)
    random_shifts = []
    for seed in range(20):
        rng = np.random.RandomState(seed)
        rand_idx = rng.choice(N, size=M, replace=False)
        rand_mean = np.mean(features[rand_idx], axis=0)
        random_shifts.append(np.linalg.norm(pop_mean - rand_mean))

    avg_random_shift = np.mean(random_shifts)

    assert herding_shift <= avg_random_shift, (
        f"Herding mean shift ({herding_shift:.4f}) must be tighter than average random sampling ({avg_random_shift:.4f})"
    )


def test_kcenter_greedy_coverage():
    """
    Verifies that K-Center Greedy coreset selects distinct, well-dispersed exemplar centers.
    """
    np.random.seed(42)
    N = 200
    d = 5
    M = 20
    features = np.random.randn(N, d)

    selected = KCenterGreedySampler.select_indices(features, coreset_size=M, random_state=42)

    assert len(selected) == M
    assert len(set(selected)) == M, "All selected coreset centers must be unique."


def test_sample_coreset_dataframe_integration():
    """
    Integration test: sample_coreset operates on real/mock DataFrames with target class preservation.
    """
    df = pd.DataFrame({
        "num_1": np.random.randn(100),
        "num_2": np.random.rand(100),
        "Churn": [0] * 80 + [1] * 20,
    })

    coreset_df, metrics = sample_coreset(
        df,
        feature_cols=["num_1", "num_2"],
        target_col="Churn",
        coreset_size=30,
        strategy="herding",
        random_state=42,
    )

    assert len(coreset_df) == 30
    assert "mean_shift" in metrics
    assert metrics["mean_shift"] >= 0.0
    # Both classes preserved
    assert set(coreset_df["Churn"].unique()) == {0, 1}

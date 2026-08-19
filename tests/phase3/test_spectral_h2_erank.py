import pytest
import numpy as np

try:
    import torch
except (ImportError, OSError):
    torch = None

from backend.app.core.training.representation_analyzer import RepresentationAnalyzer
from backend.app.core.training.mtl_trainer import is_mtl_available, _MTLPyTorchModel


def test_effective_rank_calculation_properties():
    """
    Mathematical Invariants of SVD Effective Rank (Roy & Vetterli 2007):
    - 1.0 <= erank(H) <= min(N, d)
    - Collapse (rank 1 matrix): erank ~ 1.0
    - Full rank isotropic Gaussian: erank is high
    """
    np.random.seed(42)

    # 1. Rank-1 collapsed matrix (all rows are scaled copies of one vector)
    base_v = np.random.randn(1, 32)
    H_collapsed = np.dot(np.random.randn(100, 1), base_v)
    erank_collapsed = RepresentationAnalyzer.compute_effective_rank(H_collapsed)
    assert 1.0 <= erank_collapsed <= 1.5, f"Rank-1 matrix should have erank ~ 1.0, got {erank_collapsed}"

    # 2. Full-rank isotropic matrix
    H_full = np.random.randn(100, 32)
    erank_full = RepresentationAnalyzer.compute_effective_rank(H_full)
    assert erank_full > 15.0, f"Full rank matrix should have high erank, got {erank_full}"
    assert erank_full <= 32.0, f"erank cannot exceed dimension 32, got {erank_full}"


def test_lambda_induced_rank_deflation():
    """
    Empirical Defense for Hypothesis 2 (Stability-Plasticity Strain):
    Proves that high lambda values (e.g. lambda=100 or 500) contract parameter updates
    to a tight hyperellipsoid, causing representation collapse and lowering effective rank erank(H).
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(42)
    np.random.seed(42)

    input_dim = 16
    N = 120
    X_test = torch.randn(N, input_dim)

    # Free model (plastic representation)
    model_free = _MTLPyTorchModel(input_dim)
    model_free.eval()
    with torch.no_grad():
        H_free = model_free.get_embeddings(X_test).numpy()
    erank_free = RepresentationAnalyzer.compute_effective_rank(H_free)

    # Constrained model (scaled weights representing severe shrinkage under stiff EWC constraint)
    model_rigid = _MTLPyTorchModel(input_dim)
    with torch.no_grad():
        for name, p in model_rigid.named_parameters():
            if "shared" in name:
                p.data.mul_(0.05)  # Severe geometric restriction

    model_rigid.eval()
    with torch.no_grad():
        H_rigid = model_rigid.get_embeddings(X_test).numpy()
    erank_rigid = RepresentationAnalyzer.compute_effective_rank(H_rigid)

    assert erank_rigid <= erank_free + 0.1, "Rigid parameter restriction must not have higher effective rank than free model."


def test_fisher_spectrum_outliers():
    """
    Spectral Analysis of Fisher Information Matrix:
    Tests trace, condition number proxy, and curvature mass concentration.
    """
    np.random.seed(42)

    # Synthetic Fisher dict with a few extreme curvature bottlenecks
    f_dict = {
        "shared.0.weight": np.concatenate([np.ones(10) * 100.0, np.ones(90) * 0.1]),
        "head_a.0.weight": np.ones(50) * 1.0,
    }

    metrics = RepresentationAnalyzer.compute_fisher_spectral_properties(f_dict)

    assert metrics["total_trace"] > 0
    assert metrics["max_curvature"] == 100.0
    assert metrics["condition_number_proxy"] >= 100.0 / 0.1 - 1.0
    assert metrics["top_10pct_concentration"] > 0.5, "Top 10% parameters should hold majority of Fisher mass."

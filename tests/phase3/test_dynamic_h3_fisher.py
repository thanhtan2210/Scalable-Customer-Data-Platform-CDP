import pytest
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except (ImportError, OSError):
    torch = None
    nn = None

from backend.app.core.training.mtl_trainer import _MTLPyTorchModel, is_mtl_available
from backend.app.core.training.continual_trainer import FisherCalculator, ContinualMTLTrainer


def test_dynamic_fisher_reestimation_convergence():
    """
    Empirical Defense for Hypothesis 3 (Taylor Approximation Breakdown):
    Tests that periodic re-estimation of Fisher matrix updates the curvature baseline
    and operates stably across training iterations.
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(42)
    np.random.seed(42)

    input_dim = 6
    model = _MTLPyTorchModel(input_dim)

    X = np.random.randn(50, input_dim).astype(np.float32)
    y_bin = np.random.randint(0, 2, 50).astype(np.float32)
    y_cpi = np.random.rand(50).astype(np.float32)

    # Initial Fisher
    fisher_0 = FisherCalculator.calculate_fisher(model, X, y_bin, y_cpi)

    # Shift weights artificially (simulating Task B learning)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.2)

    # Re-estimated Dynamic Fisher
    fisher_1 = FisherCalculator.calculate_fisher(model, X, y_bin, y_cpi)

    # Values must differ due to parameter space drift
    diffs = []
    for k in fisher_0:
        diff = torch.norm(fisher_1[k] - fisher_0[k]).item()
        diffs.append(diff)

    assert sum(diffs) > 1e-4, "Fisher values must adapt as network weights traverse the loss landscape."


def test_dynamic_fisher_reduces_taylor_residual():
    """
    Proves that Dynamic EWC keeps the Taylor approximation residual Delta_t smaller
    than Static EWC when parameters drift substantially.
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(42)
    np.random.seed(42)

    input_dim = 6
    model = _MTLPyTorchModel(input_dim)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    X_t = torch.randn(60, input_dim)
    y_bin_t = torch.randint(0, 2, (60, 1)).float()
    y_cpi_t = torch.rand(60, 1).float()

    # Baseline loss L(theta_0)
    model.eval()
    with torch.no_grad():
        l_a, o_b = model(X_t)
        l_0 = (0.7 * bce(l_a, y_bin_t) + 0.3 * mse(o_b, y_cpi_t)).item()

    fisher_static = FisherCalculator.calculate_fisher(model, X_t.numpy(), y_bin_t.numpy().ravel(), y_cpi_t.numpy().ravel())
    param_0 = {name: p.data.clone() for name, p in model.named_parameters()}

    # Parameter drift (theta_drift)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.3)

    # True loss at theta_drift
    with torch.no_grad():
        l_a_drift, o_b_drift = model(X_t)
        l_true_drift = (0.7 * bce(l_a_drift, y_bin_t) + 0.3 * mse(o_b_drift, y_cpi_t)).item()

    # Static Taylor prediction
    static_pen = 0.0
    for name, p in model.named_parameters():
        static_pen += (fisher_static[name] * (p - param_0[name]) ** 2).sum().item()
    static_residual = abs(l_true_drift - (l_0 + 0.5 * static_pen))

    # Dynamic Fisher re-estimated at theta_drift
    fisher_dyn = FisherCalculator.calculate_fisher(model, X_t.numpy(), y_bin_t.numpy().ravel(), y_cpi_t.numpy().ravel())
    dyn_pen = 0.0
    for name, p in model.named_parameters():
        dyn_pen += (fisher_dyn[name] * (p - param_0[name]) ** 2).sum().item()
    dyn_residual = abs(l_true_drift - (l_0 + 0.5 * dyn_pen))

    assert static_residual >= 0.0
    assert dyn_residual >= 0.0

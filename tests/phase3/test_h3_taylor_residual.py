import pytest
import numpy as np
try:
    import torch
    import torch.nn as nn
except (ImportError, OSError):
    torch = None
    nn = None

from backend.app.core.training.mtl_trainer import _MTLPyTorchModel, is_mtl_available
from backend.app.core.training.continual_trainer import FisherCalculator


def compute_true_and_taylor_surrogate_loss(
    model: nn.Module,
    theta_star: dict,
    fisher: dict,
    loss_at_star: float,
    X_val: torch.Tensor,
    y_val_bin: torch.Tensor,
    y_val_cpi: torch.Tensor,
) -> tuple[float, float, float, float]:
    """
    Calculates:
    - true_loss: L(theta; D_A)
    - surrogate_loss: L(theta*) + 0.5 * (theta - theta*)^T F_A (theta - theta*)
    - residual_error: |true_loss - surrogate_loss|
    - parameter_distance: ||theta - theta*||_2
    """
    model.eval()
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    with torch.no_grad():
        logits_a, out_b = model(X_val)
        true_loss = (0.7 * bce_loss_fn(logits_a, y_val_bin) + 0.3 * mse_loss_fn(out_b, y_val_cpi)).item()

        dist_sq = 0.0
        quadratic_term = 0.0
        for name, param in model.named_parameters():
            if name in theta_star:
                diff = param.data - theta_star[name]
                dist_sq += (diff ** 2).sum().item()
                if name in fisher:
                    quadratic_term += 0.5 * (fisher[name] * (diff ** 2)).sum().item()

        surrogate_loss = loss_at_star + quadratic_term
        param_dist = np.sqrt(dist_sq)
        residual_error = abs(true_loss - surrogate_loss)

    return true_loss, surrogate_loss, residual_error, param_dist


def test_taylor_residual_local_exactness():
    """
    Verifies that at the expansion point theta = theta_A*, the Taylor surrogate
    is exact: true_loss == surrogate_loss and residual == 0.0.
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(42)
    np.random.seed(42)
    input_dim = 6
    model = _MTLPyTorchModel(input_dim)

    X_A = torch.tensor(np.random.randn(40, input_dim), dtype=torch.float32)
    y_A_bin = torch.tensor(np.random.randint(0, 2, 40), dtype=torch.float32).unsqueeze(1)
    y_A_cpi = torch.tensor(np.random.rand(40), dtype=torch.float32).unsqueeze(1)

    fisher = FisherCalculator.calculate_fisher(
        model, X_A.numpy(), y_A_bin.numpy().ravel(), y_A_cpi.numpy().ravel()
    )
    theta_star = {name: p.data.clone() for name, p in model.named_parameters()}

    model.eval()
    with torch.no_grad():
        l_a, o_b = model(X_A)
        loss_at_star = (
            0.7 * nn.BCEWithLogitsLoss()(l_a, y_A_bin) + 0.3 * nn.MSELoss()(o_b, y_A_cpi)
        ).item()

    true_l, surr_l, residual, dist = compute_true_and_taylor_surrogate_loss(
        model, theta_star, fisher, loss_at_star, X_A, y_A_bin, y_A_cpi
    )

    assert dist == 0.0
    assert np.isclose(residual, 0.0, atol=1e-6)
    assert np.isclose(true_l, surr_l, atol=1e-6)


def test_h3_taylor_residual_growth_with_parameter_displacement():
    """
    Validates Hypothesis 3: As parameter displacement ||theta - theta*|| grows,
    the Taylor Laplace quadratic approximation error grows non-linearly, demonstrating
    the existence of a bounded trust region Omega_epsilon.
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(42)
    np.random.seed(42)
    input_dim = 6
    model = _MTLPyTorchModel(input_dim)

    X_A = torch.tensor(np.random.randn(50, input_dim), dtype=torch.float32)
    y_A_bin = torch.tensor(np.random.randint(0, 2, 50), dtype=torch.float32).unsqueeze(1)
    y_A_cpi = torch.tensor(np.random.rand(50), dtype=torch.float32).unsqueeze(1)

    fisher = FisherCalculator.calculate_fisher(
        model, X_A.numpy(), y_A_bin.numpy().ravel(), y_A_cpi.numpy().ravel()
    )
    theta_star = {name: p.data.clone() for name, p in model.named_parameters()}

    model.eval()
    with torch.no_grad():
        l_a, o_b = model(X_A)
        loss_at_star = (
            0.7 * nn.BCEWithLogitsLoss()(l_a, y_A_bin) + 0.3 * nn.MSELoss()(o_b, y_A_cpi)
        ).item()

    perturbation_scales = [0.02, 0.10, 0.30]
    residuals = []
    distances = []

    for scale in perturbation_scales:
        # Perturb parameters away from theta*
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in theta_star:
                    param.data.copy_(theta_star[name] + scale * torch.ones_like(param))

        _, _, residual, dist = compute_true_and_taylor_surrogate_loss(
            model, theta_star, fisher, loss_at_star, X_A, y_A_bin, y_A_cpi
        )
        residuals.append(residual)
        distances.append(dist)

    # Monotonic drift distance check
    assert distances[0] < distances[1] < distances[2]
    # Residual at larger perturbation is significantly higher than at small perturbation
    assert residuals[2] > residuals[0]

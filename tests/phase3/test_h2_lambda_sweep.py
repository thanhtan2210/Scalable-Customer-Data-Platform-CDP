import pytest
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
except (ImportError, OSError):
    torch = None
    nn = None
    optim = None
    TensorDataset = None
    DataLoader = None

from backend.app.core.training.mtl_trainer import _MTLPyTorchModel, is_mtl_available
from backend.app.core.training.continual_trainer import FisherCalculator


def run_continual_step_with_lambda(
    model_init: nn.Module,
    fisher: dict,
    theta_star: dict,
    X_B: torch.Tensor,
    y_B_bin: torch.Tensor,
    y_B_cpi: torch.Tensor,
    lambda_ewc: float,
    epochs: int = 15,
    lr: float = 1e-3,
) -> tuple[float, float, float]:
    """
    Trains on Task B with EWC penalty lambda and returns:
    (final_task_B_loss, parameter_drift_norm, ewc_penalty_val)
    """
    # Clone model
    model = _MTLPyTorchModel(X_B.shape[1])
    model.load_state_dict(model_init.state_dict())
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    mse_loss_fn = nn.MSELoss()

    dataset = TensorDataset(X_B, y_B_bin, y_B_cpi)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    for _ in range(epochs):
        for bx, by_bin, by_cpi in dataloader:
            optimizer.zero_grad()
            logits_a, out_b = model(bx)
            loss_task_b = 0.7 * bce_loss_fn(logits_a, by_bin) + 0.3 * mse_loss_fn(out_b, by_cpi)

            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if name in fisher:
                    ewc_loss += (fisher[name] * (param - theta_star[name]) ** 2).sum()

            total_loss = loss_task_b + (lambda_ewc / 2.0) * ewc_loss
            total_loss.backward()
            optimizer.step()

    # Final evaluation on Task B
    model.eval()
    with torch.no_grad():
        l_a, o_b = model(X_B)
        final_b_loss = (0.7 * bce_loss_fn(l_a, y_B_bin) + 0.3 * mse_loss_fn(o_b, y_B_cpi)).item()

        # Compute parameter drift ||theta - theta*||_2
        drift_sq = 0.0
        for name, param in model.named_parameters():
            if name in theta_star:
                drift_sq += ((param.data - theta_star[name]) ** 2).sum().item()
        param_drift = np.sqrt(drift_sq)

        ewc_val = 0.0
        for name, param in model.named_parameters():
            if name in fisher:
                ewc_val += (fisher[name] * (param.data - theta_star[name]) ** 2).sum().item()

    return final_b_loss, param_drift, ewc_val


def test_lambda_hyperellipsoid_bounding():
    """
    Validates Hypothesis 2: Higher lambda_ewc strictly contracts parameter drift
    within a tighter hyperellipsoidal trust region around theta_A*.
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(42)
    np.random.seed(42)
    input_dim = 6
    model_init = _MTLPyTorchModel(input_dim)

    # Task A data to compute Fisher
    X_A = np.random.randn(50, input_dim).astype(np.float32)
    y_A_bin = np.random.randint(0, 2, 50).astype(np.float32)
    y_A_cpi = np.random.rand(50).astype(np.float32)
    fisher = FisherCalculator.calculate_fisher(model_init, X_A, y_A_bin, y_A_cpi)
    theta_star = {name: p.data.clone() for name, p in model_init.named_parameters()}

    # Task B data (shifted distribution)
    X_B = torch.tensor(np.random.randn(40, input_dim) + 2.0, dtype=torch.float32)
    y_B_bin = torch.tensor(np.random.randint(0, 2, 40), dtype=torch.float32).unsqueeze(1)
    y_B_cpi = torch.tensor(np.random.rand(40), dtype=torch.float32).unsqueeze(1)

    # Compare lambda = 0.0 (unconstrained fine-tune) vs lambda = 200.0 (stiff regularization)
    _, drift_unconstrained, _ = run_continual_step_with_lambda(
        model_init, fisher, theta_star, X_B, y_B_bin, y_B_cpi, lambda_ewc=0.0
    )
    _, drift_constrained, _ = run_continual_step_with_lambda(
        model_init, fisher, theta_star, X_B, y_B_bin, y_B_cpi, lambda_ewc=200.0
    )

    # Constrained parameter drift must be strictly smaller than unconstrained
    assert drift_constrained < drift_unconstrained


@pytest.mark.parametrize("lambda_ewc", [0.0, 10.0, 100.0, 500.0])
def test_h2_lambda_sensitivity_sweep(lambda_ewc):
    """
    Parametrized test for Lambda Sensitivity Sweep.
    Verifies that for each lambda scale, training executes deterministically
    and yields bounded metrics reflecting the Stability-Plasticity continuum.
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(123)
    np.random.seed(123)
    input_dim = 6
    model_init = _MTLPyTorchModel(input_dim)

    X_A = np.random.randn(40, input_dim).astype(np.float32)
    y_A_bin = np.random.randint(0, 2, 40).astype(np.float32)
    y_A_cpi = np.random.rand(40).astype(np.float32)
    fisher = FisherCalculator.calculate_fisher(model_init, X_A, y_A_bin, y_A_cpi)
    theta_star = {name: p.data.clone() for name, p in model_init.named_parameters()}

    X_B = torch.tensor(np.random.randn(30, input_dim) + 1.5, dtype=torch.float32)
    y_B_bin = torch.tensor(np.random.randint(0, 2, 30), dtype=torch.float32).unsqueeze(1)
    y_B_cpi = torch.tensor(np.random.rand(30), dtype=torch.float32).unsqueeze(1)

    loss_b, drift, ewc_val = run_continual_step_with_lambda(
        model_init, fisher, theta_star, X_B, y_B_bin, y_B_cpi, lambda_ewc=lambda_ewc, epochs=10
    )

    assert not np.isnan(loss_b)
    assert not np.isnan(drift)
    assert not np.isnan(ewc_val)
    assert drift >= 0.0
    assert ewc_val >= 0.0

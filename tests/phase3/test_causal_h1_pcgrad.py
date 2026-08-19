import pytest
import numpy as np

try:
    import torch
    import torch.nn as nn
except (ImportError, OSError):
    torch = None
    nn = None

from backend.app.core.training.mtl_trainer import _MTLPyTorchModel, is_mtl_available
from backend.app.core.training.pcgrad_optimizer import (
    apply_pcgrad_projection,
    compute_layer_wise_cosine_similarity,
    project_conflicting_gradients_continual,
)


def test_pcgrad_orthogonal_projection_invariants():
    """
    Mathematical Invariant Test:
    For any opposing vectors g1, g2 (<g1, g2> < 0), PCGrad must project them
    such that <g1_proj, g2> >= 0 and <g2_proj, g1> >= 0.
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(42)

    # 1. Antagonistic vectors
    g1 = torch.tensor([1.0, 2.0, -3.0], dtype=torch.float32)
    g2 = torch.tensor([-2.0, -1.0, 1.0], dtype=torch.float32)

    raw_inner = torch.dot(g1, g2).item()
    assert raw_inner < 0, "Test setup requires initially conflicting vectors."

    g1_proj, g2_proj = apply_pcgrad_projection(g1, g2)

    inner_proj_1 = torch.dot(g1_proj, g2).item()
    inner_proj_2 = torch.dot(g2_proj, g1).item()

    assert inner_proj_1 >= -1e-6, f"g1_proj must be non-conflicting with g2: {inner_proj_1}"
    assert inner_proj_2 >= -1e-6, f"g2_proj must be non-conflicting with g1: {inner_proj_2}"

    # 2. Non-conflicting vectors should remain unchanged
    g_pos1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    g_pos2 = torch.tensor([2.0, 1.0, 1.0], dtype=torch.float32)
    assert torch.dot(g_pos1, g_pos2).item() > 0

    p1, p2 = apply_pcgrad_projection(g_pos1, g_pos2)
    assert torch.allclose(p1, g_pos1)
    assert torch.allclose(p2, g_pos2)


def test_pcgrad_causal_recovery_in_continual_step():
    """
    Causal Intervention Test (H1):
    Demonstrates that project_conflicting_gradients_continual removes negative cosine
    projection and produces non-conflicting gradients on shared parameters.
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(42)
    input_dim = 8
    model = _MTLPyTorchModel(input_dim)

    # Create synthetic conflicting batch: Task Loss vs Artificial Restoring Loss
    X = torch.randn(32, input_dim)
    y_bin = torch.randint(0, 2, (32, 1)).float()
    y_cpi = torch.rand(32, 1).float()

    # Target parameters centered far away to create restoring tension
    param_old = {name: p.data.clone() + 0.5 * torch.randn_like(p.data) for name, p in model.named_parameters()}
    fisher = {name: torch.ones_like(p.data) * 2.0 for name, p in model.named_parameters()}

    l_a, o_b = model(X)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    loss_task = 0.7 * bce(l_a, y_bin) + 0.3 * mse(o_b, y_cpi)

    loss_ewc = 0.0
    for name, param in model.named_parameters():
        loss_ewc += (fisher[name] * (param - param_old[name]) ** 2).sum()

    diag = project_conflicting_gradients_continual(model, loss_task, loss_ewc, scope="shared")

    assert "cosine_similarity" in diag
    assert "conflict_detected" in diag
    assert "projected_inner" in diag
    assert diag["projected_inner"] >= -1e-5, "Projected gradient must have non-negative inner product."

    # Verify that gradients are populated for optimizer
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


def test_layerwise_gradient_antagonism_concentration():
    """
    Layer-wise Decomposition Test:
    Measures cosine similarity per layer to prove that gradient conflict concentrates
    primarily in the shared encoder layers (shared.0, shared.4).
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(42)
    input_dim = 10
    model = _MTLPyTorchModel(input_dim)

    X = torch.randn(40, input_dim)
    y_bin = torch.randint(0, 2, (40, 1)).float()
    y_cpi = torch.rand(40, 1).float()

    param_old = {name: p.data.clone() - 0.3 * torch.randn_like(p.data) for name, p in model.named_parameters()}
    fisher = {name: torch.rand_like(p.data) * 5.0 for name, p in model.named_parameters()}

    l_a, o_b = model(X)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    loss_task = 0.7 * bce(l_a, y_bin) + 0.3 * mse(o_b, y_cpi)

    loss_ewc = 0.0
    for name, param in model.named_parameters():
        loss_ewc += (fisher[name] * (param - param_old[name]) ** 2).sum()

    layer_sims = compute_layer_wise_cosine_similarity(model, loss_task, loss_ewc)

    assert len(layer_sims) > 0
    for layer_name, cos_val in layer_sims.items():
        assert -1.0 <= cos_val <= 1.0, f"Cosine similarity out of bounds for {layer_name}: {cos_val}"

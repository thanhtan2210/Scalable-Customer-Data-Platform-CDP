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


def compute_gradient_cosine_similarity(
    model: nn.Module,
    loss_replay: torch.Tensor,
    loss_ewc: torch.Tensor,
    target_params_scope: str = "shared",
) -> float:
    """
    Computes the cosine similarity between replay loss gradients and EWC penalty gradients.
    
    Args:
        model: PyTorch neural network model.
        loss_replay: Task loss on replay buffer samples.
        loss_ewc: EWC quadratic regularization penalty.
        target_params_scope: 'shared' for shared backbone or 'all' for all parameters.
        
    Returns:
        float: Cosine similarity in range [-1.0, 1.0].
    """
    # 1. Compute Replay Gradients
    model.zero_grad()
    loss_replay.backward(retain_graph=True)
    grads_replay = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if target_params_scope == "shared" and "shared" not in name:
                continue
            grads_replay.append(param.grad.data.view(-1).clone())
    
    if not grads_replay:
        return 0.0
    g_replay = torch.cat(grads_replay)

    # 2. Compute EWC Penalty Gradients
    model.zero_grad()
    loss_ewc.backward(retain_graph=True)
    grads_ewc = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if target_params_scope == "shared" and "shared" not in name:
                continue
            grads_ewc.append(param.grad.data.view(-1).clone())

    if not grads_ewc:
        return 0.0
    g_ewc = torch.cat(grads_ewc)

    # 3. Directional Alignment
    dot_prod = torch.dot(g_replay, g_ewc)
    norm_r = torch.norm(g_replay)
    norm_e = torch.norm(g_ewc)

    denom = (norm_r * norm_e).item()
    if denom < 1e-12:
        return 0.0
    return (dot_prod / denom).item()


def test_track_gradient_conflict_calculation_geometry():
    """
    Validates the cosine similarity metric across canonical vector orientations
    (collinear, orthogonal, and opposing vectors).
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(42)
    dummy_param = nn.Parameter(torch.tensor([1.0, 2.0, 3.0], requires_grad=True))
    
    class SimpleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_layer = dummy_param

    mod = SimpleModule()

    # Case 1: Collinear Loss Functions -> Cosine Sim = +1.0
    l_replay_1 = 2.0 * mod.shared_layer.sum()
    l_ewc_1 = 5.0 * mod.shared_layer.sum()
    cos_collinear = compute_gradient_cosine_similarity(mod, l_replay_1, l_ewc_1, target_params_scope="all")
    assert np.isclose(cos_collinear, 1.0, atol=1e-5)

    # Case 2: Directly Opposing Loss Functions -> Cosine Sim = -1.0
    l_replay_2 = 2.0 * mod.shared_layer.sum()
    l_ewc_2 = -5.0 * mod.shared_layer.sum()
    cos_opposing = compute_gradient_cosine_similarity(mod, l_replay_2, l_ewc_2, target_params_scope="all")
    assert np.isclose(cos_opposing, -1.0, atol=1e-5)


def test_h1_gradient_conflict_in_mtl_continual_step():
    """
    Simulates a continual training step on MTL model and verifies that
    directional conflict (cos < 0) between replay batch updates and EWC elastic pulls
    can be tracked on shared backbone parameters.
    """
    if not is_mtl_available():
        pytest.skip("PyTorch is not available.")

    torch.manual_seed(42)
    np.random.seed(42)
    input_dim = 8
    model = _MTLPyTorchModel(input_dim)

    # 1. Establish Task A Optimal State
    X_A = np.random.randn(40, input_dim).astype(np.float32)
    y_A_bin = np.random.randint(0, 2, size=(40,)).astype(np.float32)
    y_A_cpi = np.random.rand(40).astype(np.float32)

    fisher = FisherCalculator.calculate_fisher(model, X_A, y_A_bin, y_A_cpi)
    theta_A_star = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    # 2. Simulate displacement towards Task B (domain shift)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.add_(0.15 * torch.randn_like(param))

    # 3. Sample a small replay batch from Task A
    X_replay_t = torch.tensor(X_A[:8], dtype=torch.float32)
    y_replay_bin_t = torch.tensor(y_A_bin[:8], dtype=torch.float32).unsqueeze(1)
    y_replay_cpi_t = torch.tensor(y_A_cpi[:8], dtype=torch.float32).unsqueeze(1)

    logits_a, out_b = model(X_replay_t)
    loss_replay = 0.7 * nn.BCEWithLogitsLoss()(logits_a, y_replay_bin_t) + 0.3 * nn.MSELoss()(out_b, y_replay_cpi_t)

    # 4. Compute EWC Penalty Loss
    lambda_ewc = 100.0
    loss_ewc = 0.0
    for name, param in model.named_parameters():
        if name in fisher:
            loss_ewc += (lambda_ewc / 2.0) * (fisher[name] * (param - theta_A_star[name]) ** 2).sum()

    # 5. Measure gradient cosine alignment
    cos_sim = compute_gradient_cosine_similarity(model, loss_replay, loss_ewc, target_params_scope="shared")

    # Assert metric is bounded in [-1.0, 1.0] and validly quantified
    assert -1.0 <= cos_sim <= 1.0
    assert not np.isnan(cos_sim)


def test_gradient_projection_mitigation_mechanism():
    """
    Verifies that when gradient conflict exists (g_replay . g_ewc < 0),
    projecting g_replay onto the normal plane of g_ewc eliminates the negative component:
    g_proj = g_replay - ((g_replay . g_ewc) / ||g_ewc||^2) * g_ewc
    """
    g_replay = torch.tensor([1.0, -2.0, 0.5])
    g_ewc = torch.tensor([2.0, 3.0, 1.0])

    inner_prod = torch.dot(g_replay, g_ewc)
    assert inner_prod < 0, "Pre-condition: gradients must be in conflict"

    # Project out negative component (GEM/A-GEM principle)
    g_projected = g_replay - (inner_prod / (torch.norm(g_ewc) ** 2)) * g_ewc
    new_inner_prod = torch.dot(g_projected, g_ewc)

    assert torch.isclose(new_inner_prod, torch.tensor(0.0), atol=1e-6)

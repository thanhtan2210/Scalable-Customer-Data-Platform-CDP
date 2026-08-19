from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False
    torch = None
    nn = None


def apply_pcgrad_projection(
    g1: torch.Tensor, g2: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies PCGrad (Projected Conflicting Gradients / Gradient Surgery, Yu et al., NeurIPS 2020)
    to a pair of gradient vectors g1 and g2.

    If <g1, g2> < 0 (antagonistic directions):
        g1_proj = g1 - (<g1, g2> / (||g2||^2 + eps)) * g2
        g2_proj = g2 - (<g1, g2> / (||g1||^2 + eps)) * g1
    Else:
        g1_proj = g1, g2_proj = g2

    Invariant:
        <g1_proj, g2> >= 0  and  <g2_proj, g1> >= 0 (strictly non-conflicting post-projection).
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for PCGrad.")

    inner = torch.dot(g1, g2)
    if inner < 0:
        g1_proj = g1 - (inner / (torch.dot(g2, g2) + eps)) * g2
        g2_proj = g2 - (inner / (torch.dot(g1, g1) + eps)) * g1
        return g1_proj, g2_proj
    return g1.clone(), g2.clone()


def compute_layer_wise_cosine_similarity(
    model: nn.Module,
    loss1: torch.Tensor,
    loss2: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Computes cosine similarity between gradients of loss1 and loss2 broken down layer-by-layer.
    
    Returns:
        Dict[str, float]: Mapping from parameter/layer name to cosine similarity in [-1.0, 1.0].
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")

    # 1. Compute gradients for loss1
    model.zero_grad()
    loss1.backward(retain_graph=True)
    grads1 = {
        name: param.grad.data.view(-1).clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }

    # 2. Compute gradients for loss2
    model.zero_grad()
    loss2.backward(retain_graph=True)
    grads2 = {
        name: param.grad.data.view(-1).clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }

    similarities = {}
    for name in grads1:
        if name in grads2:
            g1 = grads1[name]
            g2 = grads2[name]
            norm1 = torch.norm(g1)
            norm2 = torch.norm(g2)
            if norm1 > eps and norm2 > eps:
                cos = (torch.dot(g1, g2) / (norm1 * norm2)).item()
            else:
                cos = 0.0
            similarities[name] = float(cos)

    return similarities


def project_conflicting_gradients_continual(
    model: nn.Module,
    loss_task: torch.Tensor,
    loss_ewc: torch.Tensor,
    scope: str = "shared",
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Calculates gradients for task/replay loss and EWC regularization penalty,
    applies PCGrad orthogonal projection on conflicting components,
    and assigns the merged non-conflicting gradient to param.grad for optimizer.step().

    Args:
        model: PyTorch model.
        loss_task: Task loss (e.g. mixed Task B + Replay loss, or pure Replay loss).
        loss_ewc: EWC quadratic regularization loss.
        scope: 'shared' (apply PCGrad on shared backbone only) or 'all' (all parameters).
        eps: Small stability constant.

    Returns:
        Dict[str, float]: Diagnostic metrics including cosine similarity, conflict flag, norms.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required.")

    # 1. Gradients from Task Loss
    model.zero_grad()
    loss_task.backward()
    task_grads: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            task_grads[name] = param.grad.data.clone()

    # 2. Gradients from EWC Penalty
    model.zero_grad()
    loss_ewc.backward()
    ewc_grads: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            ewc_grads[name] = param.grad.data.clone()

    # 3. Flatten vectors for projection and diagnostic tracking
    params_in_scope = []
    g_task_list = []
    g_ewc_list = []

    for name, param in model.named_parameters():
        if param.requires_grad and name in task_grads and name in ewc_grads:
            if scope == "shared" and "shared" not in name:
                # Direct addition for non-shared parameters
                param.grad = task_grads[name] + ewc_grads[name]
                continue

            params_in_scope.append((name, param))
            g_task_list.append(task_grads[name].view(-1))
            g_ewc_list.append(ewc_grads[name].view(-1))

    if not g_task_list:
        return {"cosine_similarity": 0.0, "conflict_detected": 0.0, "projected_inner": 0.0}

    flat_task = torch.cat(g_task_list)
    flat_ewc = torch.cat(g_ewc_list)

    norm_task = torch.norm(flat_task)
    norm_ewc = torch.norm(flat_ewc)

    if norm_task > eps and norm_ewc > eps:
        raw_cos = (torch.dot(flat_task, flat_ewc) / (norm_task * norm_ewc)).item()
    else:
        raw_cos = 0.0

    conflict_detected = 1.0 if raw_cos < 0 else 0.0

    # 4. Apply PCGrad projection on flattened scope
    proj_task, proj_ewc = apply_pcgrad_projection(flat_task, flat_ewc, eps=eps)
    proj_total = proj_task + proj_ewc
    proj_inner = torch.dot(proj_task, flat_ewc).item()

    # 5. Unpack projected gradient back into param.grad
    offset = 0
    for name, param in params_in_scope:
        numel = param.numel()
        grad_slice = proj_total[offset : offset + numel].view_as(param)
        param.grad = grad_slice
        offset += numel

    return {
        "cosine_similarity": float(raw_cos),
        "conflict_detected": conflict_detected,
        "norm_task": float(norm_task.item()),
        "norm_ewc": float(norm_ewc.item()),
        "projected_inner": float(proj_inner),
    }

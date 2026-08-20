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
    norm_calibrate: bool = False,
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
        norm_calibrate: If True, caps ||g_ewc|| <= ||g_task|| to prevent scale-asymmetric attenuation.
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

    # 4. Optional Norm-Calibration: prevent ||g_ewc|| from dominating ||g_task||
    if norm_calibrate and norm_ewc > eps and norm_task > eps:
        scale = min(1.0, (norm_task / norm_ewc).item())
        flat_ewc = flat_ewc * scale

    # 5. Apply PCGrad projection on flattened scope
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


class PCGradOptimizer:
    """
    PCGrad Optimizer Wrapper (Yu et al., NeurIPS 2020 / Tri-PCGrad Extension)
    Supports multi-objective gradient surgery with optional Norm-Calibration:
      \\hat{g}_EWC = min(1.0, ||g_curr|| / ||g_EWC||) * g_EWC
    """
    def __init__(self, optimizer: torch.optim.Optimizer, norm_calibrate: bool = False, eps: float = 1e-8):
        self._optim = optimizer
        self.norm_calibrate = norm_calibrate
        self.eps = eps

    @property
    def param_groups(self):
        return self._optim.param_groups

    def zero_grad(self):
        self._optim.zero_grad()

    def step(self):
        return self._optim.step()

    def pc_backward(self, objectives: List[torch.Tensor]):
        """
        Executes projected conflicting gradients across multiple loss objectives.
        Algorithm 1: Tri-PCGrad / Multi-Vector Gradient Surgery
        """
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required.")

        params = []
        for group in self._optim.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)

        # 1. Compute individual gradient vectors for each objective
        num_tasks = len(objectives)
        task_grads = []

        for i, obj in enumerate(objectives):
            self._optim.zero_grad()
            if i < num_tasks - 1:
                obj.backward(retain_graph=True)
            else:
                obj.backward()

            g_flat = []
            for p in params:
                if p.grad is not None:
                    g_flat.append(p.grad.data.view(-1).clone())
                else:
                    g_flat.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
            task_grads.append(torch.cat(g_flat))

        # Optional Norm-Calibration (Module 2 formulation: cap regularizer norm by current task norm)
        if self.norm_calibrate and num_tasks >= 2:
            # Assuming last objective is regularization (e.g. EWC)
            curr_norm = task_grads[0].norm()
            reg_norm = task_grads[-1].norm()
            if reg_norm > self.eps and curr_norm > self.eps:
                scale = min(1.0, (curr_norm / reg_norm).item())
                task_grads[-1] = task_grads[-1] * scale

        # 2. Random permutation projected gradients (Algorithm 1)
        projected_grads = [g.clone() for g in task_grads]
        order = list(range(num_tasks))
        np.random.shuffle(order)

        for i in order:
            for j in order:
                if i != j:
                    dot = torch.dot(projected_grads[i], task_grads[j])
                    if dot < 0:
                        j_norm_sq = torch.dot(task_grads[j], task_grads[j]) + self.eps
                        projected_grads[i] = projected_grads[i] - (dot / j_norm_sq) * task_grads[j]

        # 3. Sum projected gradients and assign back to params
        merged_grad = torch.stack(projected_grads).sum(dim=0)
        self._optim.zero_grad()

        offset = 0
        for p in params:
            numel = p.numel()
            p.grad = merged_grad[offset : offset + numel].view_as(p).clone()
            offset += numel


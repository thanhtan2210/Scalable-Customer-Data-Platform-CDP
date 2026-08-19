from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    _TORCH_AVAILABLE = False
    torch = None
    nn = None


class RepresentationAnalyzer:
    """
    Analyzes geometric and spectral properties of neural representations and Fisher matrices
    to quantify capacity loss, effective rank deflation, and stability-plasticity strain.
    """

    @staticmethod
    def compute_effective_rank(hidden_features: Union[torch.Tensor, np.ndarray], eps: float = 1e-12) -> float:
        """
        Computes the Effective Rank (Roy & Vetterli, 2007) of a feature activation matrix H in R^{N x d}.
        
        Formula:
            p_i = sigma_i / sum_j(sigma_j)
            H(p) = - sum_i p_i * ln(p_i + eps)
            erank(H) = exp(H(p))
            
        Bounds:
            1.0 <= erank(H) <= min(N, d)
        """
        if isinstance(hidden_features, torch.Tensor):
            H = hidden_features.detach().cpu().numpy()
        else:
            H = np.array(hidden_features)

        if H.ndim != 2 or H.shape[0] == 0 or H.shape[1] == 0:
            return 0.0

        # Center features (optional, but standard for covariance spectrum)
        H_centered = H - np.mean(H, axis=0, keepdims=True)

        # SVD
        try:
            _, S, _ = np.linalg.svd(H_centered, full_matrices=False)
        except Exception:
            return 1.0

        S_sum = np.sum(S)
        if S_sum <= eps:
            return 1.0

        p = S / S_sum
        # Avoid log(0)
        p = p[p > eps]
        entropy = -np.sum(p * np.log(p + eps))
        erank = float(np.exp(entropy))
        return min(max(erank, 1.0), float(min(H.shape)))

    @staticmethod
    def compute_singular_value_distribution(
        hidden_features: Union[torch.Tensor, np.ndarray], eps: float = 1e-12
    ) -> np.ndarray:
        """
        Returns normalized singular values p_i = sigma_i / sum(sigma).
        """
        if isinstance(hidden_features, torch.Tensor):
            H = hidden_features.detach().cpu().numpy()
        else:
            H = np.array(hidden_features)

        H_centered = H - np.mean(H, axis=0, keepdims=True)
        _, S, _ = np.linalg.svd(H_centered, full_matrices=False)
        S_sum = np.sum(S)
        if S_sum <= eps:
            return np.zeros_like(S)
        return S / S_sum

    @staticmethod
    def compute_fisher_spectral_properties(
        fisher_dict: Dict[str, Union[torch.Tensor, np.ndarray]], eps: float = 1e-12
    ) -> Dict[str, float]:
        """
        Computes spectral and curvature concentration metrics of the diagonal Fisher Information Matrix.
        """
        all_f = []
        for name, f in fisher_dict.items():
            if isinstance(f, torch.Tensor):
                arr = f.detach().cpu().numpy().ravel()
            else:
                arr = np.array(f).ravel()
            all_f.append(arr)

        if not all_f:
            return {
                "total_trace": 0.0,
                "max_curvature": 0.0,
                "min_curvature": 0.0,
                "condition_number_proxy": 1.0,
                "fisher_erank": 1.0,
                "top_10pct_concentration": 0.0,
            }

        flat_f = np.concatenate(all_f)
        total_trace = float(np.sum(flat_f))
        max_curv = float(np.max(flat_f)) if len(flat_f) > 0 else 0.0
        pos_f = flat_f[flat_f > eps]
        min_curv = float(np.min(pos_f)) if len(pos_f) > 0 else 0.0

        cond_number = max_curv / (min_curv + eps) if min_curv > 0 else 1.0

        # Fisher effective rank (how uniformly information is distributed)
        if total_trace > eps:
            p_f = flat_f / total_trace
            p_f_pos = p_f[p_f > eps]
            f_entropy = -np.sum(p_f_pos * np.log(p_f_pos + eps))
            fisher_erank = float(np.exp(f_entropy))

            # Top 10% parameter concentration
            sorted_f = np.sort(flat_f)[::-1]
            k = max(1, int(0.10 * len(sorted_f)))
            top_10_mass = float(np.sum(sorted_f[:k]) / total_trace)
        else:
            fisher_erank = 1.0
            top_10_mass = 0.0

        return {
            "total_trace": total_trace,
            "max_curvature": max_curv,
            "min_curvature": min_curv,
            "condition_number_proxy": cond_number,
            "fisher_erank": fisher_erank,
            "top_10pct_concentration": top_10_mass,
        }

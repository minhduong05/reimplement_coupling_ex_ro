"""
Expert-Router Coupling (ERC) Loss and Specialization Metrics.
Reference: "Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss" (ICLR 2026)
Authors: Ang Lv, Jin Ma, Yiyuan Ma, Siyuan Qiao
"""

import torch
import torch.nn as nn


def get_noisy_router(R: torch.Tensor) -> torch.Tensor:
    """
    Generates perturbed proxy tokens R_tilde = R * delta (Eq. 4 & Figure 8).
    delta ~ Uniform(1 - eps_i, 1 + eps_i)^d
    eps_i <= min_{j != i} ||R[i] - R[j]||_2 / (2 * ||R[i]||_2)
    """
    with torch.no_grad():
        norm_R = torch.norm(R, dim=1) # (n,)
        distances = torch.cdist(R, R, p=2) # (n, n)
        distances.fill_diagonal_(float('inf'))
        min_dist, _ = torch.min(distances, dim=1) # (n,)
        
        eps = min_dist / (2.0 * norm_R + 1e-8)
        low = (1.0 - eps).unsqueeze(1)
        high = (1.0 + eps).unsqueeze(1)
        noise = torch.rand_like(R)
        delta = low + noise * (high - low)
        
    return delta * R


def compute_activation_matrix(R_noisy: torch.Tensor, W_g: torch.Tensor) -> torch.Tensor:
    """
    Computes intermediate activation matrix M in R^{n x n} (Eq. 1, 2 & Figure 8).
    M[i, j] = || R_noisy[i] * W_g[j] ||_2
    """
    acts = torch.einsum('jDd,id->ijD', W_g, R_noisy)
    M = torch.norm(acts, dim=-1) # (n, n)
    return M


def erc_loss_fn(M: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    Computes ERC Loss from activation matrix M (Eq. 3 & Figure 8).
    L_ERC = (1 / n^2) * sum_{i=1}^n sum_{j != i} [
        max(M[i, j] - alpha * M[i, i], 0) + max(M[j, i] - alpha * M[i, i], 0)
    ]
    """
    n = M.size(0)
    diag_M = torch.diag(M)

    # Constraint 1 (Intra-expert)
    row_diff = M - alpha * diag_M.unsqueeze(1)
    row_diff_clamped = torch.clamp(row_diff, min=0.0)

    # Constraint 2 (Inter-expert)
    col_diff = M - alpha * diag_M.unsqueeze(0)
    col_diff_clamped = torch.clamp(col_diff, min=0.0)

    mask = torch.ones_like(M) - torch.eye(n, device=M.device, dtype=M.dtype)
    total_diff = (row_diff_clamped + col_diff_clamped) * mask

    return total_diff.mean()


def compute_specialization_metric(M: torch.Tensor) -> float:
    """
    Computes Specialization Ratio: Mean(Diagonal M[i, i]) / Mean(Off-Diagonal M[i, j]).
    Values > 1.0 indicate strong router-expert alignment.
    """
    n = M.size(0)
    diag = torch.diag(M).mean().item()
    mask = torch.ones_like(M) - torch.eye(n, device=M.device)
    off_diag = (M * mask).sum().item() / (n * (n - 1) + 1e-8)
    return diag / (off_diag + 1e-8)

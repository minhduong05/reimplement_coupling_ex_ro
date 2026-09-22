"""
Model 3: Autonomy-of-Experts (AoE Baseline).
Reference: "Autonomy-of-Experts Models" (Lv et al., ICML 2025 / arXiv:2407.02568)
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OlmoeConfig, OlmoeForCausalLM

from .config import create_fair_moe_config


def calculate_aoe_rank(d: int, D: int) -> int:
    """Computes factorization rank r = (D * d) / (d + D) to keep parameters consistent."""
    return round((D * d) / (d + D))


class AoELayer(nn.Module):
    """
    Autonomy-of-Experts Block (Algorithm 2 & Eq. 6 in Lv et al., 2025).
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        d_low: int = None,
        aux_loss_coef: float = 0.01
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.d_low = d_low if d_low is not None else calculate_aoe_rank(hidden_size, intermediate_size)
        self.aux_loss_coef = aux_loss_coef
        self.last_aux_loss = torch.tensor(0.0)

        # Combined W_down projection for all n experts: (hidden_size, n * d_low) (Eq. 4)
        self.W_down_hat = nn.Parameter(torch.empty(hidden_size, num_experts * self.d_low))
        self.W_up = nn.Parameter(torch.empty(num_experts, self.d_low, intermediate_size))
        self.W_p = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.W_o = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for p in [self.W_down_hat, self.W_up, self.W_p, self.W_o]:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.size(0)

        # 1. Combined W_down projection (Eq. 4)
        C_raw = torch.matmul(x_flat, self.W_down_hat)
        C = C_raw.view(num_tokens, self.num_experts, self.d_low)

        # 2. L2 norm ranking (Algorithm 2 line 6-7)
        p = torch.norm(C, p=2, dim=-1)
        topk_p, topk_indices = torch.topk(p, self.top_k, dim=-1)

        # 3. Softmax over Top-K values (Algorithm 2 line 8)
        p_hat = F.softmax(topk_p, dim=-1)

        # 4. Load balancing loss (Eq. 6 in AoE Paper)
        if self.training and self.aux_loss_coef > 0:
            expert_mask = torch.zeros(num_tokens, self.num_experts, device=x.device)
            expert_mask.scatter_(1, topk_indices, 1.0)
            f = expert_mask.mean(dim=0)
            full_P = F.softmax(p, dim=-1).mean(dim=0)
            self.last_aux_loss = self.aux_loss_coef * self.num_experts * torch.sum(f * full_P)
        else:
            self.last_aux_loss = torch.tensor(0.0, device=x.device)

        # 5. Execute only for chosen Top-K experts (Algorithm 2 lines 9-13)
        out_flat = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            pos_mask = (topk_indices == expert_idx)
            if not pos_mask.any():
                continue
            token_mask = pos_mask.any(dim=-1)
            selected_tokens = x_flat[token_mask]
            selected_C = C[token_mask, expert_idx]

            gate = F.silu(torch.matmul(selected_C, self.W_up[expert_idx]))
            up = torch.matmul(selected_tokens, self.W_p[expert_idx])
            activated = gate * up
            expert_out = torch.matmul(activated, self.W_o[expert_idx])

            weights = (p_hat * pos_mask.float()).sum(dim=-1)[token_mask].unsqueeze(-1)
            out_flat[token_mask] = out_flat[token_mask] + weights * expert_out

        out = out_flat.view(orig_shape)
        return out


def build_aoe_model(config: OlmoeConfig = None, vocab_size: int = 50304) -> OlmoeForCausalLM:
    """Constructs an OlmoeForCausalLM model with all FFN blocks replaced by AoELayer."""
    if config is None:
        config = create_fair_moe_config(vocab_size=vocab_size)
    model = OlmoeForCausalLM(config)
    for layer in model.model.layers:
        layer.mlp = AoELayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            aux_loss_coef=config.router_aux_loss_coef
        )
    return model

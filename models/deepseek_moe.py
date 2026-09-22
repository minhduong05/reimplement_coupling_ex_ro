"""
Model 4: DeepSeek-MoE Architecture (Shared + Routed Experts).
Reference: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts" (Dai et al., 2024)
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OlmoeConfig, OlmoeForCausalLM

from .config import create_fair_moe_config


class SingleSwiGLUExpert(nn.Module):
    """Single expert using standard SwiGLU FFN."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoELayer(nn.Module):
    """
    DeepSeek-MoE block with 1 Shared Expert + (num_experts - 1) Routed Experts.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 16,
        top_k: int = 1, # 1 Shared + 1 Routed = 2 active experts per token
        aux_loss_coef: float = 0.01
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_routed_experts = num_experts - 1
        self.top_k = min(top_k, self.num_routed_experts)
        self.aux_loss_coef = aux_loss_coef
        self.last_aux_loss = torch.tensor(0.0)

        # 1. Shared Expert (always active)
        self.shared_expert = SingleSwiGLUExpert(hidden_size, intermediate_size)

        # 2. Routed Experts
        self.routed_experts = nn.ModuleList([
            SingleSwiGLUExpert(hidden_size, intermediate_size)
            for _ in range(self.num_routed_experts)
        ])

        # 3. Router for routed experts
        self.gate = nn.Linear(hidden_size, self.num_routed_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.size(0)

        # 1. Compute Shared Expert output
        shared_out = self.shared_expert(x_flat)

        # 2. Router for routed experts
        router_logits = self.gate(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 3. Load balancing loss on routed experts
        if self.training and self.aux_loss_coef > 0:
            expert_mask = torch.zeros(num_tokens, self.num_routed_experts, device=x.device)
            expert_mask.scatter_(1, topk_indices, 1.0)
            f = expert_mask.mean(dim=0)
            P = router_probs.mean(dim=0)
            self.last_aux_loss = self.aux_loss_coef * self.num_routed_experts * torch.sum(f * P)
        else:
            self.last_aux_loss = torch.tensor(0.0, device=x.device)

        # 4. Dispatch to selected routed experts
        routed_out = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_routed_experts):
            pos_mask = (topk_indices == expert_idx)
            if not pos_mask.any():
                continue
            token_mask = pos_mask.any(dim=-1)
            selected_tokens = x_flat[token_mask]
            expert_out = self.routed_experts[expert_idx](selected_tokens)
            weights = (topk_weights * pos_mask.float()).sum(dim=-1)[token_mask].unsqueeze(-1)
            routed_out[token_mask] = routed_out[token_mask] + weights * expert_out

        out = (shared_out + routed_out).view(orig_shape)
        return out


def build_deepseek_moe_model(config: OlmoeConfig = None, vocab_size: int = 50304) -> OlmoeForCausalLM:
    """Constructs an OlmoeForCausalLM model with DeepSeek-MoE layers (1 Shared + 15 Routed)."""
    if config is None:
        config = create_fair_moe_config(vocab_size=vocab_size)
    model = OlmoeForCausalLM(config)
    for layer in model.model.layers:
        layer.mlp = DeepSeekMoELayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            top_k=1, # 1 Shared + 1 Routed = 2 active experts
            aux_loss_coef=config.router_aux_loss_coef
        )
    return model

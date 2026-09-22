"""
Model 1: MoE with Expert-Router Coupling (ERC) Loss.
Reference: "Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss" (ICLR 2026)
"""

import types
from typing import Tuple
import torch
import torch.nn as nn
from transformers import OlmoeConfig, OlmoeForCausalLM
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from utils.erc_loss import get_noisy_router, compute_activation_matrix, erc_loss_fn
from .config import create_fair_moe_config


def compute_block_erc_loss(moe_block: nn.Module, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes ERC loss for a single OlmoeSparseMoeBlock."""
    R = moe_block.gate.weight
    intermediate_size = moe_block.experts.gate_up_proj.shape[1] // 2
    W_g = moe_block.experts.gate_up_proj[:, :intermediate_size, :]
    
    R_noisy = get_noisy_router(R)
    M = compute_activation_matrix(R_noisy, W_g)
    loss = erc_loss_fn(M, alpha=alpha)
    return loss, M


def build_erc_moe_model(
    config: OlmoeConfig = None,
    vocab_size: int = 50304,
    alpha: float = 1.0,
    erc_weight: float = 1.0,
    record_matrices: bool = True
) -> OlmoeForCausalLM:
    """Constructs and patches an OlmoeForCausalLM model with ERC auxiliary loss."""
    if config is None:
        config = create_fair_moe_config(vocab_size=vocab_size)
        
    model = OlmoeForCausalLM(config)
    model._erc_alpha = alpha
    model._erc_weight = erc_weight
    model._record_matrices = record_matrices
    model._last_erc_matrices = []
    model._last_erc_loss = torch.tensor(0.0)

    original_forward = model.forward

    def patched_forward(self, *args, **kwargs) -> MoeCausalLMOutputWithPast:
        if self.training and "output_router_logits" not in kwargs:
            kwargs["output_router_logits"] = True

        outputs: MoeCausalLMOutputWithPast = original_forward(*args, **kwargs)

        erc_loss_total = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        layer_matrices = []

        if self.training:
            num_moe_layers = 0
            for layer in self.model.layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts"):
                    layer_loss, M = compute_block_erc_loss(layer.mlp, alpha=self._erc_alpha)
                    erc_loss_total = erc_loss_total + layer_loss
                    num_moe_layers += 1
                    if self._record_matrices:
                        layer_matrices.append(M.detach().cpu())

            if num_moe_layers > 0:
                erc_loss_total = erc_loss_total / num_moe_layers

            self._last_erc_loss = erc_loss_total.detach()
            if self._record_matrices:
                self._last_erc_matrices = layer_matrices

            if outputs.loss is not None:
                outputs.loss = outputs.loss + self._erc_weight * erc_loss_total

        return outputs

    model.forward = types.MethodType(patched_forward, model)
    return model

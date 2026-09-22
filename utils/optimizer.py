"""
Optimizer and Scheduler Utilities for Fair LLM / MoE Pretraining.
Reference: Section 4.1 of ICLR 2026 Paper
"""

import math
import torch
import torch.nn as nn


def configure_adamw_optimizer(
    model: nn.Module,
    lr: float = 4e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95)
) -> torch.optim.AdamW:
    """
    Constructs AdamW optimizer with proper parameter separation:
    - 2D+ tensors (Linear weights, Embeddings): weight_decay = 0.1
    - 1D tensors (LayerNorm/RMSNorm scales, biases): weight_decay = 0.0
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "norm" in name.lower() or "bias" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)


def get_cosine_schedule_with_min_lr(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    peak_lr: float = 4e-4,
    min_lr: float = 4e-5
):
    """
    Cosine learning rate schedule decaying from peak_lr to min_lr (e.g., 4e-4 -> 4e-5).
    """
    min_ratio = min_lr / peak_lr

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

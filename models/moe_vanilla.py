"""
Model 2: Vanilla MoE (Switch Transformer Baseline).
Standard Top-K Router with Switch Load Balancing Loss (Fedus et al., 2022).
"""

from transformers import OlmoeConfig, OlmoeForCausalLM
from .config import create_fair_moe_config


def build_vanilla_moe_model(config: OlmoeConfig = None, vocab_size: int = 50304) -> OlmoeForCausalLM:
    """Constructs a standard Vanilla MoE model with Load Balancing Loss."""
    if config is None:
        config = create_fair_moe_config(vocab_size=vocab_size)
    model = OlmoeForCausalLM(config)
    return model

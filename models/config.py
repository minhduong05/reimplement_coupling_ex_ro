"""
Shared Model Architecture Configuration.
Ensures identical capacity, depth, and sparsity across all 4 models.
"""

from transformers import OlmoeConfig


def create_fair_moe_config(vocab_size: int = 50304) -> OlmoeConfig:
    """
    Standard Fair Configuration (Total 110.29M parameters, Active ~35.4M):
    - 8 Transformer decoder layers
    - hidden_size = 512
    - intermediate_size = 256
    - 8 attention heads
    - 16 total experts
    - 2 active experts per token (Top-2, 12.5% sparsity)
    - Context length: 512 tokens
    """
    return OlmoeConfig(
        num_hidden_layers=8,
        hidden_size=512,
        intermediate_size=256,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_experts=16,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
        vocab_size=vocab_size,
        max_position_embeddings=1024,
    )

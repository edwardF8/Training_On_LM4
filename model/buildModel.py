"""Model builders for the Capo (bioS) experiments.

Both `create_llama_model` and `create_gpt2_model` return a HuggingFace
`LlamaForCausalLM` — i.e. RoPE-equipped, RMSNorm, weight-tied — and differ
only in the MLP block:

    create_llama_model  → gated MLP, intermediate_size = 8d/3   (Llama-style)
    create_gpt2_model   → standard MLP, intermediate_size = 4d  (GPT2-style)

Both shapes give ~8d² params per MLP layer, so parameter counts match at a
given (layers, heads, hidden) triple. This matches the "Llama(RoPE)" vs
"GPT2(RoPE)" distinction from Allen-Zhu (2025), *Physics of Language Models:
Part 4.1*, Appendix C.

The vanilla HuggingFace GPT2 architecture (absolute positional embeddings,
no RoPE) is intentionally *not* used — the Physics of LM papers (Parts 3.1,
3.3, 4.1) all swap GPT2's positional embeddings for RoPE before training,
and we mirror that here by routing GPT2 through the Llama backbone.
"""

import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM


class StandardMLP(nn.Module):
    """Non-gated two-layer MLP: ``down_proj(SiLU(up_proj(x)))``.

    Replaces HF's gated `LlamaMLP` (which computes
    ``down_proj(SiLU(gate_proj(x)) * up_proj(x))``) to recover GPT2-style
    MLP semantics while keeping the rest of the Llama backbone (RoPE,
    RMSNorm, weight tying) intact. Paired with ``intermediate_size = 4d``
    this gives the same 8d² param count per MLP as the gated variant with
    ``intermediate_size = 8d/3``.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


def create_llama_model(
    vocab_size: int,
    block_size: int,
    hidden_size: int,
    n_layer: int,
    n_head: int,
    eos_token: int,
):
    """Build a Llama(RoPE) model with **gated** MLPs (intermediate_size = 8d/3).

    `eos_token` must be a valid id inside `vocab_size` — pass
    `CONFIG.reducedEOSToken` (the post-vocab-remap id), not the raw GPT-2
    tokenizer's 50256.
    """
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=block_size,
        hidden_size=hidden_size,
        intermediate_size=(hidden_size * 8) // 3,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        use_cache=False,
        eos_token_id=eos_token,
        bos_token_id=eos_token,
    )
    return LlamaForCausalLM(cfg)


def create_gpt2_model(
    vocab_size: int,
    block_size: int,
    n_embd: int,
    n_layer: int,
    n_head: int,
    eos_token: int,
):
    """Build a GPT2(RoPE) model: Llama backbone with **standard** (non-gated) MLPs.

    Implementation follows Allen-Zhu (2025) Part 4.1, Appendix C:

        intermediate_size = 4 * d   →  ~8d² params per MLP layer

    so parameter counts line up with `create_llama_model` at the same
    (n_layer, n_head, n_embd). The HF gated `LlamaMLP` inside each block is
    swapped for `StandardMLP` after construction.

    `eos_token` must be a valid id inside `vocab_size` — pass
    `CONFIG.reducedEOSToken`, not the raw GPT-2 tokenizer's 50256.
    """
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=block_size,
        hidden_size=n_embd,
        intermediate_size=4 * n_embd,
        num_hidden_layers=n_layer,
        num_attention_heads=n_head,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        use_cache=False,
        eos_token_id=eos_token,
        bos_token_id=eos_token,
    )
    model = LlamaForCausalLM(cfg)
    for layer in model.model.layers:
        layer.mlp = StandardMLP(cfg)
    return model

from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM
from config import Config

def create_gpt2_model(
    vocab_size: int = 3300,
    block_size: int = 512,
    n_embd: int = 512,
    n_layer: int = 8,
    n_head: int = 8,
    eos_token : int = 50257):
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size, #max sequence length
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        tie_word_embeddings=True,
        bos_token_id=eos_token,
        eos_token_id=eos_token,
    )

    return GPT2LMHeadModel(cfg)

def create_llama_model(
    vocab_size: int = 3300,
    block_size: int = 512,
    hidden_size: int = 512,
    n_layer: int = 8,
    n_head: int = 8,
    eos_token : int = 50257):
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
        bos_token_id=eos_token
    )

    return LlamaForCausalLM(cfg)
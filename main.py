"""Train Capo (bioS) models of varying sizes from scratch.

All artifacts for a single experiment are namespaced under CONFIG.NAME:

    cache/{NAME}/
        people.json          — sampled person dicts (deterministic from SEED)
        data_config.json     — full Config used for this run
        bios_prereduce.bin   — raw GPT-2 token ids (memmap)
        bios_postreduce.bin  — same tokens after reduced-vocab remap

    runs/{NAME}/
        {model_name}/        — HF Trainer checkpoints for one model size

Change CONFIG.NAME to start a clean experiment without touching anything else.
"""

import json
from pathlib import Path

from transformers import GPT2Tokenizer

from config import Config
from data.sample_people import sample_people
from data.bio_text import bio_stream
from data.tokenize_pack import (
    tokenize_and_pack,
    PackedTokenDataset,
    build_vocab_remap,
    remap_token_file,
)
from model.buildModel import create_gpt2_model, create_llama_model
from model.trainModel import train


# ---------------------------
# CONFIG
# ---------------------------

CONFIG = Config()

# Experiment name — drives cache/{NAME}/ and runs/{NAME}/. Bump for a fresh
# experiment so you don't clobber prior artifacts.
CONFIG.NAME = "default"

CONFIG.SEED         = 0
CONFIG.SHUFFLE_SEED = 1

# Data
CONFIG.N       = 50_000
CONFIG.K       = 100
CONFIG.SEQ_LEN = 512

# Derived paths — everything data-side under cache/{NAME}/.
DATA_DIR = Path("cache") / CONFIG.NAME
DATA_DIR.mkdir(parents=True, exist_ok=True)

PEOPLE_PATH      = DATA_DIR / "people.json"
DATA_CONFIG_PATH = DATA_DIR / "data_config.json"
CONFIG.PRE_REDUCE_PATH  = str(DATA_DIR / "bios_prereduce.bin")
CONFIG.POST_REDUCE_PATH = str(DATA_DIR / "bios_postreduce.bin")

# Model family applied to every entry in MODEL_CONFIGS.
CONFIG.MODEL_TYPE = "llama"

# Training
CONFIG.BATCH_SIZE   = 12
CONFIG.LR           = 1e-3
CONFIG.WEIGHT_DECAY = 0.01
CONFIG.WARMUP_STEPS = 1000
CONFIG.MAX_STEPS    = 80_000
CONFIG.GRAD_CLIP    = 1.0

MODEL_CONFIGS = [
    {"name": "2L-192D", "numLayers": 2, "dmodel": 192, "numHeads": 3},
    {"name": "4L-128D", "numLayers": 4, "dmodel": 128, "numHeads": 2},
    {"name": "6L-192D", "numLayers": 6, "dmodel": 192, "numHeads": 3},
]


# ---------------------------
# DATA
# ---------------------------

# Sample people (deterministic in SEED) and persist for later mechinterp.
people = sample_people(N=CONFIG.N, seed=CONFIG.SEED)
with open(PEOPLE_PATH, "w") as f:
    json.dump(people, f)
print(f"Saved {len(people):,} people → {PEOPLE_PATH}")

stream = bio_stream(
    people, K=CONFIG.K,
    master_seed=CONFIG.SEED, shuffle_seed=CONFIG.SHUFFLE_SEED,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
CONFIG.eosToken   = tokenizer.eos_token_id   # 50256
CONFIG.vocab_size = tokenizer.vocab_size     # 50257

n_tokens, n_seq = tokenize_and_pack(
    tokenizer,
    stream,
    n_bios_total=CONFIG.N * CONFIG.K,
    out_path=CONFIG.PRE_REDUCE_PATH,
    seq_len=CONFIG.SEQ_LEN,
)
print(f"Wrote {n_tokens:,} tokens → {n_seq:,} sequences of {CONFIG.SEQ_LEN}.")

# Reduce GPT-2's 50k vocab to only the ~3.3k tokens that appear in bioS.
old_to_new, _, CONFIG.reducedVocabSize = build_vocab_remap(CONFIG.PRE_REDUCE_PATH)
remap_token_file(CONFIG.PRE_REDUCE_PATH, CONFIG.POST_REDUCE_PATH, old_to_new)
CONFIG.reducedEOSToken = old_to_new[int(CONFIG.eosToken)]
print(f"Reduced vocab size: {CONFIG.reducedVocabSize}")

# Persist the fully-populated Config so eval/probing tools can rehydrate it.
CONFIG.save(str(DATA_CONFIG_PATH))
print(f"Saved config → {DATA_CONFIG_PATH}")

ds = PackedTokenDataset(CONFIG.POST_REDUCE_PATH, seq_len=CONFIG.SEQ_LEN)
print(f"Dataset has {len(ds):,} sequences.")


# ---------------------------
# TRAIN
# ---------------------------

for mc in MODEL_CONFIGS:
    CONFIG.numLayers = mc["numLayers"]
    CONFIG.dmodel    = mc["dmodel"]
    CONFIG.numHeads  = mc["numHeads"]

    if CONFIG.MODEL_TYPE == "llama":
        model = create_llama_model(
            CONFIG.reducedVocabSize, CONFIG.SEQ_LEN,
            CONFIG.dmodel, CONFIG.numLayers, CONFIG.numHeads,
            CONFIG.reducedEOSToken,
        )
    elif CONFIG.MODEL_TYPE == "gpt2":
        model = create_gpt2_model(
            CONFIG.reducedVocabSize, CONFIG.SEQ_LEN,
            CONFIG.dmodel, CONFIG.numLayers, CONFIG.numHeads,
            CONFIG.reducedEOSToken,
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {CONFIG.MODEL_TYPE!r}")

    out_dir = f"runs/{CONFIG.NAME}/{mc['name']}"
    print(f"\n=== Training {mc['name']} → {out_dir} ===")
    train(model, ds, CONFIG, output_dir=out_dir)
    print(f"Done. Checkpoints under {out_dir}/")

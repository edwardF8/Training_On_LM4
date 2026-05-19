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
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import wandb
import wandb.util  # `wandb.util.generate_id` isn't re-exported on the top-level module
from transformers import GPT2Tokenizer

from config import Config
from data.sample_people import sample_people
from data.bio_text import bio_stream, render_bio
from data.tokenize_pack import (
    tokenize_and_pack,
    PackedTokenDataset,
    build_vocab_remap,
    remap_token_file,
    assert_tokens_in_remap,
)
from model.buildModel import create_gpt2_model, create_llama_model
from model.trainModel import train
from eval.sequential_probe import run_probe


# ---------------------------
# CONFIG
# ---------------------------

CONFIG = Config()

# Experiment name — drives cache/{NAME}/ and runs/{NAME}/. Bump for a fresh
# experiment so you don't clobber prior artifacts.
CONFIG.NAME = "default_3eps"

# Unique stamp for this invocation of main.py. Used in wandb run names and
# the runs/ subdirectory so multiple invocations don't collide. Cache/ is
# *not* stamped — data is shared across invocations and only regenerated
# when CONFIG.NAME changes. Override INVOCATION below to use a custom tag
# (e.g. "before-bugfix") instead of the timestamp.
INVOCATION = datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"INVOCATION = {INVOCATION}")

CONFIG.SEED         = 0
CONFIG.SHUFFLE_SEED = 1

# Data
CONFIG.N       = 50_000
CONFIG.K       = 500
CONFIG.SEQ_LEN = 512

# Bio contents. Set to e.g.
#   ("birthday", "birthcity", "university", "field", "company_city", "company_name")
# for the legacy 6-field Capo bioS layout.
CONFIG.FIELDS  = ("birthday",)

# Derived paths — everything data-side under cache/{NAME}/.
DATA_DIR = Path("cache") / CONFIG.NAME
DATA_DIR.mkdir(parents=True, exist_ok=True)

PEOPLE_PATH      = DATA_DIR / "people.json"
DATA_CONFIG_PATH = DATA_DIR / "data_config.json"
REMAP_PATH       = DATA_DIR / "old_to_new.json"
CONFIG.PRE_REDUCE_PATH  = str(DATA_DIR / "bios_prereduce.bin")
CONFIG.POST_REDUCE_PATH = str(DATA_DIR / "bios_postreduce.bin")

# Model family applied to every entry in MODEL_CONFIGS.
CONFIG.MODEL_TYPE = "llama"

# Training
CONFIG.BATCH_SIZE   = 12
CONFIG.LR           = 1e-3
CONFIG.WEIGHT_DECAY = 0.01
CONFIG.WARMUP_STEPS = 1000
CONFIG.GRAD_CLIP    = 1.0

# One epoch = one full pass over the packed dataset, which renders every
# person K times (see bio_stream). So EPOCHS × K = exposures per person:
# the paper's "100 exposures" recipe is EPOCHS=1 at K=100; bump for
# memorization-leaning interp runs (6 ≈ 600 exposures).
CONFIG.EPOCHS = 1.0

# Small-model sweep matching the paper's `ℓ-h` notation
# (Allen-Zhu & Li, PhysicsForLM_4_1.pdf Appendix C + Figure 11):
#   ℓ layers, hidden = 64 · h, h heads.
# These are the small end of the Capo capacity-scaling sweep — all under
# ~3M params so training stays fast on a single GPU and the attention
# patterns are small enough for clean mechinterp.
MODEL_CONFIGS = [
    {"name": "2-3", "numLayers": 2, "dmodel": 192, "numHeads": 3},   # ~0.4M params
    {"name": "4-2", "numLayers": 4, "dmodel": 128, "numHeads": 2},   # ~0.5M params
    {"name": "4-3", "numLayers": 4, "dmodel": 192, "numHeads": 3},   # ~1.1M params
    {"name": "5-3", "numLayers": 5, "dmodel": 192, "numHeads": 3},   # ~1.4M params
    {"name": "6-3", "numLayers": 6, "dmodel": 192, "numHeads": 3},   # ~1.7M params
    {"name": "8-2", "numLayers": 8, "dmodel": 128, "numHeads": 2},   # ~1.1M params
]


# ---------------------------
# DATA
# ---------------------------

# Sample people (deterministic in SEED) .
people = sample_people(N=CONFIG.N, seed=CONFIG.SEED)
with open(PEOPLE_PATH, "w") as f:
    json.dump(people, f)
print(f"Saved {len(people):,} people → {PEOPLE_PATH}")

stream = bio_stream(
    people, K=CONFIG.K,
    master_seed=CONFIG.SEED, shuffle_seed=CONFIG.SHUFFLE_SEED,
    fields=tuple(CONFIG.FIELDS),
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

# Persist the GPT-2 → reduced id mapping. Without this, a trained checkpoint
# is unusable downstream — embedding id N has no meaning outside this dict.
with open(REMAP_PATH, "w") as f:
    json.dump({str(k): int(v) for k, v in old_to_new.items()}, f)
print(f"Saved vocab remap → {REMAP_PATH}")

# Seatbelt: every (template, person) rendering we'll evaluate on must be
# expressible in the reduced vocab. Sample a few people × every exposure
# index in [0, K) for each enabled field — that exhausts the template pool
# via the round-robin schedule render_bio uses.
sample_prompts = [
    render_bio(people[i], exposure_idx=e, fields=tuple(CONFIG.FIELDS))
    for i in range(min(5, len(people)))
    for e in range(CONFIG.K)
]
assert_tokens_in_remap(old_to_new, sample_prompts, tokenizer)
print(f"Verified all {len(sample_prompts):,} sample prompts tokenize cleanly post-remap.")

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
            seed=CONFIG.SEED,
        )
    elif CONFIG.MODEL_TYPE == "gpt2":
        model = create_gpt2_model(
            CONFIG.reducedVocabSize, CONFIG.SEQ_LEN,
            CONFIG.dmodel, CONFIG.numLayers, CONFIG.numHeads,
            CONFIG.reducedEOSToken,
            seed=CONFIG.SEED,
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {CONFIG.MODEL_TYPE!r}")

    out_dir = f"runs/{CONFIG.NAME}/{INVOCATION}/{mc['name']}"

    # Explicitly create the wandb run BEFORE train(). HF Trainer's WandbCallback
    # does `if wandb.run is None: wandb.init(...)`, so it joins this active run
    # instead of starting its own. `reinit="finish_previous"` forces wandb to
    # close any prior run state so each model genuinely gets a fresh run id,
    # even if env vars / wandb caches are sticky.
    wandb_run_id = wandb.util.generate_id()
    print(f"\n=== Training {mc['name']} → {out_dir} (wandb id {wandb_run_id}) ===")
    wandb.init(
        id=wandb_run_id,
        name=f"{mc['name']}-{INVOCATION}",
        group=INVOCATION,
        reinit="finish_previous",
        config={
            **asdict(CONFIG),
            "model_label": mc["name"],
            "INVOCATION": INVOCATION,
        },
    )

    train(model, ds, CONFIG, output_dir=out_dir)
    print(f"Done. Checkpoints under {out_dir}/")

    # Birthday memorization probe on the freshly-trained model.
    print(f"\n=== Probing {mc['name']} ===")
    results = run_probe(
        model, tokenizer, old_to_new, people,
        max_people=50,
    )
    probe_path = Path(out_dir) / "final" / "probe_results.json"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(probe_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved probe results → {probe_path}")

    # The wandb run we created before train() is still active here.
    wandb.log({
        "probe/MP":     results["macro"]["MP"],
        "probe/DayM":   results["macro"]["DayM"],
        "probe/YearMD": results["macro"]["YearMD"],
        "probe/FP":     results["macro"]["FP"],
    })
    per_t_table = wandb.Table(columns=["template_idx", "MP", "DayM", "YearMD", "FP"])
    for t_idx, accs in sorted(results["per_template"].items()):
        per_t_table.add_data(int(t_idx), accs["MP"], accs["DayM"], accs["YearMD"], accs["FP"])
    wandb.log({"probe/per_template": per_t_table})

    # Close the run so the next iteration's wandb.init starts cleanly.
    wandb.finish()

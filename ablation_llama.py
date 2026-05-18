"""Ablation sweep over Capo bioS models.

Same pipeline as main.py, but instead of MODEL_CONFIGS we expand a base
architecture against three one-axis sweeps (epochs, layers, heads). To add
a new study or change a sweep, edit BASE / ABLATIONS below — nothing else
needs to change.

Output layout:
    runs/{NAME}/{INVOCATION}/{study}/{run_name}/
wandb groups every run from one invocation by study, so the UI's group
view shows three sweep curves side-by-side.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import wandb
import wandb.util
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
from eval.birthday_probe import run_probe


# ---------------------------
# CONFIG
# ---------------------------

CONFIG = Config()

# Separate from main.py's "default" experiment so the two don't share a
# runs/ subdirectory. Cache is also separate (regenerated once at first run).
CONFIG.NAME = "bioS_name_date_small_lama_epoch_scale"

INVOCATION = datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"INVOCATION = {INVOCATION}")

# wandb umbrella for this whole ablation sweep. Every run below uses this as
# its `group`, and `job_type` is set to the study (epochs/layers/heads), so
# the UI collapses all 14 runs under one entry and lets you split by study.
SWEEP_NAME = f"{CONFIG.NAME}-{INVOCATION}"

CONFIG.SEED         = 0
CONFIG.SHUFFLE_SEED = 1

# Data
CONFIG.N       = 50_000
CONFIG.K       = 100
CONFIG.SEQ_LEN = 512
CONFIG.FIELDS  = ("birthday",)

# Derived paths
DATA_DIR = Path("cache") / CONFIG.NAME
DATA_DIR.mkdir(parents=True, exist_ok=True)

PEOPLE_PATH      = DATA_DIR / "people.json"
DATA_CONFIG_PATH = DATA_DIR / "data_config.json"
CONFIG.PRE_REDUCE_PATH  = str(DATA_DIR / "bios_prereduce.bin")
CONFIG.POST_REDUCE_PATH = str(DATA_DIR / "bios_postreduce.bin")

# Training (shared across all ablation runs)
CONFIG.MODEL_TYPE   = "llama"
CONFIG.BATCH_SIZE   = 12
CONFIG.LR           = 1e-3
CONFIG.WEIGHT_DECAY = 0.01
CONFIG.WARMUP_STEPS = 1000
CONFIG.GRAD_CLIP    = 1.0


# ---------------------------
# ABLATION SWEEP — edit me
# ---------------------------
#
# BASE is the architecture every study starts from; each ABLATIONS entry
# overrides exactly one field and sweeps it. dmodel is computed from
# numHeads via the paper's ℓ-h convention (dmodel = 64 · numHeads).
#
# Per-study overrides: add an optional "base" dict to any ABLATIONS entry
# to overlay on top of the global BASE for that study only. Useful when a
# sweep needs different fixed hyperparams (e.g. more EPOCHS to give small
# capacity models a fair chance to memorize).

BASE = {
    "numLayers": 4,
    "numHeads":  3,
    "EPOCHS":    4,
}

ABLATIONS = {
    "16EP_layer": {"axis": "numLayers", "values": [4, 6, 8, 12], "base": {"EPOCHS": 16}},
    "16EP_heads": {"axis": "numLayers", "values": [4, 6, 8, 12], "base": {"EPOCHS": 16}},
    "memorizing_attempts": {"axis": "numLayers", "values": [6, 8, 12], "base": {"EPOCHS": 16, "numHeads":6}},

}
#Format:   "numLayers": {"axis": "numLayers", "values": [4, 6, 8, 12], "base": {"EPOCHS": 16}},


def build_runs(base, ablations):
    """Expand BASE × ABLATIONS into a flat list of run specs.

    Each ablation may set "base" to override BASE for that study only;
    the swept axis value then overrides on top of that.
    """
    runs = []
    for study, spec in ablations.items():
        axis, values = spec["axis"], spec["values"]
        study_base = {**base, **spec.get("base", {})}
        for v in values:
            cfg = {**study_base, axis: v}
            cfg["dmodel"] = 64 * cfg["numHeads"]   # paper ℓ-h convention
            runs.append({
                "study": study,
                "name":  f"{study}-{v}",
                **cfg,
            })
    return runs


RUNS = build_runs(BASE, ABLATIONS)
print(f"Planned {len(RUNS)} runs across {len(ABLATIONS)} studies:")
for r in RUNS:
    print(f"  [{r['study']:>6}] {r['name']:<12} "
          f"L={r['numLayers']} H={r['numHeads']} D={r['dmodel']} E={r['EPOCHS']}")


# ---------------------------
# DATA (generated once, shared by every run)
# ---------------------------

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
CONFIG.eosToken   = tokenizer.eos_token_id
CONFIG.vocab_size = tokenizer.vocab_size

n_tokens, n_seq = tokenize_and_pack(
    tokenizer, stream,
    n_bios_total=CONFIG.N * CONFIG.K,
    out_path=CONFIG.PRE_REDUCE_PATH,
    seq_len=CONFIG.SEQ_LEN,
)
print(f"Wrote {n_tokens:,} tokens → {n_seq:,} sequences of {CONFIG.SEQ_LEN}.")

old_to_new, _, CONFIG.reducedVocabSize = build_vocab_remap(CONFIG.PRE_REDUCE_PATH)
remap_token_file(CONFIG.PRE_REDUCE_PATH, CONFIG.POST_REDUCE_PATH, old_to_new)
CONFIG.reducedEOSToken = old_to_new[int(CONFIG.eosToken)]
print(f"Reduced vocab size: {CONFIG.reducedVocabSize}")

sample_prompts = [
    render_bio(people[i], exposure_idx=e, fields=tuple(CONFIG.FIELDS))
    for i in range(min(5, len(people)))
    for e in range(CONFIG.K)
]
assert_tokens_in_remap(old_to_new, sample_prompts, tokenizer)
print(f"Verified all {len(sample_prompts):,} sample prompts tokenize cleanly post-remap.")

CONFIG.save(str(DATA_CONFIG_PATH))
print(f"Saved config → {DATA_CONFIG_PATH}")

ds = PackedTokenDataset(CONFIG.POST_REDUCE_PATH, seq_len=CONFIG.SEQ_LEN)
print(f"Dataset has {len(ds):,} sequences.")


# ---------------------------
# SWEEP
# ---------------------------

for run in RUNS:
    CONFIG.numLayers = run["numLayers"]
    CONFIG.numHeads  = run["numHeads"]
    CONFIG.dmodel    = run["dmodel"]
    CONFIG.EPOCHS    = run["EPOCHS"]

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

    out_dir = f"runs/{CONFIG.NAME}/{INVOCATION}/{run['study']}/{run['name']}"

    wandb_run_id = wandb.util.generate_id()
    print(f"\n=== Training {run['study']}/{run['name']} → {out_dir} "
          f"(wandb id {wandb_run_id}) ===")
    wandb.init(
        id=wandb_run_id,
        name=f"{run['name']}-{INVOCATION}",
        group=SWEEP_NAME,           # one umbrella for all 14 runs
        job_type=run["study"],      # sub-grouping by study within that umbrella
        tags=[run["study"], CONFIG.NAME],
        reinit="finish_previous",
        config={
            **asdict(CONFIG),
            "study":      run["study"],
            "run_name":   run["name"],
            "sweep_name": SWEEP_NAME,
            "INVOCATION": INVOCATION,
        },
    )

    train(model, ds, CONFIG, output_dir=out_dir)
    print(f"Done. Checkpoints under {out_dir}/")

    print(f"\n=== Probing {run['name']} ===")
    results = run_probe(
        model, tokenizer, old_to_new, people,
        max_people=50,
    )
    probe_path = Path(out_dir) / "final" / "probe_results.json"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(probe_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved probe results → {probe_path}")

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

    wandb.finish()

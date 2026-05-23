"""Ablation sweep over Capo bioS models — full-factorial GRID.

Same pipeline as `ablation_llama.py`, but `GRID` cross-products several
axes (numLayers × numHeads) into one big sweep instead of independent
single-axis sweeps. Run counts grow multiplicatively. Epochs are not a
swept axis: each run trains once to MAX_EPOCHS and the ProbeAtEpochs
callback probes along the way (see RUN_PROBE_AFTER).

For the single-axis version, see `ablation_llama.py`.

Output layout
-------------
    runs/{NAME}/{INVOCATION}/
        probe_curve.csv                        # one row per (run, probe epoch)
        {study}/{run_name}/
            ...                                # train checkpoints
            probes/probe_*_epoch{N}.json       # mid-training probe results
            final/probe_*.json                 # final-epoch probe results

Run names look like `grid-L4-H6` (encoded via AXIS_ABBREV below).

Probing strategies — see `ablation_llama.py` docstring for full details.
Pick any subset of {"birthday_legacy", "sequential", "separate"} via the
`PROBES` list near the top. Grid sweeps with multiple fields typically
want both `sequential` and `separate` so the cross-field vs. direct
memorization gap shows up.

wandb keys (per run, logged at every probe epoch by `eval/probes.py`)
---------------------------------------------------------------------
    birthdayProbe/{MP,DayM,YearMD,FP,LP}
    birthdayProbe/per_template               (table)
    sequentialProbe/<field>/TF
    sequentialProbe/per_template/<field>     (table)
    separateProbe/<field>/TF
    separateProbe/per_template/<field>       (table)
"""

import itertools
import json
import os
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
from eval.probe_callback import ProbeAtEpochs


# ---------------------------
# CONFIG
# ---------------------------

CONFIG = Config()

CONFIG.NAME = "bioS_N-Bd_final_grid"

# Set INVOCATION=<existing-timestamp> in the env to resume an interrupted sweep
# into the same runs/{NAME}/{INVOCATION}/ directory; otherwise a fresh one.
INVOCATION = os.environ.get("INVOCATION") or datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"INVOCATION = {INVOCATION}")

SWEEP_NAME = f"{CONFIG.NAME}-{INVOCATION}"

# Every (run, probe epoch) row from this sweep accumulates into one CSV.
CURVE_CSV = Path("runs") / CONFIG.NAME / INVOCATION / "probe_curve.csv"

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
CONFIG.BATCH_SIZE   = 24
CONFIG.LR           = 1e-3
CONFIG.WEIGHT_DECAY = 0.01
CONFIG.WARMUP_STEPS = 1000
CONFIG.GRAD_CLIP    = 1.0


# ---------------------------
# PROBE SELECTION
# ---------------------------
# Pick any subset of {"birthday_legacy", "sequential", "separate"}.
# Each probe lives in its own wandb namespace and JSON file (see eval/probes.py).

PROBES = ("birthday_legacy",)


# ---------------------------
# GRID SWEEP — edit me
# ---------------------------
#
# BASE supplies fallbacks for any axis the grid doesn't sweep. The grid
# values then cross-product to produce one run per combination.
# dmodel is computed from numHeads via the paper's ℓ-h convention
# (dmodel = 64 · numHeads).

# Each run trains once to MAX_EPOCHS; the ProbeAtEpochs callback probes the
# in-memory model after every epoch in RUN_PROBE_AFTER, plus at MAX_EPOCHS,
# so one run yields a whole probe-accuracy-vs-epoch curve.
MAX_EPOCHS      = 16
RUN_PROBE_AFTER = [1, 2, 4, 6, 8, 12]
PROBE_EPOCHS    = sorted({e for e in RUN_PROBE_AFTER if e <= MAX_EPOCHS}
                         | {MAX_EPOCHS})

BASE = {
    "numLayers": 4,
    "numHeads":  3,
}

GRID = {
    "numLayers": [1, 2, 4, 8],
    "numHeads":  [1, 2, 4, 6, 8],
}

# Short axis labels used in grid run names (e.g. grid-L4-H6-E16).
AXIS_ABBREV = {"numLayers": "L", "numHeads": "H", "EPOCHS": "E", "dmodel": "D"}


def build_runs(base, grid):
    """Cross-product `grid` axes; overlay each combo on `base`."""
    axes   = list(grid.keys())
    combos = itertools.product(*grid.values())
    runs = []
    for combo in combos:
        cfg = {**base, **dict(zip(axes, combo))}
        cfg["dmodel"] = 64 * cfg["numHeads"]   # paper ℓ-h convention
        short = "-".join(f"{AXIS_ABBREV.get(k, k)}{v}" for k, v in zip(axes, combo))
        runs.append({
            "study": "grid",
            "name":  f"grid-{short}",
            **cfg,
        })
    return runs


RUNS = build_runs(BASE, GRID)
print(f"Planned {len(RUNS)} grid runs "
      f"(MAX_EPOCHS={MAX_EPOCHS}, probe @ epochs {PROBE_EPOCHS}):")
for r in RUNS:
    print(f"  [{r['study']:>5}] {r['name']:<24} "
          f"L={r['numLayers']} H={r['numHeads']} D={r['dmodel']}")


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
    CONFIG.EPOCHS    = MAX_EPOCHS

    out_dir_path = Path(f"runs/{CONFIG.NAME}/{INVOCATION}/{run['study']}/{run['name']}")
    final_dir    = out_dir_path / "final"
    ckpts        = sorted(out_dir_path.glob("checkpoint-*"),
                          key=lambda p: int(p.name.split("-")[-1])) if out_dir_path.exists() else []

    if final_dir.exists():
        print(f"\n=== SKIP {run['study']}/{run['name']} — already completed "
              f"({final_dir} exists) ===")
        continue

    resume_from_checkpoint = bool(ckpts)
    if resume_from_checkpoint:
        print(f"\n=== RESUME {run['study']}/{run['name']} from {ckpts[-1].name} ===")

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

    out_dir = str(out_dir_path)

    wandb_run_id = wandb.util.generate_id()
    print(f"\n=== Training {run['study']}/{run['name']} → {out_dir} "
          f"(wandb id {wandb_run_id}) ===")
    wandb.init(
        id=wandb_run_id,
        name=f"{run['name']}-{INVOCATION}",
        group=SWEEP_NAME,
        job_type=run["study"],
        tags=[run["study"], CONFIG.NAME],
        reinit="finish_previous",
        config={
            **asdict(CONFIG),
            "study":      run["study"],
            "run_name":   run["name"],
            "sweep_name": SWEEP_NAME,
            "INVOCATION": INVOCATION,
            "probes":     list(PROBES),
        },
    )

    # Probe the in-memory model after each epoch in PROBE_EPOCHS (the final
    # one included). Per-epoch rows accumulate in CURVE_CSV across the grid.
    probe_cb = ProbeAtEpochs(
        probes=PROBES, tokenizer=tokenizer, old_to_new=old_to_new,
        people=people, fields=tuple(CONFIG.FIELDS),
        out_dir=out_dir, run_name=run["name"],
        probe_epochs=PROBE_EPOCHS, max_epochs=MAX_EPOCHS,
        csv_path=CURVE_CSV,
        run_info={"family": CONFIG.NAME, "study": run["study"],
                  "numLayers": CONFIG.numLayers, "numHeads": CONFIG.numHeads,
                  "dmodel": CONFIG.dmodel},
    )
    train(model, ds, CONFIG, output_dir=out_dir, callbacks=[probe_cb],
          resume_from_checkpoint=resume_from_checkpoint)
    print(f"Done. Checkpoints + per-epoch probes under {out_dir}/")

    wandb.finish()

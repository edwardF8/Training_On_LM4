"""RobustnessTest ablation sweep over Capo bioS models — full-factorial GRID.

Same pipeline as `ablation_llama_grid.py`, with one data-side twist
(RobustnessTest — docs/superpowers/specs/2026-06-10-robustness-test-design.md):

  - LIMITED_PEOPLE_FRAC of the N people are "limited": their K bios render
    from a per-person random subset of round(LIMITED_TEMPLATE_FRAC * T)
    templates (12 of the 46 birthday templates) instead of the full
    round-robin. Every person still gets exactly K bios, so corpus size and
    the reduced vocab match a normal run.
  - Which people are limited, and each one's template subset, is recorded in
    robustness_manifest.json (under cache/{NAME}/ and copied into
    runs/{NAME}/{INVOCATION}/).

Probing: ONLY the greedy + lenient judges (FP/LP), via the "birthday_robust"
probe (eval/robustness_probe.py) — it reuses birthday_probe_legacy.score_pair
and buckets pairs into total / limitedSet / fullSet / limitedSeen /
limitedUnseen. Eval set: first MAX_PEOPLE_PER_GROUP limited + first
MAX_PEOPLE_PER_GROUP full-set people, every one probed on all 46 templates.

Output layout
-------------
    runs/{NAME}/{INVOCATION}/
        probe_curve.csv                  # one row per (run, probe epoch):
                                         #   FP_/LP_ x total/limited/full/
                                         #   limitedSeen/limitedUnseen
        robustness_manifest.json         # who is limited + their templates
        {study}/{run_name}/
            ...                          # train checkpoints
            probes/probe_robust_epoch{N}.json
            final/probe_robust.json

wandb keys (per run, logged at every probe epoch)
-------------------------------------------------
    robustProbe/{FP,LP}/{total,limitedSet,fullSet,limitedSeen,limitedUnseen}
    robustProbe/per_template             (table)
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
from data.bio_text import render_bio
from data.robustness import (
    build_robustness_manifest,
    robust_bio_stream,
    save_manifest,
    verify_limited_rendering,
)
from data.tokenize_pack import (
    tokenize_and_pack,
    PackedTokenDataset,
    build_vocab_remap,
    remap_token_file,
    assert_tokens_in_remap,
)
from model.buildModel import create_gpt2_model, create_llama_model
from model.trainModel import train
from eval.probes import PROBE_REGISTRY
from eval.robustness_probe import make_robust_runner
from eval.robust_probe_callback import RobustProbeAtEpochs


# ---------------------------
# CONFIG
# ---------------------------

CONFIG = Config()

CONFIG.NAME = "bioS_N-Bd_robust_grid"

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

# RobustnessTest knobs
LIMITED_PEOPLE_FRAC   = 0.20  # fraction of people with restricted templates
LIMITED_TEMPLATE_FRAC = 0.25  # fraction of templates each limited person keeps
ROBUSTNESS_SEED       = 0     # seeds the manifest sampling
MAX_PEOPLE_PER_GROUP  = 50    # probe eval people per group (limited / full).
                              # Matches the non-robustness birthday probe's
                              # max_people=50 (each group sized like that probe);
                              # 50+50 -> 4,600 pairs/probe. The probe decode is
                              # now KV-cached + batched, so probing is no longer
                              # the sweep's bottleneck — see
                              # docs/superpowers/specs/2026-06-10-robustness-probe-optimization.md

# Derived paths
DATA_DIR = Path("cache") / CONFIG.NAME
DATA_DIR.mkdir(parents=True, exist_ok=True)

PEOPLE_PATH       = DATA_DIR / "people.json"
DATA_CONFIG_PATH  = DATA_DIR / "data_config.json"
MANIFEST_PATH     = DATA_DIR / "robustness_manifest.json"
MANIFEST_RUN_COPY = Path("runs") / CONFIG.NAME / INVOCATION / "robustness_manifest.json"
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
# RobustnessTest runs ONLY the FP/LP robustness probe ("birthday_robust",
# registered below once the manifest exists).

PROBES = ("birthday_robust",)


# ---------------------------
# GRID SWEEP — edit me
# ---------------------------
#
# Same grid semantics as ablation_llama_grid.py: BASE supplies fallbacks,
# GRID cross-products, dmodel = 64 * numHeads.

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

# Short axis labels used in grid run names (e.g. grid-L4-H6).
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

manifest = build_robustness_manifest(
    n_people=CONFIG.N, fields=tuple(CONFIG.FIELDS),
    limited_frac=LIMITED_PEOPLE_FRAC, template_frac=LIMITED_TEMPLATE_FRAC,
    seed=ROBUSTNESS_SEED,
)
save_manifest(manifest, MANIFEST_PATH)
save_manifest(manifest, MANIFEST_RUN_COPY)
n_limited = len(manifest["limited_people"])
example_subset = next(iter(manifest["limited_people"].values()))["birthday"]
print(f"RobustnessTest: {n_limited:,}/{CONFIG.N:,} limited people, "
      f"{len(example_subset)}/{manifest['n_templates']['birthday']} birthday "
      f"templates each (seed={ROBUSTNESS_SEED}).")
print(f"Manifest → {MANIFEST_PATH} (copy: {MANIFEST_RUN_COPY})")

verify_limited_rendering(people, manifest, tuple(CONFIG.FIELDS),
                         K=CONFIG.K, n_check=5)
print("Verified: first 5 limited people render only their allowed templates.")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
CONFIG.eosToken   = tokenizer.eos_token_id
CONFIG.vocab_size = tokenizer.vocab_size

# --- dataset cache guard --------------------------------------------------
# Tokenizing 50k*100 = 5M bios takes ~6 min. The bios bins depend only on the
# data knobs below (NOT on the GRID of models), so a resubmit / walltime-resume
# can reuse the existing bins instead of rebuilding them. We fingerprint those
# knobs into DATA_DIR/.data_ready.json and rebuild only on a miss or mismatch,
# so changing N/K/seed/fields or any robustness param correctly forces a fresh
# build — never a silent reuse of stale data.
DATA_READY_PATH = DATA_DIR / ".data_ready.json"
DATA_FINGERPRINT = {
    "N": CONFIG.N, "K": CONFIG.K, "SEQ_LEN": CONFIG.SEQ_LEN,
    "FIELDS": list(CONFIG.FIELDS), "SEED": CONFIG.SEED,
    "SHUFFLE_SEED": CONFIG.SHUFFLE_SEED,
    "LIMITED_PEOPLE_FRAC": LIMITED_PEOPLE_FRAC,
    "LIMITED_TEMPLATE_FRAC": LIMITED_TEMPLATE_FRAC,
    "ROBUSTNESS_SEED": ROBUSTNESS_SEED,
}


def _cached_data_matches():
    if not (Path(CONFIG.PRE_REDUCE_PATH).exists()
            and Path(CONFIG.POST_REDUCE_PATH).exists()
            and DATA_READY_PATH.exists()):
        return False
    try:
        with open(DATA_READY_PATH) as f:
            return json.load(f).get("fingerprint") == DATA_FINGERPRINT
    except (json.JSONDecodeError, OSError):
        return False


if _cached_data_matches():
    print("Reusing cached bios bins (fingerprint match) → skipping tokenize/remap.")
    # old_to_new is still needed (probes + EOS remap); rebuild it from the
    # existing prereduce bin via a fast np.unique — no re-tokenization.
    old_to_new, _, CONFIG.reducedVocabSize = build_vocab_remap(CONFIG.PRE_REDUCE_PATH)
else:
    print("No matching dataset cache → building bios bins (~6 min).")
    stream = robust_bio_stream(
        people, K=CONFIG.K, manifest=manifest,
        shuffle_seed=CONFIG.SHUFFLE_SEED, fields=tuple(CONFIG.FIELDS),
    )
    n_tokens, n_seq = tokenize_and_pack(
        tokenizer, stream,
        n_bios_total=CONFIG.N * CONFIG.K,
        out_path=CONFIG.PRE_REDUCE_PATH,
        seq_len=CONFIG.SEQ_LEN,
    )
    print(f"Wrote {n_tokens:,} tokens → {n_seq:,} sequences of {CONFIG.SEQ_LEN}.")
    old_to_new, _, CONFIG.reducedVocabSize = build_vocab_remap(CONFIG.PRE_REDUCE_PATH)
    remap_token_file(CONFIG.PRE_REDUCE_PATH, CONFIG.POST_REDUCE_PATH, old_to_new)
    with open(DATA_READY_PATH, "w") as f:
        json.dump({"fingerprint": DATA_FINGERPRINT,
                   "n_tokens": n_tokens, "n_seq": n_seq}, f, indent=2)

CONFIG.reducedEOSToken = old_to_new[int(CONFIG.eosToken)]
print(f"Reduced vocab size: {CONFIG.reducedVocabSize}")

# Probe prompts span ALL templates for EVERY eval person (that is the point
# of the robustness probe), so the seatbelt below — which renders the first
# 5 people across all K exposures, i.e. all 46 templates round-robin — is
# exactly the right vocab-coverage check, regardless of which templates those
# people actually trained on.
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
# PROBE REGISTRATION
# ---------------------------
# Additive: eval/probes.py is untouched; the runner closes over the manifest
# and its own group size (the registry's max_people default must not shrink
# the eval set — see make_robust_runner's docstring).

PROBE_REGISTRY["birthday_robust"] = make_robust_runner(
    manifest, max_people_per_group=MAX_PEOPLE_PER_GROUP)


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
        tags=[run["study"], CONFIG.NAME, "robustness"],
        reinit="finish_previous",
        config={
            **asdict(CONFIG),
            "study":      run["study"],
            "run_name":   run["name"],
            "sweep_name": SWEEP_NAME,
            "INVOCATION": INVOCATION,
            "probes":     list(PROBES),
            "limited_people_frac":   LIMITED_PEOPLE_FRAC,
            "limited_template_frac": LIMITED_TEMPLATE_FRAC,
            "robustness_seed":       ROBUSTNESS_SEED,
            "max_people_per_group":  MAX_PEOPLE_PER_GROUP,
            "n_limited_people":      n_limited,
        },
    )

    # Probe the in-memory model after each epoch in PROBE_EPOCHS (the final
    # one included). Per-epoch rows accumulate in CURVE_CSV across the grid.
    probe_cb = RobustProbeAtEpochs(
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

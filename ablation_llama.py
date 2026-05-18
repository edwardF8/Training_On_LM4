"""Ablation sweep over Capo bioS models.

Same pipeline as main.py, but instead of MODEL_CONFIGS we expand a base
architecture against single-axis or grid sweeps. To add a new study or
change a sweep, edit BASE / ABLATIONS below — nothing else needs to
change.

Output layout
-------------
    runs/{NAME}/{INVOCATION}/{study}/{run_name}/
        ...                                    # train checkpoints
        final/                                 # final checkpoint dir
            probe_sequential.json              # sequentialProbe full results
            probe_separate.json                # separateProbe full results
                                               #   (only if len(FIELDS) > 1)

wandb groups every run from one invocation by study, so the UI's group
view shows the sweep curves side-by-side.

Probing strategies
------------------
Each run is probed by one or two complementary probes:

    sequentialProbe   (eval/sequential_probe.py — always run)
        Renders the full multi-field bio exactly as it was at training
        time: fields concatenated in order, every field after the first
        using a pronoun ("He"/"She") for the subject. Per (person,
        exposure) it does ONE teacher-forced forward pass and ONE greedy
        autoregressive decode of the full bio. Metrics:
          • TF_<field>  : teacher-forced; given the TRUE bio up to the
                          field's value, does the model predict every
                          value token correctly?
          • FP_<field>  : autoregressive from [EOS]; do the generated
                          tokens at the field's value span match? Errors
                          in earlier fields compound.
          • FP_FULL     : autoregressive over the whole bio; the entire
                          generated sequence must match token-for-token.

    separateProbe     (eval/separate_probing.py — only run when
                       len(CONFIG.FIELDS) > 1)
        Renders an INDEPENDENT one-field bio for each field, ALWAYS
        substituting the full name for the subject (so the bio is
        identifiable without prior context). Per (person, field,
        exposure) it does one TF pass and one AR decode of that
        one-field bio. Metrics:
          • TF_<field> / FP_<field> as above (no FP_FULL — each field
            is its own bio).

The gap (sequential − separate) reveals how much the model relies on
cross-field context vs. direct (name → value) memorization.

wandb keys (per run)
--------------------
    sequentialProbe/FP_FULL
    sequentialProbe/<field>/{TF,FP}
    sequentialProbe/per_template/<field>    (table)
    separateProbe/<field>/{TF,FP}           (multi-field only)
    separateProbe/per_template/<field>      (table; multi-field only)
"""

import itertools
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
from eval.sequential_probe import run_probe as run_sequential_probe
from eval.separate_probing import run_separate_probe


# ---------------------------
# CONFIG
# ---------------------------

CONFIG = Config()

# Separate from main.py's "default" experiment so the two don't share a
# runs/ subdirectory. Cache is also separate (regenerated once at first run).
CONFIG.NAME = "bioS_N-Bd-Bc-lama_epoch_GRID"

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
CONFIG.FIELDS  = ("birthday","birthcity")

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
    "grid":      {"grid": {"numLayers": [2, 4, 6, 8], "numHeads": [2,4, 6, 8], "EPOCHS": [4, 8, 12, 16]}},
}
#Single-axis format:   "numLayers": {"axis": "numLayers", "values": [4, 6, 8, 12], "base": {"EPOCHS": 16}},
#Grid format:          "grid":      {"grid": {"numLayers": [...], "numHeads": [...], "EPOCHS": [...]}, "base": {...}},
#    "16EP_layer": {"axis": "numLayers", "values": [4, 6, 8, 12], "base": {"EPOCHS": 16}},
#    "16EP_heads": {"axis": "numHeads", "values": [4, 6, 8, 12], "base": {"EPOCHS": 16}},
#    "memorizing_attempts": {"axis": "numLayers", "values": [6, 8, 12], "base": {"EPOCHS": 16, "numHeads":6}},

# Short axis labels used in grid run names (e.g. grid-L4-H6-E16).
AXIS_ABBREV = {"numLayers": "L", "numHeads": "H", "EPOCHS": "E", "dmodel": "D"}


def build_runs(base, ablations):
    """Expand BASE × ABLATIONS into a flat list of run specs.

    Each ablation may set "base" to override BASE for that study only;
    the swept axis value(s) then override on top of that.

    Two ablation formats are supported:
      - single-axis: {"axis": "<field>", "values": [...]}
      - grid:        {"grid": {"<field1>": [...], "<field2>": [...], ...}}
    """
    runs = []
    for study, spec in ablations.items():
        study_base = {**base, **spec.get("base", {})}

        if "grid" in spec:
            axes   = list(spec["grid"].keys())
            combos = itertools.product(*spec["grid"].values())
            for combo in combos:
                cfg = {**study_base, **dict(zip(axes, combo))}
                cfg["dmodel"] = 64 * cfg["numHeads"]   # paper ℓ-h convention
                short = "-".join(
                    f"{AXIS_ABBREV.get(k, k)}{v}" for k, v in zip(axes, combo)
                )
                runs.append({
                    "study": study,
                    "name":  f"{study}-{short}",
                    **cfg,
                })
        else:
            axis, values = spec["axis"], spec["values"]
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

    # -------------------------------------------------------------------
    # PROBING
    # -------------------------------------------------------------------
    # We always run the sequentialProbe (multi-field bio scored in the
    # exact form the model saw at training). When there's more than one
    # field we ALSO run the separateProbe (each field as a standalone
    # one-field bio with the full name substituted for any pronoun) — the
    # gap between the two reveals how much the model leans on cross-field
    # context vs. direct (name → value) memorization.
    #
    # Outputs per run, written under {out_dir}/final/:
    #   probe_sequential.json    — full sequentialProbe results dict
    #   probe_separate.json      — full separateProbe results dict (multi-field only)
    #
    # wandb keys (per run):
    #   sequentialProbe/FP_FULL
    #   sequentialProbe/<field>/{TF,FP}
    #   sequentialProbe/per_template/<field>   (table)
    #   separateProbe/<field>/{TF,FP}          (multi-field only)
    #   separateProbe/per_template/<field>     (table; multi-field only)
    fields = tuple(CONFIG.FIELDS)

    print(f"\n=== sequentialProbe {run['name']} on fields={fields} ===")
    seq_results = run_sequential_probe(
        model, tokenizer, old_to_new, people,
        fields=fields, max_people=50,
    )
    seq_path = Path(out_dir) / "final" / "probe_sequential.json"
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    with open(seq_path, "w") as f:
        json.dump(seq_results, f, indent=2)
    print(f"Saved → {seq_path}")

    log_payload = {"sequentialProbe/FP_FULL": seq_results["FP_FULL"]}
    for field, fr in seq_results["per_field"].items():
        log_payload[f"sequentialProbe/{field}/TF"] = fr["TF"]
        log_payload[f"sequentialProbe/{field}/FP"] = fr["FP"]

    sep_results = None
    if len(fields) > 1:
        print(f"\n=== separateProbe {run['name']} on fields={fields} ===")
        sep_results = run_separate_probe(
            model, tokenizer, old_to_new, people,
            fields=fields, max_people=50,
        )
        sep_path = Path(out_dir) / "final" / "probe_separate.json"
        with open(sep_path, "w") as f:
            json.dump(sep_results, f, indent=2)
        print(f"Saved → {sep_path}")

        for field, fr in sep_results["per_field"].items():
            log_payload[f"separateProbe/{field}/TF"] = fr["TF"]
            log_payload[f"separateProbe/{field}/FP"] = fr["FP"]

    wandb.log(log_payload)

    # Per-template tables (one per (probe, field)).
    def _log_per_template(probe_ns, results):
        for field, fr in results["per_field"].items():
            if not fr["per_template"]:
                continue
            table = wandb.Table(columns=["template_idx", "TF", "FP"])
            for t_idx, accs in sorted(fr["per_template"].items()):
                table.add_data(int(t_idx), accs["TF"], accs["FP"])
            wandb.log({f"{probe_ns}/per_template/{field}": table})

    _log_per_template("sequentialProbe", seq_results)
    if sep_results is not None:
        _log_per_template("separateProbe", sep_results)

    wandb.finish()

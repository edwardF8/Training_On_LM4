"""Ablation sweep over Capo bioS models — single-axis.

Same pipeline as main.py, but instead of MODEL_CONFIGS we expand a BASE
architecture against one-axis sweeps defined in `ABLATIONS`. Each entry
overrides exactly one field (e.g. numLayers ∈ {2, 3, 4, 6, 8, 10}) while
the others stay fixed at BASE. To add a study or change a sweep, edit
BASE / ABLATIONS below — nothing else needs to change.

For the full-factorial grid version, see `ablation_llama_grid.py`.

Output layout
-------------
    runs/{NAME}/{INVOCATION}/{study}/{run_name}/
        ...                                    # train checkpoints
        final/                                 # final checkpoint dir
            probe_birthday.json                # birthdayProbe results
            probe_sequential.json              # sequentialProbe results (if enabled)
            probe_separate.json                # separateProbe results (if enabled,
                                               #   multi-field only)

wandb groups every run from one invocation by study, so the UI's group
view shows the sweep curves side-by-side.

Probing strategies
------------------
Which probes run is controlled by the `PROBES` list near the top. Pick any
subset from:

    "birthday_legacy"  (eval/birthday_probe_legacy.py)
        The original 4-metric birthday probe: MP / DayM / YearMD / FP.
        Birthday-only — ignores any other fields. Default here, since
        single-axis sweeps usually probe one field at a time.

    "sequential"       (eval/sequential_probe.py)
        Renders the full multi-field bio exactly as at training (pronouns
        after the first field). Per (person, exposure): one TF forward
        pass on the full true bio. Metric:
          • TF_<field>  — teacher-forced; 1 iff every value-span position
            argmax matches under teacher forcing (each position conditioned
            on TRUE prior tokens, including true earlier value tokens).

    "separate"         (eval/separate_probing.py)
        Each field probed independently with the FULL NAME always
        substituted for the subject (the pronoun has no antecedent when
        the field stands alone). Per (person, field, exposure): one TF
        pass on that one-field bio.
          • TF_<field>; same semantics as above.

The (sequential − separate) gap reveals how much the model leans on
cross-field context vs. direct (name → value) memorization.

wandb keys (per run, dispatched by `eval/probes.py`)
----------------------------------------------------
    birthdayProbe/{MP,DayM,YearMD,FP}
    birthdayProbe/per_template               (table)
    sequentialProbe/<field>/TF
    sequentialProbe/per_template/<field>     (table)
    separateProbe/<field>/TF
    separateProbe/per_template/<field>       (table)
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
from eval.probes import run_probes


# ---------------------------
# CONFIG
# ---------------------------

CONFIG = Config()

CONFIG.NAME = "bioS_birthday_lama_higher_capabilites"

INVOCATION = datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"INVOCATION = {INVOCATION}")

# wandb umbrella for this whole ablation sweep. Every run below uses this
# as its `group`; `job_type` is the study name so the UI can sub-group.
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


# PROBE SELECTION 
# Pick any subset of {"birthday_legacy", "sequential", "separate"}, Each probe lives in its own wandb namespace and JSON file (see eval/probes.py).
PROBES = ("birthday_legacy",)

# ABLATION SWEEP 
# BASE is the architecture every study starts from; 
# each ABLATIONS entry overrides exactly one field and sweeps it. 
# dmodel is computed from numHeads via the paper's ℓ-h convention (dmodel = 64 · numHeads).
BASE = { "numLayers": 4, "numHeads":  3, "EPOCHS":   1}

# Per-study overrides: add an optional "base" dict to any ABLATIONS entry
# to overlay on top of the global BASE for that study only.
ABLATIONS = {
    "16EP_heads": {"axis": "numHeads", "values": [8, 12], "base": {"EPOCHS": 16}},
    "memorizing_attempts": {"axis": "numLayers", "values": [8, 12], "base": {"EPOCHS": 12, "numHeads" : 6}},
}

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
    print(f"  [{r['study']:>6}] {r['name']:<14} "
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

    train(model, ds, CONFIG, output_dir=out_dir)
    print(f"Done. Checkpoints under {out_dir}/")

    run_probes(
        PROBES, model, tokenizer, old_to_new, people,
        fields=tuple(CONFIG.FIELDS),
        out_dir=out_dir,
        run_name=run["name"],
    )

    wandb.finish()

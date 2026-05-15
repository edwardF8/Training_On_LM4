"""End-to-end Capo (bioS) pipeline: data → tokenize+remap → train → eval → log.

For each model in MODEL_CONFIGS we:
  1. Train on the SAME packed/remapped dataset (so model-size effects are
     isolated from data variance).
  2. Run the bioS first-token recall probe on `EVAL_N_PEOPLE` training
     people (in-distribution prompts, matching Allen-Zhu Part 3.1 §3.1).
  3. Persist per-model results to `runs/{name}/recall.json`.

Finally we dump a cross-model summary to `runs/summary.json`.
"""

import json
from pathlib import Path

import torch
from transformers import GPT2Tokenizer, PreTrainedModel

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
from eval.recall_probe import evaluate_recall, build_remap_lut, pick_device


# ---------------------------
# CONFIG
# ---------------------------

CONFIG = Config()

CONFIG.SEED = 0
CONFIG.SHUFFLE_SEED = 1

# Data
CONFIG.N = 50_000
CONFIG.K = 100
CONFIG.SEQ_LEN = 512
CONFIG.PRE_REDUCE_PATH = "cache/bios_prereduce.bin"
CONFIG.POST_REDUCE_PATH = "cache/bios_postreduce.bin"

# Model family applied to every entry in MODEL_CONFIGS below.
CONFIG.MODEL_TYPE = "llama"

# Training
CONFIG.BATCH_SIZE = 12
CONFIG.LR = 1e-3
CONFIG.WEIGHT_DECAY = 0.01
CONFIG.WARMUP_STEPS = 1000
CONFIG.MAX_STEPS = 80_000
CONFIG.GRAD_CLIP = 1.0

# Eval (post-training recall probe)
EVAL_N_PEOPLE = 1000   # how many of the N training people to probe
EVAL_EXPOSURE = 0      # which of the K paraphrases to use (any [0, K) is fine)

MODEL_CONFIGS = [
    {"name": "2L-192D", "numLayers": 2, "dmodel": 192, "numHeads": 3},
    {"name": "4L-128D", "numLayers": 4, "dmodel": 128, "numHeads": 2},
    {"name": "6L-192D", "numLayers": 6, "dmodel": 192, "numHeads": 3},
]


# ---------------------------
# DATA
# ---------------------------

people = sample_people(N=CONFIG.N, seed=CONFIG.SEED)
stream = bio_stream(
    people, K=CONFIG.K,
    master_seed=CONFIG.SEED, shuffle_seed=CONFIG.SHUFFLE_SEED,
)

# Slow GPT2Tokenizer is used everywhere (packing + eval). The recall probe
# locates attribute token positions via subsequence matching, so it doesn't
# need offset_mapping.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

CONFIG.eosToken   = tokenizer.eos_token_id   # 50256
CONFIG.vocab_size = tokenizer.vocab_size     # 50257

n_tokens, n_seq = tokenize_and_pack(
    tokenizer,
    stream,
    n_bios_total=CONFIG.N * CONFIG.K,
    out_path=CONFIG.PRE_REDUCE_PATH,   # raw GPT-2 ids; remap step writes POST
    seq_len=CONFIG.SEQ_LEN,
)
print(f"Wrote {n_tokens:,} tokens → {n_seq:,} sequences of {CONFIG.SEQ_LEN}.")

# Token remap: shrink GPT-2's 50k vocab to only the ~3.3k tokens that
# actually appear in bioS. Both the dataset and the model embeddings use
# these reduced ids. We keep `old_to_new` in memory so the post-training
# eval can map GPT-2 ids → reduced ids without rebuilding from disk.
old_to_new, new_to_old, CONFIG.reducedVocabSize = build_vocab_remap(CONFIG.PRE_REDUCE_PATH)
remap_token_file(CONFIG.PRE_REDUCE_PATH, CONFIG.POST_REDUCE_PATH, old_to_new)
CONFIG.reducedEOSToken = old_to_new[int(CONFIG.eosToken)]
print(f"Reduced vocab size: {CONFIG.reducedVocabSize}")

ds = PackedTokenDataset(CONFIG.POST_REDUCE_PATH, seq_len=CONFIG.SEQ_LEN)
print(f"Dataset has {len(ds):,} sequences.")

# Persist config + remap so eval can also run offline against any saved
# checkpoint (e.g. `python -m eval.recall_probe runs/2L-192D/final`).
Path("cache").mkdir(parents=True, exist_ok=True)
CONFIG.save("cache/data_config.json")
with open("cache/old_to_new.json", "w") as f:
    json.dump({str(k): v for k, v in old_to_new.items()}, f)

# Build the in-memory remap lookup table once; every per-model eval reuses
# it. This is the same tensor recall_probe builds from disk in CLI mode.
lut = build_remap_lut(old_to_new)
device = pick_device()
print(f"Eval device: {device}")


# ---------------------------
# TRAIN + EVAL PER MODEL
# ---------------------------

all_results: dict[str, dict] = {}

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

    out_dir = f"runs/{mc['name']}"
    print(f"\n=== Training {mc['name']} → {out_dir} ===")
    trainer = train(model, ds, CONFIG, output_dir=out_dir)
    print(f"Done training {mc['name']}. Final checkpoint: {out_dir}/final")

    # ---- Recall probe ----
    print(f"\n=== Evaluating {mc['name']} on {EVAL_N_PEOPLE} people ===")
    # Pyright loses track of trainer.model's concrete type through HF's
    # generics; pin it here so model(ids_new) inside evaluate_recall
    # resolves to PreTrainedModel.__call__.
    trained_model: PreTrainedModel = trainer.model  # type: ignore[assignment]
    trained_model.to(device)
    trained_model.eval()
    results = evaluate_recall(
        trained_model, tokenizer, people, lut,
        master_seed=CONFIG.SEED, exposure=EVAL_EXPOSURE,
        device=device, max_people=EVAL_N_PEOPLE,
    )
    mean_acc = sum(results.values()) / len(results)

    print(f"\nRecall accuracy ({mc['name']}):")
    for attr, acc in results.items():
        print(f"  {attr:>14s}: {acc * 100:5.1f}%")
    print(f"  {'mean':>14s}: {mean_acc * 100:5.1f}%")

    # Try to attach eval metrics to the model's wandb run before closing it.
    # HF's WandbCallback leaves the run open after train() returns; finishing
    # it here keeps the next model's Trainer from inheriting this run.
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({f"recall/{a}": acc for a, acc in results.items()})
            wandb.log({"recall/mean": mean_acc})
            wandb.finish()
    except ImportError:
        pass

    # Persist per-model results alongside the checkpoint.
    results_path = Path(out_dir) / "recall.json"
    with open(results_path, "w") as f:
        json.dump({
            "model_config": mc,
            "results": results,
            "mean": mean_acc,
            "eval_n_people": EVAL_N_PEOPLE,
            "exposure": EVAL_EXPOSURE,
            "master_seed": CONFIG.SEED,
        }, f, indent=2)
    print(f"  saved → {results_path}")

    all_results[mc["name"]] = {"results": results, "mean": mean_acc}

    # Free memory between models — important on smaller GPUs.
    del trained_model, model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------
# CROSS-MODEL SUMMARY
# ---------------------------

print("\n" + "=" * 80)
print("SUMMARY: first-token recall accuracy across model sizes")
print("=" * 80)

attr_names = list(next(iter(all_results.values()))["results"].keys())

header = f"{'model':>10s} | " + " ".join(f"{a:>13s}" for a in attr_names) + f" | {'mean':>6s}"
print(header)
print("-" * len(header))
for name, payload in all_results.items():
    row = f"{name:>10s} | "
    row += " ".join(f"{payload['results'][a] * 100:>12.1f}%" for a in attr_names)
    row += f" | {payload['mean'] * 100:>5.1f}%"
    print(row)

Path("runs").mkdir(parents=True, exist_ok=True)
with open("runs/summary.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved summary → runs/summary.json")

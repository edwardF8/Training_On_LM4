"""Evaluate every trained checkpoint under runs/ and dump a clean CSV.

Walks `runs/` for every `final/` model directory, runs the birthday probe on
each, and writes one CSV row per model:

    family, run_name, path, layers, heads, dmodel, epochs, vocab_size,
    n_people, MP, DayM, YearMD, FP, LP, status

`layers` / `heads` / `dmodel` come straight from each model's config.json;
`epochs` from its training_args.bin (falling back to the run-dir name).

The probe needs the GPT-2 -> reduced-vocab remap the models were trained
with, so we rebuild the dataset exactly as `ablation_llama.py` does
(sample_people + tokenize_and_pack + build_vocab_remap). Any model whose
`vocab_size` does not match that remap was trained on a *different* dataset
(different N, or extra fields like birthcity) and is recorded with
status="skipped" rather than scored with the wrong vocab.

Re-probing (rather than reading old probe_birthday.json files) is required
to get the LP metric — checkpoints probed before LP existed only have the
old four metrics saved.

Usage
-----
    # from the Training_On_LM4/ project root
    python eval_runs_to_csv.py
    python eval_runs_to_csv.py --runs-root runs --out probe_all.csv --m 50

    # reuse an already-tokenized dataset (skips the ~15-min tokenization)
    python eval_runs_to_csv.py --pre-reduce-path cache/<NAME>/bios_prereduce.bin

    # evaluate a birthcity-trained family instead
    python eval_runs_to_csv.py --fields birthday,birthcity --out probe_bc.csv
"""

import argparse
import csv
import gc
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.sample_people import sample_people
from data.bio_text import bio_stream
from data.tokenize_pack import tokenize_and_pack, build_vocab_remap
from eval.birthday_probe_legacy import run_probe


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_remap(pre_reduce_path: Path, tokenizer, people, *,
                 K: int, seed: int, shuffle_seed: int, seq_len: int, fields):
    """Rebuild (or reuse) the token file, return (old_to_new, reduced_vocab).

    The remap is `np.unique` over the token file, so it depends only on the
    *set* of tokens — i.e. on (N, seed, fields), not on shuffle order. Matching
    ablation_llama.py's recipe therefore reproduces the exact training remap.
    """
    if pre_reduce_path.exists():
        print(f"Reusing existing token file: {pre_reduce_path}")
    else:
        n_bios = len(people) * K
        print(f"Tokenizing {n_bios:,} bios -> {pre_reduce_path} "
              f"(this is the slow step) ...")
        stream = bio_stream(people, K=K, master_seed=seed,
                            shuffle_seed=shuffle_seed, fields=tuple(fields))
        tokenize_and_pack(tokenizer, stream, n_bios_total=n_bios,
                          out_path=str(pre_reduce_path), seq_len=seq_len)
    old_to_new, _, reduced_vocab = build_vocab_remap(str(pre_reduce_path))
    return old_to_new, reduced_vocab


def find_final_dirs(runs_root: Path):
    """Every `<...>/final/` dir under runs_root that holds a loadable model."""
    out = []
    for cfg in runs_root.rglob("config.json"):
        d = cfg.parent
        if d.name != "final":
            continue                      # skip intermediate checkpoint-N dirs
        if (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists():
            out.append(d)
    return sorted(out)


def extract_epochs(final_dir: Path):
    """Best-effort training epochs: training_args.bin first, then the dir name."""
    for ta in (final_dir / "training_args.bin",
               final_dir.parent / "training_args.bin"):
        if ta.exists():
            try:
                args = torch.load(ta, weights_only=False)
                e = float(args.num_train_epochs)
                return int(e) if e == int(e) else e
            except Exception:
                pass
    name = final_dir.parent.name
    m = re.search(r"E(\d+)", name) or re.search(r"(\d+)\s*ep", name, re.I)
    return int(m.group(1)) if m else None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs-root", default="runs",
                   help="Directory to walk for final/ checkpoints.")
    p.add_argument("--out", default="probe_all.csv", help="Output CSV path.")
    p.add_argument("--m", type=int, default=50,
                   help="People per probe (legacy default 50; lower = faster).")
    p.add_argument("--pre-reduce-path", default="cache/eval_runs/bios_prereduce.bin",
                   help="Where to build/reuse the tokenized dataset.")
    # Dataset recipe — defaults mirror ablation_llama.py (birthday-only, N=50k).
    p.add_argument("--N", type=int, default=50_000)
    p.add_argument("--K", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle-seed", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--fields", default="birthday",
                   help="Comma-separated field list used at training time.")
    args = p.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.is_dir():
        sys.exit(f"runs root not found: {runs_root}")
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    device = pick_device()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print(f"Sampling {args.N:,} people (seed={args.seed}) ...")
    people = sample_people(N=args.N, seed=args.seed)

    pre = Path(args.pre_reduce_path)
    pre.parent.mkdir(parents=True, exist_ok=True)
    old_to_new, reduced_vocab = build_remap(
        pre, tokenizer, people,
        K=args.K, seed=args.seed, shuffle_seed=args.shuffle_seed,
        seq_len=args.seq_len, fields=fields,
    )
    print(f"Reduced vocab size: {reduced_vocab}  (device={device})")

    finals = find_final_dirs(runs_root)
    print(f"\nFound {len(finals)} final/ checkpoint(s) under {runs_root}/\n")

    rows = []
    for i, d in enumerate(finals, 1):
        rel      = d.relative_to(runs_root)
        family   = rel.parts[0]
        run_name = rel.parts[-2]                 # dir directly above final/
        cfg = json.loads((d / "config.json").read_text())
        layers, heads = cfg.get("num_hidden_layers"), cfg.get("num_attention_heads")
        dmodel, vocab = cfg.get("hidden_size"), cfg.get("vocab_size")
        epochs = extract_epochs(d)

        print(f"[{i}/{len(finals)}] {rel}")
        print(f"    L={layers} H={heads} D={dmodel} epochs={epochs} vocab={vocab}")

        row = {
            "family": family, "run_name": run_name, "path": str(rel),
            "layers": layers, "heads": heads, "dmodel": dmodel,
            "epochs": epochs, "vocab_size": vocab, "n_people": args.m,
            "MP": "", "DayM": "", "YearMD": "", "FP": "", "LP": "",
            "status": "ok",
        }

        if vocab != reduced_vocab:
            row["status"] = f"skipped: vocab {vocab} != remap {reduced_vocab}"
            row["n_people"] = 0
            print(f"    SKIP — trained on a different dataset "
                  f"({row['status']})\n")
            rows.append(row)
            continue

        model = None
        try:
            model = AutoModelForCausalLM.from_pretrained(d).to(device).eval()
            res = run_probe(model, tokenizer, old_to_new, people,
                            max_people=args.m, device=device)
            for k in ("MP", "DayM", "YearMD", "FP", "LP"):
                row[k] = round(res["macro"][k], 4)
        except Exception as e:
            row["status"] = f"error: {type(e).__name__}: {e}"
            print(f"    ERROR — {row['status']}")
        finally:
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
        print()

    rows.sort(key=lambda r: (r["family"], r["layers"] or 0,
                             r["heads"] or 0, r["epochs"] or 0))

    cols = ["family", "run_name", "path", "layers", "heads", "dmodel",
            "epochs", "vocab_size", "n_people",
            "MP", "DayM", "YearMD", "FP", "LP", "status"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(r["status"] == "ok" for r in rows)
    print(f"\nWrote {len(rows)} row(s) -> {args.out}  "
          f"({n_ok} probed, {len(rows) - n_ok} skipped/error)")


if __name__ == "__main__":
    main()
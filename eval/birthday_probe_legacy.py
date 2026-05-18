"""Birthday memorization probe — 4 metrics on every (person, template) pair.

For each eval person and each of the 46 birthday paraphrase templates we
build the prompt prefix `"<|endoftext|> {prefix}"` (everything before the
date placeholder) and the target span `"{month} {day}, {year}"`. We then
score four metrics:

  MP         (Month Prediction)
      Teacher-forced: argmax at the last-prefix position equals the first
      token of the month.

  Day | M    (Day given Month)
      Teacher-forced: argmax at the position right before the day equals
      the day's first token. The model sees the *true* month in context.

  Year | M,D (Year given Month, Day)
      Teacher-forced: argmax at the position right before the year equals
      the year's first token. The model sees true month + day. (If the
      year tokenizes as multiple BPE tokens, also score the rest teacher-
      forced; all must hit.)

  FP         (Full Prediction)
      Greedy autoregressive decode from the prefix for len(target) steps.
      Counted correct only if the generated token sequence exactly equals
      the target. Errors compound: a wrong month makes the day conditional
      on a wrong context.

Usage
-----
    # from project root
    python -m eval.birthday_probe runs/default/2-3/final
    python -m eval.birthday_probe runs/default/2-3/final --m 50

The probe rebuilds the GPT-2 -> reduced-vocab remap from the run's
bios_prereduce.bin (same as `recall_probe.py`).
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GPT2Tokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import Config
from data.bio_text import FIELD_SPECS
from data.sample_people import sample_people
from data.tokenize_pack import build_vocab_remap


# ----------------------------------------------------------------------------
# Per-(person, template) prompt/target construction
# ----------------------------------------------------------------------------

def build_chunks(person: dict, template: str) -> dict[str, str]:
    """Split a birthday template into chunks the probe needs to score separately.

    Concatenating prefix + month + day + sep + year + trailing reproduces the
    bio text the model saw at training time. We score MP/Day|M/Year|M,D/FP
    over month, day, and year (sep + trailing are present for context but not
    scored).

    Example template:  "{name} was born on {birthday}."
    full bio text:     " Ada Mae Lovelace was born on December 10, 1815."
    prefix:            " Ada Mae Lovelace was born on"   (no trailing space)
    month:             " December"
    day:               " 10"
    sep:               ","
    year:              " 1815"
    trailing:          "."
    """
    name = f"{person['first_name']} {person['middle_name']} {person['last_name']}"

    # The template uses a single `{birthday}` placeholder. We strip the
    # prefix's trailing space because BPE treats " December" as a single
    # token — month_text owns the leading space.
    before, after = template.split("{birthday}", 1)
    return {
        "prefix":   (" " + before.format(name=name)).rstrip(" "),
        "month":    f" {person['birthmonth']}",
        "day":      f" {person['birthday']}",
        "sep":      ",",
        "year":     f" {person['birthyear']}",
        "trailing": after,
    }


def tokenize_and_remap(text: str, tokenizer, old_to_new: dict[int, int]) -> list[int]:
    raw = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [old_to_new[int(t)] for t in raw]


# ----------------------------------------------------------------------------
# Scoring one (person, template) pair
# ----------------------------------------------------------------------------

@torch.no_grad()
def score_pair(
    model,
    tokenizer,
    old_to_new: dict[int, int],
    eos_remapped: int,
    person: dict,
    template: str,
    device: str,
) -> dict[str, int]:
    """Return {"MP": 0/1, "DayM": 0/1, "YearMD": 0/1, "FP": 0/1} for one pair."""
    chunks = build_chunks(person, template)

    prefix_ids = [eos_remapped] + tokenize_and_remap(chunks["prefix"], tokenizer, old_to_new)
    month_ids  = tokenize_and_remap(chunks["month"], tokenizer, old_to_new)
    day_ids    = tokenize_and_remap(chunks["day"],   tokenizer, old_to_new)
    sep_ids    = tokenize_and_remap(chunks["sep"],   tokenizer, old_to_new)
    year_ids   = tokenize_and_remap(chunks["year"],  tokenizer, old_to_new)

    # Target span used for FP and teacher-forcing positions:
    #   " December" + " 10" + "," + " 1815"
    target_ids = month_ids + day_ids + sep_ids + year_ids
    full_ids   = prefix_ids + target_ids

    # ---- Teacher-forced metrics: one forward pass on prefix + target ----
    x = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x).logits[0]   # (seq_len, vocab)

    # logits[i] predicts token at position i+1. To check the token at
    # position p, look at logits[p - 1].
    def argmax_at(p: int) -> int:
        return int(logits[p - 1].argmax().item())

    month_start = len(prefix_ids)
    day_start   = month_start + len(month_ids)
    year_start  = day_start   + len(day_ids) + len(sep_ids)  # skip the comma

    # MP: only score the first token of the month (months are usually 1
    # token but multi-token months still get a useful first-token check).
    mp_ok    = argmax_at(month_start) == month_ids[0]
    dayM_ok  = argmax_at(day_start)   == day_ids[0]
    # Year may be 1-2 BPE tokens (e.g. " 1815" is 1, " 1701" is " 17"+"01");
    # require every year position to argmax correctly.
    yearMD_ok = all(
        argmax_at(year_start + i) == year_ids[i]
        for i in range(len(year_ids))
    )

    # ---- FP: greedy autoregressive decode for len(target) steps ----
    cur = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated: list[int] = []
    for _ in range(len(target_ids)):
        next_tok = int(model(cur).logits[0, -1].argmax().item())
        generated.append(next_tok)
        cur = torch.cat([cur, torch.tensor([[next_tok]], device=device)], dim=1)
    fp_ok = generated == target_ids

    return {
        "MP":     int(mp_ok),
        "DayM":   int(dayM_ok),
        "YearMD": int(yearMD_ok),
        "FP":     int(fp_ok),
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_probe(model, tokenizer, old_to_new, people, *,
              max_people: int = 50, device: str | None = None):
    """Run the 4-metric birthday probe on an in-memory model.

    Reusable entry point for both the CLI (which loads from a checkpoint) and
    main.py (which calls this directly on each freshly-trained model).

    Returns a dict with the macro accuracies for MP / DayM / YearMD / FP.
    """
    if device is None:
        device = str(next(model.parameters()).device)
    model.eval()

    eos_remapped = old_to_new[int(tokenizer.eos_token_id)]
    templates = FIELD_SPECS["birthday"]["templates"]
    eval_people = people[:max_people]

    totals = defaultdict(lambda: [0, 0])
    per_template = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    n_pairs = len(eval_people) * len(templates)
    pbar = tqdm(total=n_pairs, desc="probe")
    for person in eval_people:
        for t_idx, template in enumerate(templates):
            scores = score_pair(model, tokenizer, old_to_new, eos_remapped,
                                person, template, device=device)
            for metric, ok in scores.items():
                totals[metric][0] += ok
                totals[metric][1] += 1
                per_template[t_idx][metric][0] += ok
                per_template[t_idx][metric][1] += 1
            pbar.update(1)
    pbar.close()

    print("\nMacro-average over all (person, template) pairs:")
    width = max(len(m) for m in totals)
    macro = {}
    for metric in ("MP", "DayM", "YearMD", "FP"):
        c, n = totals[metric]
        acc = c / max(n, 1)
        macro[metric] = acc
        print(f"  {metric:>{width}s}: {c}/{n}  =  {100 * acc:5.1f}%")

    # Per-template macro accuracies, suitable for JSON dump.
    per_t_acc = {
        int(t_idx): {
            m: per_template[t_idx][m][0] / max(per_template[t_idx][m][1], 1)
            for m in ("MP", "DayM", "YearMD", "FP")
        }
        for t_idx in per_template
    }

    print("\nPer-template FP (5 worst / 5 best):")
    fp_by_t = sorted(per_t_acc.items(), key=lambda kv: kv[1]["FP"])
    print("  worst:")
    for t_idx, accs in fp_by_t[:5]:
        print(f"    [{t_idx:>2d}] {accs['FP']*100:5.1f}%  {templates[t_idx]!r}")
    print("  best:")
    for t_idx, accs in fp_by_t[-5:]:
        print(f"    [{t_idx:>2d}] {accs['FP']*100:5.1f}%  {templates[t_idx]!r}")

    return {
        "macro": macro,
        "per_template": per_t_acc,
        "n_people": len(eval_people),
        "n_templates": len(templates),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ckpt", help="Checkpoint dir, e.g. runs/default/2-3/final")
    p.add_argument("--m", type=int, default=50,
                   help="Number of eval people (default 50). Total pairs = m * 46.")
    p.add_argument("--N", type=int, default=None,
                   help="Total people sampled at train time (must match)")
    p.add_argument("--seed", type=int, default=None,
                   help="Person-sampling seed (must match training)")
    p.add_argument("--pre-reduce-path", type=str, default=None,
                   help="Path to bios_prereduce.bin used to build the GPT-2 remap")
    args = p.parse_args()

    cfg = Config()
    N    = args.N    if args.N    is not None else cfg.N
    seed = args.seed if args.seed is not None else cfg.SEED
    pre_reduce = args.pre_reduce_path or cfg.PRE_REDUCE_PATH

    print(f"Sampling {N:,} people (seed={seed}) ...")
    people = sample_people(N=N, seed=seed)

    print(f"Building vocab remap from {pre_reduce} ...")
    old_to_new, _, reduced_vocab = build_vocab_remap(pre_reduce)
    print(f"  reduced vocab size: {reduced_vocab}")

    device = pick_device()
    print(f"Loading model from {args.ckpt} on {device} ...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(args.ckpt).to(device)
    model.eval()
    if model.config.vocab_size != reduced_vocab:
        print(f"  WARNING: model vocab={model.config.vocab_size} vs remap={reduced_vocab}",
              file=sys.stderr)

    run_probe(model, tokenizer, old_to_new, people,
              max_people=args.m, device=device)


if __name__ == "__main__":
    main()
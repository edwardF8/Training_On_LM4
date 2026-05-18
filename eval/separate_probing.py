"""Separate-field memorization probe — one isolated bio per (person, field).

Differs from `sequential_probe.run_probe` in two ways:

1. Each field is probed **independently** — we render a one-field bio
   `" {name} <template> <value><post>"` instead of concatenating all
   fields into one long bio. This isolates the (name → field value)
   association from any cross-field context the model might exploit.

2. Pronoun subjects are **replaced with the full name**. In the
   multi-field bio, fields like `birthcity` use "He"/"She" as the
   sentence subject (the name was already introduced by an earlier
   field). With each field standing alone, the pronoun has no
   antecedent, so the probe would be unidentifiable — we substitute
   `{first} {middle} {last}` for every field regardless of
   FIELD_SPECS[field]["subject"].

Metrics per field (no FP_FULL, since each field is its own bio):

  TF   Teacher-forced: one forward pass on the full one-field bio;
       argmax must match the true token at every value-span position.

  FP   Greedy autoregressive decode of the whole one-field bio from
       [EOS]; check the generated tokens at the value span equal the
       true value tokens.

Usage
-----
    python -m eval.separate_probing runs/.../final
    python -m eval.separate_probing runs/.../final \\
        --fields birthday,birthcity,university --m 50
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
from eval.sequential_probe import tokenize_and_remap, pick_device


# ----------------------------------------------------------------------------
# One-field bio chunking (name always substituted for the subject)
# ----------------------------------------------------------------------------

def build_single_field_chunks(person: dict, field: str, exposure_idx: int) -> dict:
    """Render one isolated bio for `field` with the full name as subject.

    Returns:
        {
          "field":  str,
          "t_idx":  int,           # template index used
          "pre":    str,           # everything before the value (no trailing space)
          "value":  str,           # " <value>" (leading space owned by value)
          "post":   str,           # everything after the value
        }

    The concatenation "pre" + "value" + "post" is the full one-field bio text;
    leading-space convention matches `build_multi_field_pieces` so BPE doesn't
    merge across chunk boundaries.
    """
    spec        = FIELD_SPECS[field]
    templates   = spec["templates"]
    t_idx       = exposure_idx % len(templates)
    template    = templates[t_idx]
    placeholder = spec["value_placeholder"]

    # Always use the full name — never the pronoun, since this field stands alone.
    name = f"{person['first_name']} {person['middle_name']} {person['last_name']}"

    before, after = template.split(f"{{{placeholder}}}", 1)
    return {
        "field":  field,
        "t_idx":  t_idx,
        "pre":    (" " + before.format(name=name)).rstrip(" "),
        "value":  " " + spec["render_value"](person),
        "post":   after,
    }


def tokenize_single_field(chunk, tokenizer, old_to_new, eos_remapped):
    """Tokenize a one-field chunk; return full_ids and the value span.

    Returns:
        full_ids:   [EOS, <pre>, <value>, <post>]
        value_span: (start, end) into full_ids covering the value tokens
    """
    full_ids = [eos_remapped]
    full_ids.extend(tokenize_and_remap(chunk["pre"], tokenizer, old_to_new))
    val_ids = tokenize_and_remap(chunk["value"], tokenizer, old_to_new)
    span = (len(full_ids), len(full_ids) + len(val_ids))
    full_ids.extend(val_ids)
    full_ids.extend(tokenize_and_remap(chunk["post"], tokenizer, old_to_new))
    return full_ids, span


# ----------------------------------------------------------------------------
# Per-(person, field, exposure) scoring
# ----------------------------------------------------------------------------

@torch.no_grad()
def score_field_pair(
    model,
    tokenizer,
    old_to_new: dict[int, int],
    eos_remapped: int,
    person: dict,
    field: str,
    exposure_idx: int,
    device: str,
) -> dict:
    """Score one (person, field, exposure) triple.

    Returns {"TF": 0/1, "FP": 0/1, "t_idx": int}.
    """
    chunk = build_single_field_chunks(person, field, exposure_idx)
    full_ids, (start, end) = tokenize_single_field(
        chunk, tokenizer, old_to_new, eos_remapped
    )

    # ---- TF: one forward pass, check argmax at every value position ----
    x = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x).logits[0]
    tf_ok = all(
        int(logits[p - 1].argmax().item()) == full_ids[p]
        for p in range(start, end)
    )

    # ---- FP: AR decode from [EOS] over the whole one-field bio ----
    cur = torch.tensor([[eos_remapped]], dtype=torch.long, device=device)
    generated: list[int] = [eos_remapped]
    for _ in range(len(full_ids) - 1):
        next_tok = int(model(cur).logits[0, -1].argmax().item())
        generated.append(next_tok)
        cur = torch.cat([cur, torch.tensor([[next_tok]], device=device)], dim=1)
    fp_ok = generated[start:end] == full_ids[start:end]

    return {"TF": int(tf_ok), "FP": int(fp_ok), "t_idx": chunk["t_idx"]}


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run_separate_probe(model, tokenizer, old_to_new, people, *,
                       fields=("birthday",),
                       max_people: int = 50,
                       n_exposures: int | None = None,
                       device: str | None = None):
    """Run the separate-field probe on an in-memory model.

    For each (person, field, exposure) triple we render an isolated one-field
    bio (full name in place of any pronoun), do one TF forward pass, and one
    greedy AR decode of bio length. Aggregates TF and FP per field, plus a
    per-template breakdown.

    `n_exposures` defaults to max(template-pool size across fields) so every
    template of every field is hit at least once; shorter pools cycle.

    Returns:
        {
            "fields":      list[str],
            "n_people":    int,
            "n_exposures": int,
            "per_field": {
                <field>: {
                    "TF":           float,
                    "FP":           float,
                    "per_template": {t_idx: {"TF": float, "FP": float}},
                    "n_templates":  int,
                }
            }
        }
    """
    if device is None:
        device = str(next(model.parameters()).device)
    model.eval()

    for f in fields:
        if f not in FIELD_SPECS:
            raise ValueError(
                f"Unknown field {f!r}; supported: {list(FIELD_SPECS)}"
            )

    if n_exposures is None:
        n_exposures = max(len(FIELD_SPECS[f]["templates"]) for f in fields)

    eos_remapped = old_to_new[int(tokenizer.eos_token_id)]
    eval_people  = people[:max_people]

    totals       = {f: {"TF": [0, 0], "FP": [0, 0]} for f in fields}
    per_template = {
        f: defaultdict(lambda: {"TF": [0, 0], "FP": [0, 0]})
        for f in fields
    }

    n_pairs = len(eval_people) * len(fields) * n_exposures
    pbar = tqdm(total=n_pairs, desc=f"sep-probe[{','.join(fields)}]")
    for person in eval_people:
        for field in fields:
            for exposure_idx in range(n_exposures):
                scores = score_field_pair(
                    model, tokenizer, old_to_new, eos_remapped,
                    person, field, exposure_idx, device=device,
                )
                tf_ok = scores["TF"]
                fp_ok = scores["FP"]
                t_idx = scores["t_idx"]
                totals[field]["TF"][0] += tf_ok
                totals[field]["TF"][1] += 1
                totals[field]["FP"][0] += fp_ok
                totals[field]["FP"][1] += 1
                per_template[field][t_idx]["TF"][0] += tf_ok
                per_template[field][t_idx]["TF"][1] += 1
                per_template[field][t_idx]["FP"][0] += fp_ok
                per_template[field][t_idx]["FP"][1] += 1
                pbar.update(1)
    pbar.close()

    per_field_results = {}
    print()
    for f in fields:
        tf_c, tf_n = totals[f]["TF"]
        fp_c, fp_n = totals[f]["FP"]
        tf_acc = tf_c / max(tf_n, 1)
        fp_acc = fp_c / max(fp_n, 1)
        print(f"  [{f:>12s}]  TF: {tf_c}/{tf_n} = {100 * tf_acc:5.1f}%   "
              f"FP: {fp_c}/{fp_n} = {100 * fp_acc:5.1f}%")
        per_t_acc = {
            int(t_idx): {
                "TF": cnts["TF"][0] / max(cnts["TF"][1], 1),
                "FP": cnts["FP"][0] / max(cnts["FP"][1], 1),
            }
            for t_idx, cnts in per_template[f].items()
        }
        per_field_results[f] = {
            "TF":           tf_acc,
            "FP":           fp_acc,
            "per_template": per_t_acc,
            "n_templates":  len(FIELD_SPECS[f]["templates"]),
        }

    return {
        "fields":      list(fields),
        "n_people":    len(eval_people),
        "n_exposures": n_exposures,
        "per_field":   per_field_results,
    }


def main():
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("ckpt", help="Checkpoint dir, e.g. runs/default/2-3/final")
    p.add_argument("--m", type=int, default=50,
                   help="Number of eval people (default 50).")
    p.add_argument("--N", type=int, default=None,
                   help="Total people sampled at train time (must match)")
    p.add_argument("--seed", type=int, default=None,
                   help="Person-sampling seed (must match training)")
    p.add_argument("--pre-reduce-path", type=str, default=None,
                   help="Path to bios_prereduce.bin used to build the GPT-2 remap")
    p.add_argument("--fields", type=str, default="birthday",
                   help="Comma-separated fields to probe independently. "
                        f"Supported: {','.join(FIELD_SPECS)}")
    p.add_argument("--exposures", type=int, default=None,
                   help="Exposures (templates) per person per field. "
                        "Default: max template-pool size across `--fields`.")
    args = p.parse_args()
    fields = tuple(f.strip() for f in args.fields.split(",") if f.strip())

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

    run_separate_probe(model, tokenizer, old_to_new, people,
                       fields=fields, max_people=args.m,
                       n_exposures=args.exposures, device=device)


if __name__ == "__main__":
    main()

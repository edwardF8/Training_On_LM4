"""Sequential probe — multi-field, TF / FP per field + FP_FULL.

For each (eval person, exposure_idx) pair we render the full multi-field
bio over the requested `fields` (in order), exactly as the model saw it
at training time. This is the SEQUENTIAL probe: every field after the
first uses a pronoun subject (He/She) and therefore depends on prior
fields to disambiguate "who". To probe each field standing alone (full
name as subject, no cross-field context), use `eval.separate_probing`.

We score three families of metrics:

  TF_{field}   (Teacher-Forced)
      One forward pass on the full TRUE bio. For each field F, check that
      argmax matches the true token at *every* position in F's value span.
      "Given the true bio up to (and including) F's value, does the model
      predict each of F's value tokens?"

  FP_{field}   (Full Prediction, AR through field F)
      Greedy autoregressive decode starting from [EOS] over the full bio
      length. Check that the generated tokens at F's value span equal the
      true tokens for F. Mistakes in prior fields can corrupt this.

  FP_FULL      (Full Prediction over the whole bio)
      Same AR decode as above; FP_FULL is 1 iff the entire generated
      sequence equals the true bio token-for-token (separators, trailing
      punctuation, everything).

One TF forward pass yields TF_F for every field at once; one AR decode
yields every FP_F and FP_FULL together. So total cost per (person, exposure)
is ~len(bio) forward passes (dominated by the AR decode).

Usage
-----
    # from project root
    python -m eval.sequential_probe runs/.../final
    python -m eval.sequential_probe runs/.../final --m 50 \\
        --fields birthday,birthcity,university
    python -m eval.sequential_probe runs/.../final --exposures 46

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
# Bio chunking + tokenization
# ----------------------------------------------------------------------------

def tokenize_and_remap(text: str, tokenizer, old_to_new: dict[int, int]) -> list[int]:
    raw = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [old_to_new[int(t)] for t in raw]


def build_multi_field_pieces(person: dict, fields, exposure_idx: int):
    """Render the bio as one dict per field with pre / value / post text.

    For field F at this exposure, the chosen template is
        template = FIELD_SPECS[F]["templates"][exposure_idx % T_F]
    and we split it on the value placeholder. The leading space convention
    matches render_bio() in data/bio_text.py, with the trailing space of
    `pre` stripped so " December"-style leading-space tokens stay with the
    value chunk (BPE doesn't merge across chunk boundaries that way).

    Concatenating pre + value + post for every field in order reproduces
    the exact bio text the model saw during training.
    """
    name    = f"{person['first_name']} {person['middle_name']} {person['last_name']}"
    pronoun = "He" if person["id"] % 2 == 0 else "She"

    out = []
    for field in fields:
        spec        = FIELD_SPECS[field]
        templates   = spec["templates"]
        t_idx       = exposure_idx % len(templates)
        template    = templates[t_idx]
        placeholder = spec["value_placeholder"]
        subject     = name if spec["subject"] == "name" else pronoun

        before, after = template.split(f"{{{placeholder}}}", 1)
        out.append({
            "field":  field,
            "t_idx":  t_idx,
            "pre":    (" " + before.format(name=subject)).rstrip(" "),
            "value":  " " + spec["render_value"](person),
            "post":   after,
        })
    return out


def tokenize_bio(field_pieces, tokenizer, old_to_new, eos_remapped):
    """Tokenize per-field chunks individually, return full_ids + value spans.

    Returns:
        full_ids:    [EOS, ...]   concatenation of all token chunks
        value_spans: list[(start, end)] in `full_ids` for each field
        t_indices:   list[int]    template index per field for this exposure
    """
    full_ids   = [eos_remapped]
    spans      = []
    t_indices  = []
    for fp in field_pieces:
        full_ids.extend(tokenize_and_remap(fp["pre"], tokenizer, old_to_new))
        val_ids = tokenize_and_remap(fp["value"], tokenizer, old_to_new)
        spans.append((len(full_ids), len(full_ids) + len(val_ids)))
        full_ids.extend(val_ids)
        full_ids.extend(tokenize_and_remap(fp["post"], tokenizer, old_to_new))
        t_indices.append(fp["t_idx"])
    return full_ids, spans, t_indices


# ----------------------------------------------------------------------------
# Per-pair scoring (one (person, exposure) pair)
# ----------------------------------------------------------------------------

@torch.no_grad()
def score_pair(
    model,
    tokenizer,
    old_to_new: dict[int, int],
    eos_remapped: int,
    person: dict,
    fields,
    exposure_idx: int,
    device: str,
) -> dict:
    """Score one (person, exposure) pair across all fields.

    Returns:
        {
            "TF":      {field: 0/1},   # teacher-forced full-value match
            "FP":      {field: 0/1},   # AR full-bio decode, field's value span match
            "FP_FULL": 0/1,            # AR full-bio decode, entire sequence match
            "t_idx":   {field: int},   # template chosen for this field at this exposure
        }
    """
    field_pieces = build_multi_field_pieces(person, fields, exposure_idx)
    full_ids, spans, t_indices = tokenize_bio(
        field_pieces, tokenizer, old_to_new, eos_remapped
    )

    # ---- TF: one forward pass on the full TRUE bio ----
    x = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits = model(x).logits[0]   # (seq_len, vocab)

    tf = {}
    for f, (start, end) in zip(fields, spans):
        # logits[p-1] predicts the token at position p.
        ok = all(
            int(logits[p - 1].argmax().item()) == full_ids[p]
            for p in range(start, end)
        )
        tf[f] = int(ok)

    # ---- FP / FP_FULL: greedy AR decode from [EOS] ----
    cur = torch.tensor([[eos_remapped]], dtype=torch.long, device=device)
    generated: list[int] = [eos_remapped]
    for _ in range(len(full_ids) - 1):
        next_tok = int(model(cur).logits[0, -1].argmax().item())
        generated.append(next_tok)
        cur = torch.cat([cur, torch.tensor([[next_tok]], device=device)], dim=1)

    fp = {
        f: int(generated[start:end] == full_ids[start:end])
        for f, (start, end) in zip(fields, spans)
    }
    fp_full = int(generated == full_ids)

    return {
        "TF":      tf,
        "FP":      fp,
        "FP_FULL": fp_full,
        "t_idx":   dict(zip(fields, t_indices)),
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
              fields=("birthday",),
              max_people: int = 50,
              n_exposures: int | None = None,
              device: str | None = None):
    """Run the multi-field TF/FP/FP_FULL probe on an in-memory model.

    Iterates `max_people * n_exposures` pairs. For each pair we run one TF
    forward pass (yields all TF_F) and one greedy AR decode of len(bio)
    steps (yields every FP_F plus FP_FULL).

    `n_exposures` defaults to max(template-pool size across fields) so every
    template of every field is hit at least once; shorter pools cycle.

    Returns:
        {
            "fields":      list[str],
            "n_people":    int,
            "n_exposures": int,
            "FP_FULL":     float,
            "per_field": {
                <field>: {
                    "TF":           float,
                    "FP":           float,
                    "per_template": {t_idx: {"TF": float, "FP": float}},
                    "n_templates":  int,
                }
            },
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

    tf_totals      = {f: [0, 0] for f in fields}
    fp_totals      = {f: [0, 0] for f in fields}
    fp_full_totals = [0, 0]
    per_template   = {
        f: defaultdict(lambda: {"TF": [0, 0], "FP": [0, 0]})
        for f in fields
    }

    n_pairs = len(eval_people) * n_exposures
    pbar = tqdm(total=n_pairs, desc=f"probe[{','.join(fields)}]")
    for person in eval_people:
        for exposure_idx in range(n_exposures):
            scores = score_pair(
                model, tokenizer, old_to_new, eos_remapped,
                person, fields, exposure_idx, device=device,
            )
            for f in fields:
                tf_ok = scores["TF"][f]
                fp_ok = scores["FP"][f]
                t_idx = scores["t_idx"][f]
                tf_totals[f][0] += tf_ok
                tf_totals[f][1] += 1
                fp_totals[f][0] += fp_ok
                fp_totals[f][1] += 1
                per_template[f][t_idx]["TF"][0] += tf_ok
                per_template[f][t_idx]["TF"][1] += 1
                per_template[f][t_idx]["FP"][0] += fp_ok
                per_template[f][t_idx]["FP"][1] += 1
            fp_full_totals[0] += scores["FP_FULL"]
            fp_full_totals[1] += 1
            pbar.update(1)
    pbar.close()

    fp_full = fp_full_totals[0] / max(fp_full_totals[1], 1)
    print(f"\nFP_FULL: {fp_full_totals[0]}/{fp_full_totals[1]}  =  "
          f"{100 * fp_full:5.1f}%")

    per_field_results = {}
    for f in fields:
        tf_c, tf_n = tf_totals[f]
        fp_c, fp_n = fp_totals[f]
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
        "FP_FULL":     fp_full,
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
                   help="Comma-separated fields rendered in the bio, in order. "
                        f"Supported: {','.join(FIELD_SPECS)}")
    p.add_argument("--exposures", type=int, default=None,
                   help="Exposures (templates) per person. "
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

    run_probe(model, tokenizer, old_to_new, people,
              fields=fields, max_people=args.m,
              n_exposures=args.exposures, device=device)


if __name__ == "__main__":
    main()

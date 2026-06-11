"""Birthday memorization probe — 5 metrics on every (person, template) pair.

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

  LP         (Lenient Prediction)
      Greedy free-decode from the prefix (up to LP_MAX_NEW_TOKENS tokens,
      stopping at <|endoftext|>), then check the true date appears somewhere
      in the generated bio as "Month D, YYYY". Unlike FP it need not come
      immediately after the prompt and the surrounding wording is ignored —
      this credits a model that knows the birthday but paraphrases its way
      there (e.g. "...born on the memorable date of July 17, 1741."). If the
      model emits a *different* date, or no date at all, it is scored wrong.

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
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GPT2Tokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import Config
from data.bio_text import FIELD_SPECS
from data.sample_people import sample_people, MONTHS
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


# Greedy budget for the lenient probe's free generation. The date reliably
# surfaces well within this; decoding still stops early at <|endoftext|>.
LP_MAX_NEW_TOKENS = 32

# Matches a date written as "Month D, YYYY" — the bios' training format.
_DATE_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2}),\s*(\d{4})\b")


def lenient_date_match(text: str, person: dict) -> bool:
    """True iff `text` states this person's birthday — and no *other* date.

    The lenient probe lets the model generate the rest of the bio freely and
    only asks that the true date surface somewhere inside it: it need not come
    immediately after the prompt, and the surrounding wording is irrelevant.

    Guard against false credit: every "Month D, YYYY" date found in `text`
    must equal the true birthday. If the model also emits a different date it
    hedged (wrong); if it emits no recognizable date it failed too.
    """
    true  = (person["birthmonth"], int(person["birthday"]), int(person["birthyear"]))
    found = {(m, int(d), int(y)) for m, d, y in _DATE_RE.findall(text)}
    return found == {true}


# ----------------------------------------------------------------------------
# Scoring one (person, template) pair
# ----------------------------------------------------------------------------

@torch.no_grad()
def score_pair(
    model,
    tokenizer,
    old_to_new: dict[int, int],
    new_to_old: dict[int, int],
    eos_remapped: int,
    person: dict,
    template: str,
    device: str,
) -> dict[str, int]:
    """Return {"MP","DayM","YearMD","FP","LP"} each 0/1 for one pair."""
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

    # ---- FP + LP: one greedy decode, scored two ways ----
    # FP wants exactly len(target) steps; LP wants room for the date to
    # surface inside a longer paraphrase. Decode the larger budget once.
    n_steps = max(len(target_ids), LP_MAX_NEW_TOKENS)
    cur = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated: list[int] = []
    for _ in range(n_steps):
        next_tok = int(model(cur).logits[0, -1].argmax().item())
        generated.append(next_tok)
        cur = torch.cat([cur, torch.tensor([[next_tok]], device=device)], dim=1)

    # FP: the first len(target) greedy tokens must exactly equal the target.
    fp_ok = generated[:len(target_ids)] == target_ids

    # LP: decode the continuation up to the first <|endoftext|> (just this
    # bio) and check the true date surfaces anywhere inside it.
    eos_cut  = generated.index(eos_remapped) if eos_remapped in generated else len(generated)
    gen_text = tokenizer.decode([new_to_old[t] for t in generated[:eos_cut]])
    lp_ok    = lenient_date_match(gen_text, person)

    return {
        "MP":     int(mp_ok),
        "DayM":   int(dayM_ok),
        "YearMD": int(yearMD_ok),
        "FP":     int(fp_ok),
        "LP":     int(lp_ok),
    }


# ----------------------------------------------------------------------------
# Batched FP/LP scoring (KV-cached) — same greedy results as score_pair
# ----------------------------------------------------------------------------

@torch.no_grad()
def score_pairs_fp_lp_batched(
    model,
    tokenizer,
    old_to_new: dict[int, int],
    new_to_old: dict[int, int],
    eos_remapped: int,
    pairs: list[tuple[dict, str]],
    device: str,
    batch_size: int = 128,
) -> list[dict[str, int]]:
    """Greedy-decode many (person, template) pairs at once; return FP/LP each.

    A drop-in, much faster path for callers that need only the two greedy
    metrics (the robustness probe). Returns a list of ``{"FP", "LP"}`` aligned
    with ``pairs``. The teacher-forced metrics (MP/DayM/YearMD) are *not*
    computed — use :func:`score_pair` if you need them.

    Why this matches :func:`score_pair` exactly. The prompt/target token ids
    are built with the *same* ``build_chunks`` + ``tokenize_and_remap`` code,
    and FP/LP are read off the generated tokens with the *same* logic. The
    only change is the decode mechanism: instead of one ungrouped, no-cache
    forward per step per pair, we

      - left-pad each batch's prefixes so every sequence's last real token sits
        in the final column (its next-token logit is therefore at ``[:, -1]``
        for the whole batch), and pass an attention mask + RoPE position ids so
        the padding is inert; and
      - carry ``past_key_values`` across steps (``use_cache=True``) so each step
        only forwards the single new token.

    Both are exact-equivalence transforms of greedy argmax decoding, so the
    per-pair FP/LP are identical to looping :func:`score_pair` (verified in
    ``tests/test_probe_batching.py``). Greedy never stops at EOS here, exactly
    like ``score_pair`` — EOS only matters when slicing the text for LP.
    """
    # Pre-tokenize every pair once (identical construction to score_pair).
    prepared = []
    for person, template in pairs:
        chunks = build_chunks(person, template)
        prefix_ids = [eos_remapped] + tokenize_and_remap(chunks["prefix"], tokenizer, old_to_new)
        target_ids = (
            tokenize_and_remap(chunks["month"], tokenizer, old_to_new)
            + tokenize_and_remap(chunks["day"],   tokenizer, old_to_new)
            + tokenize_and_remap(chunks["sep"],   tokenizer, old_to_new)
            + tokenize_and_remap(chunks["year"],  tokenizer, old_to_new)
        )
        prepared.append((prefix_ids, target_ids, person))

    results: list[dict[str, int] | None] = [None] * len(pairs)
    pbar = tqdm(total=len(pairs), desc="robust probe")
    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start:start + batch_size]
        bsz = len(chunk)
        max_prefix = max(len(p) for p, _, _ in chunk)
        # Mirror score_pair's per-pair budget (max(len(target), LP_MAX_NEW_TOKENS))
        # with a single shared length — the largest any pair in the chunk needs.
        n_steps = max(LP_MAX_NEW_TOKENS, max(len(t) for _, t, _ in chunk))

        # Left-pad: real tokens are right-aligned, pad sits on the left (inert).
        input_ids = torch.full((bsz, max_prefix), eos_remapped,
                               dtype=torch.long, device=device)
        attn = torch.zeros((bsz, max_prefix), dtype=torch.long, device=device)
        for i, (prefix_ids, _, _) in enumerate(chunk):
            L = len(prefix_ids)
            input_ids[i, max_prefix - L:] = torch.tensor(prefix_ids, device=device)
            attn[i, max_prefix - L:] = 1

        # RoPE position ids from the mask (HF convention: pad positions -> 1).
        pos = attn.long().cumsum(-1) - 1
        pos = pos.masked_fill(attn == 0, 1)

        out = model(input_ids=input_ids, attention_mask=attn,
                    position_ids=pos, use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1)          # (bsz,)
        cur_pos = pos[:, -1]                                 # last real position
        gen_cols = [next_tok]
        for _ in range(n_steps - 1):
            attn = torch.cat(
                [attn, torch.ones((bsz, 1), dtype=torch.long, device=device)], dim=1)
            cur_pos = cur_pos + 1
            out = model(input_ids=next_tok[:, None], attention_mask=attn,
                        position_ids=cur_pos[:, None],
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1)
            gen_cols.append(next_tok)
        generated = torch.stack(gen_cols, dim=1).tolist()    # (bsz, n_steps)

        for i, (_, target_ids, person) in enumerate(chunk):
            gen = generated[i]
            fp_ok = gen[:len(target_ids)] == target_ids
            eos_cut = gen.index(eos_remapped) if eos_remapped in gen else len(gen)
            gen_text = tokenizer.decode([new_to_old[t] for t in gen[:eos_cut]])
            lp_ok = lenient_date_match(gen_text, person)
            results[start + i] = {"FP": int(fp_ok), "LP": int(lp_ok)}
        pbar.update(bsz)
    pbar.close()
    return results


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
    new_to_old   = {v: k for k, v in old_to_new.items()}
    templates = FIELD_SPECS["birthday"]["templates"]
    eval_people = people[:max_people]

    totals = defaultdict(lambda: [0, 0])
    per_template = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    n_pairs = len(eval_people) * len(templates)
    pbar = tqdm(total=n_pairs, desc="probe")
    for person in eval_people:
        for t_idx, template in enumerate(templates):
            scores = score_pair(model, tokenizer, old_to_new, new_to_old,
                                eos_remapped, person, template, device=device)
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
    for metric in ("MP", "DayM", "YearMD", "FP", "LP"):
        c, n = totals[metric]
        acc = c / max(n, 1)
        macro[metric] = acc
        print(f"  {metric:>{width}s}: {c}/{n}  =  {100 * acc:5.1f}%")

    # Per-template macro accuracies, suitable for JSON dump.
    per_t_acc = {
        int(t_idx): {
            m: per_template[t_idx][m][0] / max(per_template[t_idx][m][1], 1)
            for m in ("MP", "DayM", "YearMD", "FP", "LP")
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
"""First-token recall probe for Capo (bioS) models.

Matches Allen-Zhu & Li, Physics of Language Models Part 3.1, Section 3.1:
"BIO first-token accuracy: we track the model's next-token-prediction accuracy
on the first token of each of the six attributes (birthdate, birthcity, ...)
in the BIO data."

Procedure
---------
For each person:
  1. Regenerate one biography using a fixed (master_seed, exposure) pair.
     Because get_text_simple3 is deterministic in this pair, the result is one
     of the K paraphrases the model already saw during training (in-distribution
     prompt — exactly how the paper measures memorization).
  2. Prepend "<|endoftext|>" (matching the training packer) and tokenize.
  3. For each attribute, locate the token index where its value begins by
     tokenizing " {value}" standalone and finding that subsequence in the
     bio's tokens (GPT2Tokenizer slow does not expose offset_mapping).
  4. Run a single teacher-forced forward pass on the whole bio.
  5. At position (attr_pos - 1), check whether the argmax of the logits
     matches the token id at attr_pos.

Vocab remap: the model was trained on a reduced vocab (~3275 tokens). We
rebuild the GPT-2 -> reduced map from the original pre-reduce token file and
apply it before forwarding.

Usage
-----
    # from project root
    python -m eval.recall_probe runs/2L-192D/final
    python -m eval.recall_probe runs/2L-192D/final --n 500
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, GPT2Tokenizer, PreTrainedModel

# Allow running as `python -m eval.recall_probe` from project root.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import Config
from data.bio_text import get_text_simple3
from data.sample_people import sample_people
from data.tokenize_pack import build_vocab_remap


# Attribute name -> key on the `person` dict that gives the substring whose
# first token we want to predict. For multi-token values (e.g. "Princeton, NJ",
# "Massachusetts Institute of Technology") we only score the FIRST token's
# argmax, matching the paper.
ATTR_KEYS = {
    "birthdate":    "birthmonth",   # bio format "{birthmonth} {birthday}, {birthyear}"
    "birthcity":    "birthcity",
    "university":   "university",
    "field":        "field",
    "company_name": "company1name",
    "company_city": "company1city",
}


def _find_subseq(haystack: list[int], needle: list[int], start: int = 0) -> int:
    """Return the first index >= start where `needle` appears in `haystack`, or -1."""
    if not needle:
        return -1
    n, h = len(needle), len(haystack)
    for i in range(start, h - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return -1


def find_attribute_positions(bio: str, person: dict, tokenizer: GPT2Tokenizer):
    """Tokenize `bio` and return (token_ids, {attr: token_index_of_first_token}).

    Works with the slow GPT2Tokenizer (no offset_mapping). All bioS templates
    place an attribute value right after a space, so tokenizing " {value}"
    standalone produces the same token subsequence that appears in the bio
    (GPT-2 byte-level BPE is context-free for leading-space tokens). We then
    locate that subsequence in the bio's full tokenization.
    """
    bio_tokens: list[int] = tokenizer(bio, add_special_tokens=False)["input_ids"]

    positions: dict[str, int] = {}
    cursor = 0
    for attr_name, person_key in ATTR_KEYS.items():
        value = str(person[person_key])
        target = tokenizer(" " + value, add_special_tokens=False)["input_ids"]
        if not target:
            raise ValueError(f"empty target tokens for {attr_name}={value!r}")

        # Search left-to-right; fall back to a full-bio search to handle the
        # sentence-5/6 swap (company name / company city order can flip).
        idx = _find_subseq(bio_tokens, target, start=cursor)
        if idx == -1:
            idx = _find_subseq(bio_tokens, target, start=0)
        if idx == -1:
            raise ValueError(
                f"person {person['id']}: token subseq for {attr_name}={value!r} "
                f"not found in bio"
            )
        positions[attr_name] = idx
        cursor = idx + len(target)

    return bio_tokens, positions


def build_remap_lut(old_to_new: dict[int, int]) -> torch.Tensor:
    """Index-into-able tensor: lut[old_id] -> new_id."""
    max_old = max(old_to_new.keys())
    lut = torch.zeros(max_old + 1, dtype=torch.long)
    for old, new in old_to_new.items():
        lut[old] = new
    return lut


@torch.no_grad()
def evaluate_recall(
    model: PreTrainedModel,
    tokenizer: GPT2Tokenizer,
    people: list[dict],
    lut: torch.Tensor,
    *,
    master_seed: int,
    exposure: int,
    device: str,
    max_people: int | None = None,
) -> dict[str, float]:
    """Return {attr_name: top-1 accuracy} over the (sub)list of people."""
    model.eval()
    correct: dict[str, int] = defaultdict(int)
    total:   dict[str, int] = defaultdict(int)
    skipped = 0

    if max_people is not None:
        people = people[:max_people]

    for person in tqdm(people, desc="recall"):
        bio = "<|endoftext|>" + get_text_simple3(
            person, exposure=exposure, master_seed=master_seed
        )
        try:
            token_ids, positions = find_attribute_positions(bio, person, tokenizer)
        except ValueError as err:
            skipped += 1
            print(f"  skip: {err}", file=sys.stderr)
            continue

        ids_old = torch.tensor(token_ids, dtype=torch.long)
        ids_new = lut[ids_old].unsqueeze(0).to(device)  # (1, seq_len)

        logits = model(ids_new).logits[0]  # (seq_len, vocab)

        for attr, pos in positions.items():
            if pos == 0:
                # No previous token to condition on; can't score.
                continue
            pred = logits[pos - 1].argmax().item()
            true = ids_new[0, pos].item()
            correct[attr] += int(pred == true)
            total[attr]   += 1

    if skipped:
        print(f"(skipped {skipped} people due to subseq lookup failure)",
              file=sys.stderr)

    return {a: correct[a] / max(total[a], 1) for a in ATTR_KEYS}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ckpt", help="Path to a checkpoint directory (e.g. runs/2L-192D/final)")
    p.add_argument("--n", type=int, default=1000, help="Number of people to evaluate")
    p.add_argument("--N", type=int, default=None, help="Total people sampled (must match training)")
    p.add_argument("--seed", type=int, default=None, help="Person-sampling seed (must match training)")
    p.add_argument("--master-seed", type=int, default=None,
                   help="Bio-generation master seed (must match training)")
    p.add_argument("--exposure", type=int, default=0,
                   help="Which of the K training exposures to evaluate (any in [0,K) is fine)")
    p.add_argument("--pre-reduce-path", type=str, default=None,
                   help="Path to the pre-reduce token file used to build the GPT2->reduced map")
    args = p.parse_args()

    cfg = Config()
    N = args.N if args.N is not None else cfg.N
    seed = args.seed if args.seed is not None else cfg.SEED
    master_seed = args.master_seed if args.master_seed is not None else cfg.SEED
    pre_reduce = args.pre_reduce_path or cfg.PRE_REDUCE_PATH

    print(f"Sampling {N:,} people (seed={seed}) ...")
    people = sample_people(N=N, seed=seed)

    print(f"Building vocab remap from {pre_reduce} ...")
    old_to_new, _, reduced_vocab = build_vocab_remap(pre_reduce)
    print(f"  reduced vocab size: {reduced_vocab}")
    lut = build_remap_lut(old_to_new)

    device = pick_device()
    print(f"Loading model from {args.ckpt} on {device} ...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(args.ckpt).to(device)

    # Sanity: model vocab must match remapped vocab.
    if model.config.vocab_size != reduced_vocab:
        print(f"  WARNING: model vocab={model.config.vocab_size} but remap={reduced_vocab}. "
              "Did the pre-reduce file change since training?",
              file=sys.stderr)

    n_eval = min(args.n, len(people))
    print(f"Evaluating on {n_eval} people, exposure={args.exposure} ...")
    results = evaluate_recall(
        model, tokenizer, people, lut,
        master_seed=master_seed, exposure=args.exposure,
        device=device, max_people=n_eval,
    )

    print("\nFirst-token recall accuracy (Capo bioS Section 3.1 style):")
    width = max(len(a) for a in results)
    for attr, acc in results.items():
        print(f"  {attr:>{width}s}: {acc * 100:5.1f}%")
    mean = sum(results.values()) / len(results)
    print(f"  {'mean':>{width}s}: {mean * 100:5.1f}%")


if __name__ == "__main__":
    main()

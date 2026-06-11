"""Equality test: the batched KV-cached decoder reproduces score_pair's FP/LP.

This is the correctness guarantee behind eval/robustness_probe.py's switch from
a per-pair, no-cache greedy loop to score_pairs_fp_lp_batched. It builds a tiny
*real* Llama on CPU (random weights — we test equivalence, not accuracy), then
asserts that for every (person, template) pair the batched decoder's FP/LP match
the reference score_pair exactly, including across batch boundaries and varied
prefix lengths (pairs are shuffled so each left-padded batch mixes people and
templates — the hardest case for the padding + position-id handling).

Runs on CPU in a few seconds. Requires torch + the cached gpt2 tokenizer (the
same deps the rest of the eval suite already imports).
"""
import random

import pytest

pytest.importorskip("torch")

from transformers import GPT2Tokenizer

from data.bio_text import FIELD_SPECS
from data.sample_people import sample_people
from model.buildModel import create_llama_model
from eval.birthday_probe_legacy import (
    build_chunks,
    score_pair,
    score_pairs_fp_lp_batched,
)

TEMPLATES = FIELD_SPECS["birthday"]["templates"]


def _build_remap(people, templates, tokenizer):
    """Compact GPT-2 -> reduced remap covering every token these pairs use."""
    ids = {int(tokenizer.eos_token_id)}
    for person in people:
        for template in templates:
            chunks = build_chunks(person, template)
            for part in ("prefix", "month", "day", "sep", "year"):
                ids.update(
                    tokenizer(chunks[part], add_special_tokens=False)["input_ids"])
    return {int(o): n for n, o in enumerate(sorted(ids))}


def test_batched_fp_lp_matches_score_pair():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    people = sample_people(N=12, seed=0)
    templates = TEMPLATES

    old_to_new = _build_remap(people, templates, tokenizer)
    new_to_old = {v: k for k, v in old_to_new.items()}
    eos_remapped = old_to_new[int(tokenizer.eos_token_id)]

    # Tiny real model — random weights are fine, we test path-equivalence.
    model = create_llama_model(
        vocab_size=len(old_to_new), block_size=128,
        hidden_size=32, n_layer=2, n_head=2,
        eos_token=eos_remapped, seed=0,
    )
    model.eval()
    device = "cpu"

    pairs = [(person, template) for person in people for template in templates]
    # Shuffle so each left-padded batch mixes people (different name lengths) and
    # templates (different prefix lengths).
    random.Random(0).shuffle(pairs)

    reference = [
        score_pair(model, tokenizer, old_to_new, new_to_old, eos_remapped,
                   person, template, device=device)
        for person, template in pairs
    ]
    batched = score_pairs_fp_lp_batched(
        model, tokenizer, old_to_new, new_to_old, eos_remapped,
        pairs, device, batch_size=16,
    )

    assert len(batched) == len(pairs)
    mismatches = [
        (person["id"], template, ref["FP"], ref["LP"], got["FP"], got["LP"])
        for (person, template), ref, got in zip(pairs, reference, batched)
        if ref["FP"] != got["FP"] or ref["LP"] != got["LP"]
    ]
    assert not mismatches, (
        f"{len(mismatches)}/{len(pairs)} FP/LP mismatches vs score_pair; "
        f"first few (id, template, refFP, refLP, gotFP, gotLP): {mismatches[:5]}"
    )

#!/usr/bin/env python3
"""Deterministically rebuild the bioS training corpus that is too large to ship.

The two token files (`bios_prereduce.bin`, `bios_postreduce.bin`) are 180 MB each
— over GitHub's 100 MB per-file limit — so they are NOT in this repo. They are a
pure function of the seeds in `data_config.json`, so this script rebuilds them
bit-for-bit.

This is a thin driver over the repo's OWN data functions (`data.sample_people`,
`data.bio_text`, `data.tokenize_pack`) — the same ones `main.py` calls — so it
cannot drift from the real training pipeline.

Usage
-----
    # rebuild into release/data/bioS_N-Bd_final_grid/ and verify checksums
    python release/regenerate_data.py

    # rebuild somewhere else
    python release/regenerate_data.py --out /path/to/dir

    # fast sanity check (2000 bios, no full corpus) — takes seconds
    python release/regenerate_data.py --smoke

Expected runtime for the full corpus: a few minutes, ~360 MB of output.
Requires: transformers, numpy, tqdm, torch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.bio_text import bio_stream, render_bio          # noqa: E402
from data.sample_people import sample_people              # noqa: E402
from data.tokenize_pack import (                          # noqa: E402
    build_vocab_remap,
    remap_token_file,
    tokenize_and_pack,
)

DEFAULT_OUT = REPO_ROOT / "release" / "data" / "bioS_N-Bd_final_grid"

# sha256 of the artifacts as produced by the original run. Verified 2026-08-15.
EXPECTED = {
    "people.json":         "74974c5dd0c08b20068827a903df0e613166db5bbec2c4e133fdc5bf9106ee39",
    "old_to_new.json":     "0518e6328b539da0e03d56c3c7bbbeebae32c9412478da9af16f996b5540a7b6",
    "bios_prereduce.bin":  "fed7b35fe63126f2b5c5b5006fcd323a3e81e5708fce4e96c7ab7bd90f8be88a",
    "bios_postreduce.bin": "09edc5478cc9b835b2eade3e40feb53767a8b49361ab417fe30719bf52254192",
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def report(path: Path) -> bool:
    """Print PASS/FAIL for one artifact against its known-good checksum."""
    if not path.exists():
        print(f"    {path.name:22s} MISSING")
        return False
    got = sha256(path)
    want = EXPECTED.get(path.name)
    if want is None:
        print(f"    {path.name:22s} {got}  (no reference)")
        return True
    ok = got == want
    print(f"    {path.name:22s} {'PASS' if ok else 'FAIL'}  {got}")
    if not ok:
        print(f"    {'':22s}      expected {want}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output dir (default: release/data/bioS_N-Bd_final_grid)")
    ap.add_argument("--config", type=Path, default=DEFAULT_OUT / "data_config.json",
                    help="data_config.json holding N/K/SEED/SHUFFLE_SEED/FIELDS")
    ap.add_argument("--smoke", action="store_true",
                    help="verify the first 2000 bios against people.json only; no corpus written")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    N, K = cfg["N"], cfg["K"]
    SEED, SHUFFLE_SEED = cfg["SEED"], cfg["SHUFFLE_SEED"]
    SEQ_LEN, FIELDS = cfg["SEQ_LEN"], tuple(cfg["FIELDS"])
    print(f"config: N={N:,} K={K} SEED={SEED} SHUFFLE_SEED={SHUFFLE_SEED} "
          f"SEQ_LEN={SEQ_LEN} FIELDS={FIELDS}")

    # ---- 1. people: deterministic in SEED alone -----------------------------
    print(f"\n[1/4] sampling {N:,} people (seed={SEED})")
    people = sample_people(N=N, seed=SEED)

    if args.smoke:
        blob = json.dumps(people).encode()
        got = hashlib.sha256(blob).hexdigest()
        ok = got == EXPECTED["people.json"]
        print(f"      people.json sha256 {'PASS' if ok else 'FAIL'}  {got}")
        pairs = [(i, e) for i in range(len(people)) for e in range(K)]
        random.Random(SHUFFLE_SEED).shuffle(pairs)
        print(f"      shuffled {len(pairs):,} (person, exposure) pairs; first 3 = {pairs[:3]}")
        i, e = pairs[0]
        print(f"      first bio: {render_bio(people[i], exposure_idx=e, fields=FIELDS)!r}")
        return 0 if ok else 1

    args.out.mkdir(parents=True, exist_ok=True)
    people_path = args.out / "people.json"
    with open(people_path, "w") as f:
        json.dump(people, f)

    # ---- 2. render + tokenize + pack ---------------------------------------
    # bio_stream applies random.Random(SHUFFLE_SEED).shuffle to the full
    # (person, exposure) grid, so ordering is fixed by the seed, not by
    # iteration order or thread scheduling.
    pre = args.out / "bios_prereduce.bin"
    print(f"\n[2/4] rendering + tokenizing {N * K:,} bios with the GPT-2 tokenizer")
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    stream = bio_stream(people, K=K, master_seed=SEED,
                        shuffle_seed=SHUFFLE_SEED, fields=FIELDS)
    n_tokens, n_seq = tokenize_and_pack(
        tokenizer, stream, n_bios_total=N * K, out_path=pre, seq_len=SEQ_LEN,
    )
    print(f"      {n_tokens:,} tokens -> {n_seq:,} sequences of {SEQ_LEN}")

    # ---- 3. condense the vocabulary ----------------------------------------
    # GPT-2 emits ids in [0, 50257) but bioS only ever uses ~1836 of them.
    # build_vocab_remap ranks the sorted unique ids, so the mapping is a pure
    # function of the corpus — same corpus, same map, every time.
    print("\n[3/4] condensing vocab (GPT-2 50257 -> dense rank map)")
    old_to_new, _, reduced_vocab = build_vocab_remap(pre)
    post = args.out / "bios_postreduce.bin"
    remap_token_file(pre, post, old_to_new)
    with open(args.out / "old_to_new.json", "w") as f:
        json.dump({str(k): int(v) for k, v in old_to_new.items()}, f)
    print(f"      reducedVocabSize = {reduced_vocab} (config says {cfg['reducedVocabSize']})")
    print(f"      eos {cfg['eosToken']} -> {old_to_new[int(cfg['eosToken'])]} "
          f"(config says {cfg['reducedEOSToken']})")

    # ---- 4. verify ----------------------------------------------------------
    print("\n[4/4] verifying against known-good checksums")
    results = [report(args.out / n) for n in
               ("people.json", "old_to_new.json", "bios_prereduce.bin", "bios_postreduce.bin")]
    if all(results):
        print("\nAll artifacts reproduced exactly. ✅")
        return 0
    print("\nMISMATCH — see FAIL lines above. Most likely cause: a different "
          "GPT-2 tokenizer version changing how bios tokenize.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

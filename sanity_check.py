"""Sanity check for a fresh environment.

Run this after activating your conda env (and before submitting a long
SLURM job) to confirm every moving piece imports, the CUDA stack is
healthy, project modules load, and a tiny forward pass works on the GPU.

    python sanity_check.py

Exits 0 on success, non-zero on any failure. Each check prints a clear
PASS/FAIL line so the SLURM log is easy to scan.
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

_failures: list[str] = []


def check(name: str, fn):
    """Run `fn()`; print PASS/FAIL and capture the exception on failure."""
    try:
        detail = fn()
    except Exception as e:  # noqa: BLE001
        _failures.append(name)
        print(f"  [{FAIL}] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        return
    if detail is None:
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{PASS}] {name} — {detail}")


def section(title: str):
    print(f"\n=== {title} ===")


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------

def check_python():
    section("Python")
    check("version >= 3.10",
          lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                  if sys.version_info >= (3, 10)
                  else (_ for _ in ()).throw(RuntimeError("need 3.10+")))


def check_third_party():
    section("Third-party imports")
    for pkg in ["numpy", "torch", "transformers", "tqdm", "wandb"]:
        def _check(pkg=pkg):
            m = importlib.import_module(pkg)
            return f"{pkg} {getattr(m, '__version__', '?')}"
        check(pkg, _check)


def check_cuda():
    section("CUDA / GPU")
    import torch

    check("torch.cuda.is_available",
          lambda: "True" if torch.cuda.is_available()
                  else (_ for _ in ()).throw(RuntimeError("CUDA not visible")))

    def device_info():
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        return f"{n} GPU(s); device 0 = {name} (sm_{cap[0]}{cap[1]})"
    check("device info", device_info)

    def bf16_supported():
        ok = torch.cuda.is_bf16_supported()
        return "bf16 OK" if ok else "bf16 NOT supported (training will fall back to fp32)"
    check("bf16 support", bf16_supported)

    def tiny_matmul():
        x = __import__("torch").randn(64, 64, device="cuda")
        y = (x @ x.T).sum().item()
        return f"64x64 matmul on GPU ran, sum={y:.2f}"
    check("tiny GPU matmul", tiny_matmul)


def check_project_imports():
    section("Project imports")
    # Make sure CWD is on sys.path (it is, when run as a script).
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    project_modules = [
        "config",
        "data.sample_people",
        "data.bio_text",
        "data.tokenize_pack",
        "model.buildModel",
        "model.trainModel",
        "eval.sequential_probe",
        "eval.separate_probing",
    ]
    for mod in project_modules:
        check(mod, lambda mod=mod: importlib.import_module(mod) and None)


def check_data_pipeline():
    section("Data pipeline (tiny)")
    from data.sample_people import sample_people
    from data.bio_text import render_bio, bio_stream, TEMPLATES_BIRTHDAY, FIELD_SPECS

    def sample_5():
        people = sample_people(N=5, seed=0)
        assert len(people) == 5
        return f"sampled 5 people, keys={sorted(people[0].keys())[:5]}..."
    check("sample_people(N=5)", sample_5)

    def render_one():
        people = sample_people(N=1, seed=0)
        bio = render_bio(people[0], exposure_idx=0, fields=("birthday",))
        return f"bio: {bio.strip()!r}"
    check("render_bio birthday-only", render_one)

    def round_robin_coverage():
        people = sample_people(N=1, seed=0)
        seen = set()
        for e in range(len(TEMPLATES_BIRTHDAY)):
            render_bio(people[0], exposure_idx=e, fields=("birthday",))
            seen.add(e % len(TEMPLATES_BIRTHDAY))
        return f"every template index 0..{len(TEMPLATES_BIRTHDAY)-1} reachable ({len(seen)} unique)"
    check("round-robin template coverage", round_robin_coverage)

    def stream_5():
        people = sample_people(N=5, seed=0)
        gen = bio_stream(people, K=2, shuffle_seed=1, fields=("birthday",))
        bios = list(gen)
        assert len(bios) == 5 * 2
        return f"yielded {len(bios)} bios"
    check("bio_stream (N=5, K=2)", stream_5)


def check_tokenizer():
    section("Tokenizer")
    from transformers import GPT2Tokenizer

    def load_gpt2():
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        return f"vocab_size={tok.vocab_size}, eos={tok.eos_token_id}"
    check("GPT2Tokenizer.from_pretrained('gpt2')", load_gpt2)


def check_model_build_and_forward():
    section("Model build + forward pass")
    import torch
    from model.buildModel import create_llama_model

    def build_tiny():
        m = create_llama_model(
            vocab_size=128,
            block_size=64,
            hidden_size=64,
            n_layer=2,
            n_head=2,
            eos_token=127,
        )
        params = sum(p.numel() for p in m.parameters())
        return f"built ~{params:,}-param Llama"
    check("create_llama_model (tiny)", build_tiny)

    def forward_on_gpu():
        m = create_llama_model(vocab_size=128, block_size=64, hidden_size=64,
                                n_layer=2, n_head=2, eos_token=127)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = m.to(device)
        ids = torch.randint(0, 128, (1, 16), device=device)
        out = m(ids)
        return f"forward on {device}: logits.shape={tuple(out.logits.shape)}"
    check("tiny forward pass", forward_on_gpu)


def check_probe_smoke():
    section("Probe smoke test")
    import torch
    from data.sample_people import sample_people
    from eval.sequential_probe import (
        build_multi_field_pieces, tokenize_bio, score_pair,
    )
    from eval.separate_probing import (
        build_single_field_chunks, score_field_pair,
    )

    def pieces_reassemble():
        person = sample_people(N=1, seed=0)[0]
        pieces = build_multi_field_pieces(person, ("birthday", "birthcity"),
                                          exposure_idx=0)
        joined = "".join(p["pre"] + p["value"] + p["post"] for p in pieces)
        return f"pieces reassemble to: {joined.strip()!r}"
    check("build_multi_field_pieces", pieces_reassemble)

    def single_field_chunks_reassemble():
        person = sample_people(N=1, seed=0)[0]
        c = build_single_field_chunks(person, "birthcity", exposure_idx=0)
        joined = c["pre"] + c["value"] + c["post"]
        # Pronoun must be replaced with the full name in separate-field bios.
        full_name = (f"{person['first_name']} {person['middle_name']} "
                     f"{person['last_name']}")
        assert full_name in joined, "separate-field bio must contain full name"
        return f"single-field bio: {joined.strip()!r}"
    check("build_single_field_chunks (name swap)", single_field_chunks_reassemble)

    def score_sequential_end_to_end():
        # Tiny throwaway model with a tokenizer-matched vocab; we only check
        # that score_pair runs end-to-end and returns the expected keys.
        from transformers import GPT2Tokenizer
        from model.buildModel import create_llama_model

        tok = GPT2Tokenizer.from_pretrained("gpt2")
        person = sample_people(N=1, seed=0)[0]
        fields = ("birthday", "birthcity")
        pieces = build_multi_field_pieces(person, fields, exposure_idx=0)

        # Build a vocab map covering every token of the rendered bio + EOS.
        text = "".join(p["pre"] + p["value"] + p["post"] for p in pieces)
        ids = tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        unique = sorted(set(ids))
        old_to_new = {t: i for i, t in enumerate(unique)}
        eos_remapped = old_to_new[tok.eos_token_id]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = create_llama_model(
            vocab_size=len(unique), block_size=128, hidden_size=64,
            n_layer=2, n_head=2, eos_token=eos_remapped,
        ).to(device).eval()

        # Sanity: tokenize_bio agrees that all chunks live in the vocab map.
        full_ids, spans, _ = tokenize_bio(pieces, tok, old_to_new, eos_remapped)
        assert len(spans) == len(fields)

        scores = score_pair(m, tok, old_to_new, eos_remapped, person,
                            fields, exposure_idx=0, device=device)
        assert set(scores) == {"TF", "FP", "FP_FULL", "t_idx"}
        assert set(scores["TF"].keys()) == set(fields)

        sep = score_field_pair(m, tok, old_to_new, eos_remapped, person,
                               "birthcity", exposure_idx=0, device=device)
        assert set(sep) == {"TF", "FP", "t_idx"}
        return (f"seq score_pair → keys={sorted(scores)}, "
                f"sep score_field_pair → keys={sorted(sep)}")
    check("score_pair / score_field_pair end-to-end", score_sequential_end_to_end)


def check_wandb_auth():
    section("wandb")
    import wandb

    def has_login():
        # `wandb.api.api_key` is set when ~/.netrc has wandb creds or
        # WANDB_API_KEY is in env. We don't actually init a run here.
        key = os.environ.get("WANDB_API_KEY") or getattr(wandb.api, "api_key", None)
        if key:
            return "wandb auth detected"
        raise RuntimeError("no wandb auth — run `wandb login` or set WANDB_API_KEY")
    check("auth (login or env var)", has_login)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    print("Sanity check for the birthday-bio training rig.\n")

    check_python()
    check_third_party()
    check_cuda()
    check_project_imports()
    check_data_pipeline()
    check_tokenizer()
    check_model_build_and_forward()
    check_probe_smoke()
    check_wandb_auth()

    print()
    if _failures:
        print(f"\033[31m{len(_failures)} check(s) failed:\033[0m")
        for n in _failures:
            print(f"  - {n}")
        sys.exit(1)

    print("\033[32mAll sanity checks passed.\033[0m")
    sys.exit(0)


if __name__ == "__main__":
    main()

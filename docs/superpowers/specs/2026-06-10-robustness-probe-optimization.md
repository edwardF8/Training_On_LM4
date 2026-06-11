# RobustnessTest — Probe Performance Optimization

**Date:** 2026-06-10
**Status:** implemented
**Supersedes** parts of [2026-06-10-robustness-test-design.md](2026-06-10-robustness-test-design.md)
(probe non-goals §"No KV-cache or batching optimizations"; `MAX_PEOPLE_PER_GROUP` default).

## Why

The robustness sweep on PSC (job 41279551, V100) was **not** training slowly — the
training steps ran at ~41 it/s (~48 min of compute for a full 16-epoch model).
The wall-clock was dominated by the **probe**:

- one `birthday_robust` probe = **9,200 pairs at ~4.5 it/s ≈ 35 min**, decoded one
  (person, template) pair at a time with **no KV cache** (each of the 32 greedy
  steps re-encoded the whole prefix) and **batch size 1** (the V100 sat nearly idle);
- it fires at **7 epochs** `[1,2,4,6,8,12,16]` → ~4 h of probing per model;
- across the **20-model grid** that is ~80 h of pure probe time — the sweep could not
  finish inside the 24 h walltime.

So probing was ~83% of per-model wall-clock. This change attacks the probe, not the
training loop, and does **not** change the number of epochs or probe epochs.

## What changed (minimal, additive)

1. **KV cache + batching** — new
   [`score_pairs_fp_lp_batched`](../../../eval/birthday_probe_legacy.py) next to the
   existing `score_pair`. It greedy-decodes many pairs at once:
   - prefixes are **left-padded** so every sequence's last real token sits in the
     final column (next-token logit at `[:, -1]` for the whole batch), with an
     attention mask + RoPE `position_ids` derived from that mask so the padding is
     inert;
   - `past_key_values` is carried across steps (`use_cache=True`), so each step
     forwards only the single new token.
   Both are exact-equivalence transforms of greedy argmax, so per-pair FP/LP are
   **identical** to looping `score_pair`. Only FP (greedy) and LP (lenient) are
   computed — the robustness probe never used the teacher-forced MP/DayM/YearMD, so
   that forward pass is dropped too.

   [`eval/robustness_probe.py`](../../../eval/robustness_probe.py) now flattens its
   eval plan to `(person, template)` pairs, scores them with the batched function in
   chunks of `PROBE_BATCH_SIZE = 128`, then runs the **same** bucket math as before
   (`total / limitedSet / fullSet / limitedSeen / limitedUnseen`). Bucketing is
   byte-identical; only the scoring path changed.

   `model.config.use_cache` stays `False` (set in [`model/buildModel.py`](../../../model/buildModel.py))
   — caching is enabled **per call** in the probe via `use_cache=True`, so **training is
   untouched**.

2. **Fewer eval people** —
   [`ablation_llama_grid_robustness.py`](../../../ablation_llama_grid_robustness.py)
   `MAX_PEOPLE_PER_GROUP` **100 → 50**, matching the non-robustness `birthday_legacy`
   probe's `max_people=50` (each group now sized like that probe). Pairs/probe:
   9,200 → **4,600**. This reverts the design's "bumped from 50/group" decision.

3. **Unchanged on purpose:** `MAX_EPOCHS=16`, `RUN_PROBE_AFTER=[1,2,4,6,8,12]` (+16),
   the manifest/data pipeline, the JSON/CSV/wandb schema, and the non-robustness probe
   (`birthday_legacy` still calls the original per-pair `score_pair`).

## "Same results" guarantee

[`tests/test_probe_batching.py`](../../../tests/test_probe_batching.py) builds a tiny
real Llama on CPU and asserts `score_pairs_fp_lp_batched` reproduces `score_pair`'s
FP/LP **exactly** for every pair, with pairs **shuffled** so each left-padded batch
mixes people (different name lengths) and templates (different prefix lengths) — the
hardest case for the padding + position-id handling.

`tests/test_robustness_probe.py` is updated to stub the batched scorer (same bucket-math
assertions as before).

### Verification status

- **arena-env (local Mac), torch 2.11 / transformers 4.57.6:** `pytest tests/ -q` →
  **17 passed** (incl. the equality test over 552 shuffled pairs). This is the
  authoritative proof; 4.57.6 is newer than PSC's stack.
- **PSC lm4, torch 2.12 / transformers 4.56.2:** the decode uses only version-stable
  `forward(input_ids, attention_mask, position_ids, past_key_values, use_cache=True)`
  calls, so behavior matches. A login-node run was blocked by transient `/jet` I/O
  degradation (the conda env lives on `/jet`); the source is staged at
  `~/data_storage/probe_opt_verify/` with `verify.sbatch` to run on a compute node once
  `/jet` is healthy (note: this allocation has no CPU/RM QOS, so run it inside the usual
  GPU job or fix the partition).

## Expected impact

Batched + KV-cached decode turns a ~35 min, batch-1 probe into a small fraction of that
(GPU now does real batches), and 4,600 vs 9,200 pairs halves the work again. Probing
stops being the sweep's bottleneck; the 20-model grid should fit the 24 h walltime
comfortably.

## Deploy note

Edits are to source files only; the **running job is unaffected** (its code is already
in memory). To pick up the change, **resubmit with a fresh `INVOCATION`** so every grid
run uses 50/group + the batched probe consistently (a same-`INVOCATION` resume would
skip already-`final/` runs, mixing 100- and 50-group eval sets). The cached bios bins are
reused — `MAX_PEOPLE_PER_GROUP` is a probe-only knob and is **not** part of the data
fingerprint, so no retokenize is triggered.

# RobustnessTest — Design

**Date:** 2026-06-10
**Status:** approved (brainstorm w/ Edward, 2026-06-10)

> **Update (2026-06-10):** the probe was the sweep's wall-clock bottleneck. See
> [2026-06-10-robustness-probe-optimization.md](2026-06-10-robustness-probe-optimization.md),
> which supersedes the "No KV-cache or batching" non-goal below and lowers
> `MAX_PEOPLE_PER_GROUP` 100 → 50. Probe FP/LP results are unchanged (proven by
> `tests/test_probe_batching.py`).

## Research question

Does the model still memorize a person's birthday when that person's bios appear
with **low paraphrase diversity** (12 of 46 templates) instead of full diversity —
and can it retrieve the fact through paraphrases that person was never trained
with?

## Constraints

- **Pure add-on.** No existing file is modified. New modules + a new sibling
  ablation script only. Existing scripts keep working unchanged.
- Same dataset shape as a normal run: every person still gets exactly K bios;
  total corpus size N·K is unchanged.
- Everything seeded and documented: which people are limited and which
  templates each kept must be saved to disk.

## Parameters (knobs in the new script)

| Knob | Default | Meaning |
|---|---|---|
| `LIMITED_PEOPLE_FRAC` | 0.20 | fraction of people with restricted templates (→ 10,000 of N=50,000) |
| `LIMITED_TEMPLATE_FRAC` | 0.25 | fraction of templates each limited person keeps (`round(0.25·46)` = **12**) |
| `ROBUSTNESS_SEED` | 0 | seeds the manifest sampling (its own `random.Random`) |
| `MAX_PEOPLE_PER_GROUP` | 100 | eval people per group for the probe (100 limited + 100 full) |

## 1. Data — `data/robustness.py` (new)

### Manifest

`build_robustness_manifest(n_people, fields, limited_frac, template_frac, seed)`:

- One `random.Random(seed)`: sample `round(limited_frac · n_people)` person
  indices; for each (ascending order) independently sample
  `round(template_frac · T_field)` template indices per field. Independent
  sampling ⇒ each limited person gets their own subset.
- Manifest dict (JSON-serialized):

```json
{
  "seed": 0,
  "limited_frac": 0.2,
  "template_frac": 0.25,
  "n_people": 50000,
  "fields": ["birthday"],
  "n_templates": {"birthday": 46},
  "limited_people": { "<person_index>": {"birthday": [2, 5, 9, ...12 idxs]}, ... }
}
```

- Saved to `cache/{NAME}/robustness_manifest.json` **and** copied to
  `runs/{NAME}/{INVOCATION}/robustness_manifest.json` so every run is
  self-describing. `save_manifest` / `load_manifest` helpers (load converts
  person keys back to `int`).

### Stream

`robust_bio_stream(people, K, manifest, shuffle_seed, fields)`:

- Builds the **identical** shuffled (person_idx, exposure_idx) pair sequence as
  `bio_stream` (same pair construction, same `random.Random(shuffle_seed)`
  shuffle).
- Full-set person → `render_bio(person, exposure_idx=e, fields)` — byte-identical
  to today.
- Limited person → template for each field f is `allowed_f[e % len(allowed_f)]`,
  rendered by `render_bio_limited(person, template_idx_by_field, fields)`, a
  small mirror of `render_bio`'s loop living in `data/robustness.py`
  (so `data/bio_text.py` is untouched). Each limited person still emits exactly
  K bios; with K=100 over 12 templates each allowed template appears 8–9×
  (vs 2–3× per template for full-set people).
- Properties preserved: all 46 templates still occur in the corpus (via the 80%
  full-set people) ⇒ reduced vocab is **identical** to a normal run ⇒ models
  stay comparable across normal and robustness sweeps.

### Seatbelts (run at data-gen time in the script)

- Re-render a handful of limited people's K bios and assert each matches one of
  that person's allowed templates (string-level check against all candidate
  renderings).
- Keep the existing `assert_tokens_in_remap` check over sample prompts covering
  all 46 templates.

## 2. Probe — `eval/robustness_probe.py` (new)

- **Scoring is 100% reused:** calls the existing
  `eval.birthday_probe_legacy.score_pair` unchanged and tallies only the `FP`
  and `LP` fields it returns. (It also computes MP/DayM/YearMD in the same
  pass — ~3% overhead vs the 32-step greedy decode — which we discard.)
  - **FP (greedy probe):** greedy decode from the template prefix must produce
    the date span exactly.
  - **LP (lenient probe):** free greedy generation; pass iff the true date
    appears anywhere and no other date appears.
- **Eval set (deterministic):** first `MAX_PEOPLE_PER_GROUP` limited person
  indices (ascending) + first `MAX_PEOPLE_PER_GROUP` non-limited indices
  (ascending). Every eval person is probed on **all 46 templates**.
- **Buckets** (each metric = pairs-correct / pairs-attempted, stored 0–1):

| Bucket | Pairs @ 100/group | Question answered |
|---|---|---|
| `total` | 9,200 | headline number, balanced pool |
| `fullSet` | 4,600 | control group — should match a normal run |
| `limitedSet` | 4,600 | treatment group overall |
| `limitedSeen` | 1,200 | rote recall via the person's own 12 templates (8–9 reps each) |
| `limitedUnseen` | 3,400 | generalization: templates never seen with this person |

- ×2 judges (FP, LP) ⇒ **10 scalar metrics**, all fractions 0–1.
- **Registration without touching `eval/probes.py`:**
  `make_robust_runner(manifest, max_people_per_group)` returns a closure
  matching the registry signature
  `(model, tokenizer, old_to_new, people, fields, max_people)`; the script
  inserts it:
  `PROBE_REGISTRY["birthday_robust"] = make_robust_runner(manifest, MAX_PEOPLE_PER_GROUP)`.
  The baked-in `max_people_per_group` is authoritative; the closure ignores the
  registry-passed `max_people` (which `ProbeAtEpochs` → `run_probes` leaves at
  its default of 50 — it must not silently shrink the eval set).
- **Outputs per probe epoch:**
  - wandb scalars: `robustProbe/{FP,LP}/{total,limitedSet,fullSet,limitedSeen,limitedUnseen}`
  - wandb table `robustProbe/per_template`:
    `template_idx, FP_total, FP_limited, FP_full, LP_total, LP_limited, LP_full`
  - JSON (`probes/probe_robust_epoch{N}.json`, final epoch →
    `final/probe_robust.json`):

```json
{
  "macro":  { "FP": {"total": 0.83, "limitedSet": 0.71, "fullSet": 0.95,
                     "limitedSeen": 0.88, "limitedUnseen": 0.65},
              "LP": { "...same keys..." : 0.0 } },
  "counts": { "FP": {"total": [7636, 9200], "limitedSeen": [1056, 1200], "...": []},
              "LP": { "...": [] } },
  "per_template": { "0": {"FP_total": 0.0, "FP_limited": 0.0, "FP_full": 0.0,
                          "LP_total": 0.0, "LP_limited": 0.0, "LP_full": 0.0} },
  "eval_people": { "limited": ["...100 indices..."], "full": ["...100 indices..."] },
  "n_people": {"limited": 100, "full": 100},
  "n_templates": 46
}
```

  `counts` stores `[correct, attempted]` so buckets can be re-weighted/pooled
  later without re-probing.

## 3. Callback — `eval/robust_probe_callback.py` (new)

`RobustProbeAtEpochs(ProbeAtEpochs)` — inherits epoch scheduling, wandb step
alignment, and error handling; overrides only the CSV columns/row extraction.
CSV columns:

```
family, study, run_name, numLayers, numHeads, dmodel, epoch,
FP_total, FP_limited, FP_full, FP_limitedSeen, FP_limitedUnseen,
LP_total, LP_limited, LP_full, LP_limitedSeen, LP_limitedUnseen
```

(CSV uses the short `_limited`/`_full` forms; wandb/JSON use
`limitedSet`/`fullSet`. Values rounded to 4 decimals, fractions 0–1.)

## 4. Script — `ablation_llama_grid_robustness.py` (new sibling)

Copy of `ablation_llama_grid.py` with:

- `CONFIG.NAME = "bioS_N-Bd_robust_grid"` → its own `cache/` + `runs/` trees.
- The four knobs above; robustness params added to the wandb run config.
- Manifest built (deterministic) + saved each invocation; copied into the
  invocation's runs dir.
- `robust_bio_stream` in place of `bio_stream`; limited-person seatbelt check
  before tokenization.
- `PROBES = ("birthday_robust",)` only; runner registered via
  `make_robust_runner(manifest)`; callback is `RobustProbeAtEpochs`.
- Everything else carries over unchanged: GRID axes (L×H), `MAX_EPOCHS=16`,
  `RUN_PROBE_AFTER=[1,2,4,6,8,12]`, resume-by-`INVOCATION`, skip-if-`final/`
  exists.
- Probe cost note: 9,200 pairs per probe epoch ≈ 4× the current 2,300-pair
  probe; `MAX_PEOPLE_PER_GROUP` dials it.

## 5. Testing — `tests/` (pytest, CPU-only, no training)

- **Manifest:** determinism (same seed ⇒ identical manifest); counts (10,000
  limited people, 12 template idxs each, all within range, no duplicates per
  person); subsets vary across people; save→load round-trip restores int keys.
- **Stream:** every person emits exactly K bios; total = N·K; limited people's
  texts match only their allowed templates; full people's texts byte-identical
  to `bio_stream` output; pair order identical to `bio_stream` for the same
  `shuffle_seed`. (Small N/K for speed.)
- **Probe tallies:** stub `score_pair` with deterministic fake scores; assert
  pairs land in the right buckets (total/limited/full/seen/unseen), counts and
  macros agree, and the closure uses its baked-in `max_people_per_group`
  (ignoring the registry-passed `max_people`).

## Decisions log

1. Probe limited people on **all 46 templates**, with `limitedSeen`/`limitedUnseen`
   sub-split (chosen over seen-only probing).
2. Eval set = **100 limited + 100 full**, `total` = pooled accuracy over all
   200 (chosen over population-weighted total or `people[:50]` status quo;
   bumped from 50/group on request).
3. **12 templates** per limited person (`round(11.5)`), not 11.
4. Reuse the existing `score_pair` verbatim; report **FP and LP only**.
5. All metrics stored as **fractions 0–1** (existing convention); read as %.

## Non-goals

- No changes to `ablation_llama_grid.py`, `eval/probes.py`,
  `eval/probe_callback.py`, `data/bio_text.py`, or any other existing file.
- Multi-field robustness is structurally supported by the manifest
  (per-field subsets) but only the birthday field is exercised/probed.
- No KV-cache or batching optimizations to the probe decode loop.

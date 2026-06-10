# RobustnessTest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a RobustnessTest pipeline — 20% of people train on only 12/46 birthday templates (per-person random subsets, manifest-documented), probed with FP (greedy) + LP (lenient) only, bucketed into total / limitedSet / fullSet / limitedSeen / limitedUnseen.

**Architecture:** Pure add-on, four new modules + tests; **zero existing files modified**. `data/robustness.py` builds a seeded manifest and a restricted bio stream; `eval/robustness_probe.py` reuses the existing `score_pair` verbatim and only tallies FP/LP into buckets; `eval/robust_probe_callback.py` subclasses `ProbeAtEpochs` to widen the CSV; `ablation_llama_grid_robustness.py` is a sibling of `ablation_llama_grid.py` wiring it together.

**Tech Stack:** Python, pytest, PyTorch/HF Transformers (not exercised by tests — scoring is stubbed), wandb.

**Spec:** `docs/superpowers/specs/2026-06-10-robustness-test-design.md` (approved). Read it before starting.

**Environment notes for the executor:**
- Run everything from the project root: `/Users/efmac/Code/Project Code/CRL-Interp/Training_On_LM4` (quote the path — it contains spaces).
- Use `python -m pytest` (not bare `pytest`) so the project root lands on `sys.path`.
- Tests import `torch`, `transformers`, `wandb` transitively — run them in the same Python environment used for training. If `import torch` fails, find the right conda env before proceeding (`conda env list`), do not `pip install torch`.
- `git status` already shows a pre-existing modification to `figure-generator/figuregenerator.ipynb` — leave it alone; never `git add -A`.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `conftest.py` | create (empty) | puts project root on `sys.path` for pytest |
| `data/robustness.py` | create | manifest build/save/load; `render_bio_limited`; `robust_bio_stream`; `verify_limited_rendering` seatbelt |
| `eval/robustness_probe.py` | create | eval-set selection; FP/LP bucket tallies (reusing `birthday_probe_legacy.score_pair`); `make_robust_runner` registry factory |
| `eval/robust_probe_callback.py` | create | `RobustProbeAtEpochs(ProbeAtEpochs)` — robustness CSV schema |
| `ablation_llama_grid_robustness.py` | create | sibling grid script: manifest + robust stream + robust probe only |
| `tests/test_robustness_data.py` | create | manifest + stream invariants |
| `tests/test_robustness_probe.py` | create | bucket math with stubbed `score_pair`; runner payload |
| `tests/test_robust_probe_callback.py` | create | CSV schema/row content |

Existing files referenced but **never modified**: `data/bio_text.py`, `data/sample_people.py`, `eval/birthday_probe_legacy.py`, `eval/probes.py`, `eval/probe_callback.py`, `ablation_llama_grid.py`, `config.py`.

---

### Task 1: pytest scaffolding

**Files:**
- Create: `conftest.py`

- [ ] **Step 1.1: Create root conftest**

Write `conftest.py` at the project root:

```python
# Root conftest: its presence makes pytest insert the project root into
# sys.path, so tests can `import data.robustness` / `import eval...`
# exactly like the ablation scripts do.
```

(Comment-only file is fine — its existence is what matters.)

- [ ] **Step 1.2: Verify pytest runs in the project env**

Run: `cd "/Users/efmac/Code/Project Code/CRL-Interp/Training_On_LM4" && python -m pytest --collect-only -q`
Expected: `no tests ran` / collected 0 items, exit code 5 (no tests yet) — NOT an import error. Also run `python -c "import torch, transformers, wandb; print('env ok')"` and expect `env ok`. If either fails, stop and locate the correct environment first.

- [ ] **Step 1.3: Commit**

```bash
git add conftest.py
git commit -m "Add pytest root conftest for RobustnessTest tests"
```

---

### Task 2: Manifest — build / save / load (`data/robustness.py`, part 1)

**Files:**
- Create: `data/robustness.py` (manifest functions only in this task)
- Create: `tests/test_robustness_data.py` (manifest tests only in this task)

- [ ] **Step 2.1: Write the failing tests**

Create `tests/test_robustness_data.py`:

```python
"""Tests for data/robustness.py — manifest + restricted bio stream."""
import pytest

from data.bio_text import FIELD_SPECS
from data.robustness import (
    build_robustness_manifest,
    load_manifest,
    save_manifest,
)

FIELDS = ("birthday",)
N_TEMPLATES = len(FIELD_SPECS["birthday"]["templates"])  # 46


def test_manifest_counts_and_ranges():
    m = build_robustness_manifest(1000, FIELDS, limited_frac=0.2,
                                  template_frac=0.25, seed=0)
    assert len(m["limited_people"]) == 200          # round(0.2 * 1000)
    n_keep = round(0.25 * N_TEMPLATES)              # 12
    for idx, subs in m["limited_people"].items():
        assert 0 <= idx < 1000
        tlist = subs["birthday"]
        assert len(tlist) == n_keep
        assert len(set(tlist)) == n_keep            # no duplicate templates
        assert tlist == sorted(tlist)
        assert all(0 <= t < N_TEMPLATES for t in tlist)
    assert m["n_people"] == 1000
    assert m["fields"] == ["birthday"]
    assert m["n_templates"] == {"birthday": N_TEMPLATES}


def test_manifest_deterministic_and_seed_sensitive():
    a = build_robustness_manifest(500, FIELDS, seed=7)
    b = build_robustness_manifest(500, FIELDS, seed=7)
    c = build_robustness_manifest(500, FIELDS, seed=8)
    assert a == b
    assert a != c


def test_manifest_subsets_vary_across_people():
    m = build_robustness_manifest(1000, FIELDS, seed=0)
    subsets = {tuple(s["birthday"]) for s in m["limited_people"].values()}
    assert len(subsets) > 1


def test_manifest_round_trip_restores_int_keys(tmp_path):
    m = build_robustness_manifest(100, FIELDS, seed=3)
    p = tmp_path / "robustness_manifest.json"
    save_manifest(m, p)
    loaded = load_manifest(p)
    assert loaded == m
    assert all(isinstance(k, int) for k in loaded["limited_people"])
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `python -m pytest tests/test_robustness_data.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'data.robustness'`.

- [ ] **Step 2.3: Implement the manifest functions**

Create `data/robustness.py`:

```python
# data/robustness.py
#
# RobustnessTest data generation — see
# docs/superpowers/specs/2026-06-10-robustness-test-design.md
#
# A seeded "manifest" marks a fraction of people (default 20%) as LIMITED:
# each limited person's bios are rendered from their own random subset of
# templates (default round(0.25*T), i.e. 12 of the 46 birthday templates)
# instead of the full round-robin. Every person still emits exactly K bios,
# so the corpus shape is unchanged; only the paraphrase diversity of limited
# people drops. The manifest is the on-disk record of who is limited and
# which template indices each kept.
#
# Pure add-on: data/bio_text.py is untouched. render_bio_limited mirrors
# render_bio's rendering loop with caller-chosen template indices.

import json
import random
from pathlib import Path

from data.bio_text import FIELD_SPECS, render_bio


def build_robustness_manifest(n_people, fields, limited_frac=0.2,
                              template_frac=0.25, seed=0):
    """Choose which people are limited and which templates each keeps.

    Deterministic in `seed` (one random.Random drives both draws; limited
    indices are processed in ascending order). Each limited person's subset is
    sampled independently, so subsets differ across people.

    Returns the manifest dict (JSON-serializable; person keys are ints
    in memory, strings on disk — load_manifest restores ints).
    """
    rng = random.Random(seed)
    n_limited = round(limited_frac * n_people)
    limited_indices = sorted(rng.sample(range(n_people), n_limited))

    limited_people = {}
    for idx in limited_indices:
        subsets = {}
        for field in fields:
            n_templates = len(FIELD_SPECS[field]["templates"])
            n_keep = max(1, round(template_frac * n_templates))
            subsets[field] = sorted(rng.sample(range(n_templates), n_keep))
        limited_people[idx] = subsets

    return {
        "seed": seed,
        "limited_frac": limited_frac,
        "template_frac": template_frac,
        "n_people": n_people,
        "fields": list(fields),
        "n_templates": {f: len(FIELD_SPECS[f]["templates"]) for f in fields},
        "limited_people": limited_people,
    }


def save_manifest(manifest, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path):
    """Load a manifest; person-index keys come back as ints."""
    with open(path) as f:
        manifest = json.load(f)
    manifest["limited_people"] = {
        int(k): v for k, v in manifest["limited_people"].items()
    }
    return manifest
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `python -m pytest tests/test_robustness_data.py -v`
Expected: 4 passed.

- [ ] **Step 2.5: Commit**

```bash
git add data/robustness.py tests/test_robustness_data.py
git commit -m "Add RobustnessTest manifest build/save/load"
```

---

### Task 3: Restricted bio stream (`data/robustness.py`, part 2)

**Files:**
- Modify: `data/robustness.py` (append three functions)
- Modify: `tests/test_robustness_data.py` (append stream tests)

- [ ] **Step 3.1: Write the failing tests**

Append to `tests/test_robustness_data.py` (and extend the existing import from `data.robustness` to include the three new names, plus `bio_stream` and `sample_people`):

```python
from collections import Counter

from data.bio_text import bio_stream
from data.robustness import (
    render_bio_limited,
    robust_bio_stream,
    verify_limited_rendering,
)
from data.sample_people import sample_people

N_SMALL, K_SMALL = 30, 10


@pytest.fixture(scope="module")
def small_world():
    people = sample_people(N=N_SMALL, seed=0)
    manifest = build_robustness_manifest(N_SMALL, FIELDS, limited_frac=0.2,
                                         template_frac=0.25, seed=0)
    return people, manifest


def test_stream_pair_order_matches_bio_stream(small_world):
    people, manifest = small_world
    plain = [(i, e) for i, e, _ in
             bio_stream(people, K=K_SMALL, shuffle_seed=1, fields=FIELDS)]
    robust = [(i, e) for i, e, _ in
              robust_bio_stream(people, K=K_SMALL, manifest=manifest,
                                shuffle_seed=1, fields=FIELDS)]
    assert plain == robust


def test_stream_full_people_identical_to_bio_stream(small_world):
    people, manifest = small_world
    limited = set(manifest["limited_people"])
    plain = list(bio_stream(people, K=K_SMALL, shuffle_seed=1, fields=FIELDS))
    robust = list(robust_bio_stream(people, K=K_SMALL, manifest=manifest,
                                    shuffle_seed=1, fields=FIELDS))
    checked = 0
    for (i1, e1, t1), (i2, e2, t2) in zip(plain, robust):
        if i1 not in limited:
            assert t1 == t2
            checked += 1
    assert checked == (N_SMALL - len(limited)) * K_SMALL


def test_stream_every_person_emits_exactly_K(small_world):
    people, manifest = small_world
    counts = Counter(i for i, _, _ in
                     robust_bio_stream(people, K=K_SMALL, manifest=manifest,
                                       shuffle_seed=1, fields=FIELDS))
    assert len(counts) == N_SMALL
    assert all(c == K_SMALL for c in counts.values())
    assert sum(counts.values()) == N_SMALL * K_SMALL


def test_stream_limited_people_use_only_allowed_templates(small_world):
    people, manifest = small_world
    emitted_by_person = {}
    for i, _, text in robust_bio_stream(people, K=K_SMALL, manifest=manifest,
                                        shuffle_seed=1, fields=FIELDS):
        emitted_by_person.setdefault(i, []).append(text)

    for idx, subs in manifest["limited_people"].items():
        allowed = set(subs["birthday"])
        person = people[idx]
        renders = {t: render_bio_limited(person, {"birthday": t}, FIELDS)
                   for t in range(N_TEMPLATES)}
        for text in emitted_by_person[idx]:
            matches = {t for t, r in renders.items() if r == text}
            assert matches, f"person {idx}: emitted bio matches no template"
            assert matches <= allowed, (
                f"person {idx}: bio from disallowed template(s) {matches - allowed}")


def test_render_bio_limited_matches_render_bio_for_same_template(small_world):
    people, _ = small_world
    from data.bio_text import render_bio
    person = people[0]
    for t in (0, 7, 45):
        # render_bio picks template exposure_idx % 46, so exposure_idx=t
        # selects template t for the single birthday field.
        assert (render_bio_limited(person, {"birthday": t}, FIELDS)
                == render_bio(person, exposure_idx=t, fields=FIELDS))


def test_verify_limited_rendering_passes(small_world):
    people, manifest = small_world
    assert verify_limited_rendering(people, manifest, FIELDS, K=K_SMALL,
                                    n_check=3)
```

- [ ] **Step 3.2: Run tests to verify the new ones fail**

Run: `python -m pytest tests/test_robustness_data.py -v`
Expected: the whole file fails to collect with `ImportError: cannot import name 'render_bio_limited' from 'data.robustness'` (the module-level import kills collection — the 4 manifest tests will pass again once Step 3.3 lands).

- [ ] **Step 3.3: Implement the stream functions**

Append to `data/robustness.py`:

```python
def render_bio_limited(person, template_idx_by_field, fields):
    """Mirror of data.bio_text.render_bio with caller-chosen template indices.

    render_bio derives template_idx = exposure_idx % T per field; here the
    caller supplies the index for each field explicitly (RobustnessTest
    restricts limited people to a per-person template subset). Output is
    byte-identical to render_bio for the same (person, template) choice.
    No reverse_md support — the robustness pipeline doesn't use it.
    """
    name = f"{person['first_name']} {person['middle_name']} {person['last_name']}"
    pronoun = "He" if person["id"] % 2 == 0 else "She"

    parts = []
    for field in fields:
        spec = FIELD_SPECS[field]
        template = spec["templates"][template_idx_by_field[field]]
        subject = name if spec["subject"] == "name" else pronoun
        value = spec["render_value"](person)
        parts.append(" " + template.format(
            name=subject,
            **{spec["value_placeholder"]: value},
        ))
    return "".join(parts)


def robust_bio_stream(people, K, manifest, shuffle_seed=1, fields=("birthday",)):
    """Yield (person_idx, exposure_idx, bio_text) like bio_text.bio_stream.

    The shuffled pair sequence is IDENTICAL to bio_stream's for the same
    shuffle_seed (same pair construction, same Random), so the packed corpus
    differs from a normal run only in the text of limited people's bios.

    Full-set people render exactly as bio_stream does. Limited people map
    exposure e -> their allowed template allowed[e % len(allowed)] per field,
    so each still emits exactly K bios (each allowed template ~K/len(allowed)
    times).
    """
    limited = {int(i): subs for i, subs in manifest["limited_people"].items()}
    pairs = [(i, e) for i in range(len(people)) for e in range(K)]
    random.Random(shuffle_seed).shuffle(pairs)
    for i, e in pairs:
        subs = limited.get(i)
        if subs is None:
            yield i, e, render_bio(people[i], exposure_idx=e, fields=fields)
        else:
            tidx = {f: subs[f][e % len(subs[f])] for f in fields}
            yield i, e, render_bio_limited(people[i], tidx, fields)


def verify_limited_rendering(people, manifest, fields, K, n_check=5):
    """Seatbelt for the ablation script: re-render the first n_check limited
    people's K bios and assert each is the rendering of an allowed template —
    and not of any disallowed one. (Template strings are pairwise distinct, so
    a bio's text identifies its template.) Raises AssertionError on violation;
    returns True otherwise.
    """
    limited = sorted(int(i) for i in manifest["limited_people"])
    for idx in limited[:n_check]:
        person = people[idx]
        subs = {f: list(manifest["limited_people"][idx][f]) for f in fields}
        for e in range(K):
            tidx = {f: subs[f][e % len(subs[f])] for f in fields}
            text = render_bio_limited(person, tidx, fields)
            for f in fields:
                n_templates = len(FIELD_SPECS[f]["templates"])
                allowed = set(subs[f])
                hits = {t for t in range(n_templates)
                        if render_bio_limited(person, {**tidx, f: t}, fields)
                        == text}
                if not hits & allowed:
                    raise AssertionError(
                        f"person {idx}, exposure {e}: bio matches no allowed "
                        f"{f} template {sorted(allowed)}")
                if hits - allowed:
                    raise AssertionError(
                        f"person {idx}, exposure {e}: bio matches disallowed "
                        f"{f} template(s) {sorted(hits - allowed)}")
    return True
```

- [ ] **Step 3.4: Run tests to verify they pass**

Run: `python -m pytest tests/test_robustness_data.py -v`
Expected: 10 passed.

- [ ] **Step 3.5: Commit**

```bash
git add data/robustness.py tests/test_robustness_data.py
git commit -m "Add robust_bio_stream with per-person template restriction"
```

---

### Task 4: FP/LP robustness probe (`eval/robustness_probe.py`)

**Files:**
- Create: `eval/robustness_probe.py`
- Test: `tests/test_robustness_probe.py`

- [ ] **Step 4.1: Write the failing tests**

Create `tests/test_robustness_probe.py`:

```python
"""Bucket-math tests for eval/robustness_probe.py.

score_pair is stubbed — no model, no tokenizer, no GPU. The stub makes FP
correct exactly on (limited person, seen template) pairs and LP correct
everywhere, so every bucket has a known expected value.
"""
import math

from data.bio_text import FIELD_SPECS
from data.robustness import build_robustness_manifest
from data.sample_people import sample_people
import eval.robustness_probe as rp

FIELDS = ("birthday",)
TEMPLATES = FIELD_SPECS["birthday"]["templates"]
N_T = len(TEMPLATES)
BUCKETS = ("total", "limitedSet", "fullSet", "limitedSeen", "limitedUnseen")


class FakeModel:
    def eval(self):
        pass


class FakeTokenizer:
    eos_token_id = 50256


def test_select_eval_people_first_k_per_group():
    manifest = build_robustness_manifest(200, FIELDS, limited_frac=0.2, seed=0)
    limited, full = rp.select_eval_people(manifest, 200, max_people_per_group=10)
    lim_all = set(manifest["limited_people"])
    assert limited == sorted(lim_all)[:10]
    assert full == [i for i in range(200) if i not in lim_all][:10]
    assert set(limited).isdisjoint(full)


def test_run_probe_bucket_math(monkeypatch):
    n_people, per_group = 40, 4
    people = sample_people(N=n_people, seed=0)
    manifest = build_robustness_manifest(n_people, FIELDS, limited_frac=0.2,
                                         template_frac=0.25, seed=0)
    allowed_by_idx = {i: set(s["birthday"])
                      for i, s in manifest["limited_people"].items()}
    id_to_idx = {people[i]["id"]: i for i in range(n_people)}
    t_to_idx = {t: i for i, t in enumerate(TEMPLATES)}

    def fake_score_pair(model, tokenizer, old_to_new, new_to_old, eos_remapped,
                        person, template, device):
        p_idx = id_to_idx[person["id"]]
        t_idx = t_to_idx[template]
        fp = int(p_idx in allowed_by_idx and t_idx in allowed_by_idx[p_idx])
        return {"MP": 0, "DayM": 0, "YearMD": 0, "FP": fp, "LP": 1}

    monkeypatch.setattr(rp, "score_pair", fake_score_pair)

    res = rp.run_probe(FakeModel(), FakeTokenizer(), {50256: 0}, people,
                       manifest, max_people_per_group=per_group, device="cpu")

    n_keep = round(0.25 * N_T)  # 12
    macro, counts = res["macro"], res["counts"]

    # attempted-pair counts per bucket
    assert counts["FP"]["total"][1] == 2 * per_group * N_T
    assert counts["FP"]["limitedSet"][1] == per_group * N_T
    assert counts["FP"]["fullSet"][1] == per_group * N_T
    assert counts["FP"]["limitedSeen"][1] == per_group * n_keep
    assert counts["FP"]["limitedUnseen"][1] == per_group * (N_T - n_keep)

    # FP was made correct exactly on limitedSeen pairs
    assert macro["FP"]["limitedSeen"] == 1.0
    assert macro["FP"]["limitedUnseen"] == 0.0
    assert macro["FP"]["fullSet"] == 0.0
    assert math.isclose(macro["FP"]["limitedSet"], n_keep / N_T)
    assert math.isclose(macro["FP"]["total"],
                        (per_group * n_keep) / (2 * per_group * N_T))

    # LP correct everywhere
    assert all(macro["LP"][b] == 1.0 for b in BUCKETS)

    # bookkeeping
    assert res["n_people"] == {"limited": per_group, "full": per_group}
    assert res["n_templates"] == N_T
    assert len(res["eval_people"]["limited"]) == per_group
    assert len(res["eval_people"]["full"]) == per_group
    assert set(res["per_template"]) == set(range(N_T))
    for d in res["per_template"].values():
        assert set(d) == {"FP_total", "FP_limited", "FP_full",
                          "LP_total", "LP_limited", "LP_full"}


def test_make_robust_runner_bakes_in_group_size(monkeypatch):
    manifest = build_robustness_manifest(40, FIELDS, seed=0)
    fake_results = {
        "macro": {m: {b: 0.5 for b in BUCKETS} for m in ("FP", "LP")},
        "counts": {},
        "per_template": {0: {"FP_total": 0.5, "FP_limited": 0.5, "FP_full": 0.5,
                             "LP_total": 0.5, "LP_limited": 0.5, "LP_full": 0.5}},
        "eval_people": {"limited": [], "full": []},
        "n_people": {"limited": 0, "full": 0},
        "n_templates": N_T,
    }
    captured = {}

    def fake_run_probe(model, tokenizer, old_to_new, people, mani, *,
                       max_people_per_group, device=None):
        captured["mppg"] = max_people_per_group
        return fake_results

    monkeypatch.setattr(rp, "run_probe", fake_run_probe)

    runner = rp.make_robust_runner(manifest, max_people_per_group=77)
    results, payload, tables, json_name = runner(None, None, {}, [], FIELDS, 50)

    assert captured["mppg"] == 77  # baked-in wins; registry's max_people=50 ignored
    assert json_name == "probe_robust.json"
    assert results is fake_results
    for m in ("FP", "LP"):
        for b in BUCKETS:
            assert payload[f"robustProbe/{m}/{b}"] == 0.5
    assert set(payload) == {f"robustProbe/{m}/{b}"
                            for m in ("FP", "LP") for b in BUCKETS}
    assert "robustProbe/per_template" in tables
```

- [ ] **Step 4.2: Run tests to verify they fail**

Run: `python -m pytest tests/test_robustness_probe.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'eval.robustness_probe'`.

- [ ] **Step 4.3: Implement the probe module**

Create `eval/robustness_probe.py`:

```python
"""RobustnessTest probe — FP/LP only, bucketed by template-diversity group.

See docs/superpowers/specs/2026-06-10-robustness-test-design.md.

Scoring is 100% reused: eval.birthday_probe_legacy.score_pair computes all 5
birthday metrics (one teacher-forced pass + one greedy decode); this module
tallies only FP (greedy probe) and LP (lenient probe) — the teacher-forced
metrics it also returns are discarded (~3% overhead vs the decode).

Eval set: the first `max_people_per_group` limited people plus the first
`max_people_per_group` full-set people (ascending person index), every one
probed on ALL birthday templates. Buckets per metric:

    total          all eval pairs
    limitedSet     limited people only
    fullSet        full-set people only
    limitedSeen    limited pairs whose template is in that person's subset
    limitedUnseen  limited pairs whose template is not

All metrics are fractions in [0, 1].
"""
from __future__ import annotations

import wandb
from tqdm import tqdm

from data.bio_text import FIELD_SPECS
from eval.birthday_probe_legacy import score_pair

BUCKETS = ("total", "limitedSet", "fullSet", "limitedSeen", "limitedUnseen")
METRICS = ("FP", "LP")
# per-template breakdown keys (JSON + wandb table columns)
PT_KEYS = ("FP_total", "FP_limited", "FP_full",
           "LP_total", "LP_limited", "LP_full")


def select_eval_people(manifest, n_people, max_people_per_group):
    """First k limited + first k full person indices (ascending, deterministic)."""
    limited_all = sorted(int(i) for i in manifest["limited_people"])
    limited_set = set(limited_all)
    limited_eval = limited_all[:max_people_per_group]
    full_eval = []
    for i in range(n_people):
        if i not in limited_set:
            full_eval.append(i)
            if len(full_eval) == max_people_per_group:
                break
    return limited_eval, full_eval


def run_probe(model, tokenizer, old_to_new, people, manifest, *,
              max_people_per_group=100, device=None):
    """Run the FP/LP robustness probe on an in-memory model.

    Returns a JSON-ready dict: macro (rates), counts ([correct, attempted]
    per bucket), per_template rates, eval_people indices, n_people,
    n_templates.
    """
    if device is None:
        device = str(next(model.parameters()).device)
    model.eval()

    eos_remapped = old_to_new[int(tokenizer.eos_token_id)]
    new_to_old = {v: k for k, v in old_to_new.items()}
    templates = FIELD_SPECS["birthday"]["templates"]

    limited_eval, full_eval = select_eval_people(
        manifest, len(people), max_people_per_group)
    allowed_by_person = {int(i): set(subs["birthday"])
                         for i, subs in manifest["limited_people"].items()}

    counts = {m: {b: [0, 0] for b in BUCKETS} for m in METRICS}
    per_template = {t: {k: [0, 0] for k in PT_KEYS}
                    for t in range(len(templates))}

    plan = [(i, True) for i in limited_eval] + [(i, False) for i in full_eval]
    pbar = tqdm(total=len(plan) * len(templates), desc="robust probe")
    for person_idx, is_limited in plan:
        person = people[person_idx]
        allowed = allowed_by_person.get(person_idx, set())
        group = "limited" if is_limited else "full"
        for t_idx, template in enumerate(templates):
            scores = score_pair(model, tokenizer, old_to_new, new_to_old,
                                eos_remapped, person, template, device=device)
            buckets = ["total", "limitedSet" if is_limited else "fullSet"]
            if is_limited:
                buckets.append("limitedSeen" if t_idx in allowed
                               else "limitedUnseen")
            for m in METRICS:
                ok = int(scores[m])
                for b in buckets:
                    counts[m][b][0] += ok
                    counts[m][b][1] += 1
                for key in (f"{m}_total", f"{m}_{group}"):
                    per_template[t_idx][key][0] += ok
                    per_template[t_idx][key][1] += 1
            pbar.update(1)
    pbar.close()

    macro = {m: {b: c / max(n, 1) for b, (c, n) in counts[m].items()}
             for m in METRICS}

    print("\nRobustness probe (FP = greedy, LP = lenient); fractions 0-1:")
    for m in METRICS:
        for b in BUCKETS:
            c, n = counts[m][b]
            print(f"  {m}/{b:<14s}: {c}/{n}  =  {100 * macro[m][b]:5.1f}%")

    return {
        "macro": macro,
        "counts": counts,
        "per_template": {
            t: {k: v[0] / max(v[1], 1) for k, v in d.items()}
            for t, d in per_template.items()
        },
        "eval_people": {"limited": limited_eval, "full": full_eval},
        "n_people": {"limited": len(limited_eval), "full": len(full_eval)},
        "n_templates": len(templates),
    }


def make_robust_runner(manifest, max_people_per_group=100):
    """Build a PROBE_REGISTRY-compatible runner for the robustness probe.

    The returned callable matches the registry signature
    (model, tokenizer, old_to_new, people, fields, max_people). The baked-in
    `max_people_per_group` is authoritative: ProbeAtEpochs -> run_probes
    leaves `max_people` at its default of 50, which must not silently shrink
    the robustness eval set. Register from the ablation script:

        PROBE_REGISTRY["birthday_robust"] = make_robust_runner(manifest, 100)
    """
    def _run_birthday_robust(model, tokenizer, old_to_new, people, fields,
                             max_people):
        results = run_probe(model, tokenizer, old_to_new, people, manifest,
                            max_people_per_group=max_people_per_group)
        payload = {f"robustProbe/{m}/{b}": results["macro"][m][b]
                   for m in METRICS for b in BUCKETS}
        table = wandb.Table(columns=["template_idx", *PT_KEYS])
        for t_idx in sorted(results["per_template"]):
            row = results["per_template"][t_idx]
            table.add_data(int(t_idx), *(row[k] for k in PT_KEYS))
        tables = {"robustProbe/per_template": table}
        return results, payload, tables, "probe_robust.json"
    return _run_birthday_robust
```

- [ ] **Step 4.4: Run tests to verify they pass**

Run: `python -m pytest tests/test_robustness_probe.py -v`
Expected: 3 passed.

- [ ] **Step 4.5: Commit**

```bash
git add eval/robustness_probe.py tests/test_robustness_probe.py
git commit -m "Add FP/LP robustness probe with bucket tallies"
```

---

### Task 5: CSV callback (`eval/robust_probe_callback.py`)

**Files:**
- Create: `eval/robust_probe_callback.py`
- Test: `tests/test_robust_probe_callback.py`

- [ ] **Step 5.1: Write the failing tests**

Create `tests/test_robust_probe_callback.py`:

```python
"""CSV schema tests for RobustProbeAtEpochs (scheduling is inherited and
already exercised in production by ProbeAtEpochs; only _append_csv differs)."""
import csv

from eval.robust_probe_callback import ROBUST_CSV_COLUMNS, RobustProbeAtEpochs


def _make_cb(tmp_path):
    return RobustProbeAtEpochs(
        probes=("birthday_robust",), tokenizer=None, old_to_new={},
        people=[], fields=("birthday",), out_dir=str(tmp_path / "run"),
        run_name="grid-L4-H6", probe_epochs=[1, 2], max_epochs=2,
        csv_path=tmp_path / "probe_curve.csv",
        run_info={"family": "fam", "study": "grid", "numLayers": 4,
                  "numHeads": 6, "dmodel": 384},
    )


def test_append_csv_writes_all_robust_columns(tmp_path):
    cb = _make_cb(tmp_path)
    results = {"birthday_robust": {"macro": {
        "FP": {"total": 0.83, "limitedSet": 0.71, "fullSet": 0.95,
               "limitedSeen": 0.88, "limitedUnseen": 0.65},
        "LP": {"total": 0.9, "limitedSet": 0.8, "fullSet": 0.99,
               "limitedSeen": 0.91, "limitedUnseen": 0.75},
    }}}
    cb._append_csv(epoch=2, results=results)

    with open(tmp_path / "probe_curve.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert list(row.keys()) == ROBUST_CSV_COLUMNS
    assert row["family"] == "fam"
    assert row["run_name"] == "grid-L4-H6"
    assert row["epoch"] == "2"
    assert row["FP_total"] == "0.83"
    assert row["FP_limited"] == "0.71"
    assert row["FP_full"] == "0.95"
    assert row["FP_limitedSeen"] == "0.88"
    assert row["FP_limitedUnseen"] == "0.65"
    assert row["LP_total"] == "0.9"
    assert row["LP_limitedUnseen"] == "0.75"


def test_append_csv_blank_metrics_when_probe_results_missing(tmp_path):
    cb = _make_cb(tmp_path)
    cb._append_csv(epoch=1, results={})
    with open(tmp_path / "probe_curve.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["FP_total"] == ""
    assert rows[0]["family"] == "fam"
    assert rows[0]["epoch"] == "1"


def test_append_csv_appends_without_duplicate_header(tmp_path):
    cb = _make_cb(tmp_path)
    cb._append_csv(epoch=1, results={})
    cb._append_csv(epoch=2, results={})
    with open(tmp_path / "probe_curve.csv") as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == 3  # 1 header + 2 rows
```

- [ ] **Step 5.2: Run tests to verify they fail**

Run: `python -m pytest tests/test_robust_probe_callback.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'eval.robust_probe_callback'`.

- [ ] **Step 5.3: Implement the callback**

Create `eval/robust_probe_callback.py`:

```python
"""ProbeAtEpochs variant for the RobustnessTest CSV schema.

Scheduling, wandb step alignment, and error handling are inherited from
eval.probe_callback.ProbeAtEpochs unchanged; only the CSV row differs —
instead of the birthday_legacy MP/DayM/YearMD/FP/LP columns it writes the
robustness probe's FP/LP x bucket metrics (fractions 0-1, rounded to 4dp).

CSV column names use the short _limited/_full forms; wandb and the JSON use
limitedSet/fullSet. See the spec's section 3 for the mapping.
"""
import csv

from eval.probe_callback import ProbeAtEpochs

ROBUST_CSV_COLUMNS = [
    "family", "study", "run_name", "numLayers", "numHeads", "dmodel", "epoch",
    "FP_total", "FP_limited", "FP_full", "FP_limitedSeen", "FP_limitedUnseen",
    "LP_total", "LP_limited", "LP_full", "LP_limitedSeen", "LP_limitedUnseen",
]

# macro bucket name (eval/robustness_probe.py) -> CSV column suffix
_BUCKET_TO_COL = {
    "total":         "total",
    "limitedSet":    "limited",
    "fullSet":       "full",
    "limitedSeen":   "limitedSeen",
    "limitedUnseen": "limitedUnseen",
}


class RobustProbeAtEpochs(ProbeAtEpochs):
    """ProbeAtEpochs with the robustness probe's CSV columns."""

    def _append_csv(self, epoch, results):
        row = {c: "" for c in ROBUST_CSV_COLUMNS}
        row.update(self.run_info)
        row["run_name"] = self.run_name
        row["epoch"] = epoch
        macro = results.get("birthday_robust", {}).get("macro")
        if macro:
            for metric in ("FP", "LP"):
                for bucket, col in _BUCKET_TO_COL.items():
                    val = macro.get(metric, {}).get(bucket)
                    if val is not None:
                        row[f"{metric}_{col}"] = round(val, 4)

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROBUST_CSV_COLUMNS,
                               extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(f"  RobustProbeAtEpochs: appended epoch {epoch} row "
              f"-> {self.csv_path}")
```

- [ ] **Step 5.4: Run tests to verify they pass**

Run: `python -m pytest tests/test_robust_probe_callback.py -v`
Expected: 3 passed.

- [ ] **Step 5.5: Commit**

```bash
git add eval/robust_probe_callback.py tests/test_robust_probe_callback.py
git commit -m "Add RobustProbeAtEpochs with robustness CSV schema"
```

---

### Task 6: Ablation script (`ablation_llama_grid_robustness.py`)

**Files:**
- Create: `ablation_llama_grid_robustness.py`

No unit test (the script trains models at import time). Verification = byte-compile + structured diff against the parent script + checklist review.

- [ ] **Step 6.1: Write the script**

Create `ablation_llama_grid_robustness.py`. It is `ablation_llama_grid.py` with a new docstring, the RobustnessTest knobs, manifest creation, the robust stream, and the robust probe/callback. Full content:

```python
"""RobustnessTest ablation sweep over Capo bioS models — full-factorial GRID.

Same pipeline as `ablation_llama_grid.py`, with one data-side twist
(RobustnessTest — docs/superpowers/specs/2026-06-10-robustness-test-design.md):

  - LIMITED_PEOPLE_FRAC of the N people are "limited": their K bios render
    from a per-person random subset of round(LIMITED_TEMPLATE_FRAC * T)
    templates (12 of the 46 birthday templates) instead of the full
    round-robin. Every person still gets exactly K bios, so corpus size and
    the reduced vocab match a normal run.
  - Which people are limited, and each one's template subset, is recorded in
    robustness_manifest.json (under cache/{NAME}/ and copied into
    runs/{NAME}/{INVOCATION}/).

Probing: ONLY the greedy + lenient judges (FP/LP), via the "birthday_robust"
probe (eval/robustness_probe.py) — it reuses birthday_probe_legacy.score_pair
and buckets pairs into total / limitedSet / fullSet / limitedSeen /
limitedUnseen. Eval set: first MAX_PEOPLE_PER_GROUP limited + first
MAX_PEOPLE_PER_GROUP full-set people, every one probed on all 46 templates.

Output layout
-------------
    runs/{NAME}/{INVOCATION}/
        probe_curve.csv                  # one row per (run, probe epoch):
                                         #   FP_/LP_ x total/limited/full/
                                         #   limitedSeen/limitedUnseen
        robustness_manifest.json         # who is limited + their templates
        {study}/{run_name}/
            ...                          # train checkpoints
            probes/probe_robust_epoch{N}.json
            final/probe_robust.json

wandb keys (per run, logged at every probe epoch)
-------------------------------------------------
    robustProbe/{FP,LP}/{total,limitedSet,fullSet,limitedSeen,limitedUnseen}
    robustProbe/per_template             (table)
"""

import itertools
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import wandb
import wandb.util
from transformers import GPT2Tokenizer

from config import Config
from data.sample_people import sample_people
from data.bio_text import render_bio
from data.robustness import (
    build_robustness_manifest,
    robust_bio_stream,
    save_manifest,
    verify_limited_rendering,
)
from data.tokenize_pack import (
    tokenize_and_pack,
    PackedTokenDataset,
    build_vocab_remap,
    remap_token_file,
    assert_tokens_in_remap,
)
from model.buildModel import create_gpt2_model, create_llama_model
from model.trainModel import train
from eval.probes import PROBE_REGISTRY
from eval.robustness_probe import make_robust_runner
from eval.robust_probe_callback import RobustProbeAtEpochs


# ---------------------------
# CONFIG
# ---------------------------

CONFIG = Config()

CONFIG.NAME = "bioS_N-Bd_robust_grid"

# Set INVOCATION=<existing-timestamp> in the env to resume an interrupted sweep
# into the same runs/{NAME}/{INVOCATION}/ directory; otherwise a fresh one.
INVOCATION = os.environ.get("INVOCATION") or datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"INVOCATION = {INVOCATION}")

SWEEP_NAME = f"{CONFIG.NAME}-{INVOCATION}"

# Every (run, probe epoch) row from this sweep accumulates into one CSV.
CURVE_CSV = Path("runs") / CONFIG.NAME / INVOCATION / "probe_curve.csv"

CONFIG.SEED         = 0
CONFIG.SHUFFLE_SEED = 1

# Data
CONFIG.N       = 50_000
CONFIG.K       = 100
CONFIG.SEQ_LEN = 512
CONFIG.FIELDS  = ("birthday",)

# RobustnessTest knobs
LIMITED_PEOPLE_FRAC   = 0.20  # fraction of people with restricted templates
LIMITED_TEMPLATE_FRAC = 0.25  # fraction of templates each limited person keeps
ROBUSTNESS_SEED       = 0     # seeds the manifest sampling
MAX_PEOPLE_PER_GROUP  = 100   # probe eval people per group (limited / full)

# Derived paths
DATA_DIR = Path("cache") / CONFIG.NAME
DATA_DIR.mkdir(parents=True, exist_ok=True)

PEOPLE_PATH       = DATA_DIR / "people.json"
DATA_CONFIG_PATH  = DATA_DIR / "data_config.json"
MANIFEST_PATH     = DATA_DIR / "robustness_manifest.json"
MANIFEST_RUN_COPY = Path("runs") / CONFIG.NAME / INVOCATION / "robustness_manifest.json"
CONFIG.PRE_REDUCE_PATH  = str(DATA_DIR / "bios_prereduce.bin")
CONFIG.POST_REDUCE_PATH = str(DATA_DIR / "bios_postreduce.bin")

# Training (shared across all ablation runs)
CONFIG.MODEL_TYPE   = "llama"
CONFIG.BATCH_SIZE   = 24
CONFIG.LR           = 1e-3
CONFIG.WEIGHT_DECAY = 0.01
CONFIG.WARMUP_STEPS = 1000
CONFIG.GRAD_CLIP    = 1.0


# ---------------------------
# PROBE SELECTION
# ---------------------------
# RobustnessTest runs ONLY the FP/LP robustness probe ("birthday_robust",
# registered below once the manifest exists).

PROBES = ("birthday_robust",)


# ---------------------------
# GRID SWEEP — edit me
# ---------------------------
#
# Same grid semantics as ablation_llama_grid.py: BASE supplies fallbacks,
# GRID cross-products, dmodel = 64 * numHeads.

MAX_EPOCHS      = 16
RUN_PROBE_AFTER = [1, 2, 4, 6, 8, 12]
PROBE_EPOCHS    = sorted({e for e in RUN_PROBE_AFTER if e <= MAX_EPOCHS}
                         | {MAX_EPOCHS})

BASE = {
    "numLayers": 4,
    "numHeads":  3,
}

GRID = {
    "numLayers": [1, 2, 4, 8],
    "numHeads":  [1, 2, 4, 6, 8],
}

# Short axis labels used in grid run names (e.g. grid-L4-H6).
AXIS_ABBREV = {"numLayers": "L", "numHeads": "H", "EPOCHS": "E", "dmodel": "D"}


def build_runs(base, grid):
    """Cross-product `grid` axes; overlay each combo on `base`."""
    axes   = list(grid.keys())
    combos = itertools.product(*grid.values())
    runs = []
    for combo in combos:
        cfg = {**base, **dict(zip(axes, combo))}
        cfg["dmodel"] = 64 * cfg["numHeads"]   # paper ℓ-h convention
        short = "-".join(f"{AXIS_ABBREV.get(k, k)}{v}" for k, v in zip(axes, combo))
        runs.append({
            "study": "grid",
            "name":  f"grid-{short}",
            **cfg,
        })
    return runs


RUNS = build_runs(BASE, GRID)
print(f"Planned {len(RUNS)} grid runs "
      f"(MAX_EPOCHS={MAX_EPOCHS}, probe @ epochs {PROBE_EPOCHS}):")
for r in RUNS:
    print(f"  [{r['study']:>5}] {r['name']:<24} "
          f"L={r['numLayers']} H={r['numHeads']} D={r['dmodel']}")


# ---------------------------
# DATA (generated once, shared by every run)
# ---------------------------

people = sample_people(N=CONFIG.N, seed=CONFIG.SEED)
with open(PEOPLE_PATH, "w") as f:
    json.dump(people, f)
print(f"Saved {len(people):,} people → {PEOPLE_PATH}")

manifest = build_robustness_manifest(
    n_people=CONFIG.N, fields=tuple(CONFIG.FIELDS),
    limited_frac=LIMITED_PEOPLE_FRAC, template_frac=LIMITED_TEMPLATE_FRAC,
    seed=ROBUSTNESS_SEED,
)
save_manifest(manifest, MANIFEST_PATH)
save_manifest(manifest, MANIFEST_RUN_COPY)
n_limited = len(manifest["limited_people"])
example_subset = next(iter(manifest["limited_people"].values()))["birthday"]
print(f"RobustnessTest: {n_limited:,}/{CONFIG.N:,} limited people, "
      f"{len(example_subset)}/{manifest['n_templates']['birthday']} birthday "
      f"templates each (seed={ROBUSTNESS_SEED}).")
print(f"Manifest → {MANIFEST_PATH} (copy: {MANIFEST_RUN_COPY})")

verify_limited_rendering(people, manifest, tuple(CONFIG.FIELDS),
                         K=CONFIG.K, n_check=5)
print("Verified: first 5 limited people render only their allowed templates.")

stream = robust_bio_stream(
    people, K=CONFIG.K, manifest=manifest,
    shuffle_seed=CONFIG.SHUFFLE_SEED, fields=tuple(CONFIG.FIELDS),
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
CONFIG.eosToken   = tokenizer.eos_token_id
CONFIG.vocab_size = tokenizer.vocab_size

n_tokens, n_seq = tokenize_and_pack(
    tokenizer, stream,
    n_bios_total=CONFIG.N * CONFIG.K,
    out_path=CONFIG.PRE_REDUCE_PATH,
    seq_len=CONFIG.SEQ_LEN,
)
print(f"Wrote {n_tokens:,} tokens → {n_seq:,} sequences of {CONFIG.SEQ_LEN}.")

old_to_new, _, CONFIG.reducedVocabSize = build_vocab_remap(CONFIG.PRE_REDUCE_PATH)
remap_token_file(CONFIG.PRE_REDUCE_PATH, CONFIG.POST_REDUCE_PATH, old_to_new)
CONFIG.reducedEOSToken = old_to_new[int(CONFIG.eosToken)]
print(f"Reduced vocab size: {CONFIG.reducedVocabSize}")

# Probe prompts span ALL templates for EVERY eval person (that is the point
# of the robustness probe), so the seatbelt below — which renders the first
# 5 people across all K exposures, i.e. all 46 templates round-robin — is
# exactly the right vocab-coverage check, regardless of which templates those
# people actually trained on.
sample_prompts = [
    render_bio(people[i], exposure_idx=e, fields=tuple(CONFIG.FIELDS))
    for i in range(min(5, len(people)))
    for e in range(CONFIG.K)
]
assert_tokens_in_remap(old_to_new, sample_prompts, tokenizer)
print(f"Verified all {len(sample_prompts):,} sample prompts tokenize cleanly post-remap.")

CONFIG.save(str(DATA_CONFIG_PATH))
print(f"Saved config → {DATA_CONFIG_PATH}")

ds = PackedTokenDataset(CONFIG.POST_REDUCE_PATH, seq_len=CONFIG.SEQ_LEN)
print(f"Dataset has {len(ds):,} sequences.")


# ---------------------------
# PROBE REGISTRATION
# ---------------------------
# Additive: eval/probes.py is untouched; the runner closes over the manifest
# and its own group size (the registry's max_people default must not shrink
# the eval set — see make_robust_runner's docstring).

PROBE_REGISTRY["birthday_robust"] = make_robust_runner(
    manifest, max_people_per_group=MAX_PEOPLE_PER_GROUP)


# ---------------------------
# SWEEP
# ---------------------------

for run in RUNS:
    CONFIG.numLayers = run["numLayers"]
    CONFIG.numHeads  = run["numHeads"]
    CONFIG.dmodel    = run["dmodel"]
    CONFIG.EPOCHS    = MAX_EPOCHS

    out_dir_path = Path(f"runs/{CONFIG.NAME}/{INVOCATION}/{run['study']}/{run['name']}")
    final_dir    = out_dir_path / "final"
    ckpts        = sorted(out_dir_path.glob("checkpoint-*"),
                          key=lambda p: int(p.name.split("-")[-1])) if out_dir_path.exists() else []

    if final_dir.exists():
        print(f"\n=== SKIP {run['study']}/{run['name']} — already completed "
              f"({final_dir} exists) ===")
        continue

    resume_from_checkpoint = bool(ckpts)
    if resume_from_checkpoint:
        print(f"\n=== RESUME {run['study']}/{run['name']} from {ckpts[-1].name} ===")

    if CONFIG.MODEL_TYPE == "llama":
        model = create_llama_model(
            CONFIG.reducedVocabSize, CONFIG.SEQ_LEN,
            CONFIG.dmodel, CONFIG.numLayers, CONFIG.numHeads,
            CONFIG.reducedEOSToken,
            seed=CONFIG.SEED,
        )
    elif CONFIG.MODEL_TYPE == "gpt2":
        model = create_gpt2_model(
            CONFIG.reducedVocabSize, CONFIG.SEQ_LEN,
            CONFIG.dmodel, CONFIG.numLayers, CONFIG.numHeads,
            CONFIG.reducedEOSToken,
            seed=CONFIG.SEED,
        )
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {CONFIG.MODEL_TYPE!r}")

    out_dir = str(out_dir_path)

    wandb_run_id = wandb.util.generate_id()
    print(f"\n=== Training {run['study']}/{run['name']} → {out_dir} "
          f"(wandb id {wandb_run_id}) ===")
    wandb.init(
        id=wandb_run_id,
        name=f"{run['name']}-{INVOCATION}",
        group=SWEEP_NAME,
        job_type=run["study"],
        tags=[run["study"], CONFIG.NAME, "robustness"],
        reinit="finish_previous",
        config={
            **asdict(CONFIG),
            "study":      run["study"],
            "run_name":   run["name"],
            "sweep_name": SWEEP_NAME,
            "INVOCATION": INVOCATION,
            "probes":     list(PROBES),
            "limited_people_frac":   LIMITED_PEOPLE_FRAC,
            "limited_template_frac": LIMITED_TEMPLATE_FRAC,
            "robustness_seed":       ROBUSTNESS_SEED,
            "max_people_per_group":  MAX_PEOPLE_PER_GROUP,
            "n_limited_people":      n_limited,
        },
    )

    # Probe the in-memory model after each epoch in PROBE_EPOCHS (the final
    # one included). Per-epoch rows accumulate in CURVE_CSV across the grid.
    probe_cb = RobustProbeAtEpochs(
        probes=PROBES, tokenizer=tokenizer, old_to_new=old_to_new,
        people=people, fields=tuple(CONFIG.FIELDS),
        out_dir=out_dir, run_name=run["name"],
        probe_epochs=PROBE_EPOCHS, max_epochs=MAX_EPOCHS,
        csv_path=CURVE_CSV,
        run_info={"family": CONFIG.NAME, "study": run["study"],
                  "numLayers": CONFIG.numLayers, "numHeads": CONFIG.numHeads,
                  "dmodel": CONFIG.dmodel},
    )
    train(model, ds, CONFIG, output_dir=out_dir, callbacks=[probe_cb],
          resume_from_checkpoint=resume_from_checkpoint)
    print(f"Done. Checkpoints + per-epoch probes under {out_dir}/")

    wandb.finish()
```

- [ ] **Step 6.2: Byte-compile the script**

Run: `python -m py_compile ablation_llama_grid_robustness.py && echo OK`
Expected: `OK` (no syntax errors). Do NOT run the script itself — it generates 5M bios and trains 20 models.

- [ ] **Step 6.3: Review the delta against the parent script**

Run: `diff ablation_llama_grid.py ablation_llama_grid_robustness.py | head -200`

Confirm the differences are ONLY: docstring; `CONFIG.NAME`; the four robustness knobs; the `MANIFEST_PATH`/`MANIFEST_RUN_COPY` paths; robustness imports (replacing the `bio_stream` import); `PROBES`; manifest build/save/verify block; `robust_bio_stream` call; the seatbelt comment; the `PROBE_REGISTRY["birthday_robust"]` registration block; `RobustProbeAtEpochs` instead of `ProbeAtEpochs`; the `"robustness"` wandb tag + 5 extra wandb config keys. Anything else differing from the parent is a transcription bug — fix it.

- [ ] **Step 6.4: Confirm no existing files were touched**

Run: `git status --short`
Expected: only `??` (untracked) entries for the new files plus the pre-existing ` M figure-generator/figuregenerator.ipynb`. No other ` M` lines.

- [ ] **Step 6.5: Commit**

```bash
git add ablation_llama_grid_robustness.py
git commit -m "Add ablation_llama_grid_robustness.py RobustnessTest sweep script"
```

---

### Task 7: Full verification

- [ ] **Step 7.1: Run the entire test suite**

Run: `python -m pytest tests/ -v`
Expected: **16 passed** (4 manifest + 6 stream + 3 probe + 3 callback), 0 failures.

- [ ] **Step 7.2: Smoke-import the new eval modules**

Run: `python -c "import eval.robustness_probe, eval.robust_probe_callback, data.robustness; print('imports ok')"`
Expected: `imports ok` (catches import-time errors pytest's stubs could mask).

- [ ] **Step 7.3: Spec-coverage check**

Re-read `docs/superpowers/specs/2026-06-10-robustness-test-design.md` and confirm each section maps to shipped code: manifest (Task 2), stream + seatbelt (Task 3), probe + buckets + runner (Task 4), CSV callback (Task 5), script + knobs + wandb keys (Task 6), tests (Tasks 2–5). If anything is missing, add it before finishing.

- [ ] **Step 7.4: Final commit (if anything changed in 7.1–7.3)**

```bash
git status --short   # confirm only intended files
git add <specific files>
git commit -m "RobustnessTest: verification fixes"
```

---

## Verification at sweep time (manual, not part of this plan)

When the user launches the real sweep on the training box:
1. The console must print the manifest line (`10,000/50,000 limited people, 12/46 birthday templates each`) and the seatbelt line before tokenization starts.
2. After the first probe epoch: `runs/bioS_N-Bd_robust_grid/<INVOCATION>/probe_curve.csv` has the 17 robust columns; wandb shows `robustProbe/FP/total` etc.
3. `robustness_manifest.json` exists in both `cache/bioS_N-Bd_robust_grid/` and the invocation dir.

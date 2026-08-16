# AGENTS.md — LM4 toy model, bioS data, and the interp stack

Orientation for an AI agent picking this up cold. Everything below is verified
against the actual artifacts, not inferred from docs.

---

# ⚠️ READ THIS FIRST: the training corpus is NOT in this repo

The two token files are **180 MB each**, over GitHub's 100 MB per-file limit, so
they are not committed. They are a **pure function of two integer seeds**, so you
rebuild them exactly — not approximately:

```bash
python release/regenerate_data.py
```

Writes `release/data/bioS_N-Bd_final_grid/{bios_prereduce.bin, bios_postreduce.bin}`
(~360 MB total, a few minutes) and verifies every artifact against a known-good
sha256 before exiting non-zero on any mismatch.

Sanity-check in seconds without building the corpus:

```bash
python release/regenerate_data.py --smoke
```

**These checksums must match. They were reproduced from seed on 2026-08-15:**

| Artifact | sha256 | Shipped? |
|---|---|---|
| `people.json` | `74974c5d…06ee39` | ✅ in repo (15 MB) |
| `old_to_new.json` | `0518e632…40a7b6` | ✅ in repo (25 KB) |
| `bios_prereduce.bin` | `fed7b35f…8be88a` | ❌ regenerate (180 MB) |
| `bios_postreduce.bin` | `09edc547…254192` | ❌ regenerate (180 MB) |

**This is not a theoretical claim.** The full corpus was regenerated from seed on
2026-08-15 and all four artifacts matched byte-for-byte — including both 180 MB
`.bin` files (90,000,900 tokens → 175,783 sequences of 512).

**Why it holds.** Every stage is seeded, and the chain was tested end-to-end, not
assumed:

1. `sample_people(N=50000, seed=0)` — plain `random.Random(0)`. Regenerates
   byte-identical (sha256 confirmed against the shipped `people.json`).
2. `render_bio(person, exposure_idx, fields)` — **no RNG at all**. Template is
   picked round-robin: `templates[exposure_idx % len(templates)]`.
3. `bio_stream(...)` — builds the full `(person, exposure)` grid and shuffles it
   with `random.Random(shuffle_seed=1)`. Order is fixed by the seed, not by
   iteration or thread scheduling.
4. GPT-2 tokenization — confirmed stable; the first 2,000 bios (35,970 tokens)
   re-tokenize to a byte-exact match under `transformers 5.8.1`.

**The one thing that can break it:** a GPT-2 tokenizer whose merges differ would
shift token ids, which changes `bios_prereduce.bin`, which changes the set of
unique ids, which changes `old_to_new.json` — and then **the shipped checkpoints
become unusable**, because embedding row *i* would no longer mean the same token.
If `regenerate_data.py` reports FAIL, stop and fix the tokenizer version; do not
proceed with a mismatched vocab.

---

## What you've been given

| | |
|---|---|
| **Toy model** | "LM4" — a Llama trained **from scratch** on synthetic biographies. Two checkpoints: `grid-L4-H6` (4 layers, 7.79 M params) and `grid-L8-H6` (8 layers, 14.87 M params) |
| **Data** | `bioS_N-Bd_final_grid` — 50,000 synthetic people × 100 paraphrases of their birthday = 5 M bios, 90 M tokens |
| **This repo** (`Training_On_LM4`) | Generates the data and trains the model |
| **Sibling repo** (`Interp_LM4`) | Interpretability tooling built on top: SAEs, cross-layer transcoders, a causal SAE |

The task the model is trained for is deliberately narrow: **memorize each
person's birthday and recall it through any of 46 paraphrase templates.** That
makes it a clean testbed for mechanistic interpretability — a real factual-recall
circuit, small enough to analyze exhaustively.

```
release/
  models/
    grid-L4-H6/        # BaselineL4 — the canonical model most work uses
    grid-L8-H6/        # BaselineL8 — depth-8 sibling
  data/bioS_N-Bd_final_grid/
    data_config.json         # the seeds + shapes; source of truth
    old_to_new.json          # GPT-2 id -> reduced id. REQUIRED to use the model
    people.json              # the 50k sampled people
    robustness_manifest.json # held-out template/person splits
  regenerate_data.py
```

---

## Vocabulary condensation — how the token count is reduced

**This is the single most important thing to understand before touching the
model.** These checkpoints do **not** use GPT-2's vocabulary.

The bios are written with the GPT-2 tokenizer (50,257 ids), but a corpus of
name-and-birthday sentences only ever uses a tiny slice of it. So the pipeline
collects every id that actually appears and renumbers them densely:

```python
# data/tokenize_pack.py
unique     = np.unique(tokens)              # every GPT-2 id present in the corpus
old_to_new = {old: i for i, old in enumerate(unique)}   # dense rank map
```

**50,257 → 1,836 ids.** Concretely:

| | GPT-2 | LM4 |
|---|---|---|
| vocab size | 50,257 | **1,836** |
| eos id | 50256 | **1835** |
| embedding params @ `d_model=384` | 19.3 M | **0.7 M** |

That single change is what makes the model tiny. With tied embeddings, the
50,257-row table would have been **19.3 M parameters — larger than the entire
14.87 M-parameter L8 model.** Condensing it to 1,836 rows drops that to 0.7 M and
puts the parameter mass in the transformer blocks where the interesting
computation happens.

### Consequences you must respect

- **The mapping is arbitrary and load-bearing.** It's a sorted-unique
  enumeration, so reduced id 42 has no meaning outside `old_to_new.json`. A
  checkpoint without its remap file is unusable.
- **Use `CondensedTokenizer`, never a raw GPT-2 tokenizer.** It lives in
  `Interp_LM4/util/condensed_tokenizer.py` and wraps
  `(GPT2Tokenizer, old_to_new, new_to_old)` behind a HF-shaped interface.
- **`encode` raises `KeyError` on out-of-vocab text — there is no `unk`
  fallback.** This is intentional: it fails loudly rather than silently feeding
  the model a token it never saw. If you prompt with vocabulary outside the bioS
  distribution, expect an exception, not garbage.

  You will hit this sooner than you expect, and the most common cause is not an
  exotic word — it's **a trailing space**. In this corpus every space is attached
  to the following word, so GPT-2's standalone-space token `220` never appears
  and is absent from the remap:

  ```python
  o2n[gpt2("...was born on ")["input_ids"][-1]]   # KeyError: 220
  o2n[gpt2("...was born on")["input_ids"][-1]]    # fine
  ```

  End prompts on a word, never on whitespace, and let the leading space ride on
  the next token (` February`, not `February`).
- **`bos == eos == pad == unk == 1835`.** All four roles are the same id.
- **Every bio is prefixed with one `<|endoftext|>` (1835).** Prompts should carry
  that BOS to stay on-distribution; without it the model is measurably
  off-distribution.

---

## The toy model

Both checkpoints are standard `LlamaForCausalLM` — loadable with plain
`transformers`, no custom code.

| | `grid-L4-H6` (BaselineL4) | `grid-L8-H6` (BaselineL8) |
|---|---|---|
| layers | 4 | 8 |
| `d_model` | 384 | 384 |
| heads | 6 (`head_dim` 64) | 6 |
| MLP hidden | 1024 | 1024 |
| params | 7.79 M | 14.87 M |
| vocab | 1,836 | 1,836 |
| ctx | 512 | 512 |
| tied embeddings | yes | yes |

Trained by `ablation_llama_grid.py` with `CONFIG.NAME = "bioS_N-Bd_final_grid"`;
run names encode the axes (`grid-L{layers}-H{heads}`).

`probe_birthday.json` ships next to each checkpoint — per-template
birthday-recall accuracy. BaselineL4 scores macro **MP 0.930** (month+day) and
**LP 1.000**, i.e. it has genuinely memorized the facts. Template 0 is the weak
one (MP 0.54) and is a known quirk, not a loading bug.

```python
from transformers import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained("release/models/grid-L4-H6")
```

### ⚠️ Two footguns that silently corrupt interp results

1. **fp32 everywhere.** The interp artifacts were fit in fp32. Loading in
   bf16/fp16 changes activations enough to move feature attributions.
2. **RMSNorm `eps=1e-5`, not the `1e-6` in `config.json`.** The checkpoint
   declares `rms_norm_eps=1e-6`, but every SAE/CLT in `Interp_LM4` was trained
   against TransformerLens's **default `1e-5`** rendering. `Interp_LM4` therefore
   deliberately keeps `1e-5` and asserts it in `tests/test_attribution.py`. **Do
   not "fix" this discrepancy** — matching `config.json` would invalidate every
   trained dictionary.

---

## The data

`data_config.json` is the source of truth:

```json
{"N": 50000, "K": 100, "SEQ_LEN": 512, "SEED": 0, "SHUFFLE_SEED": 1,
 "FIELDS": ["birthday"], "vocab_size": 50257, "reducedVocabSize": 1836,
 "eosToken": 50256, "reducedEOSToken": 1835}
```

- **N = 50,000 people**, sampled from name/city/university/company word lists in
  `data/fields/`. **Fully synthetic — no real personal data.** Birthdays are day
  1–28, any month, year 1700–1899.
- **K = 100 exposures** per person: the same birthday fact rendered through 100
  round-robin picks over the 46 paraphrase templates in `data/bio_text.py`.
- **`FIELDS = ("birthday",)`** — birthday only, though the renderer also supports
  `birthcity`, `university`, `field`, `company_city`, `company_name`.
- Bios are concatenated into one `uint16` stream and chunked into 512-token
  sequences (`PackedTokenDataset`).

`bios_prereduce.bin` holds raw GPT-2 ids; `bios_postreduce.bin` is the same
stream after the remap. **Train on `postreduce`** — it's the one matching the
checkpoints' 1,836-row embedding.

A gendered-pronoun invariant is baked in: person `id` parity encodes gender
(`id % 2 == 0` → "He"), and the sampler forces the middle name into the matching
half so bios stay coherent.

---

## This repo — `Training_On_LM4`

Generates data and trains models. Entry points:

| File | Does |
|---|---|
| `main.py` | The reference pipeline: sample → render → tokenize → condense vocab → train → probe. **Read this first** — it's the clearest statement of the whole flow |
| `config.py` | The `Config` dataclass (`N`, `K`, `SEED`, `SHUFFLE_SEED`, `FIELDS`, model dims) |
| `data/sample_people.py` | Deterministic person sampler |
| `data/bio_text.py` | 46 birthday templates + `render_bio` / `bio_stream`. Adapted from Meta's [PhysicsLM4](https://github.com/facebookresearch/PhysicsLM4) (Apache 2.0 — see `NOTICE`) |
| `data/tokenize_pack.py` | Tokenize/pack, **`build_vocab_remap`**, `remap_token_file`, `PackedTokenDataset` |
| `data/robustness.py` | Held-out template/person splits |
| `model/buildModel.py` | Constructs the Llama/GPT-2 configs |
| `ablation_llama_grid.py` | The sweep that produced `grid-L*-H*` |
| `eval/` | Birthday-recall probes (`MP`, `DayM`, `YearMD`, `FP`, `LP`) |

All artifacts are namespaced by `CONFIG.NAME`: data under `cache/{NAME}/`,
checkpoints under `runs/{NAME}/`. Change one knob to isolate an experiment.

---

## The sibling repo — `Interp_LM4`

<https://github.com/edwardF8/Interp_LM4> — consumes this repo's checkpoints and
data; does not train the base model.

| Subsystem | What | Hook point |
|---|---|---|
| `saes/` | JumpReLU sparse autoencoders | `blocks.{L}.hook_mlp_out` |
| `clts/` | Cross-layer transcoders — replace MLPs with a sparse dictionary; wired for Anthropic **circuit-tracer** attribution graphs + feature dashboards | reads `hook_resid_mid`, targets `hook_mlp_out` |
| `sae_CRL/` | Causal-representation SAE: dictionary + instantaneous DAG `M` + time-delayed `B_1..B_τ` | `blocks.{L}.hook_resid_post` |
| `util/` | `CondensedTokenizer`, bio sampler, loaders | — |

Structural things that will trip you up:

- **The Python project root is the parent directory**, not `Interp_LM4/`.
  `pyproject.toml`, `uv.lock`, and `.venv` live one level up and are shared with
  this repo. Run scripts **from the repo root** — imports are absolute
  (`from util.bio_sampler import …`).
- **Three environments, do not mix.** Local dev (uv, py3.13); training (conda,
  py3.11, `torch==2.12`); and a **separate circuit-tracer env** whose
  `safetensors>=0.5` / `transformers<=4.57.3` pins genuinely conflict with the
  training env. Anything importing `circuit_tracer` runs in the third one.
- **Bios must match training byte-for-byte.** `util/bio_sampler.py` puts
  `../Training_On_LM4` on `sys.path` and re-imports `render_bio` rather than
  copying the templates — so **the 46 templates live here, in this repo**, and
  editing them silently invalidates the interp artifacts.
- Heavy artifact dirs (`model/`, `data/`, `clt_storage/`, `saes/sae_runs`) are
  gitignored there; storage roots resolve via `$CLT_STORAGE_ROOT` /
  `$SAE_CRL_STORAGE_ROOT` with a repo-local fallback.

---

## Suggested first 15 minutes

```bash
# 1. prove the data pipeline is intact (seconds)
python release/regenerate_data.py --smoke

# 2. load the model and watch it recall a birthday across templates
python - <<'PY'
import json, sys, torch
from transformers import LlamaForCausalLM, GPT2Tokenizer
sys.path.insert(0, ".")
from data.bio_text import render_bio

D = "release/data/bioS_N-Bd_final_grid"
# Use the SHIPPED people.json. sample_people(N=50) is NOT the first 50 of
# N=50000 — the sampler draws from range(name_space) with N as the sample
# size, so a different N gives entirely different people.
people = json.load(open(f"{D}/people.json"))
o2n = {int(k): v for k, v in json.load(open(f"{D}/old_to_new.json")).items()}
n2o = {v: k for k, v in o2n.items()}

gpt2  = GPT2Tokenizer.from_pretrained("gpt2")
model = LlamaForCausalLM.from_pretrained("release/models/grid-L4-H6").eval()

month = people[0]["birthmonth"]
for e in (0, 1, 4, 5):
    text   = render_bio(people[0], exposure_idx=e, fields=("birthday",))
    prompt = "<|endoftext|>" + text[: text.index(" " + month)]  # cut BEFORE the space
    ids    = [o2n[t] for t in gpt2(prompt, add_special_tokens=False)["input_ids"]]
    top    = model(torch.tensor([ids])).logits[0, -1].topk(3)
    print(f"tmpl {e}: {prompt[-32:]!r:38s} -> "
          f"{[gpt2.decode([n2o[i]]) for i in top.indices.tolist()]}")
print("truth month:", month)
PY

# 3. then read main.py top-to-bottom — it is the whole system in 250 lines
```

Expected output — this is the real result, and worth understanding before you
trust or distrust the model:

```
tmpl 0: 'Gabriella Ella Rigby was born on'     -> [' the', ' February', ' August']
tmpl 1: "a Ella Rigby's birthday falls on"     -> [' February', ' August', ' December']
tmpl 4: "iella Ella Rigby's birth date is"     -> [' February', ' August', ' December']
tmpl 5: ' Gabriella Ella Rigby arrived on'     -> [' this', ' February', ' August']
truth month: February
```

The fact is memorized — ` February` is top-1 on templates 1 and 4. On templates 0
and 5 a filler token wins (`on the…`, `on this…`) and the correct month sits at
rank 2. That is a **surface-form effect, not a missing fact**, and it lines up
with the shipped `probe_birthday.json` (template 0 scores `MP 0.54` against a
macro of `0.930`). Don't read a low template-0 number as a broken checkpoint.

Note this snippet hand-rolls the remap to stay dependency-free; in real work use
`Interp_LM4/util/condensed_tokenizer.CondensedTokenizer`.

---

## Provenance

- bioS generator adapted from Meta's PhysicsLM4 (Apache 2.0); see `NOTICE`.
- Checkpoints and data config reproduced and checksum-verified 2026-08-15.
- People, names, and birthdays are **synthetic**. No real personal data.

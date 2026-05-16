# Birthday-Only Memorization Rig

This document describes the changes made to turn the full 6-field Capo bioS
rig into a configurable rig that, by default, runs **birthday-only**
biographies and evaluates memorization with four metrics over every
(person, paraphrase) pair.

## TL;DR

- Bio contents are now controlled by `CONFIG.FIELDS`. Default is
  `("birthday",)`. Set it to `("birthday", "birthcity", "university",
  "field", "company_city", "company_name")` to reproduce the legacy
  6-field bio.
- Templates are picked **round-robin** by `exposure_idx % len(templates)`,
  so when `K ≥ T_field` every paraphrase appears in the training stream at
  least `floor(K / T_field)` times per person.
- A new probe — `eval/birthday_probe.py` — scores every (eval person,
  birthday template) pair on **MP**, **Day | M**, **Year | M, D**, and
  **FP** (full-prediction greedy decode).

## File-by-file changes

### [data/bio_text.py](data/bio_text.py)
- Hoisted the 6 in-function template lists out to module scope as
  `TEMPLATES_BIRTHDAY`, `TEMPLATES_BIRTHCITY`, ..., `TEMPLATES_COMPANY_NAME`.
- `get_text_simple3` (legacy 6-field renderer) now references those
  module-level constants. Behavior is unchanged.
- Added `FIELD_SPECS` — a registry mapping each field name to its template
  pool, the placeholder name those templates use, a value-renderer
  callback, and whether the subject slot takes the full name or a pronoun.
- Added `render_bio(person, exposure_idx, fields, reverse_md=False)`. For
  each enabled field, `template_idx = exposure_idx % len(templates)` —
  round-robin coverage.
- `bio_stream(people, K, master_seed, shuffle_seed, fields=None)` now
  accepts an optional `fields` list. With `fields=None` it falls back to
  the legacy `get_text_simple3`.

### [data/sample_people.py](data/sample_people.py)
Unchanged. Continues to sample all 12 person fields, even when the bio
only uses a subset — keeps re-enabling other fields a one-line config
change.

### [config.py](config.py)
Added one field:

```python
FIELDS: tuple = ("birthday",)
```

`K` (exposures per person) keeps its existing meaning. Total bios =
`N * K`; the round-robin schedule inside `render_bio` ensures every
template is covered.

### [main.py](main.py)
- `CONFIG.FIELDS = ("birthday",)` near the other data knobs.
- `bio_stream(..., fields=tuple(CONFIG.FIELDS))`.
- After `build_vocab_remap`, render a handful of sample bios and call
  `assert_tokens_in_remap` to fail loudly if any eval-time token is
  missing from the reduced vocab.

### [data/tokenize_pack.py](data/tokenize_pack.py)
Added `assert_tokens_in_remap(old_to_new, sample_prompts, tokenizer)`.
Raises `RuntimeError` listing the missing token ids if any prompt
contains a token not present in the reduced-vocab map.

### [eval/birthday_probe.py](eval/birthday_probe.py)
New probe. Loads a checkpoint, rebuilds the GPT-2→reduced-vocab map from
the training run's `bios_prereduce.bin`, then for each (eval person,
birthday template) pair scores four metrics.

`eval/recall_probe.py` is left in place for backwards compatibility but
is no longer the primary evaluation tool.

## The four metrics

Let the bio be `" {name} was born on December 10, 1815."`. Tokenized
chunks (using GPT-2 BPE):

| Chunk     | Example tokens          |
|-----------|-------------------------|
| prefix    | `" Ada Mae Lovelace was born on"` |
| month     | `" December"` (1 token) |
| day       | `" 10"`       (1 token) |
| sep       | `","`         (1 token) |
| year      | `" 1815"`     (1–2 tokens) |
| trailing  | `"."` (varies by template; not scored) |

### MP — Month Prediction
Teacher-forced. Argmax at the last prefix position equals the first month
token.

### Day | M — Day given Month
Teacher-forced. Argmax at the position right before the day token equals
the day token. The model's context includes the **true** month — errors
do not compound from MP.

### Year | M, D — Year given Month, Day
Teacher-forced. Argmax at every year position (year is 1 or 2 BPE tokens
for the 1700–1899 range) equals the corresponding true year token. Model
context includes true month + day + comma.

### FP — Full Prediction
**Greedy autoregressive decode.** From the prefix, generate
`len(month + day + sep + year)` tokens. FP=1 only if the generated
sequence equals the true target sequence exactly. Errors compound — a
wrong month leaves the day prediction conditioned on a wrong context.

Why both teacher-forced and greedy: the teacher-forced metrics isolate
the model's per-field memorization signal; FP measures whether the model
can produce the full date end-to-end. Comparing them tells you where any
FP failure originates.

## Coverage guarantee

For each (person, field) pair, `render_bio` selects
`template_idx = exposure_idx % T_field`. The exposure loop in `main.py`
runs `exposure ∈ [0, K)`, so:

- When `K ≥ T_field`: every template appears for every person at least
  `floor(K / T_field)` times.
- When `K < T_field`: only the first `K` templates appear (per person).
  Default `K=100`, `T_birthday=46` so this isn't an issue — but worth
  knowing if you shrink `K`.

Because the eval probe iterates over *every* birthday template, the
seatbelt in `main.py` (`assert_tokens_in_remap`) will fire if any
template's tokens aren't in the reduced vocab. With the round-robin
schedule, that assertion should always pass.

## Running

Standard pipeline (default `FIELDS=("birthday",)`):

```bash
python main.py                                # generate data + train all model sizes
python -m eval.birthday_probe runs/default/2-3/final --m 50
```

To switch back to the full 6-field bio, edit one line in `main.py`:

```python
CONFIG.FIELDS = ("birthday", "birthcity", "university",
                 "field", "company_city", "company_name")
```

(or run `recall_probe.py` against the resulting checkpoints.)

## Adding a new field

1. Add the templates to [data/bio_text.py](data/bio_text.py) as a module-level
   constant `TEMPLATES_FOO = [...]`.
2. Add a `FIELD_SPECS["foo"]` entry pointing at it with a `render_value`
   callback and the right `subject` ("name" vs. "pronoun").
3. Make sure `sample_people` populates the underlying person dict key.
4. Add `"foo"` to `CONFIG.FIELDS`.

Optionally add a probe for the new field (the existing birthday probe is
field-specific; a generic version would parameterize over `FIELD_SPECS`).
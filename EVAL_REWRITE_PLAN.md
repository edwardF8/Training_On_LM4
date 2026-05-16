# Birthday-Only Memorization Rig — Change Plan

Scope: make field inclusion **configurable**, run with just `name + birthday` for now, guarantee every prompt template the eval uses is seen in training, and add a 4-metric probe (3 teacher-forced + 1 greedy).

Two key design choices the user asked for:
- **Keep `K`** (exposures per person) as the user-facing knob. Round-robin coverage is derived from `K`, not a new `C` config field.
- **Don't delete old fields**. Make field inclusion a toggle so flipping back to the 6-field setup is one config change.

---

## 1. Dataset generation — toggleable fields

### [data/sample_people.py](data/sample_people.py)
Keep the file's structure intact. The function already populates every field — leave it that way so other fields stay one config flip away. (Sampling extra fields is cheap; gating happens at render-time in `bio_text.py`.) No code changes here unless we want a `fields` arg to skip unused samplers for speed — not worth it now.

### [data/bio_text.py](data/bio_text.py)

Pull each field's templates into a registry so adding/removing fields is one entry:

```python
BIRTHDAY_TEMPLATES = [ ... sentence_structures1 ... ]    # 46
BIRTHCITY_TEMPLATES = [ ... sentence_structures2 ... ]
UNIVERSITY_TEMPLATES = [ ... sentence_structures3 ... ]
FIELD_TEMPLATES      = [ ... sentence_structures4 ... ]
COMPANY_CITY_TEMPLATES = [ ... sentence_structures5 ... ]
COMPANY_NAME_TEMPLATES = [ ... sentence_structures6 ... ]

FIELD_SPECS = {
    "birthday": {
        "templates": BIRTHDAY_TEMPLATES,
        "render_value": lambda p: f"{p['birthmonth']} {p['birthday']}, {p['birthyear']}",
        "subject": "name",   # placeholder uses full name
    },
    "birthcity":    {"templates": BIRTHCITY_TEMPLATES,    "render_value": lambda p: p["birthcity"],    "subject": "pronoun"},
    "university":   {"templates": UNIVERSITY_TEMPLATES,   "render_value": lambda p: p["university"],   "subject": "pronoun"},
    "field":        {"templates": FIELD_TEMPLATES,        "render_value": lambda p: p["field"],        "subject": "pronoun"},
    "company_city": {"templates": COMPANY_CITY_TEMPLATES, "render_value": lambda p: p["company1city"], "subject": "pronoun"},
    "company_name": {"templates": COMPANY_NAME_TEMPLATES, "render_value": lambda p: p["company1name"], "subject": "pronoun"},
}
```

`get_text_simple3` stays — unmodified, callable by anyone who wants the legacy 6-sentence bio. We add a new generator that respects a field list:

```python
def render_bio(person, exposure_idx, fields, master_seed=0):
    """Render a bio containing only the specified fields, deterministic in
    (master_seed, person['id'], exposure_idx).

    For each enabled field, template_idx = exposure_idx % T_field
    (round-robin) — guarantees every template is seen at least floor(K/T)
    times and at most ceil(K/T) times across the K exposures per person.
    """
    rng = random.Random((master_seed, person["id"], exposure_idx))
    full_name = f"{person['first_name']} {person['middle_name']} {person['last_name']}"
    pronoun = "He" if person["id"] % 2 == 0 else "She"

    parts = []
    for field in fields:
        spec = FIELD_SPECS[field]
        T = len(spec["templates"])
        t_idx = exposure_idx % T                     # round-robin coverage
        template = spec["templates"][t_idx]
        subject = full_name if spec["subject"] == "name" else pronoun
        value_key = "birthday"                       # placeholder name in every template
        # NB: all 6 sentence lists use `{name}` for the subject slot and
        # `{<field>}` for the value slot. We map by position via spec.
        value = spec["render_value"](person)
        parts.append(" " + template.format(name=subject, **{_value_placeholder(field): value}))

    # Preserve original company-city / company-name order randomization
    # only if both fields are enabled (matches legacy behavior).
    return "".join(parts)
```

(`_value_placeholder` is a tiny helper mapping `"birthday" -> "birthday"`, `"birthcity" -> "birthcity"`, etc. — just to make `.format()` work cleanly per field.)

Round-robin gives **automatic full coverage when K ≥ T_field**. For birthday-only with K=100 and T=46, every template appears 2 or 3 times per person. No new `C` knob needed.

Update `bio_stream` to accept `fields`:

```python
def bio_stream(people, fields, K=100, master_seed=0, shuffle_seed=1):
    pairs = [(i, e) for i in range(len(people)) for e in range(K)]
    random.Random(shuffle_seed).shuffle(pairs)
    for i, e in pairs:
        yield i, e, render_bio(people[i], exposure_idx=e, fields=fields,
                               master_seed=master_seed)
```

### [config.py](config.py)
Add one field, leave `K` alone:

```python
FIELDS: tuple = ("birthday",)   # subset of FIELD_SPECS keys
```

For the full legacy bio, set `FIELDS = ("birthday","birthcity","university","field","company_city","company_name")`.

### [main.py](main.py)
- Pass `CONFIG.FIELDS` through to `bio_stream`.
- `n_bios_total` stays `CONFIG.N * CONFIG.K` (unchanged — K is still exposures per person).

---

## 2. Tokenization — verify prompt coverage

### [data/tokenize_pack.py](data/tokenize_pack.py)
Add a safety check so any drift between train data and eval prompts fails loudly at data-prep time:

```python
def assert_tokens_in_remap(old_to_new, sample_prompts, tokenizer):
    """Raise if any token in `sample_prompts` is missing from `old_to_new`."""
    missing = set()
    for p in sample_prompts:
        for tok in tokenizer(p, add_special_tokens=False)["input_ids"]:
            if tok not in old_to_new:
                missing.add(tok)
    if missing:
        raise RuntimeError(
            f"{len(missing)} prompt token(s) missing from reduced vocab: {missing}"
        )
```

Call it from `main.py` right after `build_vocab_remap`, passing every (template, sample-person) rendering for every enabled field. With round-robin coverage this should always pass — the assert is the seatbelt, not the mechanism.

No changes to `remap_token_file`, `decode_from_remapped`, `PackedTokenDataset`.

---

## 3. Evaluation — `eval/birthday_probe.py`

New file. Keep [eval/recall_probe.py](eval/recall_probe.py) untouched.

### Procedure

1. Load model, tokenizer; rebuild `old_to_new` / `new_to_old` from the run's `bios_prereduce.bin`.
2. Sample `M` eval people (e.g. M=100) from the same `(N, SEED)` used at training.
3. For each `(person, template_idx)` pair in `BIRTHDAY_TEMPLATES`:
   - `prefix_text` = `template.split("{birthday}")[0]` formatted with `name=full_name`, then prepended with the leading space the bio uses.
   - `target_text` = `"{birthmonth} {birthday}, {birthyear}"`.
   - Tokenize prefix, and tokenize `" {month}"`, `" {day},"`, `" {year}"` independently so token positions for each component are explicit. Remap all token ids via `old_to_new`.
   - **One teacher-forced forward pass** on `prefix + target`:
     - **MP** = argmax at the last prefix position == true first month token.
     - **Day | M** = argmax at position right before day == true first day token.
     - **Year | M,D** = argmax at position right before year == true first year token. (If year is multi-token, also check subsequent year positions teacher-forced.)
   - **One greedy decode** from `prefix` for `len(target_tokens)` steps:
     - **FP** = generated token sequence equals `target_tokens` exactly.

### Reporting
- Macro-average across (person, template).
- Per-template breakdown (catches single-phrasing memorization).
- Optional: per-month / per-year breakdown to spot rare-value lag.

### Tokenization caveats
- `" January"` etc. should each be one GPT-2 BPE token; assert this once at probe start.
- Years 1700–1899: some are 1 token, some 2. Score the full year subsequence.
- Days 1–28 are all single-token after the leading-space prefix.

---

## 4. Metric definitions (reference)

| Metric | Conditioning | Correct when... |
|---|---|---|
| MP | true prefix | argmax at month position = true month token |
| Day \| M | true prefix + true month | argmax at day position = true day token |
| Year \| M, D | true prefix + true month + true day | argmax(es) at year positions = true year tokens |
| FP | true prefix + **model's own** generated tokens | greedy-decoded span equals `"{month} {day}, {year}"` exactly |

FP is the strict autoregressive test: errors compound. The three conditional probes diagnose where any FP failure originates.

---

## 5. Order of work

1. [data/bio_text.py](data/bio_text.py) — extract per-field template constants, build `FIELD_SPECS`, add `render_bio`, update `bio_stream` to take `fields`. Leave `get_text_simple3` in place.
2. [config.py](config.py) — add `FIELDS: tuple = ("birthday",)`.
3. [main.py](main.py) — thread `CONFIG.FIELDS` into `bio_stream`.
4. [data/tokenize_pack.py](data/tokenize_pack.py) — add `assert_tokens_in_remap`; call it from `main.py` after the remap.
5. `eval/birthday_probe.py` — new probe with the 4 metrics.
6. Smoke run with small `N` and one model size to verify end-to-end before the full sweep.
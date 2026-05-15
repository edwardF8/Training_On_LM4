# Capo (bioS) → Llama Training Roadmap

A milestone-by-milestone guide for building a Capo-style biography pretraining pipeline, training a small Llama (or GPT-2) on it, and prepping for downstream mechinterp work.

The starting point is `get_text_simple3(person, order=0, reverse_md=False)` in [`PhysicsLM4-main/data-synthetic-pretrain/Capo-bioS-bioR/Capo-bioS-bioR.py`](../PhysicsLM4-main/data-synthetic-pretrain/Capo-bioS-bioR/Capo-bioS-bioR.py). It takes one already-built `person` dict and returns one paraphrased biography string. The other two functions in that file (`generate_prompt2`, `augmentation_permutation2`) are for the *bioR* (Llama-generated) variant and can be ignored.

---

## Milestone 0 — Decisions to make first (mechinterp-driven)

These choices ripple into everything else, so worth committing now:

- **Tokenizer**: paper uses **GPT-2 tokenizer with tied embeddings** for Capo specifically (Appendix A.3), and limits vocab to ~3275 tokens since bioS doesn't use the rest. For mechinterp, **tied embeddings + small vocab is a gift** — logit-lens and unembedding analysis become cleaner. Stick with this.
- **Model size**: for mechinterp on a single workstation, aim for **8–24 layers, hidden 256–768**. The paper's `8L512D` (~25M params) is a sweet spot — big enough to actually learn, small enough that you can run every head's attention pattern in seconds.
- **Determinism**: seed `random`, `numpy`, `torch`, and **`torch.use_deterministic_algorithms(True)`**. You'll thank yourself when re-running a checkpoint to get the exact same activations.
- **Checkpoint cadence**: save every ~2k steps. Training dynamics (when does the model "grok" each attribute?) is one of the most interesting mechinterp targets here — see Allen-Zhu's Part 3.1 paper's probing experiments.

---

## Milestone 1 — Person generator

**Build**: a function `sample_people(N, seed) -> list[dict]` returning dicts with the keys `get_text_simple3` expects: `id, first_name, middle_name, last_name, birthday, birthmonth, birthyear, birthcity, university, field, company1name, company1city`.

**Tips:**
- Load the 8 files in [`fields/`](../PhysicsLM4-main/data-synthetic-pretrain/Capo-bioS-bioR/fields/). `company.txt` lines are `"Name; City, State"` — split on `';'` to get `company1name` and `company1city`.
- Sample **names from the cross product** (first × middle × last) **without replacement**. Paper has 400×400×1000 = 160M possible names, so for N≤2M collisions are rare but you should still dedupe.
- Birthdate domain: paper uses 12 months × 28 days × 200 years (1700–1899 in Part 3.3). Use month names (`"January"`, …) not numbers — they tokenize as one BPE token in GPT-2.
- `id` field: the original code uses `person['id']%2` to decide he/she. Just assign sequential IDs.

**Sanity check**: print 3 sample people, then run `get_text_simple3(person)` on each. Should look like English bios.

---

## Milestone 2 — Biography stream

**Build**: a generator that, for each person, yields `K=100` paraphrased biographies.

**Tips:**
- Each call to `get_text_simple3` re-randomizes templates internally. So just calling it K times per person gives you K paraphrases.
- The paper **shuffles all (person, exposure) pairs globally** before tokenizing — you don't want all 100 bios of person 0, then all 100 of person 1. Build a list of `(person_idx, exposure_idx)` tuples, shuffle once, then iterate.
- Memory: 100K people × 100 exposures × ~100 tokens ≈ 1B tokens. Don't materialize all bios in RAM — generate-and-tokenize on the fly, or write to disk in chunks.
- The `<bos>` separator matters. The cleanest approach: prepend `<|endoftext|>` (GPT-2's eos/bos) to each bio so the model learns each is independent. **For mechinterp this is critical** — you don't want activations from person A bleeding into a forward pass for person B.

**Sanity check**: tokenize 1000 bios, log mean/median/p99 token counts. If anything is >256 tokens you have a bug.

---

## Milestone 3 — Pack into training tensors

**Tips:**
- Pack bios end-to-end into fixed-length sequences (e.g., 512 tokens). This is standard "concatenate-and-chunk".
- **Don't mask attention across `<|endoftext|>` separators** for v1 — vanilla causal mask is fine and matches the paper. (Cross-document leakage is mild for short bios.) If you do want strict isolation later, that's a "block diagonal attention mask" and HF supports it via `attention_mask` per-sample.
- Save the tokenized dataset to disk (e.g., as a memmapped `.npy` or HF `datasets`). Re-tokenizing on every run is annoying.

---

## Milestone 4 — Model + training loop

**Tips:**
- For Llama:
  ```python
  LlamaConfig(
      vocab_size=~3300,
      hidden_size=512,
      num_hidden_layers=8,
      num_attention_heads=8,
      intermediate_size=hidden_size * 8 // 3,
      tie_word_embeddings=True,
      max_position_embeddings=512,
  )
  ```
  Then `LlamaForCausalLM(cfg)`.
- For GPT-2:
  ```python
  GPT2Config(
      vocab_size=...,
      n_embd=512,
      n_layer=8,
      n_head=8,
      n_positions=512,
      tie_word_embeddings=True,
  )
  ```
- **Loss should drop to ~3–4 nats fast**, then plateau as the model starts memorizing. Watch for it dropping below ~2.5 — that's when factual storage is happening.
- Optimizer per Capo Appendix A.3: AdamW, `lr=5e-4 to 1e-3`, `wd=0.01`, batch size scaling with N (24 for N=100K).
- Train **without dropout**. The paper relies on the data noise (paraphrases) for regularization, and dropout breaks clean mechinterp.

**Sanity check**: after a few thousand steps, prompt `"{first} {middle} {last} was born on"` for a training-set person and check if the completion is even close to the right month. If yes, the pipeline works.

---

## Milestone 5 — Evaluation (probe accuracy, not bits-per-param)

The bits-per-parameter metric in the paper is a Pareto-frontier thing across many runs — overkill for a single model. Simpler: for each attribute and each person, prompt with the canonical question and measure exact-match accuracy.

- Build a held-out set of *prompts* (not held-out people — bioS is about memorization).
- Use `model.generate(do_sample=False)` with short `max_new_tokens` matched to the attribute (e.g., 1 for birthmonth, ~5 for university name).

---

## Mechinterp prep — things to bake in *now*

1. **Save the person dict alongside the dataset** (with the same seed). You'll want to ask "what did the model store about person 42?" later.
2. **Single-token attributes are easier**: birthmonth is one token in GPT-2, university names are 3–8 tokens. Start interpretability work on birthmonth.
3. **Hook the model with [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)** if you go GPT-2 — its `HookedTransformer.from_pretrained` doesn't directly support custom checkpoints, but you can `from_pretrained_no_processing` and then `.load_state_dict()`. For Llama it's harder; vanilla forward hooks work.
4. **Run a "knowledge localization" probe early** (cf. ROME paper): mean-ablate residual stream at each layer for a "name → birthmonth" prompt and find the layer where ablation kills accuracy. This tells you where the lookup happens.

---

## Suggested file layout

```
capo_train/
  data/
    sample_people.py          # Milestone 1
    bio_stream.py             # Milestone 2 (imports get_text_simple3)
    tokenize_pack.py          # Milestone 3
  model/
    build_model.py            # Milestone 4 (Llama or GPT-2 selectable)
  train.py                    # Milestone 4 loop
  eval/
    recall_probe.py           # Milestone 5
  configs/
    small_llama.yaml          # hyperparams per model
```

Start with Milestone 1, get a clean person generator working, then move on.

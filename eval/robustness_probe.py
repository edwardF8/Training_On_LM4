"""RobustnessTest probe — FP/LP only, bucketed by template-diversity group.

See docs/superpowers/specs/2026-06-10-robustness-test-design.md.

Scoring reuses the birthday_legacy machinery: eval.birthday_probe_legacy
.score_pairs_fp_lp_batched builds the same prompts/targets as score_pair and
reads FP (greedy probe) and LP (lenient probe) off the same greedy decode —
but batched across pairs and KV-cached, so it is far faster while giving
byte-identical FP/LP (see tests/test_probe_batching.py). The teacher-forced
metrics (MP/DayM/YearMD) are not needed here and are not computed.

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

from data.bio_text import FIELD_SPECS
from eval.birthday_probe_legacy import score_pairs_fp_lp_batched

BUCKETS = ("total", "limitedSet", "fullSet", "limitedSeen", "limitedUnseen")
METRICS = ("FP", "LP")
# Pairs greedy-decoded together per forward batch. The decode is KV-cached and
# left-padded, so this only trades GPU memory for speed — results are unchanged.
PROBE_BATCH_SIZE = 128
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
    # Flatten to (person, template) pairs (person-major, ascending template) and
    # carry each pair's bucketing metadata alongside. The batched scorer returns
    # FP/LP per pair in this same order, so decoding and bucketing stay decoupled
    # yet aligned — the bucket math below is byte-identical to the old per-pair
    # loop, only the scoring is now KV-cached + batched.
    tasks = []   # (person_idx, is_limited, t_idx)
    pairs = []   # (person, template)
    for person_idx, is_limited in plan:
        person = people[person_idx]
        for t_idx, template in enumerate(templates):
            tasks.append((person_idx, is_limited, t_idx))
            pairs.append((person, template))

    scores_list = score_pairs_fp_lp_batched(
        model, tokenizer, old_to_new, new_to_old, eos_remapped,
        pairs, device, batch_size=PROBE_BATCH_SIZE)

    for (person_idx, is_limited, t_idx), scores in zip(tasks, scores_list):
        allowed = allowed_by_person.get(person_idx, set())
        group = "limited" if is_limited else "full"
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

"""Probe dispatcher — pick which probes to run on a freshly-trained model.

`PROBE_REGISTRY` maps a short name (used in each ablation script's `PROBES`
list) to a runner that:
  - calls the underlying probe with the right args
  - returns (results, wandb_payload, wandb_tables, json_filename)

`run_probes(probes, ...)` invokes each requested runner, writes the raw
results to `<out_dir>/final/<json_filename>`, then logs the merged scalar
payload + every per-template wandb.Table.

Probe namespaces (wandb keys + JSON filenames):

  birthday_legacy → eval/birthday_probe_legacy.py
      wandb:  birthdayProbe/MP, /DayM, /YearMD, /FP
              birthdayProbe/per_template               (table)
      json:   probe_birthday.json
      Birthday-only, 4 metrics: MP / DayM / YearMD / FP.

  sequential      → eval/sequential_probe.py
      wandb:  sequentialProbe/FP_FULL
              sequentialProbe/<field>/{TF,FP}
              sequentialProbe/per_template/<field>     (table)
      json:   probe_sequential.json
      Multi-field bio rendered exactly as at training (pronouns after the
      first field). Per (person, exposure): one TF pass + one AR decode of
      the whole bio → all TF_<field>, FP_<field>, and FP_FULL together.

  separate        → eval/separate_probing.py
      wandb:  separateProbe/<field>/{TF,FP}
              separateProbe/per_template/<field>       (table)
      json:   probe_separate.json
      Each field as an INDEPENDENT one-field bio with the full name always
      substituted for the subject. The gap (sequential − separate) reveals
      how much the model leans on cross-field context vs. direct
      (name → value) memorization.
"""
from __future__ import annotations

import json
from pathlib import Path

import wandb


# ---------------------------------------------------------------------------
# Per-probe runners. Each returns:
#   results:  dict       (serialized verbatim to JSON)
#   payload:  dict       (scalar metrics, one wandb.log call covers all probes)
#   tables:   dict       (per-template wandb.Table values, logged separately)
#   json_name: str       (basename under {out_dir}/final/)
# ---------------------------------------------------------------------------

def _run_birthday_legacy(model, tokenizer, old_to_new, people, fields, max_people):
    from eval.birthday_probe_legacy import run_probe
    results = run_probe(model, tokenizer, old_to_new, people, max_people=max_people)
    payload = {
        "birthdayProbe/MP":     results["macro"]["MP"],
        "birthdayProbe/DayM":   results["macro"]["DayM"],
        "birthdayProbe/YearMD": results["macro"]["YearMD"],
        "birthdayProbe/FP":     results["macro"]["FP"],
    }
    tables = {}
    if results.get("per_template"):
        table = wandb.Table(columns=["template_idx", "MP", "DayM", "YearMD", "FP"])
        for t_idx, accs in sorted(results["per_template"].items()):
            table.add_data(int(t_idx),
                           accs["MP"], accs["DayM"], accs["YearMD"], accs["FP"])
        tables["birthdayProbe/per_template"] = table
    return results, payload, tables, "probe_birthday.json"


def _run_sequential(model, tokenizer, old_to_new, people, fields, max_people):
    from eval.sequential_probe import run_probe
    results = run_probe(
        model, tokenizer, old_to_new, people,
        fields=fields, max_people=max_people,
    )
    payload = {"sequentialProbe/FP_FULL": results["FP_FULL"]}
    tables = {}
    for f, fr in results["per_field"].items():
        payload[f"sequentialProbe/{f}/TF"] = fr["TF"]
        payload[f"sequentialProbe/{f}/FP"] = fr["FP"]
        if fr["per_template"]:
            table = wandb.Table(columns=["template_idx", "TF", "FP"])
            for t_idx, accs in sorted(fr["per_template"].items()):
                table.add_data(int(t_idx), accs["TF"], accs["FP"])
            tables[f"sequentialProbe/per_template/{f}"] = table
    return results, payload, tables, "probe_sequential.json"


def _run_separate(model, tokenizer, old_to_new, people, fields, max_people):
    from eval.separate_probing import run_separate_probe
    results = run_separate_probe(
        model, tokenizer, old_to_new, people,
        fields=fields, max_people=max_people,
    )
    payload = {}
    tables = {}
    for f, fr in results["per_field"].items():
        payload[f"separateProbe/{f}/TF"] = fr["TF"]
        payload[f"separateProbe/{f}/FP"] = fr["FP"]
        if fr["per_template"]:
            table = wandb.Table(columns=["template_idx", "TF", "FP"])
            for t_idx, accs in sorted(fr["per_template"].items()):
                table.add_data(int(t_idx), accs["TF"], accs["FP"])
            tables[f"separateProbe/per_template/{f}"] = table
    return results, payload, tables, "probe_separate.json"


PROBE_REGISTRY = {
    "birthday_legacy": _run_birthday_legacy,
    "sequential":      _run_sequential,
    "separate":        _run_separate,
}


def run_probes(probes, model, tokenizer, old_to_new, people, *,
               fields, out_dir, max_people: int = 50, run_name: str = ""):
    """Run each requested probe, save JSON, and log to wandb.

    Args:
        probes: iterable of probe names (keys in PROBE_REGISTRY).
        fields: tuple of field names — passed to any probe that takes them.
        out_dir: run directory; results land under f"{out_dir}/final/".
        max_people: shared across all probes.
        run_name: just for prettier console headers.

    Unknown probe names raise ValueError so typos fail fast.
    """
    for p in probes:
        if p not in PROBE_REGISTRY:
            raise ValueError(
                f"Unknown probe {p!r}; choices: {list(PROBE_REGISTRY)}"
            )

    final_dir = Path(out_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    merged_payload = {}
    table_logs = []
    for probe_name in probes:
        runner = PROBE_REGISTRY[probe_name]
        print(f"\n=== {probe_name} probe on {run_name or '<unnamed>'} "
              f"(fields={fields}) ===")
        results, payload, tables, json_name = runner(
            model, tokenizer, old_to_new, people, fields, max_people,
        )
        json_path = final_dir / json_name
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved → {json_path}")
        merged_payload.update(payload)
        for key, tbl in tables.items():
            table_logs.append((key, tbl))

    if merged_payload:
        wandb.log(merged_payload)
    for key, tbl in table_logs:
        wandb.log({key: tbl})

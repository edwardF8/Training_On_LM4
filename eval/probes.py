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
      wandb:  birthdayProbe/MP, /DayM, /YearMD, /FP, /LP
              birthdayProbe/per_template               (table)
      json:   probe_birthday.json
      Birthday-only, 5 metrics: MP / DayM / YearMD / FP / LP.

  sequential      → eval/sequential_probe.py
      wandb:  sequentialProbe/<field>/TF
              sequentialProbe/per_template/<field>     (table)
      json:   probe_sequential.json
      Multi-field bio rendered exactly as at training (pronouns after the
      first field). Per (person, exposure): one TF forward pass on the
      full true bio → TF_<field> = 1 iff every value-span position passes
      under teacher forcing.

  separate        → eval/separate_probing.py
      wandb:  separateProbe/<field>/TF
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
        "birthdayProbe/LP":     results["macro"]["LP"],
    }
    tables = {}
    if results.get("per_template"):
        table = wandb.Table(
            columns=["template_idx", "MP", "DayM", "YearMD", "FP", "LP"])
        for t_idx, accs in sorted(results["per_template"].items()):
            table.add_data(int(t_idx), accs["MP"], accs["DayM"],
                           accs["YearMD"], accs["FP"], accs["LP"])
        tables["birthdayProbe/per_template"] = table
    return results, payload, tables, "probe_birthday.json"


def _run_sequential(model, tokenizer, old_to_new, people, fields, max_people):
    from eval.sequential_probe import run_probe
    results = run_probe(
        model, tokenizer, old_to_new, people,
        fields=fields, max_people=max_people,
    )
    payload = {}
    tables = {}
    for f, fr in results["per_field"].items():
        payload[f"sequentialProbe/{f}/TF"] = fr["TF"]
        if fr["per_template"]:
            table = wandb.Table(columns=["template_idx", "TF"])
            for t_idx, accs in sorted(fr["per_template"].items()):
                table.add_data(int(t_idx), accs["TF"])
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
        if fr["per_template"]:
            table = wandb.Table(columns=["template_idx", "TF"])
            for t_idx, accs in sorted(fr["per_template"].items()):
                table.add_data(int(t_idx), accs["TF"])
            tables[f"separateProbe/per_template/{f}"] = table
    return results, payload, tables, "probe_separate.json"


PROBE_REGISTRY = {
    "birthday_legacy": _run_birthday_legacy,
    "sequential":      _run_sequential,
    "separate":        _run_separate,
}


def run_probes(probes, model, tokenizer, old_to_new, people, *,
               fields, out_dir, max_people: int = 50, run_name: str = "",
               epoch=None, wandb_step=None):
    """Run each requested probe, save JSON, and log to wandb.

    Args:
        probes: iterable of probe names (keys in PROBE_REGISTRY).
        fields: tuple of field names — passed to any probe that takes them.
        out_dir: run directory.
        max_people: shared across all probes.
        run_name: just for prettier console headers.
        epoch: None for the final probe — JSON lands in
            f"{out_dir}/final/probe_*.json". An int for a mid-training probe
            — JSON lands in f"{out_dir}/probes/probe_*_epoch{N}.json" and the
            wandb payload is tagged with the epoch.
        wandb_step: explicit step for wandb.log (pass the trainer's
            global_step so probe points line up with the loss curve).

    Returns {probe_name: results_dict} for each probe run.
    Unknown probe names raise ValueError so typos fail fast.
    """
    for p in probes:
        if p not in PROBE_REGISTRY:
            raise ValueError(
                f"Unknown probe {p!r}; choices: {list(PROBE_REGISTRY)}"
            )

    out_dir = Path(out_dir)
    if epoch is None:
        json_dir, suffix = out_dir / "final", ""
    else:
        json_dir, suffix = out_dir / "probes", f"_epoch{int(epoch)}"
    json_dir.mkdir(parents=True, exist_ok=True)

    merged_payload = {}
    table_logs = []
    all_results = {}
    for probe_name in probes:
        runner = PROBE_REGISTRY[probe_name]
        tag = f" (epoch {epoch})" if epoch is not None else ""
        print(f"\n=== {probe_name} probe on {run_name or '<unnamed>'}{tag} "
              f"(fields={fields}) ===")
        results, payload, tables, json_name = runner(
            model, tokenizer, old_to_new, people, fields, max_people,
        )
        stem = json_name[:-5] if json_name.endswith(".json") else json_name
        json_path = json_dir / f"{stem}{suffix}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved → {json_path}")
        merged_payload.update(payload)
        all_results[probe_name] = results
        for key, tbl in tables.items():
            table_logs.append((key, tbl))

    if epoch is not None:
        merged_payload["epoch"] = epoch
    log_kw = {} if wandb_step is None else {"step": wandb_step}
    if merged_payload:
        wandb.log(merged_payload, **log_kw)
    for key, tbl in table_logs:
        wandb.log({key: tbl}, **log_kw)
    return all_results

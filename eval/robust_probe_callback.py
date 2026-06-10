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

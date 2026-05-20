"""TrainerCallback that runs the eval probes at chosen epoch boundaries.

This turns an epoch-scaling study into ONE training run. Instead of training
a fresh model for every EPOCHS value, train once to `max_epochs` and probe the
in-memory model after each epoch in `probe_epochs`. One initialization, one LR
schedule — a genuine probe-accuracy-vs-epoch curve, not six runs whose only
real difference is where the cosine schedule decayed to.

Each probe point:
  - logs the probe's scalar metrics to wandb (at the trainer's global_step),
  - writes a per-epoch JSON via `run_probes` (mid-training points land in
    `{out_dir}/probes/`, the final point in `{out_dir}/final/`),
  - appends one row to a shared CSV: family, study, run_name, numLayers,
    numHeads, dmodel, epoch, MP, DayM, YearMD, FP, LP.

The CSV is the cross-run artifact — point one CSV path at a whole sweep and
every (run, epoch) accumulates into it.
"""
import csv
from pathlib import Path

from transformers import TrainerCallback

from eval.probes import run_probes


CSV_COLUMNS = ["family", "study", "run_name", "numLayers", "numHeads",
               "dmodel", "epoch", "MP", "DayM", "YearMD", "FP", "LP"]


class ProbeAtEpochs(TrainerCallback):
    """Probe the model after each epoch in `probe_epochs` (incl. `max_epochs`)."""

    def __init__(self, *, probes, tokenizer, old_to_new, people, fields,
                 out_dir, run_name, probe_epochs, max_epochs, csv_path,
                 run_info):
        self.probes       = tuple(probes)
        self.tokenizer    = tokenizer
        self.old_to_new   = old_to_new
        self.people       = people
        self.fields       = tuple(fields)
        self.out_dir      = str(out_dir)
        self.run_name     = run_name
        self.max_epochs   = int(max_epochs)
        self.probe_epochs = sorted({int(e) for e in probe_epochs}
                                   | {int(max_epochs)})
        self.csv_path     = Path(csv_path)
        self.run_info     = dict(run_info)      # family/study/numLayers/...
        self._probed      = set()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(round(state.epoch))
        if epoch in self._probed or epoch not in self.probe_epochs:
            return
        model = kwargs.get("model")
        if model is None:
            return
        self._probed.add(epoch)
        self._run(epoch, model, getattr(state, "global_step", None))

    def _run(self, epoch, model, global_step):
        was_training = model.training
        model.eval()
        print(f"\n--- ProbeAtEpochs: probing {self.run_name} at epoch "
              f"{epoch}/{self.max_epochs} ---")
        try:
            # Final epoch -> canonical final/ JSON; earlier epochs -> probes/.
            epoch_arg = None if epoch >= self.max_epochs else epoch
            results = run_probes(
                self.probes, model, self.tokenizer, self.old_to_new,
                self.people, fields=self.fields, out_dir=self.out_dir,
                run_name=self.run_name, epoch=epoch_arg, wandb_step=global_step,
            )
            self._append_csv(epoch, results)
        except Exception as e:
            print(f"  ProbeAtEpochs: probe at epoch {epoch} FAILED — "
                  f"{type(e).__name__}: {e}  (training continues)")
        finally:
            if was_training:
                model.train()

    def _append_csv(self, epoch, results):
        """Append one row per probe point. birthday_legacy fills the metric
        columns; other probe types leave them blank (their JSON has detail)."""
        row = {c: "" for c in CSV_COLUMNS}
        row.update(self.run_info)
        row["run_name"] = self.run_name
        row["epoch"]    = epoch
        macro = results.get("birthday_legacy", {}).get("macro")
        if macro:
            for k in ("MP", "DayM", "YearMD", "FP", "LP"):
                if k in macro:
                    row[k] = round(macro[k], 4)

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(f"  ProbeAtEpochs: appended epoch {epoch} row -> {self.csv_path}")
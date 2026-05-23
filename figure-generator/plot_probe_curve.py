"""probe_curve grid plot — birthday-probe accuracy vs epoch for a L x H sweep.

Reads source-results/probe_curve.csv (written by eval/probe_callback.py) and
draws one panel per (numLayers, numHeads) cell of the grid, each plotting
MP / FP / LP against training epoch. Runs still missing from the CSV (sweep in
progress) show as empty panels; partially-probed runs just stop early.

    python figure-generator/plot_probe_curve.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV  = HERE / "source-results" / "probe_curve.csv"
OUT  = HERE / "figure" / "probe_curve_grid.png"

# The intended grid — fixed here so not-yet-run cells show as blank panels.
LAYERS = [1, 2, 4, 8]
HEADS  = [1, 2, 4, 6, 8]
EPOCHS = [1, 2, 4, 6, 8, 12, 16]
METRICS = [("MP", "tab:gray", 1.3, "MP  (month)"),
           ("FP", "tab:blue", 1.8, "FP  (full, greedy)"),
           ("LP", "tab:red",  2.2, "LP  (lenient)")]

df = pd.read_csv(CSV)

# --- coverage report (so the "missing results" are explicit) ---------------
print(f"{CSV.name}: {len(df)} rows, "
      f"{df['run_name'].nunique()}/{len(LAYERS) * len(HEADS)} runs present")
for L in LAYERS:
    for H in HEADS:
        n = len(df[(df.numLayers == L) & (df.numHeads == H)])
        if n == 0:
            print(f"  L{L}-H{H}: not run yet")
        elif n < len(EPOCHS):
            print(f"  L{L}-H{H}: partial ({n}/{len(EPOCHS)} probe epochs)")

# --- faceted plot: rows = layers, cols = heads -----------------------------
fig, axes = plt.subplots(
    len(LAYERS), len(HEADS),
    figsize=(3.1 * len(HEADS), 2.7 * len(LAYERS)),
    sharex=True, sharey=True,
)

for i, L in enumerate(LAYERS):
    for j, H in enumerate(HEADS):
        ax = axes[i][j]
        sub = df[(df.numLayers == L) & (df.numHeads == H)].sort_values("epoch")
        if len(sub):
            for col, color, lw, _ in METRICS:
                ax.plot(sub["epoch"], sub[col], "-o",
                        ms=3.5, lw=lw, color=color)
        else:
            ax.text(0.5, 0.5, "not run yet", transform=ax.transAxes,
                    ha="center", va="center", color="0.6", style="italic")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(EPOCHS)
        ax.grid(alpha=0.3, lw=0.5)
        if i == 0:
            ax.set_title(f"numHeads = {H}\n$d_{{model}}$ = {64 * H}", fontsize=10)
        if j == 0:
            ax.set_ylabel(f"numLayers = {L}\n\nprobe accuracy", fontsize=10)
        if i == len(LAYERS) - 1:
            ax.set_xlabel("training epoch")

handles = [Line2D([], [], color=c, marker="o", ms=4, lw=lw, label=lbl)
           for _, c, lw, lbl in METRICS]
fig.legend(handles=handles, ncol=3, loc="upper center",
           bbox_to_anchor=(0.5, 1.015), frameon=False, fontsize=10)
fig.suptitle("Birthday-probe accuracy vs epoch  —  layers x heads grid",
             y=1.05, fontsize=13, weight="bold")
fig.tight_layout()

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nsaved -> {OUT}")

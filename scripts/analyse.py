"""Analyse Phase 1 runs and emit a one-glance summary + headline plot."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
console = Console()

rows = []
for summary in (ROOT / "runs").glob("*/summary.json"):
    for entry in json.loads(summary.read_text()):
        rows.append({**entry, "run_root": summary.parent.name})

df = pd.DataFrame(rows)

# Normalise older arch label "multi-vote" -> "multi-vote-flat"
df["arch"] = df["arch"].replace({"multi-vote": "multi-vote-flat"})

# For each arch, keep the most recent sweep(s) up to the most recent 5 runs.
# This naturally merges a partial 3-run sweep + a 2-run top-up into n=5.
df_sorted = df.sort_values("run_root", ascending=False)
latest = (
    df_sorted.groupby("arch", as_index=False, group_keys=False)
    .head(5)
    .reset_index(drop=True)
)

console.rule("[bold cyan]Per-run results (latest sweep)[/]")
t = Table()
t.add_column("arch")
t.add_column("run", justify="right")
t.add_column("violations", justify="right")
t.add_column("ambiguity", justify="right")
for _, r in latest.iterrows():
    t.add_row(r["arch"], str(r["run"]), str(r["violations"]), str(r["ambiguity"]))
console.print(t)

console.rule("[bold cyan]Aggregates by architecture[/]")
agg = (
    latest.groupby("arch")
    .agg(
        n=("violations", "size"),
        viol_mean=("violations", "mean"),
        viol_std=("violations", "std"),
        viol_min=("violations", "min"),
        viol_max=("violations", "max"),
        amb_mean=("ambiguity", "mean"),
    )
    .round(2)
)
console.print(agg.to_string())

# Headline plot.
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
order = ["iter", "multi-vote-flat", "multi-vote-over-pipe", "multi-pipe"]
present = [a for a in order if a in agg.index]

# Colour by hand-off count: green (0), yellow (1), orange (1+pipe), red (4)
palette = {
    "iter": "#2a9d8f",
    "multi-vote-flat": "#e9c46a",
    "multi-vote-over-pipe": "#f4a261",
    "multi-pipe": "#e76f51",
}
colors = [palette[a] for a in present]

means_v = agg.loc[present, "viol_mean"]
stds_v = agg.loc[present, "viol_std"].fillna(0)
ax[0].bar(present, means_v, yerr=stds_v, capsize=6, color=colors)
ax[0].set_title("Constraint violations (mean ± std)")
ax[0].set_ylabel("violations")
ax[0].tick_params(axis="x", rotation=20)

amb = agg.loc[present, "amb_mean"]
ax[1].bar(present, amb, color=colors)
ax[1].set_title("Residual ambiguity (mean)")
ax[1].set_ylabel("underspecified steps")
ax[1].tick_params(axis="x", rotation=20)

fig.suptitle("ANSELM §10 Phase 1 — gpt-4o-mini-2024-07-18, 5 runs/arch")
fig.tight_layout()
out = ROOT / "runs" / "phase1_headline.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
console.print(f"\n[green]Plot saved to[/] {out}")

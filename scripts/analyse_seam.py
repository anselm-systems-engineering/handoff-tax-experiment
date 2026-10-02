"""Analyse cross-context runs — sweep-oriented.

Groups runs by sweep directory (one directory per CLI invocation), detects
model and brief version from the call logs inside each sweep, and emits
per-sweep aggregates + a per-family confound check. Never merges sweeps with
different models, briefs, or round-trip counts.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
console = Console()


def sweep_info(sweep: Path) -> tuple[str, str, int]:
    """Detect (model, brief_id, k) from the sweep's run logs."""
    model, brief_id, k = "unknown", "unknown", 1
    summary_path = sweep / "summary.json"
    if summary_path.exists():
        entries = json.loads(summary_path.read_text(encoding="utf-8"))
        if entries:
            k = int(entries[0].get("roundtrips", 1))
    for run_dir in sorted(sweep.glob("run_*")):
        for call_file in sorted(run_dir.glob("call_*.json")):
            try:
                payload = json.loads(call_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            model = payload.get("model", model)
            for message in payload.get("messages", []):
                match = re.search(r'"id"\s*:\s*"(atlas[a-z0-9_]*)"', message.get("content", ""))
                if match:
                    brief_id = match.group(1)
            break
        if model != "unknown" and brief_id != "unknown":
            break
    return model, brief_id, k


def sweep_aggregate(sweep: Path) -> dict:
    entries = json.loads((sweep / "summary.json").read_text(encoding="utf-8"))
    violations = [e["violations"] for e in entries]
    coverages = [e["commitment_coverage"] for e in entries]
    tokens = [e["tokens"] for e in entries]
    rejections = [
        sum(int(v) for v in (e.get("rejection_rounds") or {}).values()) for e in entries
    ]
    n = len(entries)

    fam: Counter = Counter()
    structural = 0
    cell_counts = []
    for run_dir in sorted(sweep.glob("run_*")):
        vp = run_dir / "violations.json"
        if vp.exists():
            for v in json.loads(vp.read_text(encoding="utf-8")):
                fam[v["constraint_id"]] += 1
                if v["constraint_id"] == "typed_graph_validity":
                    structural += 1
        eco = run_dir / "ecosystem.json"
        if eco.exists():
            cell_counts.append(len(json.loads(eco.read_text(encoding="utf-8")).get("cells", {})))
    return {
        "n": n,
        "viol_mean": sum(violations) / n,
        "viol_std": (sum((v - sum(violations) / n) ** 2 for v in violations) / n) ** 0.5,
        "viol_range": f"{min(violations)}–{max(violations)}",
        "coverage": sum(coverages) / n,
        "rejections": sum(rejections) / n,
        "tokens": sum(tokens) / n,
        "cell_counts": cell_counts,
        "structural": structural,
        "families": fam,
    }


sweeps = sorted(
    (
        p
        for p in (ROOT / "runs").glob("*_*")
        if (p / "summary.json").exists()
        and (p.name.endswith("_typed-seam") or p.name.endswith("_prose-seam"))
    ),
    key=lambda p: p.name,
)

table = Table(title="Cross-context sweeps")
for col in ("sweep", "model", "brief", "k", "n", "viol", "range", "cov", "rej", "cells", "struct", "tokens"):
    table.add_column(col, justify="right" if col not in ("sweep", "model", "brief") else "left")
rows_for_table = []
for sweep in sweeps:
    model, brief_id, k = sweep_info(sweep)
    agg = sweep_aggregate(sweep)
    rows_for_table.append((sweep, model, brief_id, k, agg))
    table.add_row(
        sweep.name,
        model,
        brief_id.replace("atlas_cross_context", "v1").replace("atlas_cross_context_v2", "v2"),
        str(k),
        str(agg["n"]),
        f"{agg['viol_mean']:.1f}±{agg['viol_std']:.1f}",
        agg["viol_range"],
        f"{agg['coverage']:.2f}",
        f"{agg['rejections']:.1f}",
        str(agg["cell_counts"]),
        str(agg["structural"]),
        str(int(agg["tokens"])),
    )
console.print(table)

console.rule("[bold]Compact TSV[/]")
print("sweep\tmodel\tbrief\tk\tn\tviol_mean\tviol_std\tcoverage\trejections\tcells\tstructural\ttokens")
for sweep, model, brief_id, k, agg in rows_for_table:
    print(
        f"{sweep.name}\t{model}\t{brief_id}\t{k}\t{agg['n']}\t"
        f"{agg['viol_mean']:.2f}\t{agg['viol_std']:.2f}\t{agg['coverage']:.2f}\t"
        f"{agg['rejections']:.2f}\t{','.join(map(str, agg['cell_counts']))}\t"
        f"{agg['structural']}\t{int(agg['tokens'])}"
    )

console.rule("[bold cyan]Family breakdown (newest sweep per condition)[/]")
seen: set[str] = set()
for sweep in sorted(sweeps, reverse=True):
    model, brief_id, k = sweep_info(sweep)
    key = (brief_id, k, "typed" if "typed" in sweep.name else "prose")
    if key in seen:
        continue
    seen.add(key)
    agg = sweep_aggregate(sweep)
    console.print(f"[bold]{sweep.name}[/] model={model} brief={brief_id} k={k}")
    for family, count in agg["families"].most_common():
        console.print(f"  {family}: {count}")

"""CLI for the Tier 0–1 prototype.

    python -m anselm_experiment.tier01 check DIR [--vocabulary PATH]
    python -m anselm_experiment.tier01 graph DIR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from .graph import render_mermaid
from .oracle import check_ecosystem
from .vocabulary import VocabularyError, load_vocabulary

console = Console()

_DEFAULT_VOCABULARY = Path(__file__).resolve().parents[3] / "tier01" / "vocabulary.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="anselm-tier01", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    check_p = sub.add_parser("check", help="run the deterministic coherence oracle")
    check_p.add_argument("directory", help="directory of knowledge cells + relations.yaml")
    check_p.add_argument("--vocabulary", default=str(_DEFAULT_VOCABULARY))

    graph_p = sub.add_parser("graph", help="render the typed graph as Mermaid")
    graph_p.add_argument("directory")
    graph_p.add_argument("--vocabulary", default=str(_DEFAULT_VOCABULARY))

    args = parser.parse_args(argv)

    try:
        vocabulary = load_vocabulary(args.vocabulary)
    except VocabularyError as e:
        console.print("[red]Vocabulary errors:[/red]")
        for err in e.errors:
            console.print(f"  [red]•[/red] {err}")
        return 2

    if args.command == "graph":
        report = check_ecosystem(args.directory, vocabulary)
        if not report.ok:
            console.print("[red]Graph refused — the ecosystem has mechanical errors:[/red]")
            for f in report.errors:
                console.print(f"  [red]•[/red] {f.where}: {f.message}")
            return 1
        print(render_mermaid(report.cells, report.relations))
        return 0

    # check
    report = check_ecosystem(args.directory, vocabulary)
    console.print(f"[bold]{len(report.cells)} cells, {len(report.relations)} relations[/bold]")

    if report.errors:
        console.print("[red]Errors (mechanical):[/red]")
        for f in report.errors:
            console.print(f"  [red]•[/red] {f.where}: {f.message}")
    if report.warnings:
        console.print("[yellow]Warnings (attention needed):[/yellow]")
        for f in report.warnings:
            console.print(f"  [yellow]•[/yellow] {f.where}: {f.message}")

    used = {e.get("type") for e in report.relations if e.get("type")}
    console.print("[bold]Consumer ledger (the praxeological test):[/bold]")
    for rtype_name, rtype in vocabulary.relations.items():
        status = "in use" if rtype_name in used else "unused"
        color = "green" if rtype_name in used else "dim"
        consumers = "; ".join(rtype.consumers)
        console.print(f"  [{color}]{rtype_name}[/{color}] ({status}) → {consumers}")

    if not report.ok:
        console.print("[red]Oracle verdict: FAIL[/red]")
        return 1
    console.print("[green]Oracle verdict: PASS[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

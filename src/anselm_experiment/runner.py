"""Main experiment runner.

Usage:
    python -m anselm_experiment.runner --brief briefs/credit_hold_release.yaml --arch iter --runs 3

The runner loads the brief, the schema, instantiates an LLM, and dispatches to
the chosen architecture. Each run gets its own folder under runs/<timestamp>/
containing the full transcript, all individual call logs, and the final result.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .architectures.iter_agent import run_iter
from .architectures.multi_pipe import run_multi_pipe
from .architectures.multi_vote import run_multi_vote
from .architectures.multi_vote_over_pipe import run_multi_vote_over_pipe
from .checker import check
from .llm import LLM
from .metrics import residual_ambiguity


ARCH_DISPATCH = {
    "iter": run_iter,
    "multi-pipe": run_multi_pipe,
    "multi-vote-flat": run_multi_vote,
    "multi-vote-over-pipe": run_multi_vote_over_pipe,
}


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="ANSELM Article 4 §10 experiment runner.")
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("schemas/process.schema.json"))
    parser.add_argument("--arch", choices=list(ARCH_DISPATCH), required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    brief = yaml.safe_load(args.brief.read_text(encoding="utf-8"))
    schema = json.loads(args.schema.read_text(encoding="utf-8"))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = args.out / f"{stamp}_{args.arch}"
    run_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    summary_path = run_root / "summary.json"
    sweep_t0 = time.time()
    print(f"=== {args.arch} × {args.runs} — {run_root.name} ===", flush=True)
    for r in range(args.runs):
        run_dir = run_root / f"run_{r:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_t0 = time.time()
        elapsed = time.time() - sweep_t0
        print(f"\n--- run {r+1}/{args.runs}  (sweep elapsed {elapsed:6.1f}s) ---", flush=True)
        llm = LLM(model=args.model, log_dir=run_dir)
        fn = ARCH_DISPATCH[args.arch]
        result = fn(brief=brief, schema=schema, llm=llm, log_dir=run_dir)

        process = result.get("process", {})
        final_violations = check(process, brief)
        ambiguity = residual_ambiguity(process)

        record = {
            "run": r,
            "arch": args.arch,
            "violations": len(final_violations),
            "ambiguity": ambiguity.count,
            "result": result,
        }
        (run_dir / "result.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        summary.append({k: v for k, v in record.items() if k != "result"})
        # Persist summary after every run so cancellation never wipes prior runs.
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        run_dt = time.time() - run_t0
        print(
            f"[{r+1}/{args.runs}] {args.arch}: violations={record['violations']} "
            f"ambiguity={record['ambiguity']}  ({run_dt:.1f}s)",
            flush=True,
        )

    total = time.time() - sweep_t0
    print(f"\nRun summary written to {summary_path}  (total {total:.1f}s)")


if __name__ == "__main__":
    main()

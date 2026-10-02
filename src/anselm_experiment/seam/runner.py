"""Runner for the cross-context coherence experiment.

Usage:
    python -m anselm_experiment.seam.runner \
        --brief briefs/atlas_cross_context.yaml \
        --arch prose-seam|typed-seam --runs 5 --roundtrips 1

The runner loads the brief, the tier01 vocabulary, the controlled term table,
and dispatches to the chosen architecture. Each run leaves a forensic trail
under runs/<timestamp>_<arch>/run_NN/: call logs (via the LLM wrapper), the
boundary artifacts, the final ecosystem, violations.json and metrics.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from ..llm import LLM
from .architectures import run_prose_seam, run_typed_seam
from .checker import check
from .metrics import commitment_coverage_prose, commitment_coverage_typed

ARCH_DISPATCH = {
    "prose-seam": run_prose_seam,
    "typed-seam": run_typed_seam,
}

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="ANSELM cross-context experiment runner.")
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--arch", choices=list(ARCH_DISPATCH), required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--roundtrips", type=int, default=1)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    args = parser.parse_args()

    brief = yaml.safe_load(args.brief.read_text(encoding="utf-8"))
    vocabulary_data = yaml.safe_load(
        (ROOT / "tier01" / "vocabulary.yaml").read_text(encoding="utf-8")
    )
    terms_constraint = next(
        (c for c in brief["constraints"] if c.get("kind") == "vocabulary"), {}
    )
    terms_path = ROOT / terms_constraint.get(
        "alias_table", "tier01/examples/atlas/terms.yaml"
    )
    terms = yaml.safe_load(terms_path.read_text(encoding="utf-8"))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_root = args.out / f"{stamp}_{args.arch}"
    run_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    summary_path = run_root / "summary.json"
    sweep_t0 = time.time()
    print(f"=== {args.arch} × {args.runs} (roundtrips={args.roundtrips}) — {run_root.name} ===", flush=True)
    for r in range(args.runs):
        run_dir = run_root / f"run_{r:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        run_t0 = time.time()
        print(f"\n--- run {r+1}/{args.runs}  (sweep {time.time()-sweep_t0:6.1f}s) ---", flush=True)
        llm = LLM(model=args.model, log_dir=run_dir)

        if args.arch == "typed-seam":
            result = run_typed_seam(
                brief=brief,
                vocabulary_data=vocabulary_data,
                terms=terms,
                llm=llm,
                roundtrips=args.roundtrips,
                log_dir=run_dir,
            )
        else:
            result = run_prose_seam(
                brief=brief,
                terms=terms,
                llm=llm,
                roundtrips=args.roundtrips,
                log_dir=run_dir,
            )

        ecosystem = result["ecosystem"]
        violations = check(ecosystem, brief)
        coverage = (
            commitment_coverage_typed(result.get("jane_fragment", {}), ecosystem)
            if args.arch == "typed-seam"
            else commitment_coverage_prose(
                (run_dir / "jane_pack.md").read_text(encoding="utf-8"), ecosystem
            )
        )
        record = {
            "run": r,
            "arch": args.arch,
            "roundtrips": args.roundtrips,
            "violations": len(violations),
            "commitment_coverage": round(coverage, 3),
            "rejection_rounds": result.get("rejection_rounds", {}),
            "premature_commitments": result.get("premature_commitments", 0),
            "tokens": _token_spend(run_dir),
        }
        (run_dir / "violations.json").write_text(
            json.dumps([v.__dict__ for v in violations], indent=2), encoding="utf-8"
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
        summary.append(record)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        run_dt = time.time() - run_t0
        print(
            f"[{r+1}/{args.runs}] {args.arch}: violations={record['violations']} "
            f"coverage={record['commitment_coverage']} "
            f"rejections={record['rejection_rounds']} tokens={record['tokens']}  ({run_dt:.1f}s)",
            flush=True,
        )

    total = time.time() - sweep_t0
    print(f"\nRun summary written to {summary_path}  (total {total:.1f}s)")


def _token_spend(run_dir: Path) -> int:
    total = 0
    for path in run_dir.glob("call_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            resp = payload.get("response", {})
            total += int(resp.get("prompt_tokens", 0) or 0) + int(
                resp.get("completion_tokens", 0) or 0
            )
        except (json.JSONDecodeError, OSError):
            continue
    return total


if __name__ == "__main__":
    main()

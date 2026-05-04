"""MULTI-VOTE-OVER-PIPE — the steel-man for multi-agent frameworks.

Run m independent MULTI-PIPE pipelines (each suffers its own fragmentation
losses), then an aggregator picks the best final process. This is the
architecture that real multi-agent frameworks (CrewAI, AutoGen, Du et al.'s
debate setup) most closely resemble: parallel teams, ensemble at the end.

This tests whether voting can RECOVER from per-pipeline information loss.
ITER's claim — that continuous context dominates — is genuinely at risk
against this architecture. If MVOP matches ITER, the article's prescription
has to weaken from "iterate" to "fragment-then-vote".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..checker import check
from ..llm import LLM, Message
from .multi_pipe import run_multi_pipe


def run_multi_vote_over_pipe(
    *,
    brief: dict[str, Any],
    schema: dict[str, Any],
    llm: LLM,
    m: int = 5,
    log_dir: Path,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    per_candidate_meta: list[dict[str, Any]] = []

    for k in range(m):
        sub_dir = log_dir / f"pipe_{k}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_llm = LLM(
            model=llm.model,
            log_dir=sub_dir,
            temperature=1.0,        # genuine sampling diversity across pipes
            seed=2000 + k,
        )
        try:
            pipe_result = run_multi_pipe(
                brief=brief, schema=schema, llm=sub_llm, log_dir=sub_dir,
            )
        except Exception as e:
            # One malformed candidate should not sink the sweep.
            (sub_dir / "pipe_error.txt").write_text(
                f"{type(e).__name__}: {e}", encoding="utf-8"
            )
            print(f"  ! pipe_{k} crashed: {type(e).__name__}: {e}", flush=True)
            pipe_result = {"process": {"steps": []}, "violations": 999, "info_loss": {}}
        candidates.append(pipe_result["process"])
        per_candidate_meta.append({
            "violations": pipe_result["violations"],
            "info_loss": pipe_result["info_loss"],
        })

    # Aggregator: pick by self-report, identical protocol to MULTI-VOTE-FLAT
    # so the only difference between the two is what the candidates went through.
    pick_msgs = [
        Message(
            role="system",
            content=(
                "You are an aggregator. Given several candidate process JSONs, "
                "return ONLY the index (0-based) of the best one."
            ),
        ),
        Message(role="user", content="CANDIDATES:\n" + json.dumps(candidates, indent=2)),
    ]
    pick = llm.call(pick_msgs, tag="mvop_aggregator")
    try:
        idx = int(pick.content.strip().split()[0])
    except (ValueError, IndexError):
        idx = 0
    idx = max(0, min(idx, len(candidates) - 1))
    chosen = candidates[idx]

    violations_per_candidate = []
    for c in candidates:
        try:
            violations_per_candidate.append(len(check(c, brief)))
        except Exception as e:
            print(f"  ! candidate check crashed: {type(e).__name__}: {e}", flush=True)
            violations_per_candidate.append(999)
    (log_dir / "mvop_summary.json").write_text(
        json.dumps(
            {
                "chosen_index": idx,
                "violations_per_candidate": violations_per_candidate,
                "per_candidate_meta": per_candidate_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "process": chosen,
        "violations": violations_per_candidate[idx],
        "violations_per_candidate": violations_per_candidate,
    }

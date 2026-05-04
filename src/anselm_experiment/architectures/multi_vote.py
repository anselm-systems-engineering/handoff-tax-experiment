"""MULTI-VOTE — m parallel modellers, an aggregator picks one by self-report.

Same base model, different sampling seeds. Tests the "contest of identical LLMs"
case from §6.3 of the article. Predicted to show high variance (sampling noise)
rather than diverse insight.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..checker import check
from ..llm import LLM, Message
from .prompts import ARCHITECT_SYSTEM


def run_multi_vote(
    *,
    brief: dict[str, Any],
    schema: dict[str, Any],
    llm: LLM,
    m: int = 5,
    log_dir: Path,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for k in range(m):
        msgs = [
            Message(role="system", content=ARCHITECT_SYSTEM),
            Message(
                role="user",
                content=(
                    "BRIEF:\n"
                    + json.dumps(brief, indent=2)
                    + "\n\nSCHEMA:\n"
                    + json.dumps(schema, indent=2)
                ),
            ),
        ]
        # Diversity: high temperature + per-candidate seed so the m candidates
        # are not near-identical clones.
        out = llm.call(msgs, tag=f"vote_candidate_{k}", temperature=1.0, seed=1000 + k)
        candidates.append(_safe_parse_json(out.content))

    # Aggregator: pick by self-report (the model picks "the best" given the candidates).
    pick_msgs = [
        Message(
            role="system",
            content="You are an aggregator. Given several candidate process JSONs, return ONLY the index (0-based) of the best one.",
        ),
        Message(
            role="user",
            content="CANDIDATES:\n" + json.dumps(candidates, indent=2),
        ),
    ]
    pick = llm.call(pick_msgs, tag="vote_aggregator")
    try:
        idx = int(pick.content.strip().split()[0])
    except (ValueError, IndexError):
        idx = 0
    idx = max(0, min(idx, len(candidates) - 1))
    chosen = candidates[idx]

    violations_per_candidate = [len(check(c, brief)) for c in candidates]
    (log_dir / "vote_summary.json").write_text(
        json.dumps(
            {
                "chosen_index": idx,
                "violations_per_candidate": violations_per_candidate,
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


def _safe_parse_json(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {}

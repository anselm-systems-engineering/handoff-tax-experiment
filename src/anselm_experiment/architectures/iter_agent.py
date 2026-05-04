"""ITER — single agent, continuous context.

Loop: draft -> check -> analyse violations -> revise.
The full transcript is preserved across all iterations: every prior assistant
turn and every prior tool/checker result is appended to messages and visible to
every subsequent call. This is the regime that *accumulates* in §4 of the article.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..checker import check
from ..llm import LLM, Message
from .prompts import ARCHITECT_SYSTEM


SYSTEM_PROMPT = ARCHITECT_SYSTEM + (
    "\n\nThis is an iterative loop: after each draft, a constraint checker "
    "will return the list of violations. Revise the JSON to fix every one."
)


def run_iter(
    *,
    brief: dict[str, Any],
    schema: dict[str, Any],
    llm: LLM,
    max_iterations: int = 5,
    log_dir: Path,
) -> dict[str, Any]:
    """Run the ITER architecture. Returns a result dict with:
    - process: the final process JSON (may still have violations if max_iterations hit)
    - violations_per_iter: list[int]
    - iterations: int
    """
    transcript: list[Message] = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(
            role="user",
            content=_initial_prompt(brief, schema),
        ),
    ]

    violations_per_iter: list[int] = []
    process: dict[str, Any] = {}

    for i in range(max_iterations):
        response = llm.call(transcript, tag=f"iter_draft_{i}")
        transcript.append(Message(role="assistant", content=response.content))

        process = _safe_parse_json(response.content)
        violations = check(process, brief)
        violations_per_iter.append(len(violations))

        if not violations:
            break

        transcript.append(
            Message(
                role="user",
                content=_violations_feedback(violations),
            )
        )

    (log_dir / "transcript.json").write_text(
        json.dumps([m.__dict__ for m in transcript], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "process": process,
        "violations_per_iter": violations_per_iter,
        "iterations": len(violations_per_iter),
    }


def _initial_prompt(brief: dict[str, Any], schema: dict[str, Any]) -> str:
    return (
        "BRIEF:\n"
        + json.dumps(brief, indent=2)
        + "\n\nPROCESS JSON SCHEMA:\n"
        + json.dumps(schema, indent=2)
        + "\n\nProduce the first draft of the process as a JSON object."
    )


def _violations_feedback(violations) -> str:
    return (
        "The constraint checker reported the following violations. "
        "Revise the JSON to fix every one. Output the full revised JSON.\n\n"
        + json.dumps([v.__dict__ for v in violations], indent=2)
    )


def _safe_parse_json(text: str) -> dict[str, Any]:
    """Parse JSON, tolerating fenced code blocks the model sometimes emits."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {}

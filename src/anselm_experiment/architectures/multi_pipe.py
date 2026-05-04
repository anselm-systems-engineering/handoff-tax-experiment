"""MULTI-PIPE — Discovery -> Modeller -> Compliance reviewer -> Implementation planner.

Each stage is an isolated LLM context. The downstream stage sees ONLY a summary
of the prior stage's output, never the prior raw transcript. This is the
deliberate lossy hand-off the article's §4 argument is about.

The summarisation step at each interface is itself an LLM call so that the
information-loss metric (BERTScore between raw and summary) is well-defined.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..checker import check
from ..llm import LLM, Message
from ..metrics import information_loss_bertscore
from .prompts import ARCHITECT_SYSTEM


def run_multi_pipe(
    *,
    brief: dict[str, Any],
    schema: dict[str, Any],
    llm: LLM,
    log_dir: Path,
) -> dict[str, Any]:
    discovery_raw = _stage(
        llm,
        system="You are a discovery analyst. Produce a thorough, structured discovery memo for the process below.",
        user="BRIEF:\n" + json.dumps(brief, indent=2),
        tag="discovery",
    )
    discovery_summary = _summarise(llm, discovery_raw, tag="discovery_summary")

    modeller_raw = _stage(
        llm,
        system=ARCHITECT_SYSTEM,
        user=(
            "DISCOVERY SUMMARY:\n"
            + discovery_summary
            + "\n\nSCHEMA:\n"
            + json.dumps(schema, indent=2)
        ),
        tag="modeller",
    )
    process = _safe_parse_json(modeller_raw)

    violations = check(process, brief)
    violations_summary = json.dumps([v.__dict__ for v in violations], indent=2)
    violations_compressed = _summarise(llm, violations_summary, tag="violations_summary")

    reviewer_raw = _stage(
        llm,
        system="You are a compliance reviewer. Issue concise recommendations.",
        user="MODEL:\n" + json.dumps(process, indent=2) + "\n\nVIOLATIONS:\n" + violations_compressed,
        tag="reviewer",
    )

    planner_raw = _stage(
        llm,
        system=ARCHITECT_SYSTEM + "\n\nProduce the final approved process JSON only.",
        user="APPROVED MODEL:\n" + json.dumps(process, indent=2) + "\n\nRECOMMENDATIONS:\n" + reviewer_raw,
        tag="planner",
    )
    final_process = _safe_parse_json(planner_raw)
    final_violations = check(final_process, brief)

    info_loss = {
        "discovery_to_modeller": information_loss_bertscore(discovery_raw, discovery_summary),
        "violations_to_reviewer": information_loss_bertscore(violations_summary, violations_compressed),
    }

    (log_dir / "info_loss.json").write_text(json.dumps(info_loss, indent=2), encoding="utf-8")
    return {
        "process": final_process,
        "violations": len(final_violations),
        "info_loss": info_loss,
    }


def _stage(llm: LLM, *, system: str, user: str, tag: str) -> str:
    msgs = [Message(role="system", content=system), Message(role="user", content=user)]
    return llm.call(msgs, tag=tag).content


def _summarise(llm: LLM, raw: str, *, tag: str) -> str:
    msgs = [
        Message(role="system", content="Summarise the following in 6-10 concise bullet points."),
        Message(role="user", content=raw),
    ]
    return llm.call(msgs, tag=tag).content


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

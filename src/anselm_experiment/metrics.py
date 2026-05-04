"""Metrics for the §10 experiment.

- Constraint-violation count (delegated to checker.py).
- Residual ambiguity: count of underspecified hand-offs/decisions in the process.
- Information loss at each interface (BERTScore between raw and summary).
- Total tokens (delegated to whatever the architecture records on the run log).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AmbiguityReport:
    underspecified_step_ids: list[str]

    @property
    def count(self) -> int:
        return len(self.underspecified_step_ids)


def residual_ambiguity(process: dict[str, Any]) -> AmbiguityReport:
    """A decision is underspecified if it lacks role, branches, or branch conditions.
    A non-decision step is underspecified if it lacks a role or a `next` link
    (unless it is a terminal step).
    """
    underspecified: list[str] = []
    step_ids = {s["id"] for s in process.get("steps", [])}
    for step in process.get("steps", []):
        if step.get("kind") == "decision":
            branches = step.get("branches") or []
            if not branches or any("condition" not in b or "next" not in b for b in branches):
                underspecified.append(step["id"])
            continue
        if not step.get("role"):
            underspecified.append(step["id"])
            continue
        # terminals are allowed to omit `next` only if explicitly marked
        if "next" not in step and "branches" not in step and not step.get("terminal"):
            underspecified.append(step["id"])
    return AmbiguityReport(underspecified_step_ids=underspecified)


def information_loss_bertscore(raw: str, summary: str) -> float:
    """Return 1 - F1 BERTScore as a loss (0.0 = lossless, 1.0 = total loss).

    Lazy-imports bert_score so the module loads without torch/transformers
    initialisation when this metric isn't needed (e.g. during ITER runs).
    First call downloads the embedding model (~400 MB).
    """
    if not raw or not summary:
        return 1.0
    from bert_score import score as _bert_score

    _, _, f1 = _bert_score([summary], [raw], lang="en", verbose=False)
    return float(1.0 - f1.item())

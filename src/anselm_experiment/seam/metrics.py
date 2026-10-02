"""Metrics for the cross-context experiment.

- Violations: delegated to seam/checker.py (the six-family oracle).
- Commitment coverage: fraction of the producer's explicit commitments present
  in the final deliverable. TYPED: Jane's cell ids surviving into the final
  ecosystem (deterministic). PROSE: Jane's COMMITMENT headlines recovered from
  the integrator's commitments by token Jaccard ≥ 0.5 (fuzzy but a
  deterministic function of the artifacts; biased against the hypothesis).
- Premature commitments: count of `no-fitting-type` markers (TYPED cost signal).
- Token spend: summed from the per-run call logs by the runner.
"""
from __future__ import annotations

import re
from typing import Any


def commitment_coverage_typed(jane_fragment: dict[str, Any], ecosystem: dict[str, Any]) -> float:
    jane_cells = set((jane_fragment or {}).get("cells", {}))
    if not jane_cells:
        return 0.0
    final_cells = set(ecosystem.get("cells", {}))
    return len(jane_cells & final_cells) / len(jane_cells)


def commitment_coverage_prose(jane_pack: str, ecosystem: dict[str, Any]) -> float:
    headlines = _extract_commitment_headlines(jane_pack)
    if not headlines:
        return 0.0
    integrator_claims = [
        (c.get("claim") or "") + " " + (c.get("term") or "")
        for c in ecosystem.get("commitments") or []
    ]
    matched = sum(1 for h in headlines if any(_jaccard(h, claim) >= 0.5 for claim in integrator_claims))
    return matched / len(headlines)


def _extract_commitment_headlines(pack: str) -> list[str]:
    headlines: list[str] = []
    for line in pack.splitlines():
        match = re.match(r"^\s*COMMITMENT\s+\d+\s*:\s*(.*)", line)
        if match:
            headlines.append(match.group(1).split("|", 1)[0].strip())
    return headlines


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _tokens(text: str) -> set[str]:
    # prefix-normalised (first 5 chars) so light inflection (enters/enter,
    # sensors/sensor) does not break matching; fuzzy but deterministic.
    return {t[:5] for t in re.findall(r"[a-z0-9][a-z0-9-]*", text.lower()) if len(t) > 2}

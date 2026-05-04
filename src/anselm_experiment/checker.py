"""Constraint checker — the oracle.

Takes a structured process (validated against schemas/process.schema.json) and
a brief (briefs/*.yaml) and returns a list of concrete violations.

This module is the load-bearing part of the harness. Its rules ARE the ground
truth for ITER's iterative loop and the only objective metric in the final
comparison. Treat it as version-controlled: any rule change invalidates prior runs.

The implementation below is a stub that demonstrates the shape. Each kind of
constraint declared in the brief maps to one check function. Add new kinds by
adding entries to CHECKERS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Violation:
    constraint_id: str
    severity: str  # "error" | "warning"
    where: str     # step id or "global"
    message: str


def check(process: dict[str, Any], brief: dict[str, Any]) -> list[Violation]:
    violations: list[Violation] = []
    for constraint in brief.get("constraints", []):
        kind = constraint["kind"]
        checker = CHECKERS.get(kind)
        if checker is None:
            violations.append(
                Violation(
                    constraint_id=constraint["id"],
                    severity="error",
                    where="global",
                    message=f"Unknown constraint kind: {kind}",
                )
            )
            continue
        violations.extend(checker(process, constraint))
    return violations


# ----- individual check functions ---------------------------------------------

def _check_segregation_of_duties(process, constraint) -> list[Violation]:
    """No single role may perform two activities tagged as an incompatible pair.

    `incompatible_tag_pairs` is a list of [tag_a, tag_b]. For each such pair,
    we check that the set of roles assigned to steps tagged `tag_a` is
    disjoint from the set of roles assigned to steps tagged `tag_b`.
    Steps may carry multiple tags; absent tags simply means the step is not
    in the relevant cohort.
    """
    out: list[Violation] = []
    pairs = [tuple(p) for p in constraint.get("incompatible_tag_pairs", [])]
    steps = process.get("steps", [])

    def roles_with_tag(tag: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for step in steps:
            if tag in (step.get("tags") or []):
                result.setdefault(step.get("role", ""), []).append(step["id"])
        return result

    for tag_a, tag_b in pairs:
        roles_a = roles_with_tag(tag_a)
        roles_b = roles_with_tag(tag_b)
        if not roles_a:
            out.append(Violation(constraint["id"], "error", "global",
                                 f"No step tagged '{tag_a}'."))
        if not roles_b:
            out.append(Violation(constraint["id"], "error", "global",
                                 f"No step tagged '{tag_b}'."))
        overlap = set(roles_a) & set(roles_b)
        for role in overlap:
            if not role:
                continue
            where = ",".join(sorted(set(roles_a[role] + roles_b[role])))
            out.append(Violation(
                constraint["id"], "error", where,
                f"Role '{role}' performs both '{tag_a}' and '{tag_b}'.",
            ))
    return out


def _check_sla(process, constraint) -> list[Violation]:
    """The step(s) carrying `target_tag` must declare sla_hours <= bound."""
    bound = constraint.get("bound_lte")
    target_tag = constraint.get("target_tag")
    if bound is None or target_tag is None:
        return []
    out: list[Violation] = []
    matching = [s for s in process.get("steps", []) if target_tag in (s.get("tags") or [])]
    if not matching:
        return [Violation(constraint["id"], "error", "global",
                          f"No step tagged '{target_tag}'; SLA cannot be verified.")]
    for step in matching:
        sla = step.get("sla_hours")
        if sla is None:
            out.append(Violation(constraint["id"], "error", step["id"],
                                 f"Step tagged '{target_tag}' missing sla_hours (bound {bound}h)."))
        elif sla > bound:
            out.append(Violation(constraint["id"], "error", step["id"],
                                 f"sla_hours={sla} exceeds bound {bound}."))
    return out


def _check_auditability(process, constraint) -> list[Violation]:
    """Every step carrying any of `target_tags` must produce all required_fields."""
    required = set(constraint.get("required_fields", []))
    target_tags = set(constraint.get("target_tags", []))
    out: list[Violation] = []
    matched_any = False
    for step in process.get("steps", []):
        step_tags = set(step.get("tags") or [])
        if not (step_tags & target_tags):
            continue
        matched_any = True
        produced = set(step.get("produces", []))
        missing = required - produced
        if missing:
            out.append(Violation(
                constraint["id"], "error", step["id"],
                f"Step missing required audit fields: {sorted(missing)}.",
            ))
    if not matched_any:
        out.append(Violation(
            constraint["id"], "error", "global",
            f"No step carries any of the target tags {sorted(target_tags)}.",
        ))
    return out


def _check_retention(process, constraint) -> list[Violation]:
    max_months = constraint.get("max_retention_months")
    if max_months is None:
        return []
    declared = process.get("retention", {}).get("after_close_months")
    if declared is None:
        return [Violation(constraint["id"], "error", "global", "No retention period declared.")]
    if declared > max_months:
        return [
            Violation(
                constraint_id=constraint["id"],
                severity="error",
                where="global",
                message=f"Retention {declared}m exceeds max {max_months}m.",
            )
        ]
    return []


def _check_system_reuse(process, constraint) -> list[Violation]:
    allowed = set(constraint.get("allowed_systems_of_record", []))
    declared = set(process.get("systems", []))
    extra = declared - allowed
    if extra:
        return [
            Violation(
                constraint_id=constraint["id"],
                severity="error",
                where="global",
                message=f"Disallowed systems of record introduced: {sorted(extra)}.",
            )
        ]
    return []


def _check_flow_completeness(process, constraint) -> list[Violation]:
    """Every step has an outgoing edge or is terminal; graph is connected and
    reaches a terminal from a `creates_order` start step.
    """
    out: list[Violation] = []
    steps = process.get("steps", [])
    step_ids = {s["id"] for s in steps}
    by_id = {s["id"]: s for s in steps}

    # 1. Every step has exactly one of next / branches / terminal=true.
    for s in steps:
        has_next = bool(s.get("next"))
        has_branches = bool(s.get("branches"))
        is_terminal = bool(s.get("terminal"))
        choices = sum([has_next, has_branches, is_terminal])
        if choices == 0:
            out.append(Violation(
                constraint["id"], "error", s["id"],
                "Step has no outgoing flow (need `next`, `branches`, or `terminal: true`).",
            ))
        elif choices > 1:
            out.append(Violation(
                constraint["id"], "error", s["id"],
                "Step declares more than one of `next`/`branches`/`terminal`; pick exactly one.",
            ))

    # 2. All referenced step IDs exist.
    for s in steps:
        for ref in s.get("next") or []:
            if not isinstance(ref, str):
                out.append(Violation(
                    constraint["id"], "error", s["id"],
                    f"`next` entry must be a string step id, got {type(ref).__name__}: {ref!r}.",
                ))
                continue
            if ref not in step_ids:
                out.append(Violation(
                    constraint["id"], "error", s["id"],
                    f"`next` references unknown step '{ref}'.",
                ))
        for br in s.get("branches") or []:
            if not isinstance(br, dict):
                out.append(Violation(
                    constraint["id"], "error", s["id"],
                    f"branch entry must be an object with `condition` and `next`, got {type(br).__name__}: {br!r}.",
                ))
                continue
            ref = br.get("next")
            if ref and ref not in step_ids:
                out.append(Violation(
                    constraint["id"], "error", s["id"],
                    f"branch references unknown step '{ref}'.",
                ))

    # 3. Reachability from a `creates_order`-tagged start step to a terminal.
    starts = [s["id"] for s in steps if "creates_order" in (s.get("tags") or [])]
    if not starts:
        out.append(Violation(
            constraint["id"], "error", "global",
            "No start step tagged `creates_order`.",
        ))
        return out

    visited: set[str] = set()
    stack = list(starts)
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        s = by_id.get(cur)
        if s is None:
            continue
        for ref in s.get("next") or []:
            if isinstance(ref, str) and ref in step_ids and ref not in visited:
                stack.append(ref)
        for br in s.get("branches") or []:
            if not isinstance(br, dict):
                continue
            ref = br.get("next")
            if ref and ref in step_ids and ref not in visited:
                stack.append(ref)

    unreachable = step_ids - visited
    if unreachable:
        out.append(Violation(
            constraint["id"], "error", "global",
            f"Steps unreachable from `creates_order` start: {sorted(unreachable)}.",
        ))

    if not any(by_id.get(sid, {}).get("terminal") for sid in visited):
        out.append(Violation(
            constraint["id"], "error", "global",
            "No terminal step reachable from `creates_order` start.",
        ))

    return out


CHECKERS: dict[str, Callable[[dict, dict], list[Violation]]] = {
    "segregation_of_duties": _check_segregation_of_duties,
    "sla": _check_sla,
    "auditability": _check_auditability,
    "retention": _check_retention,
    "system_reuse": _check_system_reuse,
    "flow_completeness": _check_flow_completeness,
}

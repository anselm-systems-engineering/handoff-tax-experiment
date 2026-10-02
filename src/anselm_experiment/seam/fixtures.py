"""Calibration fixtures for the cross-context checker.

KNOWN_GOOD and KNOWN_BAD ecosystems are oracle-calibration data: the checker
must pass the first and flag every family in the second. They are imported by
both the dry-run script and the unit tests, so there is one source of truth.
"""
from __future__ import annotations

from typing import Any

_DATE = "2026-10-02"


def _cell(cell_id: str, title: str, ctype: str, tags: list[str] | None = None) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "id": cell_id,
        "title": title,
        "type": ctype,
        "status": "active",
        "author": "tester",
        "date": _DATE,
    }
    if tags:
        cell["tags"] = tags
    return cell


def _relation(rid: str, rtype: str, source: str, target: str, note: str = "") -> dict[str, Any]:
    edge: dict[str, Any] = {"id": rid, "type": rtype, "source": source, "target": target}
    if note:
        edge["note"] = note
    return edge


def known_good() -> dict[str, Any]:
    """An integrated Atlas ecosystem satisfying all six families."""
    cells = {
        "jane-need-001": _cell("jane-need-001", "Operate on loose sand terrain", "need", ["mobility", "terrain"]),
        "jane-constraint-001": _cell("jane-constraint-001", "Vehicle mass below 350 kg", "constraint", ["weight"]),
        "jane-fn-001": _cell("jane-fn-001", "Propulsion — hybrid wheel-leg", "function", ["propulsion"]),
        "jane-dec-001": _cell("jane-dec-001", "Select hybrid wheel-leg concept", "decision", ["trade-study"]),
        "ahmed-constraint-002": _cell(
            "ahmed-constraint-002", "Failsafe limp-home on sensor failure", "constraint", ["safety-critical"]
        ),
        "ahmed-comp-001": _cell("ahmed-comp-001", "Battery pack X", "component", ["energy-storage"]),
        "ahmed-comp-002": _cell(
            "ahmed-comp-002", "Motor controller COTS-X", "component", ["sensor-critical", "propulsion"]
        ),
        "ahmed-if-001": _cell("ahmed-if-001", "Motor controller ↔ battery power bus", "interface", ["power"]),
        "ahmed-dec-002": _cell("ahmed-dec-002", "Select motor controller COTS-X", "decision", ["procurement"]),
    }
    relations = [
        _relation("r-001", "satisfies", "jane-fn-001", "jane-need-001"),
        _relation("r-002", "decides", "jane-dec-001", "jane-fn-001"),
        _relation("r-003", "derives_from", "jane-dec-001", "jane-need-001"),
        _relation("r-004", "derives_from", "jane-dec-001", "jane-constraint-001"),
        _relation("r-005", "constrains", "jane-constraint-001", "jane-fn-001"),
        _relation("r-006", "refines", "ahmed-if-001", "jane-fn-001"),
        _relation("r-007", "constrains", "ahmed-if-001", "ahmed-comp-001"),
        _relation("r-008", "decides", "ahmed-dec-002", "ahmed-comp-002"),
        _relation("r-009", "derives_from", "ahmed-dec-002", "jane-constraint-001"),
        _relation("r-010", "constrains", "ahmed-constraint-002", "jane-fn-001"),
    ]
    attributes = {
        "ahmed-comp-001": {"mass_kg": 18.2},
        "ahmed-comp-002": {
            "mass_kg": 2.1,
            "sensors": [
                {
                    "name": "phase-current sensor",
                    "critical": True,
                    "coverage": "failsafe_trigger",
                    "covered_by": "ahmed-constraint-002",
                }
            ],
        },
        "ahmed-if-001": {"voltage_v": 48, "peak_current_a": 60, "thermal_duty_w": 40},
    }
    assumptions = [
        {"id": "asm-001", "author": "jane", "text": "Sand capability scales with ground pressure.", "response_by": "ahmed"},
        {"id": "asm-002", "author": "ahmed", "text": "Thermal duty reflects peak, not nominal.", "response_by": "jane", "contested": False},
    ]
    commitments = [
        {
            "id": "cmt-001",
            "from": "jane",
            "to": "ahmed",
            "term": "limp-home mode",
            "claim": "Controller must enter limp-home on single-point sensor failure.",
            "agreed": True,
        },
        {
            "id": "cmt-002",
            "from": "ahmed",
            "to": "jane",
            "term": "thermal duty",
            "claim": "Interface thermal duty is declared under peak operation.",
            "agreed": True,
        },
        {
            "id": "cmt-003",
            "from": "ahmed",
            "to": "jane",
            "term": "peak current",
            "claim": "Interface peak current is 60 A.",
            "agreed": True,
        },
    ]
    return {
        "id": "atlas_cross_context",
        "title": "Integrated Atlas propulsion ↔ energy design",
        "cells": cells,
        "relations": relations,
        "attributes": attributes,
        "assumptions": assumptions,
        "commitments": commitments,
    }


def known_bad() -> dict[str, Any]:
    """Violates every one of the six families."""
    cells = {
        "jane-need-001": _cell("jane-need-001", "Operate on loose sand terrain", "need", ["mobility"]),
        "jane-fn-001": _cell("jane-fn-001", "Propulsion — hybrid wheel-leg", "function", ["propulsion"]),
        "ahmed-comp-001": _cell("ahmed-comp-001", "Battery pack X", "component", ["energy-storage"]),
        "ahmed-comp-002": _cell("ahmed-comp-002", "Motor controller COTS-X", "component", ["sensor-critical"]),
        "ahmed-if-001": _cell("ahmed-if-001", "Power bus", "interface", ["power"]),
    }
    relations = [
        # structure: dangling target
        _relation("r-001", "satisfies", "jane-fn-001", "ghost-cell"),
        # structure: open conflict in the final deliverable
        _relation("r-002", "conflicts", "ahmed-comp-002", "jane-fn-001"),
        # structure: unknown relation type
        _relation("r-003", "freeform", "ahmed-comp-001", "jane-fn-001"),
    ]
    attributes = {
        # mass: component without mass_kg (both) -> family flags; keep one to also test sum? keep simple
        "ahmed-if-001": {"voltage_v": 48},  # interface: missing peak_current_a, thermal_duty_w
    }
    assumptions = [
        {"id": "asm-001", "author": "jane", "text": "Assumption nobody acknowledged."},
    ]
    commitments = [
        {
            "id": "cmt-001",
            "from": "jane",
            "to": "ahmed",
            "term": "warp drive",
            "claim": "Uncontrolled term used at the seam.",
            "agreed": True,
        }
    ]
    return {
        "id": "atlas_cross_context_bad",
        "title": "Deliberately broken deliverable",
        "cells": cells,
        "relations": relations,
        "attributes": attributes,
        "assumptions": assumptions,
        "commitments": commitments,
    }

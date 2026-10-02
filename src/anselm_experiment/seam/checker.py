"""Extended checker — the oracle for the cross-context deliverable.

Six deterministic constraint families from briefs/atlas_cross_context.yaml:
mass, interface, safety, vocabulary, assumptions, structure.

The `structure` family reuses the tier01 oracle in-memory (schema-valid cells,
closed relation vocabulary, domain/range compliance, no dangling references).
An open `conflicts` edge in the FINAL deliverable is a violation here: the
task demands an integrated design, and carrying an unresolved contradiction
across the finish line is exactly the coherence failure this experiment
measures.

This module is version-controlled ground truth, like checker.py in the
hand-off-tax experiment: any rule change invalidates prior runs.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from ..checker import Violation
from ..tier01.oracle import (
    semantic_flags,
    validate_cells,
    validate_relations,
)
from ..tier01.vocabulary import Vocabulary

MASS_BOUND_KG = 350.0


def check(ecosystem: dict[str, Any], brief: dict[str, Any]) -> list[Violation]:
    """Run all six families. Returns the flat list of violations.

    The oracle is the measurement instrument: it must never crash on a
    malformed deliverable. A structural screen (schema validation + dropping
    malformed entries, each drop reported as a violation) runs first, so the
    six families always see sanitized data.
    """
    violations: list[Violation] = []
    structure_id = next(
        (c["id"] for c in brief.get("constraints", []) if c.get("kind") == "structure"),
        "structure",
    )
    violations.extend(_schema_screen(ecosystem, structure_id))
    ecosystem = _sanitize(ecosystem, violations, structure_id)

    for constraint in brief.get("constraints", []):
        kind = constraint["kind"]
        fn = CHECKERS.get(kind)
        if fn is None:
            violations.append(
                Violation(
                    constraint_id=constraint["id"],
                    severity="error",
                    where="global",
                    message=f"Unknown constraint kind: {kind}",
                )
            )
            continue
        violations.extend(fn(ecosystem, constraint))
    return violations


def schema_violations(ecosystem: Any, structure_id: str = "structure") -> list[Violation]:
    """Shape-level screen over the ecosystem schema. Used by the seam gate and
    the final checker. Domain families are NOT part of this screen — the gate
    stays structural, and domain semantics are measured at the finish line."""
    return _schema_screen(ecosystem, structure_id)


def _schema_screen(ecosystem: Any, structure_id: str) -> list[Violation]:
    if not isinstance(ecosystem, dict):
        return [
            Violation(structure_id, "error", "global", "deliverable is not a JSON object.")
        ]
    try:
        import jsonschema
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource

        registry = Registry().with_resource(
            "https://anselm.ing/schemas/cell.schema.json",
            Resource.from_contents(_cell_schema_local()),
        )
        Draft202012Validator(_ecosystem_schema(), registry=registry).validate(ecosystem)
    except jsonschema.ValidationError as e:
        where = "/".join(str(p) for p in e.path) or "<root>"
        return [
            Violation(
                structure_id,
                "error",
                where,
                f"deliverable fails the ecosystem schema: {e.message}",
            )
        ]
    except jsonschema.SchemaError as e:  # pragma: no cover — our schema, not runtime data
        return [Violation(structure_id, "error", "global", f"oracle schema error: {e.message}")]
    return []


def _cell_schema_local() -> dict[str, Any]:
    import json

    schema_path = Path(__file__).resolve().parents[3] / "schemas" / "cell.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _sanitize(
    ecosystem: Any, violations: list[Violation], structure_id: str
) -> dict[str, Any]:
    if not isinstance(ecosystem, dict):
        return {"cells": {}, "relations": [], "attributes": {}, "assumptions": [], "commitments": []}
    out: dict[str, Any] = {}
    for key, value in ecosystem.items():
        if key == "cells":
            if not isinstance(value, dict):
                violations.append(
                    Violation(structure_id, "error", "cells", "'cells' must be a mapping; treated as empty.")
                )
                out[key] = {}
                continue
            kept = {k: v for k, v in value.items() if isinstance(v, dict)}
            for k in value:
                if not isinstance(value[k], dict):
                    violations.append(
                        Violation(structure_id, "error", str(k), "malformed cell entry.")
                    )
            out[key] = kept
        elif key in ("relations", "assumptions", "commitments"):
            if not isinstance(value, list):
                violations.append(
                    Violation(structure_id, "error", key, f"'{key}' must be a list; treated as empty.")
                )
                out[key] = []
                continue
            kept = [v for v in value if isinstance(v, dict)]
            for v in value:
                if not isinstance(v, dict):
                    violations.append(
                        Violation(structure_id, "error", key, "malformed list entry dropped.")
                    )
            out[key] = kept
        elif key == "attributes":
            if not isinstance(value, dict):
                violations.append(
                    Violation(structure_id, "error", "attributes", "'attributes' must be a mapping; treated as empty.")
                )
                out[key] = {}
                continue
            kept = {k: v for k, v in value.items() if isinstance(v, dict)}
            for k in value:
                if not isinstance(value[k], dict):
                    violations.append(
                        Violation(structure_id, "error", str(k), "malformed attribute entry.")
                    )
            out[key] = kept
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------- structure (tier01)

def _check_structure(ecosystem: dict[str, Any], constraint: dict[str, Any]) -> list[Violation]:
    vocabulary = _load_vocabulary(constraint)
    out: list[Violation] = []
    cells = ecosystem.get("cells", {})
    relations = ecosystem.get("relations", [])

    for finding in validate_cells(cells, vocabulary):
        out.append(Violation(constraint["id"], "error", finding.where, finding.message))
    for finding in validate_relations(relations, vocabulary, cells):
        out.append(Violation(constraint["id"], "error", finding.where, finding.message))
    for edge in relations:
        if edge.get("type") == "conflicts":
            out.append(
                Violation(
                    constraint["id"],
                    "error",
                    str(edge.get("id")),
                    f"unresolved conflict {edge.get('source')} ↔ {edge.get('target')} "
                    "in the final deliverable.",
                )
            )
    if not cells:
        out.append(Violation(constraint["id"], "error", "global", "no cells in the deliverable."))
    return out


def _load_vocabulary(constraint: dict[str, Any]) -> Vocabulary:
    from ..tier01.vocabulary import load_vocabulary

    path = constraint.get("vocabulary") or (
        Path(__file__).resolve().parents[3] / "tier01" / "vocabulary.yaml"
    )
    return load_vocabulary(path)


# ---------------------------------------------------------------- mass

def _check_mass(ecosystem: dict[str, Any], constraint: dict[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    cells = ecosystem.get("cells", {})
    attributes = ecosystem.get("attributes", {})
    components = [cid for cid, c in cells.items() if c.get("type") == "component"]
    if not components:
        out.append(Violation(constraint["id"], "error", "global", "no component cells declared."))
        return out

    total = 0.0
    for cid in components:
        mass = (attributes.get(cid) or {}).get("mass_kg")
        if not isinstance(mass, (int, float)) or mass <= 0:
            out.append(
                Violation(
                    constraint["id"],
                    "error",
                    cid,
                    f"component '{cid}' must declare a positive mass_kg in attributes.",
                )
            )
            continue
        total += float(mass)
    if total > constraint.get("bound_lte", MASS_BOUND_KG):
        out.append(
            Violation(
                constraint["id"],
                "error",
                "global",
                f"total mass {total:.1f} kg exceeds {constraint.get('bound_lte', MASS_BOUND_KG)} kg.",
            )
        )
    return out


# ---------------------------------------------------------------- interface

def _check_interface(ecosystem: dict[str, Any], constraint: dict[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    cells = ecosystem.get("cells", {})
    attributes = ecosystem.get("attributes", {})
    interfaces = [cid for cid, c in cells.items() if c.get("type") == "interface"]
    if not interfaces:
        out.append(Violation(constraint["id"], "error", "global", "no interface cells declared."))
        return out

    for cid in interfaces:
        attrs = attributes.get(cid) or {}
        for field in ("voltage_v", "peak_current_a", "thermal_duty_w"):
            value = attrs.get(field)
            if not isinstance(value, (int, float)) or value <= 0:
                out.append(
                    Violation(
                        constraint["id"],
                        "error",
                        cid,
                        f"interface '{cid}' must declare a positive {field}.",
                    )
                )
    return out


# ---------------------------------------------------------------- safety

def _check_safety(ecosystem: dict[str, Any], constraint: dict[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    cells = ecosystem.get("cells", {})
    attributes = ecosystem.get("attributes", {})

    critical_cells = [cid for cid, c in cells.items() if "sensor-critical" in (c.get("tags") or [])]
    if not critical_cells:
        out.append(
            Violation(
                constraint["id"],
                "error",
                "global",
                "no cell tagged `sensor-critical`; the brief's sensor edge cases make "
                "critical sensors unavoidable, so a design without them is incomplete.",
            )
        )
        return out

    for cid in critical_cells:
        sensors = (attributes.get(cid) or {}).get("sensors") or []
        if not sensors:
            out.append(
                Violation(
                    constraint["id"],
                    "error",
                    cid,
                    f"cell '{cid}' is tagged sensor-critical but declares no sensors in attributes.",
                )
            )
            continue
        for sensor in sensors:
            if not sensor.get("critical"):
                continue
            coverage = sensor.get("coverage")
            if coverage not in ("redundant_channel", "failsafe_trigger"):
                out.append(
                    Violation(
                        constraint["id"],
                        "error",
                        cid,
                        f"sensor '{sensor.get('name')}' must declare coverage "
                        "(redundant_channel or failsafe_trigger).",
                    )
                )
            covered_by = sensor.get("covered_by")
            if not covered_by or covered_by not in cells:
                out.append(
                    Violation(
                        constraint["id"],
                        "error",
                        cid,
                        f"sensor '{sensor.get('name')}' must name a covered_by cell "
                        "that exists in the deliverable.",
                    )
                )
    return out


# ---------------------------------------------------------------- vocabulary (terms)

def _check_vocabulary(ecosystem: dict[str, Any], constraint: dict[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    commitments = ecosystem.get("commitments") or []
    if not commitments:
        out.append(
            Violation(
                constraint["id"],
                "error",
                "global",
                "no seam commitments recorded; the seam must declare what crosses it.",
            )
        )
        return out

    canonical = _load_terms(constraint)
    for commitment in commitments:
        term = (commitment.get("term") or "").strip().lower()
        if not term or term not in canonical:
            out.append(
                Violation(
                    constraint["id"],
                    "error",
                    str(commitment.get("id")),
                    f"commitment term '{commitment.get('term')}' is not in the controlled term table.",
                )
            )
    return out


def _load_terms(constraint: dict[str, Any]) -> set[str]:
    """Controlled terms + aliases, lowercased."""
    path = constraint.get("alias_table")
    if not path:
        return set()
    terms_path = Path(__file__).resolve().parents[3] / path
    data = yaml.safe_load(terms_path.read_text(encoding="utf-8"))
    known: set[str] = set()
    for entry in data.get("terms", []):
        known.add(entry["term"].strip().lower())
        for alias in entry.get("aliases", []):
            known.add(alias.strip().lower())
    return known


# ---------------------------------------------------------------- assumptions

def _check_assumptions(ecosystem: dict[str, Any], constraint: dict[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    for assumption in ecosystem.get("assumptions") or []:
        if not assumption.get("response_by"):
            out.append(
                Violation(
                    constraint["id"],
                    "error",
                    str(assumption.get("id")),
                    f"assumption '{assumption.get('text', '')[:40]}…' was never acknowledged "
                    "or contested by the other side.",
                )
            )
    return out


CHECKERS: dict[str, Callable[[dict, dict], list[Violation]]] = {
    "mass": _check_mass,
    "interface": _check_interface,
    "safety": _check_safety,
    "vocabulary": _check_vocabulary,
    "assumptions": _check_assumptions,
    "structure": _check_structure,
}


def decision_basis_warnings(ecosystem: dict[str, Any]) -> list[str]:
    """Non-violating tier01 warnings — reported as a metric, not a gate."""
    warnings = semantic_flags(ecosystem.get("cells", {}), ecosystem.get("relations", []))
    return [f"{w.where}: {w.message}" for w in warnings]


_ECOSYSTEM_SCHEMA: dict[str, Any] | None = None


def _ecosystem_schema() -> dict[str, Any]:
    import json

    global _ECOSYSTEM_SCHEMA
    if _ECOSYSTEM_SCHEMA is None:
        schema_path = Path(__file__).resolve().parents[3] / "schemas" / "ecosystem.schema.json"
        _ECOSYSTEM_SCHEMA = json.loads(schema_path.read_text(encoding="utf-8"))
    return _ECOSYSTEM_SCHEMA

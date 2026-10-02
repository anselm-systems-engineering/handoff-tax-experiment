"""Shared prompt fragments for the cross-context experiment.

Kept identical across conditions except for the seam mechanism itself: the
role framing, the brief, and the deliverable instructions are the same for
both PROSE-SEAM and TYPED-SEAM; only the boundary-artifact format differs.
"""
from __future__ import annotations

from typing import Any

JANE_SYSTEM = (
    "You are Jane, lead of the PROPULSION subsystem for Project Atlas. You hold "
    "deep context on locomotion, terrain capability, and the hybrid wheel-leg "
    "concept. You never see the other engineer's conversation — you only exchange "
    "a boundary artifact with the energy & thermal lead (Ahmed)."
)

AHMED_SYSTEM = (
    "You are Ahmed, lead of the ENERGY & THERMAL subsystem for Project Atlas. You "
    "hold deep context on the battery pack, the motor controller, thermal "
    "management, and power distribution. You never see the other engineer's "
    "conversation — you only exchange a boundary artifact with the propulsion "
    "lead (Jane)."
)

INTEGRATOR_SYSTEM = (
    "You are the integration engineer. You receive the engineers' prose packs and "
    "translate them into ONE typed ecosystem JSON that conforms to the provided "
    "schema. Preserve every commitment and every declared number. Do not invent "
    "facts that neither pack contains. Output ONLY the JSON object."
)

TYPED_FORMAT = """A typed knowledge ecosystem is a JSON object with top-level `id` and
`title` (choose concise values, e.g. "atlas_cross_context"), plus:

- `cells`: map of cell id -> frontmatter object with fields: id, title,
  type (one of {cell_types}), status (one of {statuses}), author, date
  (YYYY-MM-DD), tags (list of strings). Cell ids MUST match
  `^[a-z][a-z0-9-]*$` (lowercase kebab-case, no underscores).
- `relations`: list of {{id, type, source, target, note}} where type is one of
  {relation_types}. Domain/range rules:
    satisfies: leaves function|component, points at need
    refines:   leaves function|interface|component, points at function|interface
    constrains: leaves constraint|interface|component, points at function|component|interface
    conflicts: any -> any (route open contradictions; the final design must resolve them)
    derives_from: leaves decision, points at need|constraint|function|component|interface
    decides:   leaves decision, points at function|component|interface

CRITICAL: typed relations are commitment-level statements only — what
satisfies what, what refines what, what constrains what, what a decision
decides and what it is grounded in. PHYSICAL CONNECTIVITY (a controller
powering a motor, a sensor monitoring a bus, a cable plugging into a port) is
NOT a typed relation: describe it in the cell prose and in `attributes` with
numbers. Do not write `refines` or `satisfies` edges for wiring.

- `attributes`: map of cell id -> {{}} with:
    mass_kg on component cells;
    voltage_v, peak_current_a, thermal_duty_w on interface cells;
    sensors: list of {{name, critical, coverage, covered_by}} on cells that
    carry the EXACT tag `sensor-critical` (a cell tagged `sensor` + `critical`
    separately does not count); coverage is `redundant_channel` or
    `failsafe_trigger`, covered_by names another cell that provides the
    coverage.
- `assumptions`: list of {{id, author, text, response_by}} — every assumption
  one side records must be acknowledged or contested by the other (response_by).
- `commitments`: list of {{id, from, to, term, claim, agreed}} — commitments
  crossing the seam. `term` MUST be one of the controlled terms listed below.
  If a commitment does not fit any controlled term, write it as a note with the
  exact marker `NOTE no-fitting-type: <term>` instead of inventing a term.

Controlled terms (use the term or one of its aliases in `commitments[].term`):
{terms}

If a fact does not fit the vocabulary, do not distort it: put
`NOTE no-fitting-type: <explanation>` in the cell body or relation note.

A complete valid ecosystem looks like this (cell prose omitted for brevity):
{example}
"""


EXAMPLE_FRAGMENT = """{
  "id": "atlas-example",
  "title": "Example integrated ecosystem",
  "cells": {
    "need-001": {"id": "need-001", "title": "Operate on loose sand", "type": "need",
      "status": "active", "author": "jane", "date": "2026-10-02", "tags": ["terrain"]},
    "fn-001": {"id": "fn-001", "title": "Propulsion — hybrid wheel-leg", "type": "function",
      "status": "active", "author": "jane", "date": "2026-10-02", "tags": ["propulsion"]},
    "if-001": {"id": "if-001", "title": "Motor controller ↔ battery power bus", "type": "interface",
      "status": "active", "author": "ahmed", "date": "2026-10-02", "tags": ["power"]},
    "comp-001": {"id": "comp-001", "title": "Motor controller", "type": "component",
      "status": "active", "author": "ahmed", "date": "2026-10-02", "tags": ["sensor-critical"]},
    "dec-001": {"id": "dec-001", "title": "Select hybrid wheel-leg concept", "type": "decision",
      "status": "active", "author": "jane", "date": "2026-10-02"}
  },
  "relations": [
    {"id": "r1", "type": "satisfies", "source": "fn-001", "target": "need-001"},
    {"id": "r2", "type": "decides", "source": "dec-001", "target": "fn-001"},
    {"id": "r3", "type": "derives_from", "source": "dec-001", "target": "need-001"}
  ],
  "attributes": {
    "comp-001": {
      "mass_kg": 2.1,
      "sensors": [
        {"name": "telemetry sensor", "critical": true,
         "coverage": "failsafe_trigger", "covered_by": "dec-001"}
      ]
    },
    "if-001": {"voltage_v": 48, "peak_current_a": 60, "thermal_duty_w": 40}
  },
  "assumptions": [
    {"id": "asm-001", "author": "jane", "text": "Sand capability scales with ground pressure.",
     "response_by": "ahmed"}
  ],
  "commitments": [
    {"id": "cmt-001", "from": "jane", "to": "ahmed", "term": "limp-home mode",
     "claim": "Controller must enter limp-home on single-point sensor failure.",
     "agreed": true}
  ]
}"""


def typed_format_prompt(vocabulary: dict[str, Any], terms: dict[str, Any]) -> str:
    cell_types = ", ".join(vocabulary["cell_types"])
    statuses = ", ".join(vocabulary["statuses"])
    relation_types = ", ".join(vocabulary["relations"])
    term_lines = []
    for entry in terms.get("terms", []):
        aliases = ", ".join(entry.get("aliases", []))
        term_lines.append(f"- {entry['term']}" + (f" (aliases: {aliases})" if aliases else ""))
    return TYPED_FORMAT.format(
        cell_types=cell_types,
        statuses=statuses,
        relation_types=relation_types,
        terms="\n".join(term_lines),
        example=EXAMPLE_FRAGMENT,
    )


PROSE_FORMAT = """A prose pack is a Markdown document with sections:

# Design
Your subsystem design in prose.

# Constraints honored
The constraints you claim to satisfy, with the concrete numbers.

# Assumptions
A numbered list. Every assumption you record must be acknowledged or contested
by the other side.

# Commitments
One line per commitment crossing the seam, in exactly this form:
COMMITMENT <n>: <headline> | term: <controlled term>

The controlled terms are: {terms}
If a commitment does not fit any controlled term, write:
COMMITMENT <n>: <headline> | term: no-fitting-type:<your term>
"""


def prose_format_prompt(terms: dict[str, Any]) -> str:
    term_list = ", ".join(e["term"] for e in terms.get("terms", []))
    return PROSE_FORMAT.format(terms=term_list)


def initial_user_prompt(brief: dict[str, Any], format_instructions: str) -> str:
    return (
        "BRIEF:\n"
        + _brief_as_json(brief)
        + "\n\nDELIVERABLE FORMAT:\n"
        + format_instructions
    )


def _brief_as_json(brief: dict[str, Any]) -> str:
    import json

    return json.dumps(brief, indent=2)

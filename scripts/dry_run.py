"""Dry-run plumbing check — no API calls.

Validates:
  1. .env loads OPENAI_API_KEY (without printing it).
  2. Brief parses and schema loads.
  3. A hand-crafted KNOWN-BAD process triggers the expected violations.
  4. A hand-crafted KNOWN-GOOD process produces zero violations.
  5. residual_ambiguity behaves on both.

Run with:
    .\.venv\Scripts\python.exe scripts\dry_run.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from jsonschema import Draft202012Validator
from rich.console import Console
from rich.table import Table

from anselm_experiment.checker import check
from anselm_experiment.metrics import residual_ambiguity

ROOT = Path(__file__).resolve().parents[1]
console = Console()


def step(name: str) -> None:
    console.rule(f"[bold cyan]{name}[/]")


def ok(msg: str) -> None:
    console.print(f"[green]PASS[/] {msg}")


def fail(msg: str) -> None:
    console.print(f"[red]FAIL[/] {msg}")


# ---- 1. env -----------------------------------------------------------------
step("1. Environment")
load_dotenv(ROOT / ".env")
key = os.environ.get("OPENAI_API_KEY", "")
if key.startswith("sk-") and len(key) > 30:
    ok(f"OPENAI_API_KEY loaded (length={len(key)}, prefix={key[:7]}…)")
else:
    fail("OPENAI_API_KEY missing or malformed in .env")
    sys.exit(1)
model = os.environ.get("ANSELM_DEFAULT_MODEL", "")
ok(f"ANSELM_DEFAULT_MODEL = {model!r}")

# ---- 2. brief + schema -------------------------------------------------------
step("2. Brief + schema")
brief = yaml.safe_load((ROOT / "briefs" / "credit_hold_release.yaml").read_text(encoding="utf-8"))
ok(f"Brief parsed: {brief['id']} ({len(brief['constraints'])} constraints, "
   f"{len(brief['edge_cases'])} edge cases)")

schema = json.loads((ROOT / "schemas" / "process.schema.json").read_text(encoding="utf-8"))
validator = Draft202012Validator(schema)
ok(f"Schema loaded: {schema['title']}")

# ---- 3. KNOWN-BAD process — should violate every constraint ------------------
step("3. KNOWN-BAD process — expect violations")
bad = {
    "id": "credit_hold_release",
    "title": "Bad version",
    "roles": ["order_entry_clerk"],
    "systems": ["custom_new_db"],          # violates sap_reuse
    "retention": {"after_close_months": 60},  # violates gdpr_retention
    "steps": [
        {
            "id": "order_entry",
            "kind": "activity",
            "role": "order_entry_clerk",
            "tags": ["creates_order"],
            "next": ["credit_hold_release"],
        },
        {
            "id": "credit_hold_release",
            "kind": "decision",
            "role": "order_entry_clerk",   # same role -> SoD violation
            "tags": ["releases_hold"],
            "produces": ["decision"],       # missing role/timestamp/rationale/credit_data_ref
            "branches": [
                {"condition": "approve", "next": "done"},
                {"condition": "reject", "next": "done"},
            ],
        },
        {"id": "done", "kind": "activity", "role": "order_entry_clerk"},  # no next/terminal -> flow violation
    ],
    # No reviews_hold tag + no sla_hours -> sla violation
}
errors = list(validator.iter_errors(bad))
if errors:
    fail(f"BAD process failed schema validation: {[e.message for e in errors]}")
    sys.exit(1)
ok("BAD process passes schema (structurally valid)")

violations = check(bad, brief)
table = Table(title=f"Violations on BAD process ({len(violations)})")
table.add_column("constraint")
table.add_column("where")
table.add_column("message")
for v in violations:
    table.add_row(v.constraint_id, v.where, v.message)
console.print(table)

expected_constraints = {"sod_hold_release", "sla_review", "audit_trail", "gdpr_retention", "sap_reuse", "flow_completeness"}
got = {v.constraint_id for v in violations}
missing = expected_constraints - got
if missing:
    fail(f"Expected violations not reported: {missing}")
    sys.exit(1)
ok(f"All {len(expected_constraints)} expected constraints flagged")

amb_bad = residual_ambiguity(bad)
ok(f"Residual ambiguity on BAD: {amb_bad.count} underspecified step(s) {amb_bad.underspecified_step_ids}")

# ---- 4. KNOWN-GOOD process — should pass cleanly -----------------------------
step("4. KNOWN-GOOD process — expect 0 violations")
good = {
    "id": "credit_hold_release",
    "title": "Good version",
    "roles": ["order_entry_clerk", "credit_analyst", "finance_manager"],
    "systems": ["sap_s4_case_management"],
    "retention": {"after_close_months": 24},
    "steps": [
        {
            "id": "order_entry",
            "kind": "activity",
            "role": "order_entry_clerk",
            "system": "sap_s4_case_management",
            "tags": ["creates_order"],
            "produces": ["case_record", "amount"],
            "next": ["raise_hold"],
        },
        {
            "id": "raise_hold",
            "kind": "system",
            "role": "credit_analyst",
            "system": "sap_s4_case_management",
            "tags": ["raises_hold"],
            "sla_hours": 1,
            "next": ["review"],
        },
        {
            "id": "review",
            "kind": "activity",
            "role": "credit_analyst",
            "system": "sap_s4_case_management",
            "tags": ["reviews_hold"],
            "sla_hours": 4,
            "produces": ["credit_data_ref"],
            "next": ["credit_hold_release"],
        },
        {
            "id": "credit_hold_release",
            "kind": "decision",
            "role": "credit_analyst",
            "system": "sap_s4_case_management",
            "produces": ["role", "timestamp", "rationale", "credit_data_ref", "decision"],
            "branches": [
                {"condition": "release", "next": "release_step"},
                {"condition": "reject", "next": "reject_step"},
                {"condition": "escalate", "next": "escalate_step"},
            ],
        },
        {
            "id": "release_step",
            "kind": "activity",
            "role": "credit_analyst",
            "system": "sap_s4_case_management",
            "tags": ["releases_hold"],
            "produces": ["role", "timestamp", "rationale", "credit_data_ref"],
            "terminal": True,
        },
        {
            "id": "reject_step",
            "kind": "activity",
            "role": "credit_analyst",
            "system": "sap_s4_case_management",
            "tags": ["rejects_hold"],
            "produces": ["role", "timestamp", "rationale", "credit_data_ref"],
            "terminal": True,
        },
        {
            "id": "escalate_step",
            "kind": "activity",
            "role": "finance_manager",
            "system": "sap_s4_case_management",
            "tags": ["escalates"],
            "terminal": True,
        },
    ],
}
errors = list(validator.iter_errors(good))
if errors:
    fail(f"GOOD process failed schema validation: {[e.message for e in errors]}")
    sys.exit(1)
ok("GOOD process passes schema")

violations = check(good, brief)
if violations:
    table = Table(title=f"Unexpected violations on GOOD process ({len(violations)})")
    table.add_column("constraint")
    table.add_column("where")
    table.add_column("message")
    for v in violations:
        table.add_row(v.constraint_id, v.where, v.message)
    console.print(table)
    fail("GOOD process should produce zero violations")
    sys.exit(1)
ok("GOOD process produces 0 violations")

amb_good = residual_ambiguity(good)
if amb_good.count != 0:
    fail(f"Residual ambiguity on GOOD: {amb_good.underspecified_step_ids} (expected 0)")
    sys.exit(1)
ok("Residual ambiguity on GOOD: 0")

# ---- summary ----------------------------------------------------------------
console.rule("[bold green]All plumbing checks passed[/]")
console.print("Ready for smoke test against the live API.")

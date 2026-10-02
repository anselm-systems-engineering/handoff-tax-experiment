"""Dry-run plumbing check for the cross-context experiment — no API calls.

Validates:
  1. .env loads OPENAI_API_KEY (without printing it).
  2. Brief, tier01 vocabulary and the controlled term table load.
  3. KNOWN-BAD ecosystem triggers every one of the six families.
  4. KNOWN-GOOD ecosystem produces zero violations.
  5. PROSE-SEAM and TYPED-SEAM architectures run end-to-end on a scripted
     FakeLLM, including the oracle gate rejecting a bad draft and accepting
     the revision.

Run with:
    .\\.venv\\Scripts\\python.exe scripts\\seam_dry_run.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from anselm_experiment.llm import CallResult
from anselm_experiment.seam.architectures import run_prose_seam, run_typed_seam
from anselm_experiment.seam.checker import check
from anselm_experiment.seam.fixtures import known_bad, known_good
from anselm_experiment.seam.metrics import commitment_coverage_prose, commitment_coverage_typed

ROOT = Path(__file__).resolve().parents[1]
console = Console()


class FakeLLM:
    """Scripted stand-in for the LLM wrapper; no network, deterministic."""

    def __init__(self, script: dict[str, str]) -> None:
        self.script = script
        self.tags: list[str] = []

    def call(self, messages, *, tag: str = "untagged", temperature=None, seed=None) -> CallResult:
        self.tags.append(tag)
        content = self.script.get(tag, "{}")
        return CallResult(
            content=content, model="fake", prompt_tokens=100, completion_tokens=50
        )


def step(name: str) -> None:
    console.rule(f"[bold cyan]{name}[/]")


def ok(msg: str) -> None:
    console.print(f"[green]PASS[/] {msg}")


def fail(msg: str) -> None:
    console.print(f"[red]FAIL[/] {msg}")
    sys.exit(1)


# ---- 1. environment ----------------------------------------------------------
step("1. Environment")
load_dotenv(ROOT / ".env")
key = os.environ.get("OPENAI_API_KEY", "")
if key.startswith("sk-") and len(key) > 30:
    ok(f"OPENAI_API_KEY loaded (length={len(key)}, prefix={key[:7]}…)")
else:
    fail("OPENAI_API_KEY missing or malformed in .env")
model = os.environ.get("ANSELM_DEFAULT_MODEL", "")
ok(f"ANSELM_DEFAULT_MODEL = {model!r}")

# ---- 2. brief + vocabulary + terms ------------------------------------------
step("2. Brief + vocabulary + terms")
brief = yaml.safe_load((ROOT / "briefs" / "atlas_cross_context.yaml").read_text(encoding="utf-8"))
ok(f"Brief parsed: {brief['id']} ({len(brief['constraints'])} constraints, "
   f"{len(brief['edge_cases'])} edge cases)")
vocabulary_data = yaml.safe_load((ROOT / "tier01" / "vocabulary.yaml").read_text(encoding="utf-8"))
ok(f"Vocabulary loaded: {len(vocabulary_data['relations'])} relation types")
terms_constraint = next(c for c in brief["constraints"] if c.get("kind") == "vocabulary")
terms = yaml.safe_load((ROOT / terms_constraint["alias_table"]).read_text(encoding="utf-8"))
ok(f"Terms loaded: {len(terms['terms'])} controlled terms")

# ---- 3. KNOWN-BAD ------------------------------------------------------------
step("3. KNOWN-BAD ecosystem — expect every family flagged")
violations = check(known_bad(), brief)
table = Table(title=f"Violations on KNOWN-BAD ({len(violations)})")
table.add_column("family")
table.add_column("where")
table.add_column("message")
for v in violations:
    table.add_row(v.constraint_id, v.where, v.message)
console.print(table)
expected = {c["id"] for c in brief["constraints"]}
got = {v.constraint_id for v in violations}
missing = expected - got
if missing:
    fail(f"Families not flagged: {missing}")
ok(f"All {len(expected)} families flagged")

# ---- 4. KNOWN-GOOD -----------------------------------------------------------
step("4. KNOWN-GOOD ecosystem — expect 0 violations")
violations = check(known_good(), brief)
if violations:
    for v in violations:
        console.print(f"[red]•[/] {v.constraint_id} @ {v.where}: {v.message}")
    fail("KNOWN-GOOD produced violations")
ok("KNOWN-GOOD produces 0 violations")

# ---- 5. PROSE-SEAM with FakeLLM ---------------------------------------------
step("5. PROSE-SEAM end-to-end (FakeLLM)")
jane_pack = (
    "# Design\nHybrid wheel-leg propulsion, mass 74 kg.\n\n"
    "# Constraints honored\n- 350 kg ceiling\n\n"
    "# Assumptions\n1. Sand capability scales with ground pressure.\n\n"
    "# Commitments\n"
    "COMMITMENT 1: Controller enters limp-home on single sensor failure | term: limp-home mode\n"
    "COMMITMENT 2: Interface thermal duty under peak operation | term: thermal duty\n"
)
final_pack = jane_pack.replace("# Design", "# Design\nMotor controller COTS-X, 48 V bus, peak 60 A.")
good_ecosystem = json.dumps(known_good())
fake = FakeLLM(
    {
        "jane_r0": jane_pack,
        "ahmed_r0": final_pack,
        "integrator": good_ecosystem,
    }
)
result = run_prose_seam(brief=brief, terms=terms, llm=fake, roundtrips=1, log_dir=ROOT / "runs" / "_dry_prose")
violations = check(result["ecosystem"], brief)
coverage = commitment_coverage_prose(jane_pack, result["ecosystem"])
ok(f"prose-seam ran; violations={len(violations)}, coverage={coverage:.2f}, "
   f"premature={result['premature_commitments']}")
if len(violations) != 0:
    fail("prose-seam should integrate the scripted good ecosystem cleanly")

# ---- 6. TYPED-SEAM with FakeLLM (gate rejects once, then passes) -------------
step("6. TYPED-SEAM end-to-end with a gate rejection (FakeLLM)")
bad_fragment = json.dumps(
    {
        "cells": {
            "jane-fn-001": {
                "id": "jane-fn-001",
                "title": "Propulsion",
                "type": "function",
                "status": "active",
                "author": "jane",
                "date": "2026-10-02",
            }
        },
        "relations": [{"id": "r-x", "type": "satisfies", "source": "jane-fn-001", "target": "ghost"}],
    }
)
good_fragment = json.dumps(
    {
        "cells": {
            "jane-need-001": {
                "id": "jane-need-001",
                "title": "Operate on loose sand terrain",
                "type": "need",
                "status": "active",
                "author": "jane",
                "date": "2026-10-02",
                "tags": ["mobility"],
            },
            "jane-fn-001": {
                "id": "jane-fn-001",
                "title": "Propulsion",
                "type": "function",
                "status": "active",
                "author": "jane",
                "date": "2026-10-02",
            },
        },
        "relations": [{"id": "r-x", "type": "satisfies", "source": "jane-fn-001", "target": "jane-need-001"}],
    }
)
fake = FakeLLM(
    {
        "jane_draft_0": bad_fragment,
        "jane_draft_1": good_fragment,
        "ahmed_r0_draft_0": good_ecosystem,
    }
)
result = run_typed_seam(
    brief=brief,
    vocabulary_data=vocabulary_data,
    terms=terms,
    llm=fake,
    roundtrips=1,
    log_dir=ROOT / "runs" / "_dry_typed",
)
violations = check(result["ecosystem"], brief)
coverage = commitment_coverage_typed(result["jane_fragment"], result["ecosystem"])
rejections = result["rejection_rounds"]
ok(f"typed-seam ran; violations={len(violations)}, coverage={coverage:.2f}, "
   f"rejection_rounds={rejections}, premature={result['premature_commitments']}")
if rejections.get("jane") != 2:
    fail(f"Expected jane gate to reject once then pass (2 rounds), got {rejections}")
if len(violations) != 0:
    fail("typed-seam should end with a clean integrated ecosystem")

# ---- summary ----------------------------------------------------------------
console.rule("[bold green]All cross-context plumbing checks passed[/]")
console.print("Ready for a live smoke run: "
              "python -m anselm_experiment.seam.runner --brief briefs/atlas_cross_context.yaml "
              "--arch typed-seam --runs 1")

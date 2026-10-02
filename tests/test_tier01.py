"""Tests for the Tier 0–1 prototype: cells, relations, oracle, view."""
from __future__ import annotations

from pathlib import Path

import pytest

from anselm_experiment.tier01.graph import render_mermaid
from anselm_experiment.tier01.oracle import check_ecosystem
from anselm_experiment.tier01.vocabulary import VocabularyError, load_vocabulary, parse_vocabulary

VOCAB = Path(__file__).resolve().parents[1] / "tier01" / "vocabulary.yaml"
ATLAS = Path(__file__).resolve().parents[1] / "tier01" / "examples" / "atlas"

CELL = """---
id: {cell_id}
title: Test cell
type: {ctype}
status: active
author: tester
date: 2026-10-01
---

Body of the cell.
"""

REL = """relations:
{edges}
"""


def _write_cell(tmp: Path, cell_id: str, ctype: str) -> None:
    (tmp / f"{cell_id}.md").write_text(
        CELL.format(cell_id=cell_id, ctype=ctype), encoding="utf-8"
    )


def _write_relations(tmp: Path, edges: str) -> None:
    (tmp / "relations.yaml").write_text(REL.format(edges=edges), encoding="utf-8")


def _minimal_ecosystem(tmp: Path) -> Path:
    _write_cell(tmp, "need-a", "need")
    _write_cell(tmp, "fn-a", "function")
    _write_relations(
        tmp,
        "  - id: r1\n    type: satisfies\n    source: fn-a\n    target: need-a\n",
    )
    return tmp


def test_atlas_example_passes() -> None:
    vocab = load_vocabulary(VOCAB)
    report = check_ecosystem(ATLAS, vocab)
    assert report.ok, [f"{f.where}: {f.message}" for f in report.errors]
    assert len(report.cells) == 9
    assert len(report.relations) == 10
    conflict = [w for w in report.warnings if "routed to the Knowledge Steward" in w.message]
    assert len(conflict) == 1 and conflict[0].where == "r-010"


def test_dangling_relation_is_an_error(tmp_path: Path) -> None:
    vocab = load_vocabulary(VOCAB)
    eco = _minimal_ecosystem(tmp_path)
    _write_relations(
        tmp_path,
        "  - id: r1\n    type: satisfies\n    source: fn-a\n    target: ghost-cell\n",
    )
    report = check_ecosystem(eco, vocab)
    assert not report.ok
    assert any("ghost-cell" in e.message for e in report.errors)


def test_unknown_relation_type_is_an_error(tmp_path: Path) -> None:
    vocab = load_vocabulary(VOCAB)
    eco = _minimal_ecosystem(tmp_path)
    _write_relations(
        tmp_path,
        "  - id: r1\n    type: freeform\n    source: fn-a\n    target: need-a\n",
    )
    report = check_ecosystem(eco, vocab)
    assert any("not in the vocabulary" in e.message for e in report.errors)


def test_domain_violation_is_an_error(tmp_path: Path) -> None:
    vocab = load_vocabulary(VOCAB)
    _write_cell(tmp_path, "need-a", "need")
    _write_cell(tmp_path, "need-b", "need")
    _write_relations(
        tmp_path,
        "  - id: r1\n    type: satisfies\n    source: need-a\n    target: need-b\n",
    )
    report = check_ecosystem(tmp_path, vocab)
    assert any("cannot leave a need" in e.message for e in report.errors)


def test_invalid_cell_type_is_an_error(tmp_path: Path) -> None:
    vocab = load_vocabulary(VOCAB)
    _write_cell(tmp_path, "weird-a", "wish")
    report = check_ecosystem(tmp_path, vocab)
    assert not report.ok
    assert any("wish" in e.message for e in report.errors)


def test_decision_without_basis_is_flagged(tmp_path: Path) -> None:
    vocab = load_vocabulary(VOCAB)
    _write_cell(tmp_path, "dec-a", "decision")
    (tmp_path / "relations.yaml").write_text("relations: []\n", encoding="utf-8")
    report = check_ecosystem(tmp_path, vocab)
    assert report.ok, [f"{e.where}: {e.message}" for e in report.errors]
    assert any("without a recorded basis" in w.message for w in report.warnings)


def test_vocabulary_without_consumers_is_rejected() -> None:
    data = {
        "schema_version": 1,
        "cell_types": ["need"],
        "statuses": ["active"],
        "relations": {"satisfies": {"domain": ["need"], "range": ["need"]}},
    }
    with pytest.raises(VocabularyError, match="consumers"):
        parse_vocabulary(data)


def test_graph_renders_conflict_in_red() -> None:
    vocab = load_vocabulary(VOCAB)
    report = check_ecosystem(ATLAS, vocab)
    mermaid = render_mermaid(report.cells, report.relations)
    assert "constraint-002 -->|conflicts| dec-002" in mermaid
    assert "linkStyle" in mermaid
    assert "classDef need" in mermaid

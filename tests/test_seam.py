"""Tests for the cross-context experiment: checker, architectures, metrics."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from anselm_experiment.llm import CallResult
from anselm_experiment.seam.architectures import run_prose_seam, run_typed_seam
from anselm_experiment.seam.checker import check
from anselm_experiment.seam.fixtures import known_bad, known_good
from anselm_experiment.seam.metrics import commitment_coverage_prose, commitment_coverage_typed

ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    def __init__(self, script: dict[str, str]) -> None:
        self.script = script
        self.tags: list[str] = []

    def call(self, messages, *, tag: str = "untagged", temperature=None, seed=None) -> CallResult:
        self.tags.append(tag)
        return CallResult(
            content=self.script.get(tag, "{}"),
            model="fake",
            prompt_tokens=100,
            completion_tokens=50,
        )


@pytest.fixture(scope="module")
def brief() -> dict:
    return yaml.safe_load(
        (ROOT / "briefs" / "atlas_cross_context.yaml").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def vocabulary_data() -> dict:
    return yaml.safe_load((ROOT / "tier01" / "vocabulary.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def terms() -> dict:
    return yaml.safe_load((ROOT / "tier01" / "examples" / "atlas" / "terms.yaml").read_text(encoding="utf-8"))


def test_known_bad_flags_every_family(brief: dict) -> None:
    violations = check(known_bad(), brief)
    families = {v.constraint_id for v in violations}
    assert families == {c["id"] for c in brief["constraints"]}


def test_known_good_passes(brief: dict) -> None:
    assert check(known_good(), brief) == []


def test_coverage_typed_is_deterministic() -> None:
    fragment = {"cells": {"a": {}, "b": {}}}
    ecosystem = {"cells": {"a": {}, "b": {}, "c": {}}}
    assert commitment_coverage_typed(fragment, ecosystem) == 1.0
    ecosystem = {"cells": {"a": {}}}
    assert commitment_coverage_typed(fragment, ecosystem) == 0.5


def test_coverage_prose_matches_headlines() -> None:
    pack = "COMMITMENT 1: Controller enters limp-home on sensor failure | term: limp-home mode\n"
    ecosystem = {
        "commitments": [
            {
                "id": "cmt-001",
                "from": "jane",
                "to": "ahmed",
                "term": "limp-home mode",
                "claim": "Controller must enter limp-home mode on single sensor failure.",
            }
        ]
    }
    assert commitment_coverage_prose(pack, ecosystem) == 1.0
    assert commitment_coverage_prose("COMMITMENT 1: totally different headline | term: x", ecosystem) == 0.0


def test_typed_seam_gate_recovers(tmp_path: Path, brief: dict, vocabulary_data: dict, terms: dict) -> None:
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
            "relations": [
                {"id": "r-x", "type": "satisfies", "source": "jane-fn-001", "target": "ghost"}
            ],
        }
    )
    fake = FakeLLM({"jane_draft_0": bad_fragment, "jane_draft_1": json.dumps(known_good())})
    result = run_typed_seam(
        brief=brief,
        vocabulary_data=vocabulary_data,
        terms=terms,
        llm=fake,
        roundtrips=1,
        log_dir=tmp_path,
    )
    assert result["rejection_rounds"]["jane"] == 2


def test_typed_seam_clean_run(tmp_path: Path, brief: dict, vocabulary_data: dict, terms: dict) -> None:
    good_fragment = json.dumps(
        {
            "cells": {
                k: v for k, v in known_good()["cells"].items() if k.startswith("jane-")
            },
            "relations": [
                e
                for e in known_good()["relations"]
                if e["source"].startswith("jane-") and e["target"].startswith("jane-")
            ],
        }
    )
    fake = FakeLLM(
        {
            "jane_draft_0": good_fragment,
            "ahmed_r0_draft_0": json.dumps(known_good()),
        }
    )
    result = run_typed_seam(
        brief=brief,
        vocabulary_data=vocabulary_data,
        terms=terms,
        llm=fake,
        roundtrips=1,
        log_dir=tmp_path,
    )
    assert result["rejection_rounds"] == {"jane": 1, "ahmed_r0": 1}
    assert check(result["ecosystem"], brief) == []
    assert commitment_coverage_typed(result["jane_fragment"], result["ecosystem"]) == 1.0


def test_prose_seam_clean_run(tmp_path: Path, brief: dict, terms: dict) -> None:
    jane_pack = (
        "# Design\nHybrid wheel-leg propulsion.\n\n"
        "# Commitments\n"
        "COMMITMENT 1: Controller enters limp-home on sensor failure | term: limp-home mode\n"
    )
    fake = FakeLLM(
        {
            "jane_r0": jane_pack,
            "ahmed_r0": jane_pack,
            "integrator": json.dumps(known_good()),
        }
    )
    result = run_prose_seam(brief=brief, terms=terms, llm=fake, roundtrips=1, log_dir=tmp_path)
    assert check(result["ecosystem"], brief) == []
    assert commitment_coverage_prose(jane_pack, result["ecosystem"]) == 1.0
    assert (tmp_path / "jane_pack.md").exists()
    assert (tmp_path / "ecosystem.json").exists()

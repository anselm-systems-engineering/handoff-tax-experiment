"""The deterministic coherence oracle.

Tier 0–1 checks over a knowledge ecosystem: frontmatter validity (Tier 0) and
typed-graph coherence (Tier 1). This module is the deterministic floor under
everything stochastic — an LLM may *propose* cells and edges, but this oracle
*judges* them, and its rules are the ground truth.

Errors are mechanical defects: invalid frontmatter, dangling references,
vocabulary violations. Warnings are surfaced commitments that need human
attention: contradictions (always routed to the Knowledge Steward), decisions
without a recorded basis, orphan cells.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from .vocabulary import Vocabulary


@dataclass
class Finding:
    severity: str  # "error" | "warning"
    where: str  # cell id, relation id, or "ecosystem"
    message: str


@dataclass
class Report:
    cells: dict[str, dict[str, Any]] = field(default_factory=dict)
    relations: list[dict[str, Any]] = field(default_factory=list)
    errors: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_ecosystem(cell_dir: str | Path, vocabulary: Vocabulary) -> Report:
    """Run the full oracle over a directory of cells and its relations.yaml."""
    cell_dir = Path(cell_dir)
    report = Report()

    cells = _load_cells(cell_dir, vocabulary, report)
    relations = _load_relations(cell_dir, vocabulary, cells, report)
    report.cells = cells
    report.relations = relations

    _semantic_flags(vocabulary, cells, relations, report)
    return report


# ---------------------------------------------------------------- Tier 0

def _load_cells(
    cell_dir: Path, vocabulary: Vocabulary, report: Report
) -> dict[str, dict[str, Any]]:
    schema = _load_cell_schema()
    cells: dict[str, dict[str, Any]] = {}
    for path in sorted(cell_dir.glob("*.md")):
        frontmatter, err = _parse_frontmatter(path)
        if err:
            report.errors.append(Finding("error", path.stem, err))
            continue
        try:
            jsonschema.validate(instance=frontmatter, schema=schema)
        except jsonschema.ValidationError as e:
            where = "/".join(str(p) for p in e.path) or "<frontmatter>"
            report.errors.append(
                Finding("error", path.stem, f"invalid frontmatter at {where}: {e.message}")
            )
            continue

        cell_id = frontmatter["id"]
        if frontmatter["type"] not in vocabulary.cell_types:
            report.errors.append(
                Finding("error", cell_id, f"cell type '{frontmatter['type']}' not in vocabulary.")
            )
            continue
        if frontmatter["status"] not in vocabulary.statuses:
            report.errors.append(
                Finding("error", cell_id, f"status '{frontmatter['status']}' not in vocabulary.")
            )
            continue
        if cell_id in cells:
            report.errors.append(Finding("error", cell_id, "duplicate cell id."))
            continue
        if cell_id != path.stem:
            report.warnings.append(
                Finding("warning", cell_id, f"cell id does not match file name '{path.stem}'.")
            )
        cells[cell_id] = frontmatter
    return cells


def _load_cell_schema() -> dict[str, Any]:
    schema_path = (
        Path(__file__).resolve().parents[3] / "schemas" / "cell.schema.json"
    )
    return yaml.safe_load(schema_path.read_text(encoding="utf-8"))


def _parse_frontmatter(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return None, "missing YAML frontmatter (must start with ---)."
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return None, "unterminated YAML frontmatter."
    try:
        raw = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as e:
        return None, f"frontmatter is not valid YAML: {e}"
    if raw is None:
        return None, "empty frontmatter."
    if not isinstance(raw, dict):
        return None, "frontmatter must be a YAML mapping."
    return _stringify(raw), None


def _stringify(obj: Any) -> Any:
    """Keep YAML 1.1 scalars (dates) as strings for schema validation."""
    if isinstance(obj, datetime.datetime):
        return obj.date().isoformat()
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _stringify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify(v) for v in obj]
    return obj


# ---------------------------------------------------------------- Tier 1

def _load_relations(
    cell_dir: Path,
    vocabulary: Vocabulary,
    cells: dict[str, dict[str, Any]],
    report: Report,
) -> list[dict[str, Any]]:
    path = cell_dir / "relations.yaml"
    if not path.exists():
        report.errors.append(Finding("error", "relations", f"missing {path.name}."))
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        report.errors.append(Finding("error", "relations", f"relations.yaml is invalid: {e}"))
        return []
    if not isinstance(data, dict) or not isinstance(data.get("relations"), list):
        report.errors.append(Finding("error", "relations", "relations.yaml must map to a list."))
        return []

    relations: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for edge in data["relations"]:
        if not isinstance(edge, dict):
            report.errors.append(Finding("error", "relations", "each relation must be a mapping."))
            continue
        rid = edge.get("id")
        if not isinstance(rid, str) or not rid:
            report.errors.append(Finding("error", "relations", "relation without an id."))
            continue
        if rid in seen_ids:
            report.errors.append(Finding("error", rid, "duplicate relation id."))
            continue
        seen_ids.add(rid)

        rtype_name = edge.get("type")
        rtype = vocabulary.relations.get(rtype_name) if isinstance(rtype_name, str) else None
        if rtype is None:
            report.errors.append(
                Finding("error", rid, f"relation type '{rtype_name}' not in the vocabulary.")
            )
            continue

        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, str) or source not in cells:
            report.errors.append(Finding("error", rid, f"source '{source}' is not a known cell."))
            continue
        if not isinstance(target, str) or target not in cells:
            report.errors.append(Finding("error", rid, f"target '{target}' is not a known cell."))
            continue

        source_type, target_type = cells[source]["type"], cells[target]["type"]
        if "*" not in rtype.domain and source_type not in rtype.domain:
            report.errors.append(
                Finding(
                    "error",
                    rid,
                    f"'{rtype_name}' cannot leave a {source_type} (allowed: {', '.join(rtype.domain)}).",
                )
            )
        if "*" not in rtype.targets and target_type not in rtype.targets:
            report.errors.append(
                Finding(
                    "error",
                    rid,
                    f"'{rtype_name}' cannot point at a {target_type} (allowed: {', '.join(rtype.targets)}).",
                )
            )
        relations.append(edge)

    return relations


# ---------------------------------------------------------------- flags

def _semantic_flags(
    vocabulary: Vocabulary,
    cells: dict[str, dict[str, Any]],
    relations: list[dict[str, Any]],
    report: Report,
) -> None:
    active_ids = {cid for cid, c in cells.items() if c["status"] == "active"}
    referenced = {e.get("source") for e in relations} | {e.get("target") for e in relations}
    for cid in sorted(active_ids - referenced):
        report.warnings.append(
            Finding("warning", cid, "orphan cell: no relation references it. Steward, please review.")
        )

    decision_ids = {cid for cid, c in cells.items() if c["type"] == "decision"}
    grounded = {e.get("source") for e in relations if e.get("type") == "derives_from"}
    for cid in sorted(decision_ids - grounded):
        report.warnings.append(
            Finding("warning", cid, "decision without a recorded basis (no derives_from edge).")
        )

    for edge in relations:
        if edge.get("type") == "conflicts":
            report.warnings.append(
                Finding(
                    "warning",
                    str(edge.get("id")),
                    f"contradiction routed to the Knowledge Steward: "
                    f"{edge.get('source')} ↔ {edge.get('target')}. "
                    f"Note: {edge.get('note', '(no note)')}",
                )
            )

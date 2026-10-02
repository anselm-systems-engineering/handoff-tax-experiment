"""The deterministic coherence oracle.

Tier 0–1 checks over a knowledge ecosystem: frontmatter validity (Tier 0) and
typed-graph coherence (Tier 1). This module is the deterministic floor under
everything stochastic — an LLM may *propose* cells and edges, but this oracle
*judges* them, and its rules are the ground truth.

Errors are mechanical defects: invalid frontmatter, dangling references,
vocabulary violations. Warnings are surfaced commitments that need human
attention: contradictions (always routed to the Knowledge Steward), decisions
without a recorded basis, orphan cells.

The module is split into parsing (file I/O) and validation (in-memory), so the
cross-context experiment can reuse the validation half directly at a seam.
"""
from __future__ import annotations

import datetime
import json
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

    cells, parse_errors, cell_files = parse_cells(cell_dir)
    relations, relation_errors = parse_relations(cell_dir)

    report = Report(cells=cells, relations=relations or [])
    report.errors.extend(parse_errors)
    report.errors.extend(relation_errors)

    report.errors.extend(validate_cells(cells, vocabulary))
    if relations is not None:
        report.errors.extend(validate_relations(relations, vocabulary, cells))
    report.warnings.extend(semantic_flags(cells, relations or []))

    for cell_id in cells:
        filename = cell_files.get(cell_id)
        if filename is not None and cell_id != filename:
            report.warnings.append(
                Finding("warning", cell_id, f"cell id does not match file name '{filename}'.")
            )
    return report


# ---------------------------------------------------------------- parsing

def parse_cells(cell_dir: Path) -> tuple[dict[str, dict[str, Any]], list[Finding], dict[str, str]]:
    """Parse all *.md cells in a directory. Returns (cells, errors, id→stem)."""
    schema = _load_cell_schema()
    cells: dict[str, dict[str, Any]] = {}
    cell_files: dict[str, str] = {}
    errors: list[Finding] = []
    for path in sorted(cell_dir.glob("*.md")):
        frontmatter, err = _parse_frontmatter(path)
        if err:
            errors.append(Finding("error", path.stem, err))
            continue
        try:
            jsonschema.validate(instance=frontmatter, schema=schema)
        except jsonschema.ValidationError as e:
            where = "/".join(str(p) for p in e.path) or "<frontmatter>"
            errors.append(
                Finding("error", path.stem, f"invalid frontmatter at {where}: {e.message}")
            )
            continue
        cell_id = frontmatter["id"]
        if cell_id in cells:
            errors.append(Finding("error", cell_id, "duplicate cell id."))
            continue
        cells[cell_id] = frontmatter
        cell_files[cell_id] = path.stem
    return cells, errors, cell_files


def parse_relations(cell_dir: Path) -> tuple[list[dict[str, Any]] | None, list[Finding]]:
    path = cell_dir / "relations.yaml"
    if not path.exists():
        return None, [Finding("error", "relations", f"missing {path.name}.")]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        return None, [Finding("error", "relations", f"relations.yaml is invalid: {e}")]
    if not isinstance(data, dict) or not isinstance(data.get("relations"), list):
        return None, [Finding("error", "relations", "relations.yaml must map to a list.")]
    return data["relations"], []


def _load_cell_schema() -> dict[str, Any]:
    schema_path = Path(__file__).resolve().parents[3] / "schemas" / "cell.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


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


# ---------------------------------------------------------------- validation (in-memory)

def validate_cells(cells: dict[str, dict[str, Any]], vocabulary: Vocabulary) -> list[Finding]:
    """Schema + vocabulary cell checks. Schema checks run again here so the
    in-memory path (the seam gate) enforces the same Tier 0 contract as the
    file-based path."""
    errors: list[Finding] = []
    for cell_id, frontmatter in cells.items():
        try:
            jsonschema.validate(instance=frontmatter, schema=_cell_schema())
        except jsonschema.ValidationError as e:
            where = "/".join(str(p) for p in e.path) or "<frontmatter>"
            errors.append(
                Finding("error", cell_id, f"invalid frontmatter at {where}: {e.message}")
            )
            continue
        if frontmatter["type"] not in vocabulary.cell_types:
            errors.append(
                Finding("error", cell_id, f"cell type '{frontmatter['type']}' not in vocabulary.")
            )
        if frontmatter["status"] not in vocabulary.statuses:
            errors.append(
                Finding("error", cell_id, f"status '{frontmatter['status']}' not in vocabulary.")
            )
    return errors


_CELL_SCHEMA: dict[str, Any] | None = None


def _cell_schema() -> dict[str, Any]:
    global _CELL_SCHEMA
    if _CELL_SCHEMA is None:
        schema_path = Path(__file__).resolve().parents[3] / "schemas" / "cell.schema.json"
        _CELL_SCHEMA = json.loads(schema_path.read_text(encoding="utf-8"))
    return _CELL_SCHEMA


def validate_relations(
    relations: list[dict[str, Any]],
    vocabulary: Vocabulary,
    cells: dict[str, dict[str, Any]],
) -> list[Finding]:
    """Closed vocabulary, dangling references, domain/range compliance."""
    errors: list[Finding] = []
    seen_ids: set[str] = set()
    for edge in relations:
        if not isinstance(edge, dict):
            errors.append(Finding("error", "relations", "each relation must be a mapping."))
            continue
        rid = edge.get("id")
        if not isinstance(rid, str) or not rid:
            errors.append(Finding("error", "relations", "relation without an id."))
            continue
        if rid in seen_ids:
            errors.append(Finding("error", rid, "duplicate relation id."))
            continue
        seen_ids.add(rid)

        rtype_name = edge.get("type")
        rtype = vocabulary.relations.get(rtype_name) if isinstance(rtype_name, str) else None
        if rtype is None:
            errors.append(
                Finding("error", rid, f"relation type '{rtype_name}' not in the vocabulary.")
            )
            continue

        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, str) or source not in cells:
            errors.append(Finding("error", rid, f"source '{source}' is not a known cell."))
            continue
        if not isinstance(target, str) or target not in cells:
            errors.append(Finding("error", rid, f"target '{target}' is not a known cell."))
            continue

        source_type, target_type = cells[source]["type"], cells[target]["type"]
        if "*" not in rtype.domain and source_type not in rtype.domain:
            errors.append(
                Finding(
                    "error",
                    rid,
                    f"'{rtype_name}' cannot leave a {source_type} (allowed: {', '.join(rtype.domain)}).",
                )
            )
        if "*" not in rtype.targets and target_type not in rtype.targets:
            errors.append(
                Finding(
                    "error",
                    rid,
                    f"'{rtype_name}' cannot point at a {target_type} (allowed: {', '.join(rtype.targets)}).",
                )
            )
    return errors


def semantic_flags(
    cells: dict[str, dict[str, Any]], relations: list[dict[str, Any]]
) -> list[Finding]:
    """Warnings, not errors: commitments surfaced for human attention."""
    warnings: list[Finding] = []

    active_ids = {cid for cid, c in cells.items() if c["status"] == "active"}
    referenced = {e.get("source") for e in relations} | {e.get("target") for e in relations}
    for cid in sorted(active_ids - referenced):
        warnings.append(
            Finding("warning", cid, "orphan cell: no relation references it. Steward, please review.")
        )

    decision_ids = {cid for cid, c in cells.items() if c["type"] == "decision"}
    grounded = {e.get("source") for e in relations if e.get("type") == "derives_from"}
    for cid in sorted(decision_ids - grounded):
        warnings.append(
            Finding("warning", cid, "decision without a recorded basis (no derives_from edge).")
        )

    for edge in relations:
        if edge.get("type") == "conflicts":
            warnings.append(
                Finding(
                    "warning",
                    str(edge.get("id")),
                    f"contradiction routed to the Knowledge Steward: "
                    f"{edge.get('source')} ↔ {edge.get('target')}. "
                    f"Note: {edge.get('note', '(no note)')}",
                )
            )
    return warnings

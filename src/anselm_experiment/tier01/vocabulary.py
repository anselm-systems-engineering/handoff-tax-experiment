"""Tier 1 vocabulary loading and validation.

The vocabulary is a versioned YAML file (tier01/vocabulary.yaml) defining:
cell types, statuses, and the closed relation set. Every relation type must
name its consumers — the praxeological rule from the article: a formal element
earns its keep only through the acts that consume it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RelationType:
    name: str
    domain: tuple[str, ...]
    targets: tuple[str, ...]  # the YAML key is "range"
    consumers: tuple[str, ...]


@dataclass(frozen=True)
class Vocabulary:
    schema_version: int
    cell_types: tuple[str, ...]
    statuses: tuple[str, ...]
    relations: dict[str, RelationType]


class VocabularyError(Exception):
    """The vocabulary file itself violates the method's rules."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


def parse_vocabulary(data: dict[str, Any]) -> Vocabulary:
    """Parse and validate a vocabulary mapping. Raises VocabularyError."""
    errors: list[str] = []

    version = data.get("schema_version")
    if not isinstance(version, int):
        errors.append("`schema_version` must be an integer.")

    cell_types = _string_list(data.get("cell_types"), "cell_types", errors)
    statuses = _string_list(data.get("statuses"), "statuses", errors)

    if cell_types is not None and not cell_types:
        errors.append("`cell_types` must not be empty.")
    if statuses is not None and not statuses:
        errors.append("`statuses` must not be empty.")

    raw_relations = data.get("relations")
    if not isinstance(raw_relations, dict) or not raw_relations:
        errors.append("`relations` must be a non-empty mapping.")

    relations: dict[str, RelationType] = {}
    if isinstance(raw_relations, dict) and cell_types is not None:
        for name, spec in raw_relations.items():
            if not isinstance(spec, dict):
                errors.append(f"relation '{name}': specification must be a mapping.")
                continue
            domain = _type_set(spec.get("domain"), name, "domain", cell_types, errors)
            targets = _type_set(spec.get("range"), name, "range", cell_types, errors)
            consumers = _string_list(spec.get("consumers"), f"relation '{name}' consumers", errors)
            if consumers is not None and not consumers:
                errors.append(
                    f"relation '{name}' has no consumers. "
                    "Every formal element must name the acts that consume it."
                )
            if domain is None or targets is None or consumers is None:
                continue
            relations[name] = RelationType(
                name=name, domain=tuple(domain), targets=tuple(targets), consumers=tuple(consumers)
            )

    if errors:
        raise VocabularyError(errors)

    return Vocabulary(
        schema_version=version if isinstance(version, int) else 0,
        cell_types=tuple(cell_types or ()),
        statuses=tuple(statuses or ()),
        relations=relations,
    )


def load_vocabulary(path: str | Path) -> Vocabulary:
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise VocabularyError(["Vocabulary file must contain a YAML mapping."])
    return parse_vocabulary(data)


def _string_list(value: Any, label: str, errors: list[str]) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        errors.append(f"`{label}` must be a list of non-empty strings.")
        return None
    if len(set(value)) != len(value):
        errors.append(f"`{label}` contains duplicates.")
    return list(value)


def _type_set(
    value: Any, relation: str, key: str, cell_types: list[str], errors: list[str]
) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        errors.append(f"relation '{relation}': `{key}` must be a non-empty list.")
        return None
    unknown = [v for v in value if v != "*" and v not in cell_types]
    if unknown:
        errors.append(
            f"relation '{relation}': `{key}` names unknown cell types: {', '.join(unknown)}."
        )
        return None
    return list(value)

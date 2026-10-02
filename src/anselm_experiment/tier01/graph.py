"""Disposable view generation — the graph as a report, not a source.

Renders the typed graph as a Mermaid flowchart. ANSELM views are disposable:
regenerate, never redraw. The committed source of truth is the cells and the
typed relations; this function only projects them.
"""
from __future__ import annotations

from typing import Any

_TYPE_STYLES: dict[str, str] = {
    "need": "fill:#fde68a,stroke:#b45309",
    "function": "fill:#bfdbfe,stroke:#1d4ed8",
    "component": "fill:#bbf7d0,stroke:#15803d",
    "decision": "fill:#e9d5ff,stroke:#7e22ce",
    "constraint": "fill:#fecaca,stroke:#b91c1c",
    "interface": "fill:#cffafe,stroke:#0e7490",
}

_CONFLICT_STYLE = "stroke:#dc2626,stroke-width:3px"


def render_mermaid(cells: dict[str, dict[str, Any]], relations: list[dict[str, Any]]) -> str:
    lines = ["flowchart LR"]
    for cid, cell in cells.items():
        label = f"{cid}<br/>{_escape(cell['title'])}"
        lines.append(f'    {cid}["{label}"]')

    typed = {cid for cid, cell in cells.items() if cell["type"] in _TYPE_STYLES}
    for ctype, style in _TYPE_STYLES.items():
        members = [cid for cid in typed if cells[cid]["type"] == ctype]
        if members:
            lines.append(f"    class {','.join(members)} {ctype}")
    for ctype, style in _TYPE_STYLES.items():
        lines.append(f"    classDef {ctype} {style}")

    conflict_indices: list[int] = []
    edge_index = 0
    for edge in relations:
        src, tgt, rtype = edge.get("source"), edge.get("target"), edge.get("type")
        if src is None or tgt is None or rtype is None:
            continue
        lines.append(f"    {src} -->|{rtype}| {tgt}")
        if rtype == "conflicts":
            conflict_indices.append(edge_index)
        edge_index += 1

    if conflict_indices:
        lines.append("")
        for i in conflict_indices:
            lines.append(f"    linkStyle {i} {_CONFLICT_STYLE}")
    return "\n".join(lines)


def _escape(text: str) -> str:
    return text.replace('"', "'").replace("<", "&lt;").replace(">", "&gt;")

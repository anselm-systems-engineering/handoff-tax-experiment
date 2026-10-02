"""PROSE-SEAM and TYPED-SEAM architectures.

Two isolated reasoning contexts (Jane: propulsion; Ahmed: energy & thermal)
integrate through a boundary artifact. The independent variable is the
formalism of the seam:

- PROSE-SEAM: free-form Markdown packs; an integrator translates the final
  pack into the typed deliverable. The translation is itself a lossy hand-off.
- TYPED-SEAM: the boundary artifact is typed cells + relations, gated at every
  crossing by the deterministic tier01 oracle (structure only — open
  `conflicts` edges pass the gate so they reach the receiver; they are
  violations only in the final deliverable).

Both conditions end with the same artefact: a typed ecosystem evaluated by the
six-family extended checker.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..llm import LLM, Message
from ..tier01.oracle import validate_cells, validate_relations
from ..tier01.vocabulary import Vocabulary
from . import prompts as P

MAX_GATE_ROUNDS = 8


def run_prose_seam(
    *,
    brief: dict[str, Any],
    terms: dict[str, Any],
    llm: LLM,
    roundtrips: int = 1,
    log_dir: Path,
) -> dict[str, Any]:
    """PROSE-SEAM: prose packs across the seam, integrator at the end."""
    log_dir.mkdir(parents=True, exist_ok=True)
    format_instructions = P.prose_format_prompt(terms)
    jane_pack = _call(
        llm,
        system=P.JANE_SYSTEM,
        user=P.initial_user_prompt(brief, format_instructions)
        + "\n\nProduce YOUR pack (propulsion).",
        tag="jane_r0",
    )
    packs = [jane_pack]

    current = jane_pack
    for r in range(roundtrips):
        receiver_system = P.AHMED_SYSTEM if r % 2 == 0 else P.JANE_SYSTEM
        receiver_name = "ahmed" if r % 2 == 0 else "jane"
        current = _call(
            llm,
            system=receiver_system,
            user=(
                "THE OTHER SIDE'S PACK:\n"
                + current
                + "\n\nProduce YOUR integrated pack: respond to every assumption and "
                "commitment of the other side, and keep the same format sections."
            ),
            tag=f"{receiver_name}_r{r}",
        )
        packs.append(current)

    integrator_raw = _call(
        llm,
        system=P.INTEGRATOR_SYSTEM,
        user=(
            "JANE'S PACK:\n"
            + jane_pack
            + "\n\nFINAL PACK:\n"
            + current
            + "\n\nTranslate both into the typed ecosystem JSON (schema below).\nSCHEMA:\n"
            + json.dumps(_ecosystem_schema(), indent=2)
        ),
        tag="integrator",
    )
    ecosystem = _safe_parse_json(integrator_raw)

    (log_dir / "jane_pack.md").write_text(jane_pack, encoding="utf-8")
    (log_dir / "final_pack.md").write_text(current, encoding="utf-8")
    (log_dir / "ecosystem.json").write_text(
        json.dumps(ecosystem, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "ecosystem": ecosystem,
        "packs": packs,
        "rejection_rounds": {},
        "premature_commitments": _count_no_fitting_type([jane_pack, current, integrator_raw]),
    }


def run_typed_seam(
    *,
    brief: dict[str, Any],
    vocabulary_data: dict[str, Any],
    terms: dict[str, Any],
    llm: LLM,
    roundtrips: int = 1,
    log_dir: Path,
) -> dict[str, Any]:
    """TYPED-SEAM: typed fragments, oracle-gated at every crossing."""
    log_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = _vocabulary(vocabulary_data)
    format_instructions = P.typed_format_prompt(vocabulary_data, terms)
    rejection_rounds: dict[str, int] = {}
    assistant_texts: list[str] = []

    # --- Jane produces her subsystem fragment, gated at the seam --------------
    jane_scope, jane_excluded = _role_scope(brief, "jane")
    jane_user = P.initial_user_prompt(brief, format_instructions) + "\n\n"
    if jane_scope:
        jane_user += (
            f"Produce YOUR subsystem as a typed fragment: {jane_scope}. Ids prefixed "
            f"`jane-`. {jane_excluded} are AHMED'S subsystem — do not create them. "
            "Relations only among your own cells. Output ONLY the JSON."
        )
    else:
        jane_user += (
            "Produce YOUR subsystem as a typed fragment: the terrain-capability "
            "need, the 350 kg mass constraint, the propulsion function (hybrid "
            "wheel-leg), and your concept decision with its rationale. Ids prefixed "
            "`jane-`. The motor controller, battery, thermal management and power "
            "distribution are AHMED'S subsystem — do not create them. Relations only "
            "among your own cells. Output ONLY the JSON."
        )
    jane_fragment, rounds, texts = _gated_json(
        llm,
        system=P.JANE_SYSTEM,
        user=jane_user,
        tag="jane",
        vocabulary=vocabulary,
        schema_check=False,
    )
    rejection_rounds["jane"] = rounds
    assistant_texts.extend(texts)

    # --- alternating crossings: Ahmed integrates, then Jane revises, ... -------
    current: dict[str, Any] = jane_fragment
    for r in range(roundtrips):
        if r % 2 == 0:
            system, name = P.AHMED_SYSTEM, "ahmed"
            ahmed_scope, _ = _role_scope(brief, "ahmed")
            scope_text = (
                ahmed_scope
                or "battery pack, motor controller with the EXACT tag `sensor-critical` "
                "and its sensors attribute, thermal management, the power-bus "
                "INTERFACE cell, and the failsafe constraint"
            )
            instruction = (
                "JANE'S VALIDATED FRAGMENT:\n"
                + json.dumps(current, indent=2)
                + "\n\nProduce the FULL integrated ecosystem JSON (with top-level `id` and "
                f"`title`): keep Jane's cells as-is, add your own cells (ids prefixed "
                f"`ahmed-`: {scope_text}), write the integrated `relations` "
                "(commitment-level only), `attributes` (masses, interface numbers, "
                "sensor coverage), `assumptions`, and `commitments`. Respond to every "
                "Jane assumption and commitment (set response_by / agreed). Output "
                "ONLY the JSON."
            )
        else:
            system, name = P.JANE_SYSTEM, "jane"
            instruction = (
                "AHMED'S INTEGRATED ECOSYSTEM:\n"
                + json.dumps(current, indent=2)
                + "\n\nRevise the full integrated ecosystem: fix any structural errors "
                "reported below, keep both sides' cells, and resolve open conflicts. "
                "Output ONLY the JSON."
            )
        current, rounds, texts = _gated_json(
            llm,
            system=system,
            user=instruction,
            tag=f"{name}_r{r}",
            vocabulary=vocabulary,
            schema_check=True,
        )
        rejection_rounds[f"{name}_r{r}"] = rounds
        assistant_texts.extend(texts)

    (log_dir / "jane_fragment.json").write_text(
        json.dumps(jane_fragment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (log_dir / "ecosystem.json").write_text(
        json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "ecosystem": current,
        "jane_fragment": jane_fragment,
        "rejection_rounds": rejection_rounds,
        "premature_commitments": _count_no_fitting_type(assistant_texts),
    }


# ---------------------------------------------------------------- helpers

def _gated_json(
    llm: LLM,
    *,
    system: str,
    user: str,
    tag: str,
    vocabulary: Vocabulary,
    schema_check: bool = False,
) -> tuple[dict[str, Any], int, list[str]]:
    """Draft → deterministic gate → feedback loop. Returns (json, rounds, texts).

    The gate is structural: tier01 cell/relation checks, plus (when
    `schema_check` is set) the ecosystem shape schema. Domain families are
    deliberately excluded — they are the measurement, not the treatment.
    """
    transcript: list[Message] = [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]
    assistant_texts: list[str] = []
    for i in range(MAX_GATE_ROUNDS):
        response = llm.call(transcript, tag=f"{tag}_draft_{i}")
        transcript.append(Message(role="assistant", content=response.content))
        assistant_texts.append(response.content)
        parsed = _safe_parse_json(response.content)
        errors = list(_structure_errors(parsed, vocabulary))
        if schema_check:
            from .checker import schema_violations

            errors.extend(schema_violations(parsed, structure_id="structure"))
        if not errors:
            return parsed, i + 1, assistant_texts
        error_text = " ".join(getattr(e, "message", str(e)) for e in errors)
        feedback = (
            "The deterministic gate rejected the JSON with these structural "
            "errors. Fix every one and output ONLY the full JSON again.\n\n"
            + json.dumps([e.__dict__ for e in errors], indent=2)
            + "\n\n"
            + _shape_hints(errors)
            + "\nReminder — the ONLY allowed relation types are: "
            + ", ".join(sorted(vocabulary.relations))
            + ". Typed relations are commitment-level only; physical "
            "connectivity (wiring, powering, monitoring) belongs in cell "
            "prose and in `attributes`, never in `relations`. Keep "
            "`attributes`, `assumptions`, and `commitments` present and "
            "well-formed — the final checker verifies them."
        )
        if any(
            kw in error_text
            for kw in (
                "Additional properties",
                "required property",
                "each relation must be a mapping",
                "relation without an id",
            )
        ):
            feedback += "\n\nCanonical shape to follow:\n" + P.EXAMPLE_FRAGMENT
        transcript.append(Message(role="user", content=feedback))
    return _safe_parse_json(assistant_texts[-1]), MAX_GATE_ROUNDS, assistant_texts


def _shape_hints(errors: list[Any]) -> str:
    """Targeted hints derived from the error text, so the model fixes the
    shape instead of oscillating between wrong shapes."""
    messages = " ".join(getattr(e, "message", str(e)) for e in errors)
    hints: list[str] = []
    if "each relation must be a mapping" in messages or "relation without an id" in messages:
        hints.append(
            "`relations` must be a JSON ARRAY of objects, each with fields "
            "`id`, `type`, `source`, `target` — NOT a map keyed by relation id."
        )
    if "'attributes' was unexpected" in messages:
        hints.append(
            "`attributes` is a TOP-LEVEL map keyed by cell id — do NOT nest an "
            "`attributes` object inside an individual cell."
        )
    if "Additional properties are not allowed" in messages:
        hints.append(
            "Numeric fields (`mass_kg`, `voltage_v`, `peak_current_a`, "
            "`thermal_duty_w`) and `sensors` belong in the top-level "
            "`attributes` map, NOT inside cell frontmatter."
        )
    if "'array'" in messages or "not of type 'array'" in messages:
        hints.append("`assumptions` and `commitments` must be JSON ARRAYS of objects.")
    if "is a required property" in messages:
        hints.append(
            "Required fields per object type — cells: id, title, type, status, "
            "author, date; relations: id, type, source, target; assumptions: "
            "id, author, text; commitments: id, from, to, term, claim."
        )
    return "\n".join(hints) + ("\n" if hints else "")


def _role_scope(brief: dict[str, Any], role: str) -> tuple[str, str]:
    """Role scoping from the brief (v2), so architecture prompts are not
    hard-coded to one task. Falls back to empty strings for v1 briefs."""
    spec = (brief.get("roles") or {}).get(role) or {}
    return spec.get("scope", ""), spec.get("excluded", "")


def _structure_errors(ecosystem: dict[str, Any], vocabulary: Vocabulary):
    cells = ecosystem.get("cells", {})
    relations = ecosystem.get("relations", [])
    errors = validate_cells(cells, vocabulary) + validate_relations(relations, vocabulary, cells)
    if not cells:
        from ..tier01.oracle import Finding

        errors.append(Finding("error", "global", "no cells in the fragment."))
    return errors


def _vocabulary(vocabulary_data: dict[str, Any]) -> Vocabulary:
    from ..tier01.vocabulary import parse_vocabulary

    return parse_vocabulary(vocabulary_data)


def _ecosystem_schema() -> dict[str, Any]:
    schema_path = Path(__file__).resolve().parents[3] / "schemas" / "ecosystem.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _count_no_fitting_type(texts: list[str]) -> int:
    return sum(t.lower().count("no-fitting-type") for t in texts)


def _call(llm: LLM, *, system: str, user: str, tag: str) -> str:
    return llm.call(
        [Message(role="system", content=system), Message(role="user", content=user)],
        tag=tag,
    ).content


def _safe_parse_json(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return {}

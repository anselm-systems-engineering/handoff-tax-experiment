# ANSELM Tier 0–1 prototype

Reference implementation of the "Ontology at the Seams" article: typed
knowledge-cell frontmatter (Tier 0), a closed relation vocabulary (Tier 1),
and a deterministic coherence oracle. No LLM calls, no modeling suite — this
is the formal layer that earns its place.

## What is here

| Path | Role |
|------|------|
| `../schemas/cell.schema.json` | Tier 0: JSON Schema for cell frontmatter (closed `type` and `status` vocabularies, open `tags`). |
| `vocabulary.yaml` | Tier 1: the closed relation set. **Every relation must name its consumers** — the praxeological test from the article. |
| `examples/atlas/` | A worked ecosystem: 9 knowledge cells + 10 typed relations, including an open contradiction (constraint-002 ↔ dec-002). |
| `../src/anselm_experiment/tier01/` | The oracle, the vocabulary loader, and the Mermaid view generator. |
| `../tests/test_tier01.py` | 8 deterministic tests. |

## Usage

```powershell
# run the oracle
python -m anselm_experiment.tier01 check tier01/examples/atlas

# render the typed graph as a disposable Mermaid view
python -m anselm_experiment.tier01 graph tier01/examples/atlas
```

## What the oracle enforces

**Errors (mechanical, deterministic):**

- frontmatter validity against `cell.schema.json`;
- cell id uniqueness and filename agreement;
- cell `type` / `status` inside the vocabulary;
- every relation edge references existing cells;
- relation types are closed to the vocabulary;
- domain/range compliance (a `satisfies` edge cannot leave a `decision`);
- the vocabulary itself: every relation type must declare at least one
  consumer.

**Warnings (commitments surfaced for human attention):**

- `conflicts` edges — always routed to the Knowledge Steward;
- decisions without a `derives_from` basis;
- active cells referenced by no relation.

## Design notes

- **Extraction is a proposal; the oracle is the truth.** An LLM (or a human)
  proposes cells and edges; this checker decides. The committed graph is
  versioned like code.
- **The consumer ledger is printed on every run.** It makes the
  praxeological rule visible: a relation type nobody uses is a decoration
  waiting to be deleted.
- **The Mermaid graph is a view, not a source.** It is regenerated from the
  committed cells and relations; editing the picture changes nothing.

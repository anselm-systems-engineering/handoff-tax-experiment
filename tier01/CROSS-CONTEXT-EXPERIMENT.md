# Cross-Context Coherence Experiment (design)

> Pre-registered design for measuring whether the Tier 0–1 typed layer reduces
> coherence loss when reasoning must cross a seam. No runs yet — this document
> is the contract that future runs must satisfy.

## 1. Research question

Article 4 established that a single continuous context beats fragmented
pipelines: the hand-off tax is real, and the remedy is *don't fragment*. But
some seams are unavoidable — two teams, two institutions, a regulator, a
programme resumed after three years. Article 5 ("Ontology at the Seams")
claims that the capacity of such a seam is set by the **shared formal
vocabulary**: typed cells (Tier 0) plus a closed relation set with a
deterministic oracle (Tier 1) let meaning cross a lossy channel with bounded
loss, while bare prose does not.

This experiment tests that claim directly:

> **Does the formalism of the seam reduce coherence violations in the
> integrated deliverable — and at what cost in token spend, rejection
> rounds, and premature commitments?**

## 2. Pre-registered predictions

1. **Main effect.** Integrated deliverables produced through a typed seam
   have fewer or equal checker violations than those produced through a prose
   seam: `V_TYPED ≤ V_PROSE` at every round-trip count.
2. **Scaling.** The gap widens with the number of round-trips `k`: at
   `k = 1` the advantage is small; at `k = 3` it is large, because prose-seam
   loss compounds while typed-seam loss is bounded by the oracle at each
   crossing.
3. **The schema tax is real but cheap.** The typed condition pays rejection
   rounds (oracle failures the producer must fix) and some token overhead.
   Prediction: total tokens of TYPED within 1.5× of PROSE, and ≥ 80% of
   rejection rounds are mechanical (schema/domain errors), not semantic.
4. **Commitment coverage.** The fraction of the producer's explicit
   commitments present in the final integrated design is higher in TYPED.
5. **Honest inversion risk (the cost side of formalism).** The closed
   vocabulary may *distort* meaning when nothing fits — producers forced to
   mis-type commitments. Measured as "premature commitment" events: producer
   notes of the form "no fitting type". Prediction: present in TYPED, absent
   in PROSE, but small (< 1 per run on average). If it is large, the
   experiment concludes against a closed Tier 1 vocabulary as specified, and
   the vocabulary must shrink or open — this is the anti-cathedral control.

## 3. Conditions (the independent variable is the seam)

Same base model, same brief, same two roles (Jane = propulsion, Ahmed =
energy & thermal). Only the boundary artifact differs.

| Condition | Boundary artifact | Oracle at the seam | Integrator step |
| --------- | ----------------- | ------------------ | --------------- |
| **PROSE-SEAM** | Free-form Markdown pack | none | yes — translates prose into the typed deliverable |
| **TYPED-SEAM** | Knowledge cells + relations (`tier01` format) | yes — the deterministic oracle gates every crossing | no — validation only |

Both conditions end with the same artefact: a typed ecosystem directory
(cells + `relations.yaml`) checked by the **extended checker** (tier01 oracle + the six domain constraint families from the brief). The PROSE integrator
step is deliberate: a prose seam that must be consumed by a typed downstream
pays a translation hand-off, and that tax is part of what is being measured.

Both contexts are isolated: neither agent sees the other's conversation.
They exchange only the boundary artifact, `k` times.

## 4. Task

`briefs/atlas_cross_context.yaml` — Project Atlas propulsion ↔ energy/thermal
integration. Six deterministic constraint families:

1. `mass_budget` — sum of declared component masses ≤ 350 kg;
2. `interface_completeness` — the power interface declares voltage, peak
   current, thermal duty;
3. `redundancy` — every sensor-critical element has declared coverage
   (redundant channel or failsafe trigger);
4. `term_consistency` — seam-crossing terms match the controlled table
   (`tier01/examples/atlas/terms.yaml`);
5. `no_dangling_assumptions` — every recorded assumption acknowledged or
   contested by the other side;
6. `typed_graph_validity` — the deliverable passes the tier01 oracle.

Edge cases: thermal peak, sensor loss mid-climb, depleted battery on sand.

## 5. Metrics (per run)

| Metric | Definition |
| ------ | ---------- |
| **Violations** | Count of checker failures in the final deliverable (objective; the extended checker is the oracle). |
| **Commitment coverage** | Fraction of the producer's explicit commitments present in the final deliverable. In TYPED, commitments are the typed cells/edges that crossed the seam; in PROSE, the same list is recovered by a deterministic extractor over the prose pack. |
| **Residual ambiguity** | Underspecified decision points in the deliverable (a decision must name role, data, and outcome paths — as in the hand-off-tax experiment). |
| **Token spend** | Total tokens across both contexts, all round-trips, and the integrator. |
| **Rejection rounds** | Oracle failures at the seam in TYPED (mechanical vs semantic, by failure class). |
| **Premature commitments** | Producer notes declaring "no fitting type" (TYPED only). |

## 6. Procedure

- **Model:** pinned, same for both conditions (`gpt-4o-mini` for Phase 1).
- **Temperature:** 0 everywhere; each condition runs `n = 5`.
- **Round-trips:** `k = 1` for Phase 1; `k = 3` for Phase 2.
- **Oracle:** the extended checker is deterministic and version-controlled;
  any rule change bumps the schema/checker version and invalidates prior
  runs.
- **Forensic trail:** every run leaves the full call log, the boundary
  artifacts of every crossing, the rejection messages, and `violations.json`
  — same layout discipline as the hand-off-tax experiment.

## 7. Phases

- **Phase 1** (this design): `k = 1`, `n = 5`, one model family.
- **Phase 2:** `k = 3`, add a second, heterogeneous model family (a genuine
  `K_0` comparison, not sampling jitter).
- **Phase 3 (open):** a third condition, **OPEN-SEAM** — prose plus the
  controlled term table but no closed relation vocabulary — to separate the
  contribution of Tier 0 (vocabulary) from Tier 1 (typed relations).

## 8. Caveats

- One task, one subsystem pair, small `n`. The structural signal should be
  readable through it, but the magnitude of the gap is open.
- The checker is partial: it scores the six families, not "is this a good
  design in every sense".
- The prose-side extractor for commitment coverage is deterministic but
  shallow; it may under-recover prose commitments, which biases *against*
  the hypothesis. Report it as a floor, not an estimate.

## 9. What would falsify the claim

- `V_TYPED > V_PROSE` at `k = 1`, or premature commitments dominating the
  typed runs (the vocabulary distorts more than it transports).
- Token spend of TYPED exceeding 2× PROSE with no violation advantage.
- The integrator step recovering prose commitments so well that the typed
  seam adds nothing beyond Tier 0 — which would push the recommendation down
  the commitment ladder, not up.

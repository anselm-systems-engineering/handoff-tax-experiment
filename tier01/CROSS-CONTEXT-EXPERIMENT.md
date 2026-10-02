# Cross-Context Coherence Experiment (design)

> Pre-registered design for measuring whether the Tier 0–1 typed layer reduces
> coherence loss when reasoning must cross a seam. This document is the
> contract that runs must satisfy.

## 0. Implementation status

The harness is implemented and plumbing-verified (no live runs yet):

- `src/anselm_experiment/seam/` — extended checker (six families), PROSE-SEAM
  and TYPED-SEAM architectures, metrics, runner.
- `schemas/ecosystem.schema.json` — the typed deliverable.
- `scripts/seam_dry_run.py` — no-API plumbing check (KNOWN-BAD flags all six
  families, KNOWN-GOOD is clean, both architectures run on a scripted LLM
  including a gate rejection/revision cycle).
- `tests/test_seam.py` — 7 unit tests.

```powershell
# plumbing check (no API cost)
python scripts/seam_dry_run.py

# live runs (API cost; see §6)
python -m anselm_experiment.seam.runner --brief briefs/atlas_cross_context.yaml --arch prose-seam --runs 5
python -m anselm_experiment.seam.runner --brief briefs/atlas_cross_context.yaml --arch typed-seam --runs 5
```

## 0.2 Phase 1 results (2026-10-02, n=5 per condition, k=1, gpt-4o-mini-2024-07-18)

| Condition | Violations (mean ± std) | Range | Cell preservation | Structural violations | Rejection rounds | Tokens |
|-----------|-------------------------|-------|--------------------|-----------------------|------------------|--------|
| typed-seam | 9.8 ± 3.5 | 4–13 | **9/9 cells, 5/5 runs** | **0** | 5.2 | 21 890 |
| prose-seam | 8.0 ± 2.0 | 6–11 | **1–2 cells, 5/5 runs** | 20 | — | 6 614 |

**The pre-registered main prediction was not confirmed by the naive metric at
k=1 — and the phase exposed a metric confound that matters more than the
comparison.** The violation count rewards information loss: the prose
integrator dropped the design almost entirely (1–2 of 9 cells per run), and a
dropped subsystem costs one `no X declared` violation while a present-but-
incomplete subsystem costs one violation per missing field. The typed seam
preserved all 9 cells with zero structural violations in every run; its
violations are genuine domain-completeness gaps (sensor coverage, interface
numbers, seam commitments) — precisely the things the structural gate does not
enforce, which is a designed property (the gate stays structural so the final
checker remains a measurement, not a treatment).

Revised reading, metric v2 (information preservation + structural validity):
the typed seam is decisive at k=1 — 9/9 vs 1–2 cells, 0 vs 20 structural
violations, 5/5 runs both sides. The open question moves to the scaling
prediction: does the gap grow with round-trips `k`? That is what Phase 2 is
for. The domain-completeness gaps in TYPED are a separate, honest finding:
agents optimize for what the gate enforces, so a structural-only seam needs
the completeness work to happen in the conversation, not at the gate.

## 0.3 Phase 2a results (2026-10-02, k=3, n=5 per condition, same model)

| Condition | k | Violations (mean ± std) | Cell preservation | Structural violations | Coverage | Rejections | Tokens |
|-----------|---|-------------------------|-------------------|-----------------------|----------|------------|--------|
| typed-seam | 1 | 9.8 ± 3.5 | 9/9, 5/5 runs | 0 | 1.00 | 5.2 | 21 890 |
| typed-seam | 3 | 8.2 ± 3.0 | 9,9,8,9,9 | 0 | 0.95 | 7.8 | 27 994 |
| prose-seam | 1 | 8.0 ± 2.0 | 1–2, 5/5 runs | 20 | 0.80 | — | 6 614 |
| prose-seam | 3 | 7.8 ± 2.4 | 1–2, 5/5 runs | 14 | 0.60 | — | 9 522 |

**The scaling prediction needs a refinement.** The preservation gap did not
*widen* with k — it could not: the prose seam collapsed to 1–2 cells already
at k=1 and stayed there. Prose-seam loss is not linear in k; it is a **step
function** — the first translation collapses the design, and further
round-trips only accumulate *vocabulary drift* (term violations 5 → 10, and
headline coverage 0.80 → 0.60). The typed seam held its shape across k:
8–9 of 9 cells every run, zero structural violations at both k, and its
residual domain-completeness violations actually *decreased* (9.8 → 8.2) as
revision rounds repaired gaps. Rejection rounds scale with crossings
(5.2 → 7.8) — the price of the gate, paid per seam.

Refined pre-registration for Phase 2b (open): test whether the step-function
reading survives a second, heterogeneous model family, and whether a
moderately *larger* brief (more cells) lets prose degrade below its current
floor in a way that compounds with k.

## 0.4 Phase 2b results (2026-10-02, n=5 per condition throughout)

| Sweep | Model | Brief | k | Violations | Coverage | Cells | Structural |
|-------|-------|-------|---|------------|----------|-------|-------------|
| typed | gpt-4o-2024-08-06 | v1 | 3 | 14.4 ± 2.2 | 1.00 | 9,9,9,9,9 | 0 |
| prose | gpt-4o-2024-08-06 | v1 | 3 | 10.8 ± 1.0 | 1.00 | 2,2,2,2,2 | 22 |
| typed | mini | v2 | 1 | 6.2 ± 3.4 | 1.00 | 12×5 | 0 |
| prose | mini | v2 | 1 | 8.0 ± 1.3 | 0.30 | 2,2,2,2,2 | 17 |
| typed | mini | v2 | 3 | 10.8 ± 1.0 | 1.00 | 12×5 | 0 |
| prose | mini | v2 | 3 | 5.4 ± 0.5 | 0.80 | 2,2,1,1,1 | 12 |

**The step-function reading survives the model change.** gpt-4o's prose seam
still collapsed to exactly 2 cells in 5/5 runs — with *more* structural
violations (22 vs 14) — while its typed seam held 9/9 with zero structural
violations. The collapse is architectural (the lossy translation channel),
not a weakness of one model family. gpt-4o's higher naive violation counts on
the typed side (14.4 vs 8.2) come from producing *more* countable content,
not less coherence — the same confound from §0.2, in the other direction.

**The larger brief answered the floor question, weakly.** At k=1 the prose
floor did not move in absolute terms (2 cells), so the *relative* loss grew
(~22% → ~17% preserved). At k=3 the larger brief pushed prose below the v1
floor: 1 cell in 3/5 runs (v1: 1/5) — the first sign of compounding depth,
consistent with, but weaker than, the pre-registered scaling hope.

Phase 2b closes the empirical loop for this task family: **the typed seam
preserves the design across k, model families, and brief size; the prose seam
collapses to 1–2 cells immediately and stays there, accumulating vocabulary
drift.** Open for later phases: a genuinely larger brief (30+ cells), a third
model family, and a second seam topology (three-way integration).


## 0.1 Pilot (2026-10-02, n=1 per condition, k=1, gpt-4o-mini-2024-07-18)

| Condition | Violations | Tokens | Gate rounds | Notes |
|-----------|-----------|--------|-------------|-------|
| typed-seam | 10 | 18 790 | jane: 2, ahmed: 3 | Failures are *domain completeness*: missing interface numbers, sensor coverage, no commitments — the agent optimizes for what the gate enforces. |
| prose-seam | 6 | 5 953 | — | Failures are *structure*: the integrator translated prose into partially malformed JSON (missing `type`/`id`, invented term). Translation loss at the seam. |

n=1, so no conclusions — but the profiles differ as the mechanism predicts:
the typed gate cleans shape but not completeness; the prose translator loses
shape and vocabulary. Phase 1 (n=5) will tell whether the count comparison
follows the pre-registered prediction or its falsification.


## 0.5 Formal amendment — metric v2 and confirmatory phase 2c

*Written before any DeepSeek runs were executed, after phase 1/2a/2b data were
collected. This amendment supersedes the naive metric of §2 and §5 and
re-registers the analysis for the confirmatory phase.*

**Why:** phase 1 proved that the pre-registered metric (raw violation count)
rewards information loss — a dropped subsystem costs one violation while a
present-but-incomplete subsystem costs one per missing field. The revision
below fixes the confound structurally, not interpretively.

**Metric v2 (counting rules):**

1. **Preservation** (primary): number of valid cells in the final deliverable
   (`len(ecosystem.cells)`). Assumption-free; high preservation = the design
   survived the seam. Reported per run as a distribution; compared with an
   exact two-sample permutation test.
2. **Domain violations per preserved cell** (secondary): non-structural
   violations ÷ cell count. Normalises domain completeness by how much design
   actually crossed the seam — a condition can no longer win by dropping
   content.
3. **Structural violations** (tertiary): `typed_graph_validity` count.
4. Cost: gate rejection rounds, tokens — reported, not tested.

**Statistical plan:** exact two-sample permutation test (n=5 vs n=5, 252
combinations, two-sided) on preservation and on domain-violations-per-cell;
effect sizes reported alongside. No claim is made from raw violation counts,
which remain underpowered at n=5.

**Confirmatory/exploratory split:**
- *Exploratory (hypothesis-generating)*: all readings of phases 1, 2a, 2b —
  including the step-function refinement and the confound discovery. These
  were learned from data and may not be re-sold as confirmations.
- *Confirmatory (hypothesis-testing)*: phase 2c (DeepSeek, v1 brief, k=3,
  n=5 per condition), pre-registered here against metric v2:
  - H1: typed-seam preservation > prose-seam preservation;
  - H2: typed-seam domain-violations-per-cell ≤ prose-seam;
  - H3: typed-seam structural violations = 0.
  Falsification of any of H1–H3 on fresh data is to be reported as such.

**Declared limitations carried into any publication:** one task family, one
seam topology, n=5, treatment/measurement coupling (the gate shares the
tier01 validator with the final checker), oracle measures declarations not
truth, interpretation unblinded, two-vendor model pool.

## 0.6 Phase 2c results — confirmatory (2026-10-03, DeepSeek deepseek-chat, v1 brief, k=3, n=5)

| Condition | Violations (mean ± std) | Valid cells | Structural violations |
|-----------|-------------------------|-------------|-----------------------|
| typed-seam | 4.0 ± 2.3 | 9,9,9,9,9 | 0 |
| prose-seam | 20.8 ± 8.4 | 0,0,0,0,0 | 88 |

**Verdict on the pre-registered hypotheses (fresh data, metric v2):**

- **H1 (preservation) — CONFIRMED.** Typed: 9 valid cells in 5/5 runs; prose:
  0 valid cells in 5/5 runs. Exact permutation test p = 0.0079 (complete
  separation).
- **H2 (domain violations per valid cell) — CONFIRMED, in strong form.** The
  prose seam carries no valid content at all, so the ratio is undefined (∞);
  typed is finite (0.22–0.89).
- **H3 (typed structural validity) — CONFIRMED.** 0 in every run, matching
  all earlier phases.

**A failure-mode taxonomy, honestly recorded.** The prose seam fails in two
distinct ways depending on the model family: the GPT family *collapses*
(1–2 raw cells, valid 0), DeepSeek *over-produces* (7–16 raw cells, valid 0 —
the integrator invents its own vocabulary, e.g. `type: subsystem`, and omits
required fields). A raw-cell count would have falsified H1 on DeepSeek
(prose 10.2 vs typed 9.0 mean raw cells); the amendment's specification of
*valid* cells is the load-bearing definition. Implementation note: the first
v2 implementation counted raw cells; it was corrected to the written
specification (valid = passing tier01 schema + vocabulary) before this
confirmatory analysis was finalised. Both readings are recorded here rather
than silently merged.

**Status of the whole experiment after 2c:** the claim "the typed seam
preserves the design across model families, brief sizes, and round-trips,
while the prose seam fails to carry structurally valid content at all" is now
**confirmatory** on one pre-registered phase; the step-function reading and
the failure-mode taxonomy remain exploratory. Replication across a second
seam topology and a second task family is open and pre-declared in §7.


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

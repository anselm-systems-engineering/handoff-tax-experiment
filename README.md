# handoff-tax-experiment

> Empirical study of the *hand‑off tax* in LLM systems: single‑agent iteration vs. multi‑agent pipelines and voting ensembles.

This repository is the empirical backbone of [**ANSELM Article 4 — *Not a Committee, a Conversation***](https://anselm.ing/articles/not-a-committee-a-conversation/). The article argues, on information‑theoretic grounds, that fragmenting a task across role‑bound LLM agents pays a **hand‑off tax** — each summary‑mediated interface is a lossy channel, and the data‑processing inequality is unforgiving. This repo runs the experiment that measures that tax.

## Headline result (Phase 1, n = 5 per architecture)

Same base model (`gpt-4o-mini-2024-07-18`) across all four architectures; only hand‑off structure varies. Task: redesign the **credit‑hold release** sub‑process under six constraint families (segregation of duties, SLA, audit trail, GDPR retention, system‑of‑record reuse, structural flow completeness).

| Architecture          | Hand‑offs   | Violations (mean ± std) | Range |
|-----------------------|-------------|-------------------------|-------|
| **ITER**              | 0           | **0.0 ± 0.0**           | 0–0   |
| MULTI‑VOTE‑OVER‑PIPE  | 4 + vote    | 1.4 ± 0.9               | 1–3   |
| MULTI‑VOTE‑FLAT       | 1 (vote)    | 2.0 ± 0.0               | 2–2   |
| MULTI‑PIPE            | 4 (chain)   | 17.0 ± 20.1             | 1–39  |

![Phase 1 headline plot](runs/phase1_headline.png)

Three things to read out of this:

1. **Violations track hand‑off count and structure**, not "amount of fragmentation." Zero hand‑offs → zero violations; four sequential hand‑offs → mean 17, range up to 39.
2. **Same‑model voting harvests noise, not diversity.** MV‑flat's five candidates were genuinely distinct samples, yet *every one* produced exactly two violations. Variance = 0 is the cleanest possible signature of "candidates differ in surface form but not in their relationship to the constraint set."
3. **Voting partially compensates for fragmentation but cannot recover ITER quality.** MVOP (the steel‑man for real multi‑agent frameworks) softens MULTI‑PIPE but stays strictly above the ITER floor — at considerable token cost.

The full discussion, including the honest pre‑registration inversion, is in §10 of the article.

## Architectures

All four use the same base model. The only independent variable is hand‑off structure.

- **ITER** — one agent, one continuous context. Loop: draft → check (constraint checker as a tool) → analyse violations → revise. Capped at 5 iterations. **0 hand‑offs.**
- **MULTI‑VOTE‑FLAT** — *m* = 5 parallel modellers given the same brief at temperature 1.0 with seeds 1000+k. Aggregator picks one by self‑report. **1 hand‑off** (vote only); each candidate sees the brief whole.
- **MULTI‑VOTE‑OVER‑PIPE** — *m* = 5 parallel MULTI‑PIPE chains (seeds 2000+k), then an aggregator picks. **4 hand‑offs per candidate, plus a final vote.** Steel‑man for production multi‑agent frameworks (CrewAI, AutoGen).
- **MULTI‑PIPE** — Discovery → Modeller (sees only Discovery's summary) → Reviewer (sees only the model + a violations summary) → Planner (sees only the approved model). **4 sequential hand‑offs**; each lossy.

## Layout

```
.
├── briefs/
│   └── credit_hold_release.yaml      # task definition (constraints + edge cases)
├── schemas/
│   └── process.schema.json           # closed tag vocabulary; deliverable target
├── src/anselm_experiment/
│   ├── llm.py                        # litellm wrapper; per-call logs + progress
│   ├── checker.py                    # deterministic oracle for the 6 families
│   ├── metrics.py                    # BERTScore at hand-offs; ambiguity; tokens
│   ├── runner.py                     # dispatch + sweep; writes summary.json incrementally
│   └── architectures/
│       ├── iter_agent.py
│       ├── multi_pipe.py
│       ├── multi_vote.py             # MULTI-VOTE-FLAT
│       ├── multi_vote_over_pipe.py   # MVOP — m parallel multi_pipe chains + aggregator
│       └── prompts.py
├── scripts/
│   ├── analyse.py                    # aggregates runs/*/summary.json → table + headline plot
│   └── dry_run.py
├── runs/                             # one directory per run; full forensic trail
│   └── phase1_headline.png
└── notebooks/
```

## Quick start

```powershell
cd handoff-tax-experiment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env                 # then edit: OPENAI_API_KEY=sk-...

# Run all four architectures, n=5 each
python -m anselm_experiment.runner --brief briefs/credit_hold_release.yaml --arch iter                 --runs 5
python -m anselm_experiment.runner --brief briefs/credit_hold_release.yaml --arch multi-pipe           --runs 5
python -m anselm_experiment.runner --brief briefs/credit_hold_release.yaml --arch multi-vote-flat      --runs 5
python -m anselm_experiment.runner --brief briefs/credit_hold_release.yaml --arch multi-vote-over-pipe --runs 5

# Aggregate + regenerate the headline plot
python scripts/analyse.py
```

`summary.json` is written after every completed run, so cancelling a sweep never wipes prior data — re‑running with `--runs N` continues; `analyse.py` keeps the most recent 5 rows per architecture.

## What every run leaves on disk

```
runs/<timestamp>_<arch>/
├── summary.json                   # one row per completed run; written incrementally
└── run_NN/
    ├── call_0001_<role>.json      # full request + response + latency + tokens
    ├── call_0002_<role>.json
    ├── ...
    ├── info_loss.json             # BERTScore at each hand-off (pipeline archs)
    ├── result.json                # final deliverable (process JSON)
    └── violations.json            # structured: which step broke which rule
```

MVOP additionally keeps `pipe_0/` … `pipe_4/` per candidate chain plus `mvop_summary.json` from the aggregator, so the "which candidate did the vote pick, and why" question is answerable without re‑running anything.

## Reproducibility knobs

- **Pinned model:** `gpt-4o-mini-2024-07-18` for *every* architecture (so $K_0$ in §5 of the article is held constant).
- **Temperature:** 0 everywhere except vote candidates — 1.0 with **seeds 1000+k for MV‑flat, 2000+k for MVOP** (distinct seed bands so the two voting architectures don't share samples).
- **ITER iteration cap:** 5.
- **Constraint checker is the oracle.** It is deterministic and version‑controlled. Any change to the rules invalidates prior runs — bump the schema/checker version when you change them.

## Caveats

- One sub‑process, one model family, n = 5. The structural signal is large enough to read through that, but the gap on the full order‑to‑cash brief is open.
- The oracle is partial — it scores the six rules listed, not "is this a good redesign in every sense."
- Same‑model is the deliberately weakest test of voting. A heterogeneous ensemble might do better than MVOP did here.

## Phase 2 (open)

- Scale the brief to full order‑to‑cash.
- Add at least one heterogeneous model family (a genuine $K_0$ comparison rather than sampling jitter).
- Stress MVOP with non‑trivial aggregator strategies (e.g. cross‑critique instead of argmax).

## Citation

If you use or reference this experiment, please cite the article it backs:

> ANSELM (2026). *Not a Committee, a Conversation: Why Iterative Refinement Beats Agent Fragmentation*. <https://anselm.ing/articles/not-a-committee-a-conversation/>

## License

See [`LICENSE`](LICENSE).

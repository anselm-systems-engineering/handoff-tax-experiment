# ANSELM Article 4 — Empirical Sketch (Phase 1)

Scaffold for the experiment described in §10 of *Not a Committee, a Conversation*.
The harness compares three agent architectures on a constrained business‑process
redesign task and measures the prediction from §4 of the article.

## Architectures

- **ITER** — one agent, one continuous context, full transcript preserved across
  draft → check → analyse violations → revise.
- **MULTI‑PIPE** — Discovery → Modeller → Compliance reviewer → Implementation
  planner. Each stage is an isolated context that receives only a *summary* of
  what came before.
- **MULTI‑VOTE** — *m* parallel modellers, an aggregator picks one by self‑report.

## Phase 1 (toy) vs. Phase 2 (full §10)

Phase 1 uses a deliberately small task — redesigning the **credit‑hold release**
sub‑process under five constraints — to validate the harness, the schema, and
the metrics. Phase 2 scales to the full order‑to‑cash task once Phase 1 reproduces
the predicted ordering.

## Layout

```
experiment/
├── pyproject.toml
├── README.md
├── .env.example
├── briefs/                    # task definitions (constraints + edge cases)
│   └── credit_hold_release.yaml
├── schemas/                   # process schema (BPMN‑lite as JSON Schema)
│   └── process.schema.json
├── src/
│   └── anselm_experiment/
│       ├── __init__.py
│       ├── llm.py             # thin LLM client wrapper (litellm)
│       ├── checker.py         # constraint checker (the oracle)
│       ├── metrics.py         # information loss, ambiguity, tokens
│       ├── architectures/
│       │   ├── __init__.py
│       │   ├── iter_agent.py
│       │   ├── multi_pipe.py
│       │   └── multi_vote.py
│       └── runner.py          # main entry point
├── runs/                      # one folder per run, holds prompts/responses/results
└── notebooks/
    └── analysis.ipynb         # results + plots
```

## Quick start

```powershell
cd experiment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env         # then edit with your API keys
python -m anselm_experiment.runner --brief briefs/credit_hold_release.yaml --arch iter --runs 3
```

## Notes

- The constraint checker is the load‑bearing part of the harness — it is the
  oracle. Treat its rules as version‑controlled: any change invalidates prior runs.
- All prompts and responses are logged verbatim under `runs/<timestamp>/` so
  results can be re‑analysed without re‑calling the model.
- Set fixed seeds and record model name + version per run for reproducibility.

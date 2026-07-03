# VALIDATION_GAP — experiments

Not one of the six core Lár-JEPA papers. This is the empirical piece behind the
Lár vs. LangChain regulated-workflows comparison
(`personal/pharma/LAR_VS_LANGCHAIN_REGULATED_WORKFLOWS.md`) — a head-to-head
pipeline comparison on the same adverse-event test cases, used to generate the
report's compliance/audit-trail evidence.

## Status

| File | Purpose | Status |
|---|---|---|
| `ae_test_cases.py` | Shared adverse-event test cases (SAE, routine, injection, out-of-scope) | **committed** |
| `langchain_pipeline.py` | LangChain-side pipeline under test | **committed** |
| `lar_pipeline.py` | Lár-side pipeline under test (deterministic router + HMAC audit) | **committed** |
| `run_experiment.py` | Runs both pipelines over `ae_test_cases.py`, diffs behaviour | **committed** |
| `results.json` | Raw run output | **committed** |
| `results_table.tex` | LaTeX table generated from `results.json` | **committed** |

## Reproducing

```
python run_experiment.py
```

Self-contained — no Lár-core import bootstrap, no external dataset.

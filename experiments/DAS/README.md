# DAS — experiments

This directory holds the experiments for the first paper of the Lár series:

> **Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion and the Safety-Learning Equivalence**
> (Sajeev 2026). Zenodo, [doi:10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781). **Published.**

DAS establishes the core claim the rest of the series builds on: divergence
between independent stream encoders (V1) is a content-blind (V4), scale-invariant
(V2) routing signal, and halting on high divergence is the maximum-gradient
training signal for the same system (V6, Safety-Learning Equivalence).

## Status

| Location | Purpose | Status |
|---|---|---|
| `lar_divergence_exp/run_experiment.py` | Base CXR divergence experiment — ViT + BioBERT streams | **committed** |
| `lar_divergence_exp/run_experiment_{a,a2,b,c,c2,e}.py` | Lettered ablations (see `EXPERIMENT_LOG.md`) | **committed** |
| `lar_divergence_exp/run_experiment_{d,f,f2-f5}.py` | Self-contained ablations, no Lár core dependency | **committed** |
| `lar_divergence_exp/run_tier1.py` | Tier-1 scaled run | **committed** |
| `lar_divergence_exp/results/experiment_results*.json` | One result file per lettered run (cited in paper as C, C2, E, F, F2–F5, Tier-1) | **committed** |
| `lar_divergence_exp/EXPERIMENT_LOG.md` | Per-run notes and parameter changes | **committed** |

Raw data: `lar_divergence_exp/data/NLMCXR_png.tgz` and `NLMCXR_reports.tar`
(Indiana University Chest X-ray collection). `data/dataset.py` extracts these
into `data/images/` and `data/reports/` on first run.

## Reproducing

```
cd lar_divergence_exp
python run_experiment.py        # base run — extracts data on first call
python run_experiment_c.py      # Experiment C (cited in paper, §Results)
python run_tier1.py             # Tier-1 scaled run
```

Each script resolves `sys.path` back to the shared Lár engine at
`JEPA_Playground/lar_jepa/` relative to its own location. If you relocate this
folder, update the `_PLAY = _HERE.parent...` line at the top of each
`run_experiment*.py` that imports `core.types` accordingly.

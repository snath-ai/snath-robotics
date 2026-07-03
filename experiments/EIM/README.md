# EIM — experiments

This directory holds the experiments for the fourth paper of the Lár series:

> **The Encoder Is Not the Memory: World-Grounded Difficulty Representations for Encoder-Invariant and Predictive Continual Learning**
> (Sajeev 2026). Zenodo, [doi:10.5281/zenodo.20583318](https://doi.org/10.5281/zenodo.20583318). **Published.**

EIM separates the modal encoder (`AbstractModalEncoder`, M1–M3) from the
context bridge (`AbstractContextBridge`, CB1–CB2) and shows that stale
embeddings — not stale encoders — are the actual failure mode across encoder
upgrades. NB4 (Experiment 4) is the completeness proof: a single Lár graph
instantiating all ten core-contract ABCs simultaneously.

## Status

| File | Purpose | Status |
|---|---|---|
| `notebooks/eim_exp1_ltl_difficulty_invariance_dhard_curriculum.ipynb` | Exp 1 — difficulty invariance across encoder swap | **run (Kaggle T4)** |
| `notebooks/eim_exp2_exp3_jepa_predictor_projection_head.ipynb` | Exp 2/3 — JEPA predictor vs. projection-head ablation | **run (Kaggle T4)** |
| `notebooks/eim_exp4_nb4_oracle_5seed.ipynb` | Exp 4 (NB4) — 5-seed attention oracle run | **run (Kaggle T4)** |
| `notebooks/eim_exp4_nb4_oracle_extended.ipynb` | Exp 4 extended follow-up | **run (Kaggle T4)** |
| `notebooks/nb4_lar_graph.py` | NB4 as an executable Lár graph — all 10 ABCs in one trace | **committed** |
| `nb4_checkpoint.json` | HMAC-sealed NB4 run checkpoint | **committed** |
| `nb4_lar_logs/` | HMAC-signed audit logs from NB4 runs | **committed** |

Per the paper's own Code and Data Availability note, Experiments 1–3 and NB4
were run as Kaggle notebooks (T4 GPU) — the `.ipynb` files here are the primary
artifacts; `nb4_lar_graph.py` is the local/reproducible version of the NB4 run.

**Confirmed result:** oracle k-NN AUROC = 0.7410 across 5 seeds (seed-varied
oracle evaluation split), exceeding the pre-registered target of 0.70.

## Reproducing

`nb4_lar_graph.py` resolves `sys.path` back to the shared Lár engine at
`JEPA_Playground/lar_jepa/` relative to its own location. The `.ipynb` files
were run on Kaggle (T4 GPU) and reference Kaggle-local paths; adjust the
path-setup cell for local execution.

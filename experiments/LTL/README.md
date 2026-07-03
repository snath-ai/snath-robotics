# LTL — experiments

This directory holds supplementary experiments for the third paper of the Lár series:

> **The Lár Training Loop: Routing Flags as Gradient Signals**
> (Sajeev 2026). Zenodo, [doi:10.5281/zenodo.20581128](https://doi.org/10.5281/zenodo.20581128). Version 2, June 2026. **Published.**

LTL proves routing flags double as an annotation-free training curriculum:
the same divergence that triggers `TRIGGER_REPLAN` at inference time is the
signal that marks an example as curriculum-worthy at training time, with no
human labels at any stage.

**The paper's primary, citable experiment set lives in `../continual_learning/`**
(`prove_learning.py`, `ablation_proof.py`, `prove_transfer.py`, `prove_policy.py`,
`coco_proof.py`, `curriculum_proof.py`, and `coco_results/`) — that location is
what the paper's own Code and Data Availability section cites directly
(`snath-robotics/experiments/continual_learning/`), so it was left in place
rather than moved here.

## What's actually in this folder

| File | Purpose | Status |
|---|---|---|
| `ltl_sigreg_5seed.ipynb` | SIGReg 5-seed run backing the isotropic regularisation results | **run** |
| `ltl_experiment1_winoground.py` | LTL Experiment 1 — COCO-Order vision-language result (AUROC 0.7245±0.0013 vs 0.6360±0.0049 random) | **committed** |
| `cb_bridge_validation/` | CB1–CB2 (`AbstractContextBridge`) empirical closure — added to the paper 2026-07-02 (§Empirical Closure of CB1–CB2) | **committed** |

### `cb_bridge_validation/`

Ablations for the two invariants that had no direct ablation before this:

- `validate_cb1_statelessness.py` — stateful vs. stateless bridge; measures
  false `TRIGGER_REPLAN` from cross-stream contamination (37/100 stateful
  vs. 0/100 stateless, G=8)
- `validate_cb2_pure_function.py` — deterministic vs. stochastic bridge;
  measures HMAC replay-verification pass rate (100/100 vs. 0/100)
- `run_validation.py` — runs both and prints the combined paper table
  (Table in §Empirical Closure of CB1–CB2)

## Reproducing

```
cd cb_bridge_validation
python run_validation.py
```

For the primary LTL curriculum proofs, see `../continual_learning/`.

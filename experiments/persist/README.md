# PERSIST — experiments

This directory holds the experiments for the sixth paper of the Lár series:

> **PERSIST: Proprioceptive Error Resolution with Scope-bounded Invariant Signal Tracking**
> (Sajeev 2026). Zenodo, [doi:10.5281/zenodo.20820042](https://doi.org/10.5281/zenodo.20820042). Version 1, June 2026. **Published.**

PAV (`../pav/`) proved that proprioceptive divergence **detects** a physics
assumption violation. PERSIST closes the loop: the same divergence signal that
detected the violation also **verifies** whether the adaptive response resolved
it — across a bounded persistence loop (adaptor tournament → incremental
adaptation → composition → scope boundary → escalation → memory consolidation).

## Status

Both experiment files are committed and validated:

| File | Purpose | Status |
|---|---|---|
| `experiments/persist/run_experiment.py` | 7-phase IceWorld persistence-loop driver (5 seeds) | **committed** |
| `experiments/persist/ice_world.py` | MuJoCo Hopper IceWorld env — four physics perturbation zones | **committed** |

Run `python experiments/persist/run_experiment.py` to reproduce all phases and
regenerate both paper figures (`divergence_curves.pdf`, `zone_detection.pdf`).
`replot.py` regenerates the figures from an existing results JSON without
re-running the MuJoCo rollouts.

The protocol was extended from 6 to 7 phases during implementation: Phase 3b
(Force zone, wind adaptor) was added to close the validation gap for the Force
zone, which was present in IceWorld's design but lacked a dedicated experiment
in the original draft. The paper (§Validation, Table 2) reflects the 7-phase
protocol. As published, every phase escalates (0/5 resolved): the SAC base
policy's hopping variance keeps D above the D_norm=0.8 normalisation threshold
in all cases, so the scope boundary fires at delta_max before full resolution.
Detection, directionality, scope-boundary, and tournament components are all
validated; see the paper's Table 2 note for the full explanation.

## Reference parameters (as implemented in `run_experiment.py`)

```
Persistence loop:  δ₀ = 0.05,  Δ = 0.06,  δ_max = 1.0
                   D_norm = 0.8 (normalisation),  D_esc = 3.0 (escalation)
                   patience P = 60 steps (Phases 2, 3, 3b, 5, 6) / 20 (Phases 1, 4)
Tournament:        trial steps K = 5,  search steps K = 8,  retrieval band ε = 0.3
Environment:       MuJoCo Hopper, 4 physics perturbation zones, 5 seeds (42, 7, 13, 99, 2026)
```

## Shared infrastructure (already in the repo)

The persistence loop reuses the same routing/learning spine as PAV:

- `divergence_router.py`, `dhard.py`, `core/`, `dmn/` — routing + D-hard queue + DMN consolidation
- `encoders/robotics/` — proprioceptive concept encoders
- `models/jepa_predictor.py`, `models/jepa_loop.py` — JEPA predictor

Trained checkpoints for the base policy live under `models/persist/`, following
the `_ROOT`-relative path convention used by `../pav/` and `../continual_learning/`
(compute repo root as `Path(__file__).parent.parent.parent`).

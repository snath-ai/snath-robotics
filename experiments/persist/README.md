# PERSIST — experiments (pending)

This directory is the home for the experiments in the sixth paper of the Lár series:

> **PERSIST: Proprioceptive Error Resolution with Scope-bounded Invariant Signal Tracking**
> (Sajeev 2026) — *forthcoming, DOI pending.*

PAV (`../pav/`) proved that proprioceptive divergence **detects** a physics
assumption violation. PERSIST closes the loop: the same divergence signal that
detected the violation also **verifies** whether the adaptive response resolved
it — across a bounded persistence loop (adaptor tournament → incremental
adaptation → composition → scope boundary → escalation → memory consolidation).

## Status

Both experiment files are committed:

| File | Purpose | Status |
|---|---|---|
| `experiments/persist/run_experiment.py` | 7-phase IceWorld persistence-loop driver (5 seeds) | **committed** |
| `experiments/persist/ice_world.py` | MuJoCo Hopper IceWorld env — four physics perturbation zones | **committed** |

Run `python experiments/persist/run_experiment.py` to reproduce all phases and
regenerate both paper figures (`divergence_curves.pdf`, `zone_detection.pdf`).

The protocol was extended from 6 to 7 phases during implementation: Phase 3b
(Force zone, wind adaptor) was added to close the validation gap for the Force
zone, which was present in IceWorld's design but lacked a dedicated experiment
in the original draft. The paper (§Validation, Table 2) has been updated to
reflect the 7-phase protocol.

## Reference parameters (from the manuscript, for when the code is added)

```
Persistence loop:  δ₀ = 0.05,  Δ = 0.06,  δ_max = 0.5
                   D_norm = 0.8 (normalisation),  D_esc = 3.0 (escalation)
                   patience P = 40 steps (Phases 2,3,5,6) / 20 (Phases 1,4)
Tournament:        trial steps K = 5,  retrieval band ε = 0.3
Environment:       MuJoCo Hopper, 4 physics perturbation zones, 5 seeds
```

## Shared infrastructure (already in the repo)

The persistence loop reuses the same routing/learning spine as PAV:

- `divergence_router.py`, `dhard.py`, `core/`, `dmn/` — routing + D-hard queue + DMN consolidation
- `encoders/robotics/` — proprioceptive concept encoders
- `models/jepa_predictor.py`, `models/jepa_loop.py` — JEPA predictor

When the IceWorld code is added, place trained checkpoints under
`models/persist/` and follow the `_ROOT`-relative path convention used by
`../pav/` and `../continual_learning/` (compute repo root as
`Path(__file__).parent.parent.parent`).

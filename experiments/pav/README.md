# PAV — experiments

This directory holds the experiments for the fifth paper of the Lár series:

> **Physics Assumption Violations: Label-Free Detection via Concept-Space Routing in Deployed Robotic Systems**
> (Sajeev 2026). Zenodo, [doi:10.5281/zenodo.20682615](https://doi.org/10.5281/zenodo.20682615). **Published.**

PAV proves the V1–V6 dual-stream routing contract detects real physics
assumption violations in MuJoCo Walker2d — friction changes the visual stream
cannot see but the proprioceptive stream detects through slip — with no human
labels or reward signal. `../persist/` (PERSIST) closes the loop by proving the
same signal also verifies whether an adaptive response resolved the violation.

## Status

| File | Purpose | Status |
|---|---|---|
| `mujoco_bipedal_proof.py` | Base V1–V6 detection proof — normal vs. ice friction | **committed** |
| `mujoco_bipedal_proof_gru.py` | GRU-encoder variant of the detection proof | **committed** |
| `mujoco_bipedal_proof_trained.py` | Trained-policy variant | **committed** |
| `mujoco_selfsup_proof.py` | Self-supervised variant | **committed** |
| `train_jepa_walker2d.py` | JEPA predictor training on Walker2d | **committed** |
| `train_gru_walker2d.py` | GRU baseline training | **committed** |
| `train_contrastive_walker2d.py` | Contrastive baseline training | **committed** |
| `train_momentum_walker2d.py` | Momentum-encoder baseline training | **committed** |
| `train_cls_walker2d.py` | Classifier baseline training | **committed** |
| `train_selfsup_walker2d.py` | Self-supervised baseline training | **committed** |
| `mujoco_proof_*.json`, `gru_proof_*.json`, `selfsup_proof_*.json`, `trained_proof_*.json` | Timestamped run outputs backing the paper's tables | **committed** |

Cited directly in the paper as `experiments/pav/mujoco_bipedal_proof_gru.py`
and `experiments/pav/train_gru_walker2d.py`.

## Shared infrastructure

Reuses the same routing/learning spine as `../persist/` and
`../continual_learning/`: `divergence_router.py`, `dhard.py`, `core/`, `dmn/`,
`encoders/robotics/`, `models/jepa_predictor.py` — compute repo root as
`Path(__file__).parent.parent.parent` (the `_ROOT`-relative convention used
across all `Snath Robotics/experiments/` subfolders).

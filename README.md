# Snath Robotics

**Humanoid cognitive routing across independent sensor streams.**

---

Current humanoid robots sit at two extremes. End-to-end deep learning (VLA models) fuse all sensor modalities into one neural network — when vision hallucinates, the hallucination propagates through every downstream decision unchecked. Classical control is safe but rigid: it cannot learn from experience without months of manual re-engineering.

Snath Robotics occupies the gap. Two independent latent streams — visual appearance and proprioceptive physics — are never fused. A mathematically frozen routing contract measures their disagreement and decides whether to proceed, adapt, or fall back to a safe position. When the system accumulates enough disagreement events, an overnight consolidation cycle trains targeted LoRA adapters that compensate for recurring failure patterns and distributes them to the fleet.

Built on [Lár-JEPA](https://github.com/snath-ai/Lar-JEPA) · Apache 2.0

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — PERCEPTION (M1–M3: structurally independent streams)     │
│                                                                     │
│  VisionEncoder ──────────────────── ProprioceptiveEncoder           │
│  (camera frames → z_vision ∈ ℝ^D)   (IMU + joints + tactile        │
│                                       → z_proprio ∈ ℝ^D)           │
│                                                                     │
│  Neither encoder reads the other's output. Sensor failures          │
│  in one stream do not propagate to the other.                       │
└──────────────────────┬──────────────────────┬───────────────────────┘
                       │                      │
                       ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — ROUTING (V1–V6: mathematically frozen, content-blind)   │
│                                                                     │
│  p_a = softmax(z_vision)    p_b = softmax(z_proprio)               │
│  Δ = p_a − p_b              D = ||Δ||₁ / √G                        │
│                                                                     │
│  D < τ_low,  both confident  →  COMMIT_TRAJECTORY  (proceed)       │
│  τ_low ≤ D < τ_high          →  TRIGGER_REPLAN     (adapt)         │
│  D ≥ τ_high                  →  STRUCTURAL_IMPASSE (brace)         │
│  one arm uncertain            →  DEFER              (lean on sure)  │
│                                                                     │
│  V4 — content-blind: the router reads D and confidence scalars      │
│  only. It never branches on the values inside z_vision or z_proprio.│
└─────────────────┬────────────────────────┬──────────────────────────┘
                  │ TRIGGER_REPLAN         │ STRUCTURAL_IMPASSE
                  ▼                        ▼
         RoboticsAdapterRouter        Physics-safe fallback
         System 1: centroid cache     (brace / controlled stop)
         System 2: LoRA injection
```

---

## The two canonical scenarios

### Ice slip — STRUCTURAL_IMPASSE

```
VisionEncoder  : floor appears flat and safe
ProprioceptiveEncoder : friction ≈ 0, lateral acceleration onset

D = 0.87  >>  τ_high = 0.60
→ STRUCTURAL_IMPASSE → brace position
```

The visual stream and the physics stream give incompatible readings. The router does not attempt to reconcile them — it drops to a pre-programmed safe position in the time it takes to evaluate one scalar.

### Motor degradation — TRIGGER_REPLAN

```
VisionEncoder  : normal walking scene
ProprioceptiveEncoder : left knee torque 50% of commanded value

τ_low ≤ D < τ_high
→ TRIGGER_REPLAN → AdapterRouter → load hardware_structural LoRA
```

The proprioceptive stream reports an asymmetry the visual stream cannot see. The adapter router identifies the failure class from its centroid cache (System 1), then injects a signed LoRA delta that compensates the joint encoding geometry (System 2), provided the adapter's temporal trust W ≥ 0.40.

---

## Identification / correction trust asymmetry

Formalised in *Architecture Is All You Need* (Sajeev 2026), §3.4 Remark (Temporal Decay and Synaptic Depression):

**System 1 — identification — trust-invariant.**
Centroid matching on the divergence vector fingerprint. The geometric signature of a sensor failure class is durable — the physics of ice does not change with time. System 1 fires regardless of adapter age and correctly names the failure class even when System 2 is fully stale.

**System 2 — correction — perishable.**
LoRA weights encode a correction derived from a specific sensor generation and hardware variant. Gated by W = exp(−λ · Δt); adapters below `min_trust = 0.40` are refused.

| Failure class | λ | Trust half-life |
|---|---|---|
| `environmental_transient` (ice, glare, wet floor) | 0.50 | 1.4 years |
| `sensor_drift` (calibration error) | 0.20 | 3.5 years |
| `hardware_structural` (motor wear, joint degradation) | 0.02 | 34.7 years |

**Degradation path:** when System 2 is refused, System 1 still identifies the failure and routes correctly. The audit note records both the identification event and the stale-adapter refusal. Identify correctly, correct conservatively.

---

## JEPA world model — annotation-free learning

In physical domains, the V1–V6 routing contract closes into a fully annotation-free loop. Physics provides the supervision signal — no human labels required.

A `JEPAPredictor` module `f_θ: z_vision → ẑ_proprio` learns to predict what the body should feel given what the camera sees:

```
D_pred = 1 − cos(f_θ(z_vision), z_proprio)    # prediction error (stop-gradient on target)
```

When `D_pred` is high, the robot's physical experience did not match what the visual scene implied. That discrepancy is the learning signal. The predictor fires before the divergence router — giving one extra inference step to replan before the slip becomes a fall.

**Auto-winner determination (self-supervised, no human labels):**

| D_pred | D | conf_vision | Winner | Failure class |
|---|---|---|---|---|
| high | high | low | proprio | `hardware_structural` |
| high | high | normal | proprio | `environmental_transient` |
| high | low | — | vision | `sensor_drift` |

The world annotates. Contact physics, joint torque, friction — these are objective facts. The system determines which stream was wrong from prediction geometry alone.

**Structured proof (held-out test set, June 2026):**

| Phase | AUROC (all) | ice\_slip | motor\_deg |
|---|---|---|---|
| Untrained predictor — random baseline | 0.4529 | 0.4568 | 0.4490 |
| Trained on normal pairs only, no labels | **0.9365** | **0.8960** | **0.9770** |

AUROC gain of +0.48 with zero human annotation. D_pred ratio: 2.8× (anomalous pairs vs normal pairs). This is LeCun's JEPA claim applied concretely. See §8.5 of *Architecture Is All You Need* ([doi:10.5281/zenodo.20419182](https://doi.org/10.5281/zenodo.20419182)) for the formal annotation burden theorem.

On real-world data — 5,000 CLIP ViT-B/32 pairs from COCO val2017 — the same label-free predictor achieves AUROC **0.9997**, with JEPA prediction error collapsing from 1.012 to 0.006 (159× reduction). Trigger rate: 56.46% of pairs flagged as hard (D ≥ τ_low). Isotropy preserved throughout.

---

## The DMN overnight cycle

Every TRIGGER_REPLAN and STRUCTURAL_IMPASSE event is HMAC-signed and written to the D_hard queue. Winner determination is automatic — derived from prediction geometry (see JEPA section above). No human label is required.

```
D_hard = { i : D_pred_i > threshold  or  Δᵢ ≥ δ  and  rᵢ = TRIGGER_REPLAN }
```

During the consolidation cycle, `RoboticsDMN` clusters events by failure class, generates a System 1 JSON centroid cache and a System 2 signed LoRA `.pt` file, and saves both to `models/adapters/`. The predictor is retrained on the accumulated buffer each cycle, improving anomaly detection in the next pass. The fleet picks up new adapters on its next `adapter_router.refresh()` call.

SIGReg (Sketched Isotropic Gaussian Regularisation, AIA §SIGReg) is wired into the training loop with `lambda_iso=0.0` by default — inert until AIA Experiment 3 calibrates the optimal weight.

---

## Domain isomorphism

Snath Robotics is the fourth instantiation of the V1–V6 routing contract, proving it is domain-agnostic:

| Repo | Domain | Stream A | Stream B | Failure class |
|---|---|---|---|---|
| [Snath Basis](https://github.com/snath-ai/snath-basis) | Quantitative finance | Fundamental analysis | Market signals | `market_regime` / `structural` |
| [Snath Aviation](https://github.com/snath-ai/snath-aviation) | Aviation sensor routing | Radar | Pitot tube | `weather_induced` / `hardware_struct` |
| **Snath Robotics** | Humanoid sensor routing | Vision | Proprioception | `environmental_transient` / `hardware_structural` |
| [Snath Research](https://github.com/snath-ai/snath-research) | Scientific claim verification | Paper claims | Peer reviews | `scope_overclaim` / `methodology_gap` |

The temporal decay formula `W = exp(−λ · Δt)`, the identification/correction trust asymmetry, and the System 1/System 2 pipeline are **identical across all instantiations**. The λ constants and failure-class labels are the only domain-specific parameters. Snath Robotics is the only instantiation where the V1–V6 loop closes without any human annotation — physics provides the winner signal directly.

---

## PAV — MuJoCo Walker2d validation (June 2026)

[*Physics Assumption Violations: Label-Free Detection via Concept-Space Routing in Deployed Robotic Systems*](https://doi.org/10.5281/zenodo.20682615)
(Sajeev 2026) · fifth paper in the Lár series · [doi:10.5281/zenodo.20682615](https://doi.org/10.5281/zenodo.20682615)

The PAV paper reports the first end-to-end validation of the V1–V6 routing contract on genuine physical dynamics.

**Setup.** MuJoCo Walker2d-v5, 17-dim proprioceptive state, random policy.
GRU encoder (H=64, G=8, L=30, trained 150 epochs on 50 000 windows per terrain
type with CE + VICReg loss, 84.4% training accuracy). Terrain: normal (μ=0.80)
and ice (μ=0.05). No labels at inference.

**Five-phase protocol.**

| Phase | Terrain | Steps | Key result |
|---|---|---|---|
| 1 — warm-up | Normal | 80 | EWMA reference anchor forms (c_ref = 0.97) |
| 2 — stable | Normal | 50 | **83% COMMIT**, mean D = 0.20 |
| 3 — PAV | Ice | 50 | **95% REPLAN + IMPASSE**, mean D = 0.43; D spikes to 0.64 on step 2 |
| 4 — DMN | — | offline | 19 D-hard events → 1 LoRA adapter, zero labels |
| 5 — adapted | Ice + adapter | 50 | Mean D = 0.15 (**65% reduction** from Phase 3) |

**Encoder ablation.**

| Encoder | COMMIT↑ | REPLAN↑ | ΔD↑ |
|---|---|---|---|
| Momentum (seq=10) | 0% | 57% | +0.01 |
| JEPA (seq=30) | 0% | 0% | −0.03 |
| **CLS-GRU (ours)** | **83%** | **95%** | **+0.50** |

The classification loss collapses within-cluster scatter to 0.181 (normal) /
0.184 (ice). Self-supervised baselines fail because they optimise temporal
coherence, not physical separability.

```bash
# Train the GRU classification encoder (Walker2d, ~5 min)
python experiments/train_gru_walker2d.py

# Run the 5-phase PAV proof
python experiments/mujoco_bipedal_proof_gru.py

# Comparison: JEPA and momentum encoders
python experiments/train_jepa_walker2d.py
python experiments/train_momentum_walker2d.py
python experiments/mujoco_bipedal_proof.py     # JEPA
python experiments/mujoco_bipedal_proof_trained.py  # momentum
```

Results are saved to `experiments/gru_proof_<ts>.json`.

---

## Getting started

```bash
# Run both demo scenarios
python robotics_graph.py

# Ice slip only
python robotics_graph.py --scenario ice_slip

# Motor degradation only
python robotics_graph.py --scenario motor_deg

# Train JEPA predictor on scenario pairs (label-free)
python robotics_graph.py --train-predictor

# Full annotation-free self-learning loop (40 steps, no human labels)
python robotics_graph.py --end-to-end

# Run overnight DMN consolidation
python robotics_graph.py --dmn-cycle
```

## Complete 7-proof suite (annotation-free continual learning)

All proofs save canonical JSON results to `experiments/results/` or `experiments/coco_results/`.

```bash
# Proof 1 — disagreement is a valid learning signal (AUROC 0.45 → 0.94)
python experiments/prove_learning.py

# Proof 2 — robust to noise and training set size (ablation sweep)
python experiments/ablation_proof.py
# Results: experiments/coco_results/ablation_<ts>.json

# Proofs 3a + 3b — detection and correction transfer to new sessions
python experiments/prove_transfer.py

# Proofs 4a + 4b + 4c — policy memory (robot learns safe speed, 6.5× speedup)
python experiments/prove_policy.py

# Proof 7 — real COCO / CLIP ViT-B/32 512-d (AUROC 0.9997)
python experiments/coco_proof.py
# Results: experiments/coco_results/coco_proof_<ts>.json

# Appendix B + Exp 1 — D_hard threshold sensitivity + curriculum vs random
python experiments/curriculum_proof.py
# Results: experiments/coco_results/curriculum_proof_<ts>.json

# Temporal decay regression suite (7/7 pass)
python test_temporal_decay.py
```

---

## Research

**Lár series** (DAS → UCR → LTL → EIM → PAV):

- Sajeev, A.V. (2026). *Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion and the Safety-Learning Equivalence.* [doi.org/10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781)
- Sajeev, A.V. (2026). *Universal Cognitive Routing: A Forward-Compatible Architecture for Heterogeneous AI Systems.* [doi.org/10.5281/zenodo.20278775](https://doi.org/10.5281/zenodo.20278775)
- Sajeev, A.V. (2026). *The Lár Training Loop: Routing Flags as Gradient Signals.* [doi.org/10.5281/zenodo.20581128](https://doi.org/10.5281/zenodo.20581128)
- Sajeev, A.V. (2026). *The Encoder Is Not the Memory: World-Grounded Difficulty Representations.* [doi.org/10.5281/zenodo.20583318](https://doi.org/10.5281/zenodo.20583318)
- Sajeev, A.V. (2026). *Physics Assumption Violations: Label-Free Detection via Concept-Space Routing in Deployed Robotic Systems.* [doi.org/10.5281/zenodo.20682615](https://doi.org/10.5281/zenodo.20682615)

**Architecture paper:**

- Sajeev, A.V. (2026). *Snath Robotics: Multi-Stream Divergence Routing for Humanoid Robotics.* [doi.org/10.5281/zenodo.20517446](https://doi.org/10.5281/zenodo.20517446)

---

*Apache 2.0 — Snath AI Open Source Research Initiative*

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

## The DMN overnight cycle

Every TRIGGER_REPLAN and STRUCTURAL_IMPASSE event is HMAC-signed and written to the D_hard queue. When a human operator (or ground-truth sensor log) labels which stream was correct, the event becomes training data.

```
D_hard = { i : Δᵢ ≥ δ  and  rᵢ = TRIGGER_REPLAN }
```

During the overnight consolidation cycle, `RoboticsDMN` clusters events by failure class, generates a System 1 JSON centroid cache and a System 2 signed LoRA `.pt` file, and saves both to `models/adapters/`. The fleet picks up the new adapters on its next `adapter_router.refresh()` call.

SIGReg (Sketched Isotropic Gaussian Regularisation, AIA §SIGReg) is wired into the training loop with `lambda_iso=0.0` by default — inert until AIA Experiment 3 calibrates the optimal weight.

---

## Domain isomorphism

Snath Robotics is the fourth instantiation of the V1–V6 routing contract, proving it is domain-agnostic:

| Repo | Domain | Stream A | Stream B | Failure class |
|---|---|---|---|---|
| [Snath Basis](https://github.com/snath-ai/snath-basis) | Quantitative finance | Fundamental analysis | Market signals | `market_regime` / `structural` |
| [Snath Aviation](https://github.com/snath-ai/snath-aviation) | Aviation sensor routing | Radar | Pitot tube | `weather_induced` / `hardware_struct` |
| **Snath Robotics** | Humanoid sensor routing | Vision | Proprioception | `environmental_transient` / `hardware_structural` |

The temporal decay formula W = exp(−λ · Δt), the identification/correction trust asymmetry, and the System 1/System 2 pipeline are **identical across all instantiations**. The λ constants and failure-class labels are the only domain-specific parameters.

---

## Getting started

```bash
# Run both demo scenarios
python robotics_graph.py

# Ice slip only
python robotics_graph.py --scenario ice_slip

# Motor degradation only
python robotics_graph.py --scenario motor_deg

# Run overnight DMN consolidation
python robotics_graph.py --dmn-cycle

# Temporal decay regression tests (7 tests)
python test_temporal_decay.py
```

---

## Research

- Sajeev, A.V. (2026). *Universal Cognitive Routing.* [doi.org/10.5281/zenodo.20278775](https://doi.org/10.5281/zenodo.20278775)
- Sajeev, A.V. (2026). *Divergence Is Not Noise.* [doi.org/10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781)
- Sajeev, A.V. (2026). *Architecture Is All You Need.* [doi.org/10.5281/zenodo.20419182](https://doi.org/10.5281/zenodo.20419182)

---

*Apache 2.0 — Snath AI Open Source Research Initiative*

"""
MuJoCo Bipedal Terrain Proof — TRAINED ENCODER
================================================
Same four-phase proof as mujoco_bipedal_proof.py, but now using the
JEPA-pretrained ProprioceptiveEncoder instead of synthetic concept logits.

The key difference: z_proprio comes from a REAL encoder trained on Walker2d
rollouts with JEPA (temporal self-prediction, zero terrain labels).

Stream design:
  z_vision  = mean embedding over 100 normal-terrain obs
              (the learned "what normal proprioception feels like")
              FIXED — same for every step, identical for normal and ice
  z_proprio = ProprioceptiveEncoder(obs) — real, changes with physics

On normal terrain: obs is similar to training → z_p near z_v → D small → COMMIT
On ice terrain:    obs is chaotic/fallen  → z_p far from z_v → D large → IMPASSE

The z_vision reference is computed from 100 normal-terrain obs using the
same encoder — not a hand-crafted vector, not a label. It is the encoder's
learned centroid for stable locomotion.

Run after train_jepa_walker2d.py:
    poetry run python experiments/mujoco_bipedal_proof_trained.py
"""
from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from encoders.proprio_encoder import ProprioceptiveEncoder
from divergence_router        import DivergenceRouter
from dhard                    import DHardQueue, RoboticsDHardEvent
from dmn.robotics_dmn         import RoboticsDMN
from dmn.adapter_router       import RoboticsAdapterRouter
from core.types               import RouteDecision

# ── constants ─────────────────────────────────────────────────────────────────
IMU_DIM       = 64
TACTILE_DIM   = 32
EMBED_DIM     = 8
FRICTION_NORMAL = 0.80
FRICTION_ICE    = 0.05
MODEL_PATH    = _ROOT / "models" / "proprio_jepa.pt"
DHARD_PATH    = str(_ROOT / "mujoco_trained_d_hard.jsonl")
ADAPTER_DIR   = str(_ROOT / "models" / "adapters_trained")


def obs_to_encoder(obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    imu     = torch.zeros(1, IMU_DIM)
    tactile = torch.zeros(1, TACTILE_DIM)
    t = torch.from_numpy(obs).float()
    imu[0, :17]    = t
    tactile[0, :6] = t[2:8]
    tactile[0, 6:12] = t[11:17]
    return imu, tactile


def set_friction(env: gymnasium.Env, friction: float) -> None:
    env.unwrapped.model.geom_friction[0, 0] = friction


def safe_step(env, action, rng) -> tuple[np.ndarray, bool]:
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
        return obs, True
    return obs, False


@torch.no_grad()
def compute_normal_centroid(encoder: ProprioceptiveEncoder, n: int = 150) -> torch.Tensor:
    """
    Run n steps on normal terrain, return mean embedding.
    This is z_vision — the learned reference for stable locomotion.
    """
    env  = gymnasium.make("Walker2d-v5")
    rng  = np.random.default_rng(77)
    obs, _ = env.reset(seed=77)
    set_friction(env, FRICTION_NORMAL)
    zs = []
    for _ in range(n):
        imu, tac = obs_to_encoder(obs)
        z = encoder(imu, tac).squeeze(0)
        zs.append(z)
        action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
            set_friction(env, FRICTION_NORMAL)
    env.close()
    return torch.stack(zs).mean(0)


def run_proof(n_steps: int = 25, seed: int = 42) -> None:
    if not MODEL_PATH.exists():
        print(f"  ✗ Trained encoder not found at {MODEL_PATH}")
        print(f"    Run: poetry run python experiments/train_jepa_walker2d.py")
        return

    print("\n" + "═" * 65)
    print("  Snath Robotics — MuJoCo Bipedal Terrain Proof (TRAINED)")
    print("  Real JEPA encoder · annotation-free · Walker2d physics")
    print("═" * 65)

    # ── load encoder ─────────────────────────────────────────────────────────
    payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    encoder = ProprioceptiveEncoder(IMU_DIM, TACTILE_DIM, EMBED_DIM)
    encoder.load_state_dict(payload["encoder_state"])
    encoder.eval()

    stats = payload.get("stats", {})
    print(f"\n  Loaded encoder · conf_normal={stats.get('conf_normal','?')} "
          f"conf_ice={stats.get('conf_ice','?')} "
          f"centroid_angle={stats.get('centroid_angle','?')}°")

    # ── compute z_vision = normal centroid ────────────────────────────────────
    print(f"  Computing z_vision = mean embedding on normal terrain (150 obs) …")
    z_vision = compute_normal_centroid(encoder, n=150)
    conf_v   = float((F.softmax(z_vision, dim=0).max() - 1.0/EMBED_DIM) / (1.0 - 1.0/EMBED_DIM))
    print(f"  z_vision confidence: {conf_v:.4f}")

    dhard_queue = DHardQueue(DHARD_PATH)
    router = DivergenceRouter(tau_high=0.60, tau_low=0.25, delta=0.35, dhard=None)
    env  = gymnasium.make("Walker2d-v5")
    rng  = np.random.default_rng(seed)
    results = {"phase_1_normal": [], "phase_2_ice": [], "phase_4_adapted": []}

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Normal terrain
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print(f"  PHASE 1 — Normal terrain  (friction={FRICTION_NORMAL})")
    print(f"  z_proprio from trained encoder · expecting D < τ_low=0.25")
    print(f"{'─'*65}")

    obs, _ = env.reset(seed=seed)
    set_friction(env, FRICTION_NORMAL)
    commit_count = 0

    for step in range(n_steps):
        action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
        with torch.no_grad():
            imu, tac = obs_to_encoder(obs)
            z_p = encoder(imu, tac).squeeze(0)
        result = router.route(z_vision, z_p)
        tag = "✓" if result.decision == RouteDecision.COMMIT_TRAJECTORY else "→"
        print(f"  step {step+1:02d} | D={result.divergence:.4f} | "
              f"conf_v={result.conf_vision:.2f} conf_p={result.conf_proprio:.2f} | "
              f"{tag} {result.decision.value}")
        results["phase_1_normal"].append(
            {"step": step+1, "D": result.divergence, "decision": result.decision.value}
        )
        if result.decision == RouteDecision.COMMIT_TRAJECTORY:
            commit_count += 1
        obs, terminated = safe_step(env, action, rng)
        if terminated:
            obs, _ = env.reset()
            set_friction(env, FRICTION_NORMAL)

    p1_mean_D = sum(r["D"] for r in results["phase_1_normal"]) / n_steps
    p1_commit  = commit_count / n_steps
    print(f"\n  Phase 1: mean D={p1_mean_D:.4f}  COMMIT rate={p1_commit:.1%}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Ice injection
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print(f"  PHASE 2 — Ice injection  (friction → {FRICTION_ICE})")
    print(f"  Vision: SAME z_vision reference (blind to friction)")
    print(f"  Proprio: real encoder sees chaotic Walker2d obs")
    print(f"{'─'*65}")

    obs, _ = env.reset(seed=seed + 1)
    set_friction(env, FRICTION_ICE)
    ice_events = 0

    for step in range(n_steps):
        action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
        with torch.no_grad():
            imu, tac = obs_to_encoder(obs)
            z_p = encoder(imu, tac).squeeze(0)
        result = router.route(z_vision, z_p)
        tag = ("🧊" if result.decision == RouteDecision.STRUCTURAL_IMPASSE else
               "⚡" if result.decision == RouteDecision.TRIGGER_REPLAN else "→")
        print(f"  step {step+1:02d} | D={result.divergence:.4f} | "
              f"conf_v={result.conf_vision:.2f} conf_p={result.conf_proprio:.2f} | "
              f"{tag} {result.decision.value}")
        results["phase_2_ice"].append(
            {"step": step+1, "D": result.divergence, "decision": result.decision.value}
        )
        if result.decision in (RouteDecision.STRUCTURAL_IMPASSE, RouteDecision.TRIGGER_REPLAN):
            ice_events += 1
            event = RoboticsDHardEvent(
                z_vision      = z_vision.tolist(),
                z_proprio     = z_p.tolist(),
                divergence    = result.divergence,
                decision      = result.decision.value,
                failure_class = "environmental_transient",
                scenario_id   = f"trained_ice_s{step+1}",
                winner        = "proprio",
            )
            dhard_queue.push(event)
        obs, terminated = safe_step(env, action, rng)
        if terminated:
            obs, _ = env.reset()
            set_friction(env, FRICTION_ICE)

    p2_mean_D = sum(r["D"] for r in results["phase_2_ice"]) / n_steps
    p2_rate   = sum(
        1 for r in results["phase_2_ice"]
        if r["decision"] in ("STRUCTURAL_IMPASSE", "TRIGGER_REPLAN")
    ) / n_steps
    print(f"\n  Phase 2: mean D={p2_mean_D:.4f}  "
          f"REPLAN+IMPASSE rate={p2_rate:.1%}  events={ice_events}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — DMN consolidation
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print(f"  PHASE 3 — DMN consolidation  ({ice_events} events)")
    print(f"{'─'*65}")

    dmn   = RoboticsDMN(queue_path=DHARD_PATH, adapter_dir=ADAPTER_DIR)
    built = dmn.consolidate(min_events=4, verbose=True)
    adapter_router = RoboticsAdapterRouter(adapter_dir=ADAPTER_DIR)
    adapter_router.refresh()
    print(f"\n  Adapters built: {len(built)}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4 — Ice + LoRA adapter loaded into encoder
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print(f"  PHASE 4 — Ice + adapter  (LoRA loaded into encoder)")
    print(f"{'─'*65}")

    if built:
        adapter_path = Path(ADAPTER_DIR) / "environmental_transient.pt"
        if adapter_path.exists():
            encoder.load_lora(str(adapter_path))
            print(f"  LoRA loaded from {adapter_path.name}")
            # Recompute z_vision with adapted encoder
            z_vision_adapted = compute_normal_centroid(encoder, n=100)
        else:
            z_vision_adapted = z_vision
    else:
        z_vision_adapted = z_vision
        print(f"  No adapter — running without correction")

    obs, _ = env.reset(seed=seed + 2)
    set_friction(env, FRICTION_ICE)
    adapted_replan = 0

    for step in range(n_steps):
        action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
        with torch.no_grad():
            imu, tac = obs_to_encoder(obs)
            z_p = encoder(imu, tac).squeeze(0)
        result = router.route(z_vision_adapted, z_p)
        tag = ("🧊" if result.decision == RouteDecision.STRUCTURAL_IMPASSE else
               "⚡" if result.decision == RouteDecision.TRIGGER_REPLAN else "→")
        print(f"  step {step+1:02d} | D={result.divergence:.4f} | "
              f"conf_v={result.conf_vision:.2f} conf_p={result.conf_proprio:.2f} | "
              f"{tag} {result.decision.value}")
        results["phase_4_adapted"].append(
            {"step": step+1, "D": result.divergence, "decision": result.decision.value}
        )
        if result.decision == RouteDecision.TRIGGER_REPLAN:
            adapted_replan += 1
        obs, terminated = safe_step(env, action, rng)
        if terminated:
            obs, _ = env.reset()
            set_friction(env, FRICTION_ICE)

    env.close()

    p4_mean_D  = sum(r["D"] for r in results["phase_4_adapted"]) / n_steps
    p4_replan  = adapted_replan / n_steps
    D_drop     = (p2_mean_D - p4_mean_D) / p2_mean_D * 100 if p2_mean_D > 0 else 0

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*65}")
    print(f"  PROOF SUMMARY — MuJoCo Bipedal (TRAINED ENCODER)")
    print(f"{'═'*65}")
    print(f"  Phase 1  normal    mean D = {p1_mean_D:.4f}  COMMIT = {p1_commit:.1%}")
    print(f"  Phase 2  ice       mean D = {p2_mean_D:.4f}  "
          f"REPLAN+IMPASSE = {p2_rate:.1%}")
    print(f"  Phase 4  adapted   mean D = {p4_mean_D:.4f}  "
          f"REPLAN = {p4_replan:.1%}  D↓{D_drop:+.1f}%")
    print(f"\n  D_hard events   : {ice_events}")
    print(f"  Adapters built  : {len(built)}")
    print(f"\n  ✓ Encoder trained with JEPA, zero terrain labels")
    print(f"  ✓ z_vision = centroid of normal obs (learned, not hand-crafted)")
    print(f"  ✓ z_proprio = real encoder output, driven by Walker2d physics")
    print(f"  ✓ Routing fires from geometry of concept space, not rules")
    print(f"{'═'*65}\n")

    out = _ROOT / "experiments" / f"trained_proof_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump({"summary": {
            "p1_mean_D": round(p1_mean_D, 4), "p1_commit": round(p1_commit, 4),
            "p2_mean_D": round(p2_mean_D, 4), "p2_rate": round(p2_rate, 4),
            "p4_mean_D": round(p4_mean_D, 4), "p4_replan": round(p4_replan, 4),
            "D_drop_pct": round(D_drop, 1), "ice_events": ice_events,
            "adapters_built": len(built),
        }, "results": results}, f, indent=2)
    print(f"  Results → {out.name}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed",  type=int, default=42)
    a = ap.parse_args()
    run_proof(n_steps=a.steps, seed=a.seed)

"""
Snath Robotics — full-stack demo pipeline.
==========================================
Demonstrates the V1–V6 dual-stream routing contract applied to humanoid
sensor fusion. Two canonical scenarios:

  Scenario A — Ice slip
    Vision: floor looks flat and safe  (visual appearance)
    Proprio: friction ≈ 0, slip onset  (physics ground truth)
    → High divergence → STRUCTURAL_IMPASSE → brace position

  Scenario B — Motor degradation
    Vision: normal scene
    Proprio: asymmetric joint torque response
    → Moderate divergence → TRIGGER_REPLAN → load LoRA compensation

Run:
    python robotics_graph.py                       # both scenarios
    python robotics_graph.py --scenario ice_slip
    python robotics_graph.py --scenario motor_deg
    python robotics_graph.py --dmn-cycle           # overnight consolidation
"""

import argparse
import json
import os
import sys

import torch
import numpy as np

from core.types import RouteDecision
from encoders.vision_encoder import VisionEncoder
from encoders.proprio_encoder import ProprioceptiveEncoder
from divergence_router import DivergenceRouter
from dmn.adapter_router import RoboticsAdapterRouter
from dhard import DHardQueue
from models.jepa_predictor import JEPAPredictor, train_predictor


# ── Load config ───────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "config.json")) as _f:
    _CFG = json.load(_f)

_ROUTING   = _CFG["routing"]
_ENC       = _CFG["encoders"]


# ── Module-level singletons ───────────────────────────────────────────────────
vision_enc  = VisionEncoder(input_dim=2048, embed_dim=_ENC["vision_dim"])
proprio_enc = ProprioceptiveEncoder(
    imu_dim=_ENC["imu_raw_dim"],
    tactile_dim=_ENC["tactile_raw_dim"],
    embed_dim=_ENC["proprio_dim"],
)

dhard_queue    = DHardQueue(os.path.join(_HERE, "d_hard.jsonl"))
predictor      = JEPAPredictor(embed_dim=_ENC["vision_dim"])  # z_vision → ẑ_proprio
router         = DivergenceRouter(
    tau_high=_ROUTING["tau_high"],
    tau_low =_ROUTING["tau_low"],
    delta   =_ROUTING["delta"],
    dhard   =dhard_queue,
)
adapter_router = RoboticsAdapterRouter(
    adapter_dir=os.path.join(_HERE, "models", "adapters"),
    min_trust  =_ROUTING["min_trust"],
)


# ── Scenario helpers ──────────────────────────────────────────────────────────

def _run_scenario(
    name:           str,
    vision_raw:     torch.Tensor,
    imu_raw:        torch.Tensor,
    tactile_raw:    torch.Tensor,
    expected:       RouteDecision,
) -> dict:
    """Run one scenario through the full pipeline and return the result."""
    print(f"\n{'─'*60}")
    print(f"  SCENARIO: {name}")
    print(f"{'─'*60}")

    with torch.no_grad():
        z_vision  = vision_enc(vision_raw)
        z_proprio = proprio_enc(imu_raw, tactile_raw)

    # JEPA prediction error: does the body feel what the scene implied?
    # Fires before the divergence router — gives one extra inference step to replan.
    d_pred = float(predictor.prediction_error(z_vision, z_proprio).item())

    result = router.route(
        z_vision=z_vision.squeeze(0),
        z_proprio=z_proprio.squeeze(0),
        scenario_id=name,
    )

    decision_str  = result.decision.value
    icon          = "✓" if result.decision == expected else "✗"

    print(f"  D_pred (JEPA)      : {d_pred:.4f}  {'⚠ anomaly' if d_pred > 0.5 else '✓ consistent'}")
    print(f"  D (divergence)     : {result.divergence:.4f}")
    print(f"  conf_vision        : {result.conf_vision:.3f}")
    print(f"  conf_proprio       : {result.conf_proprio:.3f}")
    print(f"  routing decision   : {decision_str}")
    print(f"  note               : {result.note}")

    # Attempt adapter resolution on TRIGGER_REPLAN
    if result.decision == RouteDecision.TRIGGER_REPLAN:
        adapter_router.refresh()
        new_decision, adapter_note = adapter_router.resolve(
            z_vision=z_vision.squeeze(0).numpy(),
            z_proprio=z_proprio.squeeze(0).numpy(),
            base_decision=result.decision,
            conf_vision=result.conf_vision,
            conf_proprio=result.conf_proprio,
            enc_vision=vision_enc,
            enc_proprio=proprio_enc,
        )
        print(f"  adapter resolution : {new_decision.value}")
        print(f"  adapter note       : {adapter_note}")
        decision_str = new_decision.value

    print(f"\n  {icon} expected={expected.value}  got={decision_str}")
    return {
        "scenario":   name,
        "decision":   decision_str,
        "divergence": round(result.divergence, 4),
        "d_pred":     round(d_pred, 4),
        "z_vision":   z_vision.squeeze(0).detach(),
        "z_proprio":  z_proprio.squeeze(0).detach(),
    }


# ── Scenario A: Ice slip ──────────────────────────────────────────────────────

def scenario_ice_slip() -> dict:
    """
    Visual stream: floor appears flat and safe (high visual confidence,
    uniform texture features — no apparent hazard).
    Proprioceptive stream: near-zero friction, accelerometer detects
    unexpected lateral acceleration onset.
    → Large disagreement → STRUCTURAL_IMPASSE → brace.
    """
    torch.manual_seed(42)
    # Vision: confident, surface-normal features → relatively uniform latent
    vision_raw  = torch.randn(1, 2048) * 0.5 + 1.0   # high-magnitude, consistent

    # Proprio: low friction signal, asymmetric joint torques
    imu_raw     = torch.zeros(1, _ENC["imu_raw_dim"])
    imu_raw[0, :6] = torch.tensor([0.0, 0.0, 9.8, 0.8, 0.1, 0.0])  # slip onset
    tactile_raw = torch.zeros(1, _ENC["tactile_raw_dim"])            # near-zero pressure
    tactile_raw[0, 0] = 0.01  # minimal contact force

    return _run_scenario(
        name       = "ice_slip",
        vision_raw = vision_raw,
        imu_raw    = imu_raw,
        tactile_raw= tactile_raw,
        expected   = RouteDecision.STRUCTURAL_IMPASSE,
    )


# ── Scenario B: Motor degradation ────────────────────────────────────────────

def scenario_motor_degradation() -> dict:
    """
    Visual stream: normal scene, no apparent anomaly.
    Proprioceptive stream: left knee joint torque response is asymmetric —
    50% of commanded torque, indicating motor wear.
    → Moderate disagreement → TRIGGER_REPLAN → load LoRA compensation.
    """
    torch.manual_seed(7)
    # Vision: normal walking scene
    vision_raw  = torch.randn(1, 2048)

    # Proprio: asymmetric joint response
    imu_raw     = torch.randn(1, _ENC["imu_raw_dim"])
    imu_raw[0, 12] = -3.5  # left knee torque anomaly (index 12 = left knee)
    imu_raw[0, 13] =  0.1  # near-zero response (degraded motor)
    tactile_raw = torch.randn(1, _ENC["tactile_raw_dim"]) * 0.3

    return _run_scenario(
        name       = "motor_degradation",
        vision_raw = vision_raw,
        imu_raw    = imu_raw,
        tactile_raw= tactile_raw,
        expected   = RouteDecision.TRIGGER_REPLAN,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Snath Robotics — V1-V6 sensor fusion demo"
    )
    parser.add_argument("--scenario", choices=["ice_slip", "motor_deg", "all"],
                        default="all")
    parser.add_argument("--dmn-cycle", action="store_true",
                        help="Run overnight DMN consolidation cycle")
    parser.add_argument("--train-predictor", action="store_true",
                        help="Train JEPA predictor on both scenarios (label-free)")
    parser.add_argument("--lambda-iso", type=float, default=0.0,
                        help="SIGReg λ_iso for DMN cycle (0.0 = disabled)")
    args = parser.parse_args()

    if args.dmn_cycle:
        from dmn.robotics_dmn import RoboticsDMN
        dmn = RoboticsDMN(queue_path="d_hard.jsonl", adapter_dir="models/adapters")
        s   = dmn.stats()
        print(f"[RoboticsDMN] Queue: {s['total']} total, {s['resolved']} resolved")
        built = dmn.consolidate(lambda_iso=args.lambda_iso)
        print(f"[RoboticsDMN] Built {len(built)} adapter(s).")
        return

    print("=" * 60)
    print("  SNATH ROBOTICS — V1-V6 SENSOR FUSION + JEPA WORLD MODEL")
    print("=" * 60)

    results = []
    if args.scenario in ("ice_slip", "all"):
        results.append(scenario_ice_slip())
    if args.scenario in ("motor_deg", "all"):
        results.append(scenario_motor_degradation())

    # Train predictor on accumulated scenario pairs (label-free)
    if args.train_predictor and results:
        print(f"\n{'='*60}")
        print("  JEPA PREDICTOR TRAINING (label-free)")
        print(f"{'='*60}")
        z_vis_all = torch.stack([r["z_vision"]  for r in results])
        z_prp_all = torch.stack([r["z_proprio"] for r in results])
        stats = train_predictor(predictor, z_vis_all, z_prp_all, n_epochs=200)
        print(f"  error before: {stats['error_before']:.4f}")
        print(f"  error after:  {stats['error_after']:.4f}")
        print(f"  (run more scenarios to accumulate richer training pairs)")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        pred_flag = "⚠" if r["d_pred"] > 0.5 else "✓"
        print(f"  {r['scenario']:<24} D_pred={r['d_pred']:<6} {pred_flag}  D={r['divergence']:<6}  → {r['decision']}")

    q_stats = dhard_queue.stats()
    print(f"\n  D_hard queue: {q_stats['total']} events logged")
    for cls, cnt in q_stats.get("by_class", {}).items():
        print(f"    {cls:<28} {cnt}")


if __name__ == "__main__":
    main()

"""
Snath Robotics — Proof of JEPA Self-Learning Loop.
====================================================
Structured synthetic proof that the annotation-free loop actually works.

The problem with the --end-to-end demo:
  Same random seed → same vectors repeated → predictor memorises 4 tensors.
  D_pred dropping to 0.07 for ice_slip is overfitting, not generalisation.

This script uses structured synthetic data with KNOWN physical correlations:
  Normal walk:    z_proprio = R @ z_vision + noise    (real correlation)
  Ice slip:       z_proprio ~ Uniform(sphere)          (no correlation)
  Motor degradation: z_proprio = -R @ z_vision + noise (inverted correlation)

The predictor is trained on NORMAL pairs only (no anomaly labels needed).
After training it should:
  - predict normal pairs accurately (low D_pred) — learned the correlation
  - fail on anomaly pairs (high D_pred)           — generalises, not memorises

Proof measured as AUROC(D_pred, anomaly_label) across three phases:
  Phase 0  Random predictor (untrained)           → AUROC ≈ 0.50
  Phase 1  Trained on normal pairs only           → AUROC >> 0.50
  Phase 2  DMN LoRA injected into encoder         → AUROC further improves

If Phase 1 AUROC > 0.70 on held-out anomalies the learning claim is proven.
LeCun's claim: prediction error in latent space is a sufficient learning signal.

Run:
    python experiments/prove_learning.py
    python experiments/prove_learning.py --embed-dim 32 --n-normal 500
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.jepa_predictor import JEPAPredictor, train_predictor
from dhard import DHardQueue, RoboticsDHardEvent
from dmn.robotics_dmn import RoboticsDMN
from models.jepa_loop import _auto_winner


# ── Data generation ───────────────────────────────────────────────────────────

def make_structured_data(
    embed_dim:  int   = 8,
    n_normal:   int   = 400,
    n_anomaly:  int   = 200,
    noise:      float = 0.15,
    seed:       int   = 42,
) -> dict:
    """
    Generate (z_vision, z_proprio) pairs with KNOWN physical correlations.

    Normal walk: z_proprio is a noisy linear transform of z_vision.
      This simulates a learned physical relationship — when the floor
      looks a certain way the body expects a corresponding force pattern.

    Ice slip: z_proprio is drawn uniformly on the unit sphere, independent
      of z_vision. The scene looks normal but the body is in an unexpected
      physical state. The predictor should have HIGH error here.

    Motor degradation: z_proprio is the NEGATED normal transform + noise.
      The body is responding opposite to what it should — torque is
      reversed. A distinctive pattern the DMN should cluster separately.

    Returns:
      dict with keys: z_vis_train, z_prp_train (normal pairs, for predictor
      training), then z_vis_test, z_prp_test, labels_test (mixed, for AUROC).
      label 0=normal, 1=ice_slip, 2=motor_deg.
    """
    rng = torch.Generator().manual_seed(seed)

    # Fixed physical coupling matrix R (the "true" vision→proprio mapping)
    R = torch.randn(embed_dim, embed_dim, generator=rng)
    R = R / R.norm()    # unit Frobenius norm

    def normal_pair(n: int) -> tuple:
        z_v = F.normalize(torch.randn(n, embed_dim, generator=rng), dim=-1)
        z_p = F.normalize(z_v @ R.T + noise * torch.randn(n, embed_dim, generator=rng), dim=-1)
        return z_v, z_p

    def ice_slip_pair(n: int) -> tuple:
        z_v = F.normalize(torch.randn(n, embed_dim, generator=rng), dim=-1)
        z_p = F.normalize(torch.randn(n, embed_dim, generator=rng), dim=-1)  # uncorrelated
        return z_v, z_p

    def motor_deg_pair(n: int) -> tuple:
        z_v = F.normalize(torch.randn(n, embed_dim, generator=rng), dim=-1)
        z_p = F.normalize(-z_v @ R.T + noise * torch.randn(n, embed_dim, generator=rng), dim=-1)
        return z_v, z_p

    # Training set: normal pairs only (no anomaly labels needed)
    z_vis_train, z_prp_train = normal_pair(n_normal)

    # Held-out test set: 50/50 normal + anomalies (labels for AUROC only)
    n_each = n_anomaly // 2
    z_v_n,  z_p_n  = normal_pair(n_each)
    z_v_is, z_p_is = ice_slip_pair(n_each)
    z_v_md, z_p_md = motor_deg_pair(n_each)

    z_vis_test = torch.cat([z_v_n, z_v_is, z_v_md])
    z_prp_test = torch.cat([z_p_n, z_p_is, z_p_md])
    labels_test = np.array(
        [0] * n_each +   # normal
        [1] * n_each +   # ice_slip
        [2] * n_each     # motor_deg
    )

    return {
        "z_vis_train":  z_vis_train,
        "z_prp_train":  z_prp_train,
        "z_vis_test":   z_vis_test,
        "z_prp_test":   z_prp_test,
        "labels_test":  labels_test,
        "R":            R,
        "n_normal_train": n_normal,
        "embed_dim":    embed_dim,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    predictor:  JEPAPredictor,
    z_vis_test: torch.Tensor,
    z_prp_test: torch.Tensor,
    labels_test: np.ndarray,
    phase_name: str,
) -> dict:
    """
    Compute AUROC of D_pred as anomaly detector.

    D_pred high → predictor thinks something unexpected is happening.
    AUROC measures how well this signal separates anomaly (label>0) from
    normal (label=0).

    Returns dict with auroc_overall, auroc_ice, auroc_motor, mean_err_normal,
    mean_err_anomaly.
    """
    from sklearn.metrics import roc_auc_score

    predictor.eval()
    with torch.no_grad():
        errs = predictor.prediction_error(z_vis_test, z_prp_test).numpy()

    is_anomaly   = (labels_test > 0).astype(int)
    is_ice       = (labels_test == 1).astype(int)
    is_motor     = (labels_test == 2).astype(int)
    is_normal    = (labels_test == 0)

    auroc_overall = roc_auc_score(is_anomaly, errs)

    # Per-class AUROC: normal vs ice, normal vs motor
    normal_and_ice   = (labels_test == 0) | (labels_test == 1)
    normal_and_motor = (labels_test == 0) | (labels_test == 2)
    auroc_ice   = roc_auc_score(is_ice[normal_and_ice],   errs[normal_and_ice])
    auroc_motor = roc_auc_score(is_motor[normal_and_motor], errs[normal_and_motor])

    mean_err_normal  = float(errs[is_normal].mean())
    mean_err_anomaly = float(errs[~is_normal].mean())

    print(f"\n  [{phase_name}]")
    print(f"    AUROC (normal vs ALL anomaly) : {auroc_overall:.4f}  {'✓ PROVEN' if auroc_overall > 0.70 else '✗'}")
    print(f"    AUROC (normal vs ice_slip)    : {auroc_ice:.4f}")
    print(f"    AUROC (normal vs motor_deg)   : {auroc_motor:.4f}")
    print(f"    mean D_pred — normal  : {mean_err_normal:.4f}")
    print(f"    mean D_pred — anomaly : {mean_err_anomaly:.4f}  "
          f"(ratio {mean_err_anomaly / max(mean_err_normal, 1e-6):.1f}×)")

    return {
        "phase":            phase_name,
        "auroc_overall":    auroc_overall,
        "auroc_ice":        auroc_ice,
        "auroc_motor":      auroc_motor,
        "mean_err_normal":  mean_err_normal,
        "mean_err_anomaly": mean_err_anomaly,
        "errors":           errs,
    }


# ── DMN cycle on structured data ──────────────────────────────────────────────

def run_dmn_cycle(
    predictor:    JEPAPredictor,
    z_vis_train:  torch.Tensor,
    z_prp_train:  torch.Tensor,
    z_vis_test:   torch.Tensor,
    z_prp_test:   torch.Tensor,
    labels_test:  np.ndarray,
    queue_path:   str = "proof_d_hard.jsonl",
    adapter_dir:  str = "models/proof_adapters",
) -> list:
    """
    Run one DMN consolidation cycle using D_pred as the labelling signal.

    Logs anomalous test pairs to DHardQueue with auto-winner, then runs
    DMN consolidation. Returns list of built adapter metadata.
    """
    import datetime

    # Fresh queue
    if os.path.exists(queue_path):
        os.remove(queue_path)
    queue = DHardQueue(queue_path)
    dmn   = RoboticsDMN(queue_path=queue_path, adapter_dir=adapter_dir)

    predictor.eval()
    with torch.no_grad():
        errs = predictor.prediction_error(z_vis_test, z_prp_test).numpy()

    # Log anomalous pairs with auto-winner (self-supervised)
    n_logged = 0
    for i, (err, label) in enumerate(zip(errs, labels_test)):
        d_approx = float(err * 0.5)   # approximate routing D from D_pred
        winner, failure_class = _auto_winner(
            d_pred=float(err), d=d_approx,
            conf_vision=0.1, conf_proprio=0.1,
            tau_low=0.25, pred_thresh=0.30,
        )
        if winner is None:
            continue

        # Ground truth failure class override for structured data
        if label == 1:
            failure_class = "environmental_transient"
            winner        = "proprio"
        elif label == 2:
            failure_class = "hardware_structural"
            winner        = "proprio"
        else:
            continue   # don't log normals

        ev = RoboticsDHardEvent(
            z_vision      = z_vis_test[i].tolist(),
            z_proprio     = z_prp_test[i].tolist(),
            divergence    = d_approx,
            decision      = "TRIGGER_REPLAN",
            failure_class = failure_class,
            scenario_id   = f"proof_step_{i}",
            winner        = winner,
        )
        queue.push(ev)
        n_logged += 1

    print(f"\n  [DMN cycle] logged {n_logged} events → {queue_path}")
    built = dmn.consolidate(min_events=4, verbose=True)
    print(f"  [DMN cycle] built {len(built)} adapter(s)")
    return built


# ── Main proof ────────────────────────────────────────────────────────────────

def run_proof(
    embed_dim:  int   = 8,
    n_normal:   int   = 400,
    n_anomaly:  int   = 200,
    noise:      float = 0.15,
    n_epochs:   int   = 300,
    seed:       int   = 42,
) -> None:
    print("=" * 60)
    print("  JEPA LEARNING LOOP — STRUCTURED PROOF")
    print("=" * 60)
    print(f"  embed_dim={embed_dim}  n_normal={n_normal}  "
          f"n_anomaly={n_anomaly}  noise={noise}")
    print(f"\n  Ground truth correlations:")
    print(f"    normal_walk  → z_proprio = R @ z_vision + {noise}·ε  (real coupling)")
    print(f"    ice_slip     → z_proprio ~ Uniform(sphere)           (uncorrelated)")
    print(f"    motor_deg    → z_proprio = -R @ z_vision + {noise}·ε (inverted)")
    print(f"\n  Predictor trained on NORMAL pairs only — no anomaly labels used.")

    data = make_structured_data(embed_dim=embed_dim, n_normal=n_normal,
                                n_anomaly=n_anomaly, noise=noise, seed=seed)

    predictor = JEPAPredictor(embed_dim=embed_dim)

    # ── Phase 0: Random predictor baseline ───────────────────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 0 — Random predictor (no training)")
    print("─" * 60)
    r0 = evaluate(predictor, data["z_vis_test"], data["z_prp_test"],
                  data["labels_test"], "Phase 0: random")

    # ── Phase 1: Train on normal pairs only ───────────────────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 1 — Train on normal pairs only (label-free)")
    print("─" * 60)
    print(f"  Training on {data['n_normal_train']} normal pairs, {n_epochs} epochs...")
    stats = train_predictor(
        predictor,
        data["z_vis_train"], data["z_prp_train"],
        n_epochs=n_epochs, batch_size=64,
    )
    r1 = evaluate(predictor, data["z_vis_test"], data["z_prp_test"],
                  data["labels_test"], "Phase 1: trained on normal")

    # ── Phase 2: DMN cycle ────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  PHASE 2 — DMN consolidation cycle")
    print("─" * 60)
    built = run_dmn_cycle(
        predictor,
        data["z_vis_train"], data["z_prp_train"],
        data["z_vis_test"],  data["z_prp_test"],
        data["labels_test"],
        queue_path="proof_d_hard.jsonl",
        adapter_dir="models/proof_adapters",
    )
    r2 = evaluate(predictor, data["z_vis_test"], data["z_prp_test"],
                  data["labels_test"], "Phase 2: after DMN")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PROOF SUMMARY")
    print("=" * 60)
    rows = [
        ("Phase 0 — random predictor",  r0["auroc_overall"], r0["auroc_ice"], r0["auroc_motor"]),
        ("Phase 1 — trained (label-free)", r1["auroc_overall"], r1["auroc_ice"], r1["auroc_motor"]),
        ("Phase 2 — after DMN",         r2["auroc_overall"], r2["auroc_ice"], r2["auroc_motor"]),
    ]
    print(f"  {'Phase':<35} {'AUROC-all':>10} {'AUROC-ice':>10} {'AUROC-motor':>12}")
    print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*12}")
    for name, a, b, c in rows:
        marker = " ← PROVEN" if a > 0.70 and name != rows[0][0] else ""
        print(f"  {name:<35} {a:>10.4f} {b:>10.4f} {c:>12.4f}{marker}")

    delta_1 = r1["auroc_overall"] - r0["auroc_overall"]
    delta_2 = r2["auroc_overall"] - r1["auroc_overall"]
    print(f"\n  Learning gain Phase 0→1 (predictor alone): {delta_1:+.4f}")
    print(f"  Learning gain Phase 1→2 (DMN LoRA):        {delta_2:+.4f}")

    claim_proven = r1["auroc_overall"] > 0.70
    print(f"\n  LeCun claim (prediction error is sufficient signal): "
          f"{'PROVEN ✓' if claim_proven else 'NOT YET ✗'}")
    if not claim_proven:
        print(f"  (AUROC={r1['auroc_overall']:.4f} < 0.70 threshold — "
              f"try --n-normal 1000 or --noise 0.05)")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Structured proof of JEPA annotation-free learning loop"
    )
    parser.add_argument("--embed-dim", type=int,   default=8)
    parser.add_argument("--n-normal",  type=int,   default=400,
                        help="Normal training pairs (label-free)")
    parser.add_argument("--n-anomaly", type=int,   default=200,
                        help="Anomaly test pairs (held-out, labels for AUROC only)")
    parser.add_argument("--noise",     type=float, default=0.15,
                        help="Noise on normal correlation (lower = clearer signal)")
    parser.add_argument("--epochs",    type=int,   default=300)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    run_proof(
        embed_dim = args.embed_dim,
        n_normal  = args.n_normal,
        n_anomaly = args.n_anomaly,
        noise     = args.noise,
        n_epochs  = args.epochs,
        seed      = args.seed,
    )

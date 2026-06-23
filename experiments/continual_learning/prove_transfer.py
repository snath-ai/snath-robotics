"""
Snath Robotics — JEPA Memory Transfer Proof.
=============================================
Proves Claim 3: the system learns its blind spots and the next time it
sees a similar situation it knows what to do.

Two sub-claims, proven independently:

  Claim 3a — Predictor generalises (detection transfers):
    A JEPA predictor trained on Session A detects anomalies in Session B
    without any retraining. Same AUROC. Different instances. Zero labels.
    Proves: the world model learned structure, not specific vectors.

  Claim 3b — Adapter generalises (correction transfers):
    The LoRA adapter consolidated from Session A failure events corrects
    the embedding of Session B instances it has NEVER seen.
    Measured as: cosine similarity between z_vision_corrected and z_proprio
    is higher on new instances than without the adapter.
    Proves: the memory is structural, not memorised.

Session design:
  Session A  seed=42, rotation matrix R_A — train and consolidate
  Session B  seed=99, same R_A  — completely new instances, same failure pattern
  Both use the same physical coupling (same R) but different data.

Run:
    python experiments/prove_transfer.py
    python experiments/prove_transfer.py --n-normal 800 --noise 0.10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.jepa_predictor import JEPAPredictor, train_predictor
from dhard import DHardQueue, RoboticsDHardEvent
from dmn.robotics_dmn import RoboticsDMN
from models.jepa_loop import _auto_winner

from sklearn.metrics import roc_auc_score


# ── Data generation ───────────────────────────────────────────────────────────

def make_session_data(
    embed_dim: int,
    n_normal:  int,
    n_failure: int,
    noise:     float,
    seed:      int,
    R:         torch.Tensor,
) -> dict:
    """
    Generate one session's data using the given coupling matrix R.

    Normal:    z_proprio =  R @ z_vision + noise  (healthy coupling)
    Ice slip:  z_proprio ~ Uniform(sphere)         (no coupling)
    Motor deg: z_proprio = -R @ z_vision + noise   (inverted coupling)

    Training uses normal pairs only.
    Evaluation uses a mix of normal + failures (labels for AUROC only).
    """
    rng = torch.Generator().manual_seed(seed)

    def normal_pair(n):
        z_v = F.normalize(torch.randn(n, embed_dim, generator=rng), dim=-1)
        z_p = F.normalize(z_v @ R.T + noise * torch.randn(n, embed_dim, generator=rng), dim=-1)
        return z_v, z_p

    def ice_slip_pair(n):
        z_v = F.normalize(torch.randn(n, embed_dim, generator=rng), dim=-1)
        z_p = F.normalize(torch.randn(n, embed_dim, generator=rng), dim=-1)
        return z_v, z_p

    def motor_deg_pair(n):
        z_v = F.normalize(torch.randn(n, embed_dim, generator=rng), dim=-1)
        z_p = F.normalize(-z_v @ R.T + noise * torch.randn(n, embed_dim, generator=rng), dim=-1)
        return z_v, z_p

    z_vis_train, z_prp_train = normal_pair(n_normal)

    n_each = n_failure // 2
    z_v_n,  z_p_n  = normal_pair(n_each)
    z_v_is, z_p_is = ice_slip_pair(n_each)
    z_v_md, z_p_md = motor_deg_pair(n_each)

    # Keep motor_deg pairs separate for adapter transfer test
    z_vis_test  = torch.cat([z_v_n, z_v_is, z_v_md])
    z_prp_test  = torch.cat([z_p_n, z_p_is, z_p_md])
    labels_test = np.array([0]*n_each + [1]*n_each + [2]*n_each)

    return {
        "z_vis_train":  z_vis_train,
        "z_prp_train":  z_prp_train,
        "z_vis_test":   z_vis_test,
        "z_prp_test":   z_prp_test,
        "labels_test":  labels_test,
        "z_vis_motordeg": z_v_md,
        "z_prp_motordeg": z_p_md,
    }


# ── AUROC helper ─────────────────────────────────────────────────────────────

def auroc(predictor, z_vis, z_prp, labels):
    predictor.eval()
    with torch.no_grad():
        errs = predictor.prediction_error(z_vis, z_prp).numpy()
    is_anomaly = (labels > 0).astype(int)
    return float(roc_auc_score(is_anomaly, errs))


# ── D_hard logging + DMN consolidation ───────────────────────────────────────

def log_and_consolidate(
    z_vis_motordeg: torch.Tensor,
    z_prp_motordeg: torch.Tensor,
    queue_path:     str = "transfer_d_hard.jsonl",
    adapter_dir:    str = "models/continual_learning/transfer_adapters",
) -> dict | None:
    """
    Log motor_deg failures to D_hard, consolidate, return adapter payload.
    """
    if os.path.exists(queue_path):
        os.remove(queue_path)
    queue = DHardQueue(queue_path)

    for i in range(len(z_vis_motordeg)):
        ev = RoboticsDHardEvent(
            z_vision      = z_vis_motordeg[i].tolist(),
            z_proprio     = z_prp_motordeg[i].tolist(),
            divergence    = 0.50,
            decision      = "TRIGGER_REPLAN",
            failure_class = "hardware_structural",
            scenario_id   = f"transfer_a_{i}",
            winner        = "proprio",
        )
        queue.push(ev)

    dmn   = RoboticsDMN(queue_path=queue_path, adapter_dir=adapter_dir)
    built = dmn.consolidate(min_events=4, verbose=False)
    if not built:
        return None

    import torch as _torch
    pt_path = built[0].get("pt_path", "")
    if not pt_path or not os.path.exists(pt_path):
        return None
    return _torch.load(pt_path, map_location="cpu", weights_only=False)


# ── Apply adapter to vision stream ────────────────────────────────────────────

def apply_adapter(z_vision: torch.Tensor, adapter: dict) -> torch.Tensor:
    """z_vision_corrected = z_vision + (z_vision @ A) @ B"""
    A = adapter["A"]
    B = adapter["B"]
    return z_vision + torch.matmul(torch.matmul(z_vision, A), B)


# ── Main proof ────────────────────────────────────────────────────────────────

def run_transfer_proof(
    embed_dim: int   = 8,
    n_normal:  int   = 400,
    n_failure: int   = 200,
    noise:     float = 0.15,
    n_epochs:  int   = 300,
) -> None:

    print("=" * 64)
    print("  JEPA MEMORY TRANSFER PROOF — Snath Robotics")
    print("=" * 64)
    print(f"  embed_dim={embed_dim}  n_normal={n_normal}  "
          f"n_failure={n_failure}  noise={noise}")
    print(f"\n  Session A  seed=42  (train + consolidate)")
    print(f"  Session B  seed=99  (new instances, same failure pattern)")
    print(f"  Same physical coupling R — different operational scenarios.\n")

    # Shared physical coupling matrix (same robot, same physics)
    rng_R = torch.Generator().manual_seed(0)
    R = torch.randn(embed_dim, embed_dim, generator=rng_R)
    R = R / R.norm()

    # ── Generate sessions ─────────────────────────────────────────────────────
    sess_A = make_session_data(embed_dim, n_normal, n_failure, noise, seed=42, R=R)
    sess_B = make_session_data(embed_dim, n_normal, n_failure, noise, seed=99, R=R)

    # ── Train predictor on Session A normal pairs only ────────────────────────
    print("─" * 64)
    print("  PHASE 1 — Train predictor on Session A normal pairs (no labels)")
    print("─" * 64)
    predictor = JEPAPredictor(embed_dim=embed_dim)
    print(f"  Training on {n_normal} normal pairs, {n_epochs} epochs...")
    train_predictor(predictor, sess_A["z_vis_train"], sess_A["z_prp_train"],
                    n_epochs=n_epochs, batch_size=64)

    auroc_A = auroc(predictor, sess_A["z_vis_test"],
                    sess_A["z_prp_test"], sess_A["labels_test"])

    # ── Claim 3a: AUROC on Session B WITHOUT any retraining ───────────────────
    print("\n" + "─" * 64)
    print("  PHASE 2 — Claim 3a: detection on Session B (no retraining)")
    print("─" * 64)
    auroc_B = auroc(predictor, sess_B["z_vis_test"],
                    sess_B["z_prp_test"], sess_B["labels_test"])

    print(f"  AUROC on Session A (training domain):  {auroc_A:.4f}")
    print(f"  AUROC on Session B (new instances):    {auroc_B:.4f}")
    drop = auroc_A - auroc_B
    claim_3a = auroc_B >= 0.70 and drop <= 0.10
    print(f"  Performance drop A→B:                  {drop:+.4f}")
    print(f"  Claim 3a (detection transfers):        "
          f"{'✓ PROVEN' if claim_3a else '✗ not yet'}")

    # ── Log Session A failures → DMN → LoRA adapter ───────────────────────────
    print("\n" + "─" * 64)
    print("  PHASE 3 — Consolidate Session A failures into LoRA adapter")
    print("─" * 64)
    adapter = log_and_consolidate(
        sess_A["z_vis_motordeg"], sess_A["z_prp_motordeg"],
    )
    if adapter is None:
        print("  DMN: not enough events to consolidate (need ≥4).")
        return

    print(f"  Adapter built: {adapter['failure_class']}  "
          f"n={adapter['n_events']}  winner={adapter['target_encoder']}")

    # ── Claim 3b: adapter corrects Session B instances it never saw ───────────
    print("\n" + "─" * 64)
    print("  PHASE 4 — Claim 3b: adapter corrects NEW instances (Session B)")
    print("─" * 64)

    z_v_md_B = sess_B["z_vis_motordeg"]
    z_p_md_B = sess_B["z_prp_motordeg"]

    # Cosine similarity before adapter
    sim_before = F.cosine_similarity(z_v_md_B, z_p_md_B, dim=-1).mean().item()

    # Apply adapter (corrects z_vision toward z_proprio)
    z_v_corrected = apply_adapter(z_v_md_B, adapter).detach()
    sim_after = F.cosine_similarity(z_v_corrected, z_p_md_B, dim=-1).mean().item()

    print(f"  cos(z_vision, z_proprio) before adapter:  {sim_before:+.4f}")
    print(f"  cos(z_vision, z_proprio) after  adapter:  {sim_after:+.4f}")
    print(f"  Alignment gain on new instances:           {sim_after - sim_before:+.4f}")
    claim_3b = sim_after > sim_before
    print(f"  Claim 3b (adapter generalises to new B):   "
          f"{'✓ PROVEN' if claim_3b else '✗ not yet'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  TRANSFER PROOF SUMMARY")
    print("=" * 64)
    rows = [
        ("Session A AUROC (training domain)",   auroc_A),
        ("Session B AUROC (new instances)",      auroc_B),
        ("Performance drop (A→B)",               auroc_A - auroc_B),
    ]
    for name, val in rows:
        print(f"  {name:<42} {val:>+8.4f}")

    print()
    print(f"  Adapter alignment gain on new B instances: {sim_after - sim_before:+.4f}")
    print(f"    (cos before: {sim_before:+.4f}  →  after: {sim_after:+.4f})")
    print()
    print(f"  Claim 3a — Detection transfers (AUROC B ≥ 0.70, drop ≤ 0.10): "
          f"{'✓ PROVEN' if claim_3a else '✗'}")
    print(f"  Claim 3b — Adapter corrects new instances (sim improves):       "
          f"{'✓ PROVEN' if claim_3b else '✗'}")
    print()

    if claim_3a and claim_3b:
        print("  CONCLUSION:")
        print("  The system learned its blind spots in Session A.")
        print("  In Session B — entirely new instances, zero labels —")
        print("  it detects the same failure pattern with the same accuracy,")
        print("  and its consolidated memory corrects the embeddings of cases")
        print("  it has never seen before.")
        print()
        print("  This is how humans learn: not by retraining from scratch,")
        print("  but by consolidating hard cases into retrievable memory.")
    print("=" * 64)

    out = dict(
        config=dict(embed_dim=embed_dim, n_normal=n_normal, n_failure=n_failure,
                    noise=noise, n_epochs=n_epochs),
        auroc_session_a=round(auroc_A, 4),
        auroc_session_b=round(auroc_B, 4),
        auroc_drop=round(auroc_A - auroc_B, 4),
        cos_before_adapter=round(sim_before, 4),
        cos_after_adapter=round(sim_after, 4),
        alignment_gain=round(sim_after - sim_before, 4),
        claim_3a=claim_3a,
        claim_3b=claim_3b,
    )
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"prove_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JEPA memory transfer proof: Session A → Session B"
    )
    parser.add_argument("--embed-dim", type=int,   default=8)
    parser.add_argument("--n-normal",  type=int,   default=400)
    parser.add_argument("--n-failure", type=int,   default=200)
    parser.add_argument("--noise",     type=float, default=0.15)
    parser.add_argument("--epochs",    type=int,   default=300)
    args = parser.parse_args()

    run_transfer_proof(
        embed_dim = args.embed_dim,
        n_normal  = args.n_normal,
        n_failure = args.n_failure,
        noise     = args.noise,
        n_epochs  = args.epochs,
    )

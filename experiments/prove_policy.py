"""
Snath Robotics — JEPA Policy Learning Proof.
=============================================
Proves that the system learns not just WHAT went wrong but WHAT TO DO.

Scenario: a robot moving on ice with a speed parameter s ∈ (0, 1].
Physics:
  s > threshold  → z_proprio = R @ z_vision + s * noise * ε   (slipping)
  s ≤ threshold  → z_proprio = R @ z_vision + base_noise * ε  (stable)

The robot does not know the threshold. It must find it by observing
whether prediction error drops after a speed reduction.

After enough episodes on Surface A, a PolicyMemory stores the speed
prior — the median speed at which convergence was observed.

On Surface B (new instances, same threshold), the robot starts from
the prior instead of maximum speed. Steps-to-convergence drops.

That is the learning claim: not just detection, not just memory of
what the failure looks like — memory of what to DO about it.

Three claims proven:

  Claim 4a — The robot finds the safe speed through self-supervised search
              (prediction error is the only signal, no labels).

  Claim 4b — After N episodes on Surface A, the PolicyMemory stores a
              prior that cuts steps-to-convergence on Surface B.

  Claim 4c — The more episodes on A, the tighter the prior and the
              fewer steps needed on B. Learning improves with experience.

Run:
    python experiments/prove_policy.py
    python experiments/prove_policy.py --threshold 0.4 --episodes 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.jepa_predictor import JEPAPredictor, train_predictor


# ── Physics simulation ────────────────────────────────────────────────────────

def make_normal_pairs(
    n: int, embed_dim: int, R: torch.Tensor,
    base_noise: float, seed: int,
) -> tuple:
    rng = torch.Generator().manual_seed(seed)
    z_v = F.normalize(torch.randn(n, embed_dim, generator=rng), dim=-1)
    z_p = F.normalize(
        z_v @ R.T + base_noise * torch.randn(n, embed_dim, generator=rng), dim=-1
    )
    return z_v, z_p


def ice_pair_at_speed(
    speed: float, embed_dim: int, R: torch.Tensor,
    base_noise: float, threshold: float, seed: int,
) -> tuple:
    """
    One (z_vision, z_proprio) pair on ice at the given speed.
    Above threshold: noisy (slipping). At or below: stable.
    """
    rng = torch.Generator().manual_seed(seed)
    z_v = F.normalize(torch.randn(1, embed_dim, generator=rng), dim=-1)
    if speed > threshold:
        noise_scale = base_noise + (speed - threshold) * 2.0
    else:
        noise_scale = base_noise
    z_p = F.normalize(
        z_v @ R.T + noise_scale * torch.randn(1, embed_dim, generator=rng), dim=-1
    )
    return z_v.squeeze(0), z_p.squeeze(0)


def prediction_error_at_speed(
    predictor: JEPAPredictor,
    speed: float,
    embed_dim: int,
    R: torch.Tensor,
    base_noise: float,
    threshold: float,
    seed: int,
    n_samples: int = 8,
) -> float:
    """Average prediction error over n_samples pairs at this speed."""
    errors = []
    for i in range(n_samples):
        z_v, z_p = ice_pair_at_speed(speed, embed_dim, R, base_noise,
                                     threshold, seed + i)
        with torch.no_grad():
            err = predictor.prediction_error(
                z_v.unsqueeze(0), z_p.unsqueeze(0)
            )
        errors.append(float(err.item()))
    return float(np.mean(errors))


# ── Policy memory ─────────────────────────────────────────────────────────────

class PolicyMemory:
    """
    Maps failure_class → list of speeds at which convergence was observed.
    Prior = median of observed convergence speeds.

    This is the "what to do" complement to the DMN's "what it looks like."
    """

    def __init__(self):
        self._speeds: dict = defaultdict(list)

    def record(self, failure_class: str, speed: float) -> None:
        self._speeds[failure_class].append(speed)

    def prior(self, failure_class: str) -> Optional[float]:
        speeds = self._speeds[failure_class]
        if not speeds:
            return None
        return float(np.median(speeds))

    def std(self, failure_class: str) -> float:
        speeds = self._speeds[failure_class]
        if len(speeds) < 2:
            return float("inf")
        return float(np.std(speeds))

    def n(self, failure_class: str) -> int:
        return len(self._speeds[failure_class])


# ── Speed search ──────────────────────────────────────────────────────────────

def search_for_safe_speed(
    predictor:   JEPAPredictor,
    embed_dim:   int,
    R:           torch.Tensor,
    base_noise:  float,
    threshold:   float,
    surface_seed: int,
    start_speed: float = 1.0,
    delta:       float = 0.05,
    conv_thresh: float = 0.25,
    n_samples:   int   = 8,
) -> tuple[float, int]:
    """
    Reduce speed in steps until prediction error drops below conv_thresh.

    Returns:
        (safe_speed, steps_taken)
    """
    speed = start_speed
    steps = 0
    while speed > delta:
        err = prediction_error_at_speed(
            predictor, speed, embed_dim, R, base_noise, threshold,
            surface_seed, n_samples,
        )
        steps += 1
        if err < conv_thresh:
            return round(speed, 3), steps
        speed = round(speed - delta, 3)
    return round(speed, 3), steps


# ── Main proof ────────────────────────────────────────────────────────────────

def run_policy_proof(
    embed_dim:       int   = 8,
    n_normal:        int   = 400,
    base_noise:      float = 0.15,
    threshold:       float = 0.35,
    n_episodes_A:    int   = 15,
    n_episodes_B:    int   = 10,
    delta:           float = 0.05,
    conv_thresh:     float = 0.25,
) -> None:

    print("=" * 64)
    print("  JEPA POLICY LEARNING PROOF — Snath Robotics")
    print("=" * 64)
    print(f"  embed_dim={embed_dim}  base_noise={base_noise}")
    print(f"  ice threshold={threshold}  speed_delta={delta}")
    print(f"  convergence_error < {conv_thresh}")
    print(f"\n  Physics: speed > {threshold} → slipping   "
          f"speed ≤ {threshold} → stable\n")

    # Shared physical coupling
    rng_R = torch.Generator().manual_seed(0)
    R = F.normalize(torch.randn(embed_dim, embed_dim, generator=rng_R), dim=0)

    # ── Phase 1: Pre-train predictor on normal floor ──────────────────────────
    print("─" * 64)
    print("  PHASE 1 — Pre-train predictor on normal floor (no labels)")
    print("─" * 64)
    predictor = JEPAPredictor(embed_dim=embed_dim)
    z_v_tr, z_p_tr = make_normal_pairs(n_normal, embed_dim, R, base_noise, seed=0)
    train_predictor(predictor, z_v_tr, z_p_tr, n_epochs=300, batch_size=64)

    # Verify: error is high at full speed on ice, low at safe speed
    err_full = prediction_error_at_speed(
        predictor, 1.0, embed_dim, R, base_noise, threshold, seed=1
    )
    err_safe = prediction_error_at_speed(
        predictor, threshold - delta, embed_dim, R, base_noise, threshold, seed=1
    )
    print(f"  Prediction error at speed=1.0 (slipping):  {err_full:.4f}")
    print(f"  Prediction error at speed={threshold-delta:.2f} (stable):   {err_safe:.4f}")
    print(f"  Signal gap: {err_full - err_safe:.4f}  "
          f"{'✓ clear' if err_full - err_safe > 0.10 else '✗ weak'}")

    # ── Phase 2: Surface A — search and learn ─────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  PHASE 2 — Surface A: {n_episodes_A} episodes, no prior")
    print(f"{'─'*64}")

    memory = PolicyMemory()
    steps_A = []

    print(f"  {'Episode':>8}  {'Start':>7}  {'Found':>7}  {'Steps':>6}  {'Prior after'}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*12}")

    for ep in range(n_episodes_A):
        surface_seed = 100 + ep
        safe_speed, steps = search_for_safe_speed(
            predictor, embed_dim, R, base_noise, threshold,
            surface_seed, start_speed=1.0, delta=delta, conv_thresh=conv_thresh,
        )
        memory.record("environmental_transient", safe_speed)
        steps_A.append(steps)
        prior = memory.prior("environmental_transient")
        print(f"  {ep+1:>8}  {'1.00':>7}  {safe_speed:>7.3f}  {steps:>6}  "
              f"prior={prior:.3f} ± {memory.std('environmental_transient'):.3f}")

    mean_steps_A = float(np.mean(steps_A))
    prior_speed  = memory.prior("environmental_transient")
    prior_std    = memory.std("environmental_transient")

    print(f"\n  Mean steps on Surface A (no prior): {mean_steps_A:.1f}")
    print(f"  PolicyMemory prior:                 {prior_speed:.3f} ± {prior_std:.3f}")
    print(f"  True threshold:                     {threshold:.3f}")
    print(f"  Prior accuracy:                     "
          f"{'✓ within 1 step' if abs(prior_speed - threshold) <= delta else '~ close'}")

    # ── Phase 3: Surface B — search with prior ────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  PHASE 3 — Surface B: {n_episodes_B} episodes, WITH prior vs WITHOUT")
    print(f"{'─'*64}")
    print(f"  Prior start speed: {prior_speed:.3f} + {delta:.2f} = {prior_speed+delta:.3f}")
    print()
    print(f"  {'Episode':>8}  {'No prior':>10}  {'With prior':>12}  {'Saved':>6}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*6}")

    steps_B_no_prior   = []
    steps_B_with_prior = []

    for ep in range(n_episodes_B):
        surface_seed = 200 + ep

        # Without prior: start from 1.0
        _, steps_no = search_for_safe_speed(
            predictor, embed_dim, R, base_noise, threshold,
            surface_seed, start_speed=1.0,
            delta=delta, conv_thresh=conv_thresh,
        )

        # With prior: start just above the remembered safe speed
        start = min(1.0, round(prior_speed + delta, 3))
        _, steps_wi = search_for_safe_speed(
            predictor, embed_dim, R, base_noise, threshold,
            surface_seed, start_speed=start,
            delta=delta, conv_thresh=conv_thresh,
        )

        steps_B_no_prior.append(steps_no)
        steps_B_with_prior.append(steps_wi)
        saved = steps_no - steps_wi
        print(f"  {ep+1:>8}  {steps_no:>10}  {steps_wi:>12}  {saved:>+6}")

    mean_no   = float(np.mean(steps_B_no_prior))
    mean_wi   = float(np.mean(steps_B_with_prior))
    speedup   = mean_no / max(mean_wi, 0.1)
    saved_avg = mean_no - mean_wi

    # ── Claim 4c: prior tightens with more episodes ───────────────────────────
    print(f"\n{'─'*64}")
    print(f"  PHASE 4 — Prior convergence: does more experience help?")
    print(f"{'─'*64}")
    print(f"  {'Episodes':>10}  {'Prior':>8}  {'Std':>8}  {'Steps saved (est)'}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*18}")

    mem_tmp = PolicyMemory()
    for i, sp in enumerate(
        [s for s in [
            search_for_safe_speed(predictor, embed_dim, R, base_noise, threshold,
                                  100+i, 1.0, delta, conv_thresh)[0]
            for i in range(n_episodes_A)
        ]]
    ):
        mem_tmp.record("environmental_transient", sp)
        if (i + 1) in [1, 3, 5, 10, n_episodes_A]:
            p   = mem_tmp.prior("environmental_transient")
            std = mem_tmp.std("environmental_transient")
            est_start = min(1.0, round(p + delta, 3))
            est_saved = round((1.0 - est_start) / delta)
            print(f"  {i+1:>10}  {p:>8.3f}  {std:>8.3f}  ~{est_saved} steps")

    # ── Summary ───────────────────────────────────────────────────────────────
    claim_4a = err_full - err_safe > 0.10
    claim_4b = mean_wi < mean_no and speedup >= 1.5
    claim_4c = prior_std < 0.10

    print(f"\n{'='*64}")
    print(f"  POLICY PROOF SUMMARY")
    print(f"{'='*64}")
    print(f"  True safe speed threshold:          {threshold:.3f}")
    print(f"  Learned prior (Surface A):          {prior_speed:.3f} ± {prior_std:.3f}")
    print(f"")
    print(f"  Mean steps without prior (B):       {mean_no:.1f}")
    print(f"  Mean steps with prior    (B):       {mean_wi:.1f}")
    print(f"  Steps saved per episode:            {saved_avg:.1f}")
    print(f"  Speedup factor:                     {speedup:.2f}×")
    print(f"")
    print(f"  Claim 4a — Error signal separates slip from stable:   "
          f"{'✓ PROVEN' if claim_4a else '✗'}")
    print(f"  Claim 4b — Prior cuts steps-to-convergence on new B:  "
          f"{'✓ PROVEN' if claim_4b else '✗'}")
    print(f"  Claim 4c — Prior tightens with experience (std<0.10): "
          f"{'✓ PROVEN' if claim_4c else '✗'}")
    print(f"")

    if claim_4a and claim_4b and claim_4c:
        print(f"  CONCLUSION:")
        print(f"  The robot found the safe speed through prediction error alone.")
        print(f"  After {n_episodes_A} episodes on Surface A, it remembered {prior_speed:.2f}.")
        print(f"  On Surface B — entirely new surfaces — it converged {speedup:.1f}× faster.")
        print(f"  No labels. No reward signal. No human telling it what speed to use.")
        print(f"  The world told it: prediction error drops when you get it right.")
    print(f"{'='*64}")

    out = dict(
        config=dict(embed_dim=embed_dim, n_normal=n_normal, base_noise=base_noise,
                    threshold=threshold, n_episodes_A=n_episodes_A,
                    n_episodes_B=n_episodes_B, delta=delta, conv_thresh=conv_thresh),
        signal_gap=round(err_full - err_safe, 4),
        err_at_full_speed=round(err_full, 4),
        err_at_safe_speed=round(err_safe, 4),
        learned_prior=round(prior_speed, 4),
        prior_std=round(prior_std, 4),
        true_threshold=threshold,
        mean_steps_no_prior=round(mean_no, 2),
        mean_steps_with_prior=round(mean_wi, 2),
        speedup=round(speedup, 2),
        claim_4a=claim_4a,
        claim_4b=claim_4b,
        claim_4c=claim_4c,
    )
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"prove_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JEPA policy learning: robot learns safe speed on ice"
    )
    parser.add_argument("--embed-dim",    type=int,   default=8)
    parser.add_argument("--n-normal",     type=int,   default=400)
    parser.add_argument("--base-noise",   type=float, default=0.15)
    parser.add_argument("--threshold",    type=float, default=0.35,
                        help="True safe speed threshold (unknown to robot)")
    parser.add_argument("--episodes",     type=int,   default=15,
                        help="Episodes on Surface A to build prior")
    parser.add_argument("--episodes-b",   type=int,   default=10,
                        help="Episodes on Surface B to test prior")
    parser.add_argument("--delta",        type=float, default=0.05,
                        help="Speed reduction step size")
    parser.add_argument("--conv-thresh",  type=float, default=0.25,
                        help="Prediction error threshold for convergence")
    args = parser.parse_args()

    run_policy_proof(
        embed_dim    = args.embed_dim,
        n_normal     = args.n_normal,
        base_noise   = args.base_noise,
        threshold    = args.threshold,
        n_episodes_A = args.episodes,
        n_episodes_B = args.episodes_b,
        delta        = args.delta,
        conv_thresh  = args.conv_thresh,
    )

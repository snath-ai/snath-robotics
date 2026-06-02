"""
RoboticsDMN — overnight consolidation cycle.
============================================
Reads the D_hard queue of sensor-disagreement events, clusters them by
failure class and divergence pattern, and generates signed LoRA adapters
stored in models/adapters/.

This is the swarm learning mechanism described in the position paper:
every TRIGGER_REPLAN and STRUCTURAL_IMPASSE event is logged; overnight
the DMN trains adapters that compensate for recurring failure patterns;
by morning the adapters are distributed to the fleet.

System 1 (fast JSON centroid cache) and System 2 (LoRA .pt file) are
both generated here, following the identical pattern as Snath Aviation,
Snath Basis, and Snath Locus.

SIGReg — Sketched Isotropic Gaussian Regularisation
----------------------------------------------------
lambda_iso=0.0 by default (inert until AIA Experiment 3 calibrates the
optimal weight from {0.01, 0.1, 1.0}). Wire in via consolidate(lambda_iso=0.1).
"""

from __future__ import annotations

import os
import json
import hmac as _hmac
import hashlib
import datetime
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from dhard import DHardQueue, RoboticsDHardEvent
from dmn.sigreg import SIGRegLoss

_ADAPTER_KEY = b"snath_robotics_adapter_sovereignty_2026"
_MIN_EVENTS  = 4


class RoboticsDMN:
    """
    Default Mode Network for Snath Robotics.

    Run nightly (or after a session):
        python -m dmn.robotics_dmn --run-cycle
    """

    def __init__(
        self,
        queue_path:  str = "d_hard.jsonl",
        adapter_dir: str = "models/adapters",
    ):
        self.queue       = DHardQueue(queue_path)
        self.adapter_dir = Path(adapter_dir)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate(
        self,
        min_events:  int   = _MIN_EVENTS,
        n_epochs:    int   = 100,
        lr:          float = 0.1,
        lambda_iso:  float = 0.0,
        verbose:     bool  = True,
    ) -> List[dict]:
        """
        Cluster resolved D_hard events → generate signed adapters.

        Args:
            min_events:  Minimum resolved events per cluster before fitting.
            n_epochs:    LoRA training epochs.
            lr:          Adam learning rate.
            lambda_iso:  SIGReg isotropy penalty. 0.0 = disabled.
            verbose:     Print per-cluster summary.

        Returns:
            List of adapter metadata dicts for logging.
        """
        events    = self.queue.resolved()
        by_class  = defaultdict(list)
        for e in events:
            by_class[e.failure_class].append(e)

        built = []
        sigreg = SIGRegLoss(lambda_iso=lambda_iso)

        for failure_class, group in sorted(by_class.items()):
            if len(group) < min_events:
                if verbose:
                    print(f"  · {failure_class:<28} {len(group)} events "
                          f"— too few (need {min_events}), skipped")
                continue

            # ── SYSTEM 1: JSON centroid ────────────────────────────────────
            centroid_a = [
                round(sum(e.z_vision[i]  for e in group) / len(group), 6)
                for i in range(len(group[0].z_vision))
            ]
            centroid_b = [
                round(sum(e.z_proprio[i] for e in group) / len(group), 6)
                for i in range(len(group[0].z_proprio))
            ]
            winner_counts = {}
            for e in group:
                if e.winner:
                    winner_counts[e.winner] = winner_counts.get(e.winner, 0) + 1
            winner = max(winner_counts, key=winner_counts.get) if winner_counts else "unknown"
            win_rate = round(winner_counts.get(winner, 0) / len(group), 3)

            json_payload = {
                "failure_class": failure_class,
                "centroid_vision":  centroid_a,
                "centroid_proprio": centroid_b,
                "winner":    winner,
                "win_rate":  win_rate,
                "n_events":  len(group),
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            }
            json_path = self.adapter_dir / f"{failure_class}.json"
            json_path.write_text(json.dumps(json_payload, indent=2))

            # ── SYSTEM 2: LoRA .pt ────────────────────────────────────────
            # Target stream is the winner; faulty stream is the loser.
            if winner == "vision":
                target_vecs = [e.z_vision  for e in group]
                faulty_vecs = [e.z_proprio for e in group]
                target_enc  = "proprio"
            else:
                target_vecs = [e.z_proprio for e in group]
                faulty_vecs = [e.z_vision  for e in group]
                target_enc  = "vision"

            target_t = torch.tensor(target_vecs, dtype=torch.float32)
            faulty_t = torch.tensor(faulty_vecs, dtype=torch.float32)
            dim      = faulty_t.shape[1]

            A = nn.Parameter(torch.randn(dim, 1) * 0.01)
            B = nn.Parameter(torch.randn(1, dim) * 0.01)
            opt = optim.AdamW([A, B], lr=lr)

            for _ in range(n_epochs):
                opt.zero_grad()
                adapted = faulty_t + torch.matmul(torch.matmul(faulty_t, A), B)
                loss = torch.nn.functional.l1_loss(adapted, target_t)
                # SIGReg: penalise anisotropy in the adapted latent space so
                # that Δ = softmax(z_a) − softmax(z_b) is a reliable signal.
                # No-op at lambda_iso=0.0 (default) until AIA Experiment 3.
                loss = loss + sigreg(adapted)
                loss.backward()
                opt.step()

            # HMAC sign the LoRA weights
            a_hash = hashlib.sha256(A.detach().numpy().tobytes()).hexdigest()[:16]
            b_hash = hashlib.sha256(B.detach().numpy().tobytes()).hexdigest()[:16]
            sig = _hmac.new(
                _ADAPTER_KEY,
                f"{failure_class}|{target_enc}|{a_hash}|{b_hash}".encode(),
                hashlib.sha256,
            ).hexdigest()

            pt_payload = {
                "A":             A.detach(),
                "B":             B.detach(),
                "target_encoder": target_enc,
                "failure_class":  failure_class,
                "created_at":     datetime.datetime.utcnow().isoformat() + "Z",
                "n_events":       len(group),
                "win_rate":       win_rate,
                "final_loss":     round(float(loss.item()), 6),
                "hmac_hex":       sig,
            }
            pt_path = self.adapter_dir / f"{failure_class}.pt"
            torch.save(pt_payload, str(pt_path))

            meta = {**json_payload, "pt_path": str(pt_path),
                    "final_loss": pt_payload["final_loss"]}
            built.append(meta)

            if verbose:
                print(f"  ✓ {failure_class:<28} n={len(group):<3} "
                      f"winner={winner:<8} win_rate={win_rate:<5} "
                      f"LoRA loss={loss.item():.4f}")

        return built

    def stats(self) -> dict:
        return self.queue.stats()


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="RoboticsDMN — overnight D_hard → LoRA consolidation cycle"
    )
    parser.add_argument("--run-cycle",   action="store_true")
    parser.add_argument("--queue-path",  default="d_hard.jsonl")
    parser.add_argument("--adapter-dir", default="models/adapters")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--lambda-iso",  type=float, default=0.0,
                        help="SIGReg λ_iso. 0.0=disabled. AIA Exp 3: {0.01,0.1,1.0}")
    parser.add_argument("--verbose",     action="store_true", default=True)
    args = parser.parse_args()

    if args.run_cycle:
        dmn = RoboticsDMN(queue_path=args.queue_path, adapter_dir=args.adapter_dir)
        s   = dmn.stats()
        print(f"[RoboticsDMN] D_hard queue: {s['total']} total, "
              f"{s['resolved']} resolved")
        built = dmn.consolidate(
            n_epochs=args.epochs, lambda_iso=args.lambda_iso, verbose=args.verbose
        )
        print(f"[RoboticsDMN] Built {len(built)} adapter(s).")
        sys.exit(0)

    parser.print_help()

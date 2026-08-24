"""
System-1 (raw centroid) vs. System-2 (LoRA-corrected) anchor comparison.
==========================================================================
Finding find:system1system2 in the FLEET paper (09_FLEET/main.tex) claims
a specific cosine-similarity / L2-distance range between (a) the DMN
service's raw consolidated centroid and (b) the original, validated
pipeline's actual LoRA-corrected ice anchor for Robot B. A 2026-08-2x audit
could not locate any saved script or computation producing those exact
numbers anywhere in the repo — they were asserted without a saved
provenance trail.

This script recomputes the comparison for real, from artifacts that ARE
still on disk from the bit-for-bit local-file rerun on 2026-08-20
(fleet_n2_results_20260820T162650.json / ...726.json):

  System-2 (LoRA-corrected) anchor, per seed:
    experiments/fleet/runs/seed_<seed>/adapters_b/environmental_transient.pt
    — the actual rank-1 (A, B) correction Robot B trained on the received
    events in that run (100 AdamW epochs, L1 loss; RoboticsDMN.consolidate,
    torch.manual_seed(seed)). Applied to a freshly-rederived z_ref via
    Robot.warmup() (deterministic: same seed_b = seed + 1000, same MuJoCo
    physics, same policy RNG, same encoder in eval() mode — no stochastic
    ops in the forward pass) exactly as fleet_n2_transfer.py's run_seed()
    does: z_ice_learned = z_ref + z_ref @ A @ B.

  System-1 (raw) centroid, per seed:
    the arithmetic mean of the same seed's shipped z_proprio vectors,
    computed two ways that must and do agree:
      (a) locally, straight from runs/seed_<seed>/robot_a_dhard.jsonl
          (the exact events Robot A shipped in this run)
      (b) the DMN service's own served centroid, already saved in
          fleet_n2_service_results_20260820T163112.json's
          results[i].service_entry.centroid — produced independently by
          dmn_service/consolidation.py's consolidate_class() (plain
          arithmetic mean, LTL paper's Delta_bar_c) over the same events,
          shipped over real HTTP in the service-routed rerun.

Nothing here is reverse-engineered to hit a target number. Run this, read
whatever it prints/saves — that is what goes in the paper.

Run:
    .venv/bin/python experiments/fleet/compare_system1_system2_anchors.py
"""
from __future__ import annotations

import sys, json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from experiments.fleet.fleet_n2_transfer import (
    Robot, load_base_encoder, adapted_anchor, DEFAULT_MODEL, RUNS_DIR,
)

SEEDS = [42, 7, 13, 99, 2026]
SERVICE_RESULTS = Path(__file__).parent / "fleet_n2_service_results_20260820T163112.json"


def raw_centroid_from_jsonl(path: Path) -> np.ndarray:
    vecs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vecs.append(rec["z_proprio"])
    return np.array(vecs).mean(axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    with open(SERVICE_RESULTS) as f:
        service = json.load(f)
    service_by_seed = {r["seed"]: r for r in service["results"]}

    encoder = load_base_encoder(DEFAULT_MODEL)

    rows = []
    for seed in SEEDS:
        run_dir = RUNS_DIR / f"seed_{seed}"
        adapter_path = run_dir / "adapters_b" / "environmental_transient.pt"
        events_path = run_dir / "robot_a_dhard.jsonl"
        if not adapter_path.exists() or not events_path.exists():
            print(f"  seed {seed}: MISSING artifacts ({adapter_path.exists()=}, "
                  f"{events_path.exists()=}) — skipping")
            continue

        # Re-derive Robot B's normal anchor exactly as fleet_n2_transfer.py
        # run_seed() does (same seed_b, same warmup()).
        seed_b = seed + 1000
        B = Robot("B-fleet", encoder, seed_b)
        B.warmup()
        z_ref = B.anchors[0][1]
        B.close()

        system2 = adapted_anchor(z_ref, adapter_path).numpy()
        system1_local = raw_centroid_from_jsonl(events_path)

        service_entry = service_by_seed[seed]["service_entry"]
        system1_service = np.array(service_entry["centroid"])

        cos_local_vs_service = cosine(system1_local, system1_service)
        cos_sys1_sys2 = cosine(system1_service, system2)
        l2_sys1_sys2 = float(np.linalg.norm(system1_service - system2))

        rows.append({
            "seed": seed,
            "cosine_system1_local_vs_service": cos_local_vs_service,
            "cosine_system1_vs_system2": cos_sys1_sys2,
            "l2_system1_vs_system2": l2_sys1_sys2,
        })
        print(f"  seed {seed:5d}: System1(local)-vs-System1(service) cos="
              f"{cos_local_vs_service:.6f}  |  System1-vs-System2 cos="
              f"{cos_sys1_sys2:.4f}  L2={l2_sys1_sys2:.4f}")

    cos_vals = [r["cosine_system1_vs_system2"] for r in rows]
    l2_vals = [r["l2_system1_vs_system2"] for r in rows]
    print("\n  Aggregate (System1 raw centroid vs. System2 LoRA-corrected anchor):")
    print(f"    cosine: min={min(cos_vals):.4f} max={max(cos_vals):.4f} "
          f"mean={mean(cos_vals):.4f} ± {stdev(cos_vals) if len(cos_vals) > 1 else 0:.4f}")
    print(f"    L2:     min={min(l2_vals):.4f} max={max(l2_vals):.4f} "
          f"mean={mean(l2_vals):.4f} ± {stdev(l2_vals) if len(l2_vals) > 1 else 0:.4f}")

    out_path = Path(__file__).parent / "system1_system2_anchor_comparison.json"
    with open(out_path, "w") as f:
        json.dump({"rows": rows,
                    "cosine_range": [min(cos_vals), max(cos_vals)],
                    "l2_range": [min(l2_vals), max(l2_vals)]}, f, indent=2)
    print(f"\n  Results -> {out_path.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()

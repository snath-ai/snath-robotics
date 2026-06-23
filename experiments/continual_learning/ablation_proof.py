"""
Snath Robotics — JEPA Ablation Study.
======================================
Sweeps two axes to characterise the label-free learning claim:

  Axis 1 — Noise level:       how much sensor noise can the predictor tolerate?
  Axis 2 — Training set size: how few normal pairs does the predictor need?

Each cell is the mean AUROC over N_SEEDS independent runs.

The result is two tables suitable for direct inclusion in a paper:

  Table 1: AUROC vs noise (σ), N_normal=400 fixed
  Table 2: AUROC vs N_normal, noise=0.15 fixed (default from prove_learning.py)

Usage
-----
  python experiments/ablation_proof.py           # full sweep, ~60s
  python experiments/ablation_proof.py --quick   # fewer seeds, ~15s
  python experiments/ablation_proof.py --save    # write results/ablation_<ts>.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.continual_learning.prove_learning import make_structured_data, evaluate
from models.jepa_predictor import JEPAPredictor, train_predictor


# ── Sweep configuration ───────────────────────────────────────────────────────

NOISE_LEVELS  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
TRAIN_SIZES   = [25, 50, 100, 200, 400, 800]

N_SEEDS_FULL  = 5
N_SEEDS_QUICK = 3

EMBED_DIM     = 8
N_ANOMALY     = 200   # held-out anomaly pairs per run (fixed)
N_EPOCHS      = 300
LR            = 1e-3


# ── Core: single run ─────────────────────────────────────────────────────────

def single_run(
    embed_dim: int,
    n_normal:  int,
    n_anomaly: int,
    noise:     float,
    seed:      int,
    n_epochs:  int = N_EPOCHS,
) -> Tuple[float, float]:
    """
    One training run.

    Returns:
        (auroc_before, auroc_after) — prediction-error AUROC on held-out anomalies.
    """
    data = make_structured_data(
        embed_dim=embed_dim,
        n_normal=n_normal,
        n_anomaly=n_anomaly,
        noise=noise,
        seed=seed,
    )
    z_vis_tr  = data["z_vis_train"]
    z_prp_tr  = data["z_prp_train"]
    z_vis_te  = data["z_vis_test"]
    z_prp_te  = data["z_prp_test"]
    labels_te = data["labels_test"]

    predictor = JEPAPredictor(embed_dim=embed_dim)

    # Suppress per-phase prints during sweep by redirecting stdout temporarily
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        stats_before = evaluate(predictor, z_vis_te, z_prp_te, labels_te, "")
        train_predictor(predictor, z_vis_tr, z_prp_tr,
                        n_epochs=n_epochs, lr=LR, batch_size=min(32, n_normal))
        stats_after = evaluate(predictor, z_vis_te, z_prp_te, labels_te, "")

    return stats_before["auroc_overall"], stats_after["auroc_overall"]


# ── Sweep 1: noise level ──────────────────────────────────────────────────────

def sweep_noise(
    noise_levels: List[float] = NOISE_LEVELS,
    n_normal:     int         = 400,
    n_seeds:      int         = N_SEEDS_FULL,
) -> dict:
    """
    Fix N_normal=400, sweep noise σ.
    Returns dict keyed by noise value with {before_mean, after_mean, after_std, gain}.
    """
    results = {}
    for noise in noise_levels:
        befores, afters = [], []
        for seed in range(n_seeds):
            b, a = single_run(EMBED_DIM, n_normal, N_ANOMALY, noise, seed)
            befores.append(b)
            afters.append(a)
        results[noise] = {
            "before_mean": float(np.mean(befores)),
            "after_mean":  float(np.mean(afters)),
            "after_std":   float(np.std(afters)),
            "gain":        float(np.mean(afters) - np.mean(befores)),
        }
        print(f"  noise={noise:.2f}  "
              f"before={np.mean(befores):.4f}  "
              f"after={np.mean(afters):.4f} ± {np.std(afters):.4f}  "
              f"gain={np.mean(afters)-np.mean(befores):+.4f}")
    return results


# ── Sweep 2: training set size ────────────────────────────────────────────────

def sweep_n_normal(
    train_sizes: List[int] = TRAIN_SIZES,
    noise:       float     = 0.15,
    n_seeds:     int       = N_SEEDS_FULL,
) -> dict:
    """
    Fix noise=0.15, sweep N_normal.
    Returns dict keyed by N_normal.
    """
    results = {}
    for n_normal in train_sizes:
        befores, afters = [], []
        for seed in range(n_seeds):
            b, a = single_run(EMBED_DIM, n_normal, N_ANOMALY, noise, seed)
            befores.append(b)
            afters.append(a)
        results[n_normal] = {
            "before_mean": float(np.mean(befores)),
            "after_mean":  float(np.mean(afters)),
            "after_std":   float(np.std(afters)),
            "gain":        float(np.mean(afters) - np.mean(befores)),
        }
        print(f"  N_normal={n_normal:4d}  "
              f"before={np.mean(befores):.4f}  "
              f"after={np.mean(afters):.4f} ± {np.std(afters):.4f}  "
              f"gain={np.mean(afters)-np.mean(befores):+.4f}")
    return results


# ── Pretty-print table ────────────────────────────────────────────────────────

def print_noise_table(results: dict) -> None:
    print()
    print("  Table 1: AUROC vs noise  (N_normal=400, label-free predictor)")
    print(f"  {'Noise σ':>8}  {'Before':>8}  {'After':>8}  {'±':>6}  {'Gain':>7}  {'Verdict'}")
    print("  " + "─" * 58)
    for noise, r in sorted(results.items()):
        verdict = "✓ PROVEN" if r["after_mean"] >= 0.80 else ("~ partial" if r["after_mean"] >= 0.65 else "✗ weak")
        print(f"  {noise:>8.2f}  {r['before_mean']:>8.4f}  {r['after_mean']:>8.4f}  "
              f"{r['after_std']:>6.4f}  {r['gain']:>+7.4f}  {verdict}")
    print()


def print_size_table(results: dict) -> None:
    print()
    print("  Table 2: AUROC vs training set size  (noise=0.15, label-free predictor)")
    print(f"  {'N_normal':>8}  {'Before':>8}  {'After':>8}  {'±':>6}  {'Gain':>7}  {'Verdict'}")
    print("  " + "─" * 58)
    for n, r in sorted(results.items()):
        verdict = "✓ PROVEN" if r["after_mean"] >= 0.80 else ("~ partial" if r["after_mean"] >= 0.65 else "✗ weak")
        print(f"  {n:>8d}  {r['before_mean']:>8.4f}  {r['after_mean']:>8.4f}  "
              f"{r['after_std']:>6.4f}  {r['gain']:>+7.4f}  {verdict}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def run_ablation(quick: bool = False, save: bool = False) -> dict:
    n_seeds = N_SEEDS_QUICK if quick else N_SEEDS_FULL

    print("=" * 62)
    print("  JEPA ABLATION STUDY — Snath Robotics")
    print(f"  embed_dim={EMBED_DIM}  n_anomaly={N_ANOMALY}  "
          f"n_seeds={n_seeds}  epochs={N_EPOCHS}")
    print("=" * 62)

    print(f"\n── Sweep 1: noise level (N_normal=400, {n_seeds} seeds each) ──")
    noise_results = sweep_noise(n_seeds=n_seeds)
    print_noise_table(noise_results)

    print(f"── Sweep 2: training set size (noise=0.15, {n_seeds} seeds each) ──")
    size_results = sweep_n_normal(n_seeds=n_seeds)
    print_size_table(size_results)

    # Summary stats
    proven_noise = sum(1 for r in noise_results.values() if r["after_mean"] >= 0.80)
    proven_size  = sum(1 for r in size_results.values()  if r["after_mean"] >= 0.80)
    max_noise    = max(n for n, r in noise_results.items() if r["after_mean"] >= 0.80)
    min_n        = min(n for n, r in size_results.items()  if r["after_mean"] >= 0.80)

    print("=" * 62)
    print("  ABLATION SUMMARY")
    print("=" * 62)
    print(f"  Robust up to noise σ = {max_noise:.2f}  ({proven_noise}/{len(noise_results)} levels ≥ 0.80)")
    print(f"  Works from N_normal  = {min_n}  ({proven_size}/{len(size_results)} sizes ≥ 0.80)")
    print(f"  Claim: JEPA prediction error is a sufficient anomaly signal")
    print(f"         with zero labels, robust to sensor noise up to σ={max_noise:.2f}")
    print(f"         and requiring as few as {min_n} normal training pairs.")
    print("=" * 62)

    all_results = {
        "config": {
            "embed_dim": EMBED_DIM,
            "n_anomaly": N_ANOMALY,
            "n_seeds":   n_seeds,
            "n_epochs":  N_EPOCHS,
        },
        "noise_sweep":  {str(k): v for k, v in noise_results.items()},
        "size_sweep":   {str(k): v for k, v in size_results.items()},
        "summary": {
            "max_noise_proven": max_noise,
            "min_n_proven":     min_n,
        },
    }

    if save:
        import datetime
        ts       = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir  = Path(os.path.dirname(os.path.abspath(__file__))) / "coco_results"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"ablation_{ts}.json"
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"\n  Saved → {out_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JEPA ablation: AUROC vs noise and vs training set size"
    )
    parser.add_argument("--quick", action="store_true",
                        help=f"Fewer seeds ({N_SEEDS_QUICK} instead of {N_SEEDS_FULL})")
    parser.add_argument("--save",  action="store_true",
                        help="Save results to experiments/coco_results/ablation_<ts>.json")
    args = parser.parse_args()
    run_ablation(quick=args.quick, save=args.save)

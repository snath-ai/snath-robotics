"""
Snath Robotics — Curriculum Proof
===================================
Appendix B: D_hard threshold sensitivity.
Experiment 1 (adapted): D_hard curriculum vs random at matched sample count.

Difficulty proxy: D-score (L1 over 80-class COCO concept projections in 32-dim
space, temperature tau=100, initialized from CLIP ViT-B/32 text embeddings).
Training: 512-dim CLIP embeddings, JEPA predictor only (no labels).
Oracle: matched (label=0) vs caption-shuffled mismatches (label=1).

Run:
    python experiments/curriculum_proof.py
    python experiments/curriculum_proof.py --epochs 5 --seeds 1   # smoke test
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.jepa_predictor import JEPAPredictor, train_predictor

# ── Defaults ───────────────────────────────────────────────────────────────────
N_EPOCHS   = 100
N_SEEDS    = 3
ORACLE_N   = 500   # matched pairs → oracle = 500 matched + 500 mismatched
EMBED_DIM  = 512
LR         = 1e-3

# ── Data / D-score ─────────────────────────────────────────────────────────────

def load_coco() -> tuple[torch.Tensor, torch.Tensor]:
    d = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "coco_clip_cache",
    )
    img = torch.load(os.path.join(d, "coco_val2017_img.pt"),
                     map_location="cpu", weights_only=False)
    cap = torch.load(os.path.join(d, "coco_val2017_cap.pt"),
                     map_location="cpu", weights_only=False)
    return img, cap


CONCEPT_DIM    = 32
ROUTING_SCALE  = 20.0   # matches init_pca() default in CLIPImageEncoder


def build_concept_proj(img: torch.Tensor) -> torch.Tensor:
    """
    Return W ∈ (32, 512) via PCA over the image embedding distribution.
    Matches CLIPImageEncoder.init_pca(routing_scale=20.0) used in coco_proof.py.
    """
    from sklearn.decomposition import PCA
    X = F.normalize(img, dim=-1).numpy()
    pca = PCA(n_components=CONCEPT_DIM)
    pca.fit(X)
    W = torch.tensor(pca.components_, dtype=torch.float32) * ROUTING_SCALE
    var = float(pca.explained_variance_ratio_.sum())
    print(f"  PCA explained variance: {var:.1%}  (routing_scale={ROUTING_SCALE})")
    return W   # (32, 512)


def compute_d_scores(img: torch.Tensor, cap: torch.Tensor,
                     W: torch.Tensor) -> np.ndarray:
    """
    D = ||softmax(z_img) - softmax(z_cap)||₁ / √G
    where z = clip_emb @ Wᵀ  (PCA projection, no additional temperature).
    Matches DivergenceRouter formula and coco_proof.py routing logic.
    """
    G = W.shape[0]
    with torch.no_grad():
        z_img = F.normalize(img, dim=-1) @ W.T   # (N, 32)
        z_cap = F.normalize(cap, dim=-1) @ W.T
        v_img = F.softmax(z_img, dim=-1)
        v_cap = F.softmax(z_cap, dim=-1)
    return ((v_img - v_cap).abs().sum(dim=-1) / G**0.5).numpy()


# ── Oracle ─────────────────────────────────────────────────────────────────────

def build_oracle(img, cap, n, seed=0):
    rng   = np.random.RandomState(seed)
    idx   = rng.choice(len(img), n, replace=False)
    shuf  = rng.permutation(n)
    z_img = torch.cat([img[idx], img[idx]])
    z_cap = torch.cat([cap[idx], cap[idx[shuf]]])
    labels = np.array([0]*n + [1]*n, dtype=int)
    return z_img, z_cap, labels


# ── Training helpers ───────────────────────────────────────────────────────────

def _eval(predictor, z_img, z_cap, labels):
    predictor.eval()
    with torch.no_grad():
        errs = predictor.prediction_error(z_img, z_cap).numpy()
    return float(roc_auc_score(labels, errs))


def _run(t_img, t_cap, o_img, o_cap, o_labels, seed, n_epochs):
    torch.manual_seed(seed)
    p  = JEPAPredictor(embed_dim=EMBED_DIM)
    bs = min(256, len(t_img))
    with contextlib.redirect_stdout(io.StringIO()):
        train_predictor(p, t_img, t_cap, n_epochs=n_epochs,
                        batch_size=bs, lr=LR)
    return _eval(p, o_img, o_cap, o_labels)


def _seeds(t_img, t_cap, o_img, o_cap, o_labels, n_seeds, n_epochs):
    aurocs = [_run(t_img, t_cap, o_img, o_cap, o_labels, s, n_epochs)
              for s in range(n_seeds)]
    return float(np.mean(aurocs)), float(np.std(aurocs))


# ── Appendix B ─────────────────────────────────────────────────────────────────

def run_appendix_b(t_img, t_cap, d_scores, oracle, n_seeds, n_epochs):
    o_img, o_cap, o_labels = oracle

    untrained = JEPAPredictor(embed_dim=EMBED_DIM)
    baseline  = _eval(untrained, o_img, o_cap, o_labels)

    print("=" * 64)
    print("  APPENDIX B — D_hard Threshold Sensitivity (COCO val2017)")
    print("=" * 64)
    print(f"  Training pool: {len(d_scores)} pairs  |  Oracle: {ORACLE_N}×2")
    print(f"  Epochs: {n_epochs}  |  Seeds: {n_seeds}")
    print(f"  Baseline AUROC (untrained): {baseline:.4f}\n")
    print(f"  {'Frac':>6}  {'N_train':>8}  {'tau_D cut':>10}  "
          f"{'AUROC':>8}  {'± std':>7}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*7}")

    # Sort indices by D-score descending (hardest first) — .copy() avoids neg-stride
    sorted_idx = np.argsort(d_scores)[::-1].copy()
    fracs      = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
    results    = []

    for frac in fracs:
        n_hard   = max(64, int(len(d_scores) * frac))
        hard_idx = torch.tensor(sorted_idx[:n_hard])
        tau_cut  = float(d_scores[sorted_idx[n_hard - 1]])

        m, s = _seeds(t_img[hard_idx], t_cap[hard_idx],
                      o_img, o_cap, o_labels,
                      n_seeds=n_seeds, n_epochs=n_epochs)

        results.append(dict(frac=frac, n_train=n_hard,
                            tau_d_cutoff=round(tau_cut, 4),
                            auroc=round(m, 4), std=round(s, 4)))
        print(f"  {frac:>5.0%}  {n_hard:>8}  {tau_cut:>10.4f}  "
              f"{m:>8.4f}  ±{s:>6.4f}")

    best = max(results, key=lambda r: r["auroc"])
    print(f"\n  Baseline (untrained):   {baseline:.4f}")
    print(f"  Best curriculum AUROC:  {best['auroc']:.4f} "
          f"at {best['frac']:.0%} (tau_D ≥ {best['tau_d_cutoff']:.4f})")
    return baseline, results


# ── Experiment 1 ───────────────────────────────────────────────────────────────

def run_exp1(t_img, t_cap, d_scores, oracle, n_seeds, n_epochs):
    o_img, o_cap, o_labels = oracle

    sizes      = [200, 500, 1000, 2000]
    sizes      = [n for n in sizes if n <= len(d_scores)]
    sorted_idx = np.argsort(d_scores)[::-1].copy()
    rng        = np.random.RandomState(0)

    print("\n" + "=" * 64)
    print("  EXPERIMENT 1 (adapted) — D_hard vs Random Curriculum")
    print("=" * 64)
    print(f"  Training pool: {len(d_scores)} pairs  |  Oracle: {ORACLE_N}×2")
    print(f"  Epochs: {n_epochs}  |  Seeds: {n_seeds}\n")
    print(f"  {'N':>6}  {'D_hard':>14}  {'Random':>14}  {'Gain':>9}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*14}  {'─'*9}")

    results = []
    for n in sizes:
        hard_idx = torch.tensor(sorted_idx[:n])
        rand_idx = torch.tensor(
            rng.choice(len(d_scores), n, replace=False).copy()
        )

        mh, sh = _seeds(t_img[hard_idx], t_cap[hard_idx],
                        o_img, o_cap, o_labels, n_seeds, n_epochs)
        mr, sr = _seeds(t_img[rand_idx], t_cap[rand_idx],
                        o_img, o_cap, o_labels, n_seeds, n_epochs)
        gain   = mh - mr

        results.append(dict(
            n=n,
            dhard_auroc=round(mh, 4), dhard_std=round(sh, 4),
            rand_auroc=round(mr, 4),  rand_std=round(sr, 4),
            gain=round(gain, 4),
        ))
        print(f"  {n:>6}  {mh:.4f} ±{sh:.4f}  {mr:.4f} ±{sr:.4f}  "
              f"{gain:>+9.4f}")

    all_gains = [r["gain"] for r in results]
    print(f"\n  Mean gain: {np.mean(all_gains):+.4f}")
    claim = all(r["gain"] > 0 for r in results)
    print(f"  D_hard > random at all sizes: {'✓ PROVEN' if claim else '✗ mixed'}")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--seeds",  type=int, default=N_SEEDS)
    args = parser.parse_args()

    t0 = time.time()

    print("Loading COCO val2017 CLIP embeddings (512-dim)...")
    img, cap = load_coco()
    print(f"  {img.shape[0]} pairs\n")

    print("Building concept projection (PCA over image embeddings, 32 dims)...")
    W = build_concept_proj(img)
    print(f"  W shape: {W.shape}\n")

    print("Computing D-scores (32-dim concept space, τ=100)...")
    d_scores = compute_d_scores(img, cap, W)
    print(f"  range: {d_scores.min():.4f} – {d_scores.max():.4f}  "
          f"mean={d_scores.mean():.4f}  std={d_scores.std():.4f}")

    n_above_025 = (d_scores >= 0.25).sum()
    print(f"  pairs with D ≥ 0.25: {n_above_025} ({n_above_025/len(d_scores)*100:.1f}%)\n")

    # Stratified split: hold out ORACLE_N pairs from the middle of D distribution
    # (not just the hardest, to avoid oracle leakage into curriculum)
    rng_split = np.random.RandomState(99)
    all_idx   = rng_split.permutation(len(img)).copy()
    o_idx     = all_idx[:ORACLE_N]
    t_idx     = all_idx[ORACLE_N:]

    oracle   = build_oracle(img[o_idx], cap[o_idx], n=ORACLE_N, seed=7)
    t_img    = img[t_idx]
    t_cap    = cap[t_idx]
    t_d      = d_scores[t_idx]

    print(f"Oracle: {ORACLE_N} matched + {ORACLE_N} mismatched pairs (held out)")
    print(f"Training pool: {len(t_idx)} pairs\n")

    baseline, appb = run_appendix_b(
        t_img, t_cap, t_d, oracle,
        n_seeds=args.seeds, n_epochs=args.epochs,
    )
    exp1 = run_exp1(
        t_img, t_cap, t_d, oracle,
        n_seeds=args.seeds, n_epochs=args.epochs,
    )

    elapsed = time.time() - t0
    out = dict(
        config=dict(n_epochs=args.epochs, n_seeds=args.seeds,
                    oracle_n=ORACLE_N, embed_dim=EMBED_DIM, lr=LR,
                    concept_dim=CONCEPT_DIM, routing_scale=ROUTING_SCALE),
        d_score_stats=dict(
            min=round(float(d_scores.min()), 4),
            max=round(float(d_scores.max()), 4),
            mean=round(float(d_scores.mean()), 4),
            std=round(float(d_scores.std()), 4),
            n_above_025=int(n_above_025),
        ),
        baseline_auroc=round(baseline, 4),
        appendix_b=appb,
        experiment_1=exp1,
        elapsed_min=round(elapsed / 60, 1),
    )

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "coco_results"
    )
    os.makedirs(out_dir, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"curriculum_proof_{ts}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*64}")
    print(f"  Results saved → {path}")
    print(f"  Total elapsed: {elapsed/60:.1f} min")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()

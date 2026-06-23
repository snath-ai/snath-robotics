"""
Classification-trained GRU Encoder — Walker2d Terrain
======================================================
Train GRU with cross-entropy loss on terrain type (normal=0, ice=1).

Why this works for the routing proof:
  - CE trains z_normal to peak at concept dim 0 → softmax ≈ [0.9, 0.1/7, ...]
  - CE trains z_ice    to peak at concept dim 1 → softmax ≈ [0.1/7, 0.9, ...]
  - D(z_normal, z_ice) ≈ 2/sqrt(8) ≈ 0.71  >> delta=0.35  → REPLAN/IMPASSE
  - D(z_vision, z_p_normal) ≈ 0              << delta=0.35  → COMMIT

Terrain labels are used during training only.
At inference the classifier head is discarded; routing fires purely from
concept-space geometry — no terrain labels required.

Run:
    poetry run python experiments/train_cls_walker2d.py
"""
from __future__ import annotations

import sys, math, argparse
from pathlib import Path
from collections import deque

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import gymnasium
from encoders.robotics.gru_proprio_encoder import GRUProprioEncoder

OBS_DIM    = 17
SEQ_LEN    = 30
EMBED_DIM  = 8
HIDDEN_DIM = 64
FRICTION_NORMAL = 0.80
FRICTION_ICE    = 0.05
MODEL_PATH = _ROOT / "models" / "pav" / "gru_cls.pt"


def collect_windows(n: int, friction: float, seed: int, noise_std: float = 0.02):
    env = gymnasium.make("Walker2d-v5")
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    env.unwrapped.model.geom_friction[0, 0] = friction
    buf = deque(maxlen=SEQ_LEN)
    wins = []
    while len(wins) < n:
        buf.append(obs.copy())
        if len(buf) == SEQ_LEN:
            w = np.array(buf)
            if noise_std > 0:
                w = w + rng.normal(0, noise_std, w.shape)
            wins.append(w)
        action = rng.uniform(-0.4, 0.4, size=env.action_space.shape)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
            env.unwrapped.model.geom_friction[0, 0] = friction
            buf.clear()
    env.close()
    return wins


def variance_loss(z: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    return F.relu(torch.tensor(gamma) - z.std(dim=0)).mean()


def main(n_each: int = 50000, epochs: int = 150, batch: int = 512,
         lr: float = 3e-4) -> None:

    print("\n" + "═" * 65)
    print("  Classification GRU Encoder — Walker2d Terrain")
    print(f"  CE loss · hidden={HIDDEN_DIM} · seq_len={SEQ_LEN}")
    print("═" * 65)

    print(f"\n  Collecting {n_each} normal windows …")
    normal_wins = collect_windows(n_each, FRICTION_NORMAL, seed=0)
    print(f"  Collecting {n_each} ice windows …")
    ice_wins    = collect_windows(n_each, FRICTION_ICE, seed=1)

    windows = np.array(normal_wins + ice_wins, dtype=np.float32)
    labels  = np.array([0]*n_each + [1]*n_each, dtype=np.int64)
    N = len(windows)
    print(f"  Total: {N} windows\n")

    encoder = GRUProprioEncoder(OBS_DIM, HIDDEN_DIM, EMBED_DIM, SEQ_LEN)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr/20)

    print(f"  Training {epochs} epochs · batch={batch} · CE + 0.5*VICReg\n")

    for epoch in range(epochs):
        idx = np.random.permutation(N)
        t_ce = t_var = t_acc = 0.0
        steps = 0

        for start in range(0, N - batch, batch):
            bi = idx[start : start + batch]
            w  = torch.from_numpy(windows[bi])
            y  = torch.from_numpy(labels[bi])

            z    = encoder(w)                           # (B, EMBED_DIM)
            ce   = F.cross_entropy(z, y)
            vl   = variance_loss(z)
            loss = ce + 0.5 * vl

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()

            t_ce  += ce.item()
            t_var += vl.item()
            t_acc += (z.argmax(dim=1) == y).float().mean().item()
            steps += 1

        scheduler.step()
        if (epoch + 1) % 30 == 0:
            print(f"  epoch {epoch+1:3d}  ce={t_ce/steps:.4f}  "
                  f"var={t_var/steps:.4f}  acc={t_acc/steps:.3f}")

    # ── Verification ─────────────────────────────────────────────────────────
    encoder.eval()
    with torch.no_grad():
        zn = encoder(torch.from_numpy(np.array(normal_wins[:300], dtype=np.float32)))
        zi = encoder(torch.from_numpy(np.array(ice_wins[:300],    dtype=np.float32)))

    def cdist(a, b):
        cos = F.cosine_similarity(
            a.unsqueeze(1).expand(-1, len(b), -1).reshape(-1, EMBED_DIM),
            b.unsqueeze(0).expand(len(a), -1, -1).reshape(-1, EMBED_DIM), dim=-1)
        return float(1 - cos.mean())

    def mean_conf(z):
        p = F.softmax(z, dim=-1)
        return float(((p.max(dim=-1).values - 1/EMBED_DIM) / (1 - 1/EMBED_DIM)).clamp(0).mean())

    def router_D(z_v, zs):
        return float(np.mean([
            float(((F.softmax(z_v, dim=0) - F.softmax(z, dim=0)).abs().sum()
                   / math.sqrt(EMBED_DIM)))
            for z in zs[:100]
        ]))

    wn = cdist(zn, zn); wi = cdist(zi, zi); bw = cdist(zn, zi)
    mn = F.normalize(zn.mean(0), dim=0); mi = F.normalize(zi.mean(0), dim=0)
    angle = math.degrees(math.acos(
        float(F.cosine_similarity(mn.unsqueeze(0), mi.unsqueeze(0)).clamp(-1, 1))
    ))

    # z_vision = last z_p from warmup (single vector, as used in proof)
    z_v = zn[-1].clone()
    conf_v = float((F.softmax(z_v, dim=0).max() - 1/EMBED_DIM) / (1 - 1/EMBED_DIM))
    dn = router_D(z_v, zn); di = router_D(z_v, zi)

    stats = {
        "within_normal": round(wn, 4), "within_ice": round(wi, 4),
        "between": round(bw, 4), "separation": round(bw - max(wn, wi), 4),
        "centroid_angle": round(angle, 1),
        "conf_normal": round(mean_conf(zn), 4), "conf_ice": round(mean_conf(zi), 4),
        "zv_conf": round(conf_v, 4),
        "router_D_normal": round(dn, 4), "router_D_ice": round(di, 4),
    }

    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  CLS GRU Encoder — Verification                      │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  Within normal:  {wn:.4f}   (want < 0.20)              │")
    print(f"  │  Within ice:     {wi:.4f}   (want < 0.20)              │")
    print(f"  │  Between:        {bw:.4f}                              │")
    print(f"  │  Centroid angle: {angle:.1f}°    (want > 80°)             │")
    print(f"  │  Conf normal:    {mean_conf(zn):.4f}   (want > 0.40)              │")
    print(f"  │  Conf ice:       {mean_conf(zi):.4f}   (want > 0.40)              │")
    print(f"  │  z_vision conf:  {conf_v:.4f}   (want > 0.25)              │")
    print(f"  │  D normal:       {dn:.4f}   (want < 0.25)              │")
    print(f"  │  D ice:          {di:.4f}   (want > 0.40)              │")
    print(f"  │  D gap:          {di-dn:+.4f}   (want > 0.20)             │")
    print(f"  └─────────────────────────────────────────────────────┘")

    ok = conf_v > 0.25 and (di - dn) > 0.15 and dn < 0.30
    print(f"\n  {'✓ Ready for proof' if ok else '✗ Not ready — see gaps above'}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        "encoder_state": encoder.state_dict(),
        "obs_dim": OBS_DIM, "hidden_dim": HIDDEN_DIM,
        "embed_dim": EMBED_DIM, "seq_len": SEQ_LEN,
        "stats": stats,
    }, MODEL_PATH)
    print(f"  Saved → {MODEL_PATH.relative_to(_ROOT)}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",      type=int,   default=50000)
    ap.add_argument("--epochs", type=int,   default=150)
    ap.add_argument("--batch",  type=int,   default=512)
    ap.add_argument("--lr",     type=float, default=3e-4)
    a = ap.parse_args()
    main(a.n, a.epochs, a.batch, a.lr)
